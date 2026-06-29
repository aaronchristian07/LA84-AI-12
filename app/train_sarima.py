"""
Run once offline. Produces:
    models/model.pkl              — single / level_shift
    models/model_pre.pkl          — two_model only (pre-break; audit only)
    models/model_post.pkl         — two_model only (post-break; production)
    forecasts/forecast_12w.csv    — 12-week ahead forecast
    outputs/metrics.json          — evaluation metrics
    outputs/pca_scaler.pkl        — written by preprocess_store via fit_pca
    outputs/pca_model.pkl         — written by preprocess_store via fit_pca

Store ID is not referenced anywhere in this script.
model_config.json is read internally by preprocess_store using the fixed key '14'.

Do not run from main.py. main.py loads pre-built artifacts only.

Execution:
    python train_sarima.py

Prerequisites:
    - data/walmart.csv present
    - outputs/model_config.json present (corrected version)
    - outputs/, models/, forecasts/ directories exist
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import parse_and_sort, preprocess_store
from utils.sarima_model import (
    fit_single,
    fit_two_model,
    diagnose_residuals,
    evaluate,
    naive_benchmark,
    walk_forward_cv,
    generate_forecast,
)

# ── Directory guard ───────────────────────────────────────────────────────────
for d in ['models', 'forecasts', 'outputs']:
    os.makedirs(d, exist_ok=True)

# ── Load config ───────────────────────────────────────────────────────────────
with open('outputs/model_config.json') as f:
    config = json.load(f)

cfg = config['14']

# ── Load and parse dataset ────────────────────────────────────────────────────
df = parse_and_sort(pd.read_csv('data/Walmart.csv'))

print(f"{'=' * 60}")
print(f"Training  |  model_type={cfg['model_type']}  d={cfg['d']}  m={cfg['m']}")

# ── Preprocess ────────────────────────────────────────────────────────────────
data = preprocess_store(df, config)

# ── Fit ───────────────────────────────────────────────────────────────────────
if cfg['model_type'] == 'two_model':
    pre_model, model = fit_two_model(data)
    diag = diagnose_residuals(model, label='post-break')

    # Auto-remediation: Ljung-Box failure → increase max_p by 1, refit once.
    # Do NOT call preprocess_store again — re-running fit_pca overwrites the
    # scaler on disk and misaligns the exog matrix for test/forecast.
    if not diag['lb_pass']:
        print(f"  Remediating: max_p {config['14']['max_p']} → {config['14']['max_p'] + 1}")
        config['14']['max_p'] += 1
        data['cfg'] = config['14']
        _, model = fit_two_model(data)
        diag = diagnose_residuals(model, label='post-break [remediated]')
        if not diag['lb_pass']:
            print("  WARNING: residual autocorrelation persists after remediation. "
                  "Consider max_Q=1.")

    joblib.dump(pre_model, 'models/model_pre.pkl')
    joblib.dump(model,     'models/model_post.pkl')

else:
    # single or level_shift
    model = fit_single(data)
    diag  = diagnose_residuals(model)

    # Auto-remediation: increase max_p by 1, refit on same preprocessed data.
    # Do NOT call preprocess_store again — re-running fit_pca overwrites the
    # scaler on disk and misaligns the exog matrix for test/forecast.
    if not diag['lb_pass']:
        print(f"  Remediating: max_p {config['14']['max_p']} → {config['14']['max_p'] + 1}")
        config['14']['max_p'] += 1
        data['cfg'] = config['14']
        model = fit_single(data)
        diag  = diagnose_residuals(model, label='[remediated]')
        if not diag['lb_pass']:
            print("  WARNING: residual autocorrelation persists after remediation. "
                  "Consider max_Q=1.")

    joblib.dump(model, 'models/model.pkl')

# ── Evaluate on held-out test set ─────────────────────────────────────────────
metrics    = evaluate(data, model)
naive_pred = naive_benchmark(data)

if cfg['model_type'] == 'two_model':
    test_true = np.expm1(
        data['log_sales'].iloc[data['break_idx']:].iloc[-cfg['forecast_horizon']:].values
    )
else:
    test_true = np.expm1(data['test_y'].values)

naive_wmape = float(
    np.sum(np.abs(test_true - naive_pred)) / (np.sum(np.abs(test_true)) + 1e-8)
)
metrics['naive_wMAPE'] = round(naive_wmape, 4)
metrics['beats_naive'] = metrics['wMAPE'] < naive_wmape

if not metrics['beats_naive']:
    print(f"  WARNING: wMAPE={metrics['wMAPE']:.4f} does not beat "
          f"naive={naive_wmape:.4f}. Apply BIC pass or ensemble fallback.")

# ── Walk-forward cross-validation ─────────────────────────────────────────────
cv_results = walk_forward_cv(data, model)

# ── Production forecast ───────────────────────────────────────────────────────
forecast_dict = generate_forecast(model, data)

pd.DataFrame({
    'Week':           forecast_dict['dates'],
    'Forecast_Sales': forecast_dict['forecast'],
    'Lower_95':       forecast_dict['lower_95'],
    'Upper_95':       forecast_dict['upper_95'],
    'Is_Holiday':     forecast_dict['is_holiday'],
}).to_csv('forecasts/forecast_12w.csv', index=False)

# ── Write metrics ─────────────────────────────────────────────────────────────
all_metrics = {
    **metrics,
    **cv_results,
    **diag,
    'sarima_order':   list(model.order),
    'seasonal_order': list(model.seasonal_order),
    'model_type':     cfg['model_type'],
    'd':              cfg['d'],
    'm':              cfg['m'],
}

with open('outputs/metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"Training complete")
print(f"wMAPE:        {metrics['wMAPE']:.4f}")
print(f"SMAPE:        {metrics['SMAPE']:.4f}")
print(f"DirAcc:       {metrics['DirectionalAccuracy']:.4f}")
print(f"Beats naive:  {metrics['beats_naive']}")
print(f"LB pass:      {diag['lb_pass']}")
print(f"Forecast:     forecasts/forecast_12w.csv")
print(f"Metrics:      outputs/metrics.json")
