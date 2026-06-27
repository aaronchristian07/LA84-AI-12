"""
Walmart SARIMA — Offline Training Script

Run once offline. Produces:
    models/store_{id}.pkl             — single / level_shift stores
    models/store_{id}_pre.pkl         — two_model stores (pre-break; audit only)
    models/store_{id}_post.pkl        — two_model stores (post-break; production)
    forecasts/store_{id}_12w.csv      — 12-week ahead forecast per store
    outputs/metrics.json              — evaluation metrics for all stores
    outputs/pca_scaler_{id}.pkl       — written by preprocess_store via fit_pca
    outputs/pca_model_{id}.pkl        — written by preprocess_store via fit_pca

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

# ── Load and parse dataset ────────────────────────────────────────────────────
df = parse_and_sort(pd.read_csv('data/walmart.csv'))

all_metrics = {}

# ── Per-store training loop ───────────────────────────────────────────────────
for store_id in sorted(df['Store'].unique()):
    sid = str(store_id)
    cfg = config[sid]

    print(f"\n{'=' * 60}")
    print(f"Store {store_id}  |  model_type={cfg['model_type']}  "
          f"d={cfg['d']}  m={cfg['m']}")

    # ── Preprocess ────────────────────────────────────────────────────
    data = preprocess_store(df, store_id, config)

    # ── Fit ───────────────────────────────────────────────────────────
    if cfg['model_type'] == 'two_model':
        pre_model, model = fit_two_model(data)
        diag = diagnose_residuals(model, store_id, 'post-break')

        # Auto-remediation: Ljung-Box failure → increase max_p by 1, refit once.
        # Do NOT call preprocess_store again — re-running fit_pca overwrites the
        # scaler on disk and misaligns the exog matrix for test/forecast.
        if not diag['lb_pass']:
            print(f"  Remediating Store {store_id}: max_p "
                  f"{config[sid]['max_p']} → {config[sid]['max_p'] + 1}")
            config[sid]['max_p'] += 1
            data['cfg'] = config[sid]   # update cfg reference in data dict
            _, model = fit_two_model(data)
            diag = diagnose_residuals(model, store_id, 'post-break [remediated]')
            if not diag['lb_pass']:
                print(f"  WARNING Store {store_id}: residual autocorrelation "
                      f"persists after remediation. Consider max_Q=1.")

        joblib.dump(pre_model, f'models/store_{store_id}_pre.pkl')
        joblib.dump(model,     f'models/store_{store_id}_post.pkl')

    else:
        # single or level_shift
        model = fit_single(data)
        diag  = diagnose_residuals(model, store_id)

        # Auto-remediation: increase max_p by 1, refit on same preprocessed data.
        # Do NOT call preprocess_store again — re-running fit_pca overwrites the
        # scaler on disk and misaligns the exog matrix for test/forecast.
        if not diag['lb_pass']:
            print(f"  Remediating Store {store_id}: max_p "
                  f"{config[sid]['max_p']} → {config[sid]['max_p'] + 1}")
            config[sid]['max_p'] += 1
            data['cfg'] = config[sid]   # update cfg reference in data dict
            model = fit_single(data)
            diag  = diagnose_residuals(model, store_id, '[remediated]')
            if not diag['lb_pass']:
                print(f"  WARNING Store {store_id}: residual autocorrelation "
                      f"persists after remediation. Consider max_Q=1.")

        joblib.dump(model, f'models/store_{store_id}.pkl')

    # ── Evaluate on held-out test set ─────────────────────────────────
    metrics     = evaluate(data, model)
    naive_pred  = naive_benchmark(data)

    # Compute naive wMAPE on original scale
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
        print(f"  WARNING Store {store_id}: wMAPE={metrics['wMAPE']:.4f} does not beat "
              f"naive={naive_wmape:.4f}. Apply BIC pass or ensemble fallback.")

    # ── Walk-forward cross-validation ─────────────────────────────────
    cv_results = walk_forward_cv(data, model)

    # ── Production forecast (refit on full series inside generate_forecast) ──
    forecast_dict = generate_forecast(model, data)

    pd.DataFrame({
        'Week':           forecast_dict['dates'],
        'Forecast_Sales': forecast_dict['forecast'],
        'Lower_95':       forecast_dict['lower_95'],
        'Upper_95':       forecast_dict['upper_95'],
        'Is_Holiday':     forecast_dict['is_holiday'],
    }).to_csv(f'forecasts/store_{store_id}_12w.csv', index=False)

    # ── Accumulate metrics ─────────────────────────────────────────────
    all_metrics[sid] = {
        **metrics,
        **cv_results,
        **diag,
        'sarima_order':   list(model.order),
        'seasonal_order': list(model.seasonal_order),
        'model_type':     cfg['model_type'],
        'd':              cfg['d'],
        'm':              cfg['m'],
    }

    print(f"  wMAPE={metrics['wMAPE']:.4f}  "
          f"SMAPE={metrics['SMAPE']:.4f}  "
          f"DirAcc={metrics['DirectionalAccuracy']:.4f}  "
          f"beats_naive={metrics['beats_naive']}")

# ── Write metrics ─────────────────────────────────────────────────────────────
with open('outputs/metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
beat_count  = sum(1 for v in all_metrics.values() if v.get('beats_naive'))
lb_failures = [sid for sid, v in all_metrics.items() if not v.get('lb_pass')]
wmapes      = [v['wMAPE'] for v in all_metrics.values()]

print(f"\n{'=' * 60}")
print(f"Training complete — {len(all_metrics)} stores")
print(f"Beats naive:        {beat_count} / {len(all_metrics)}")
print(f"Mean wMAPE:         {np.mean(wmapes):.4f}")
print(f"wMAPE range:        {np.min(wmapes):.4f} – {np.max(wmapes):.4f}")
print(f"Ljung-Box failures: {lb_failures if lb_failures else 'none'}")
print(f"Metrics written to: outputs/metrics.json")
