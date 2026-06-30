"""
Model Fitting, Evaluation, CV, Forecast

All functions operate on the data dict returned by preprocess_store.
No function accepts or surfaces a store_id argument.
Artifact paths are suffix-free: models/model.pkl, app/outputs/pca_scaler.pkl, etc.

Functions:
    fit_single          — fit SARIMA for single / level_shift model_type
    fit_two_model       — fit pre/post-break models for two_model model_type
    diagnose_residuals  — Ljung-Box + normality checks on model residuals
    evaluate            — wMAPE, SMAPE, DirectionalAccuracy on test set
    naive_benchmark     — seasonal naive baseline (lag-52 on log scale)
    walk_forward_cv     — expanding-window cross-validation
    generate_forecast   — refit on full series, produce 12-week forecast
"""

import copy
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm

from utils.preprocessing import (
    build_exog_matrix,
    inverse_log_transform,
    MACRO_COLS,
)

warnings.filterwarnings('ignore')

_MODEL_PATH      = 'models/model.pkl'
_MODEL_POST_PATH = 'models/model_post.pkl'
_SCALER_PATH     = 'app/outputs/pca_scaler.pkl'
_PCA_PATH        = 'app/outputs/pca_model.pkl'


# ── Fit: single / level_shift ─────────────────────────────────────────────────

def fit_single(data: dict):
    """
    Fit one SARIMA model using pmdarima.auto_arima.
    Used for model_type in {'single', 'level_shift'}.

    d is fixed from config (EDA-validated; no unit-root test at fit time).
    m=52 for all stores confirmed from EDA periodogram.
    max_p, max_q, max_P, max_Q bounds from config.
    information_criterion='bic' — penalises over-parameterisation at m=52.
    stepwise=True — exhaustive search infeasible at m=52 parameter space.
    """
    cfg        = data['cfg']
    train_y    = data['train_y']
    train_exog = data['train_exog']

    model = pm.auto_arima(
        train_y,
        exogenous=train_exog,
        d=cfg['d'],
        D=cfg.get('D', 1),
        m=cfg['m'],
        max_p=cfg.get('max_p', 3),
        max_q=cfg.get('max_q', 3),
        max_P=cfg.get('max_P', 2),
        max_Q=cfg.get('max_Q', 1),
        seasonal=True,
        stepwise=True,
        information_criterion='bic',
        error_action='ignore',
        suppress_warnings=True,
        with_intercept=cfg.get('with_intercept', True),
    )

    print(f"  Fitted SARIMA{model.order}x{model.seasonal_order}")
    return model


# ── Fit: two_model ────────────────────────────────────────────────────────────

def fit_two_model(data: dict) -> tuple:
    """
    Fit separate pre-break and post-break SARIMA models.
    The post-break model is the production model used for inference.
    The pre-break model is saved for audit only.

    Series is split at break_idx; no level_shift column is included
    in exog (the split itself handles the regime change).

    Returns:
        (pre_model, post_model)
    """
    cfg       = data['cfg']
    log_sales = data['log_sales']
    store_df  = data['store_df']
    scaler    = data['scaler']
    pca       = data['pca']
    break_idx = data['break_idx']

    pre_y    = log_sales.iloc[:break_idx]
    post_y   = log_sales.iloc[break_idx:]
    pre_idx  = log_sales.index[:break_idx]
    post_idx = log_sales.index[break_idx:]

    pre_exog  = build_exog_matrix(pre_idx,  store_df, scaler, pca)
    post_exog = build_exog_matrix(post_idx, store_df, scaler, pca)

    # Pre-break: audit only — no level_shift, no d override
    pre_model = pm.auto_arima(
        pre_y,
        exogenous=pre_exog,
        d=cfg['d'],
        D=cfg.get('D', 1),
        m=cfg['m'],
        max_p=cfg.get('max_p', 3),
        max_q=cfg.get('max_q', 3),
        max_P=cfg.get('max_P', 2),
        max_Q=cfg.get('max_Q', 1),
        seasonal=True,
        stepwise=True,
        information_criterion='bic',
        error_action='ignore',
        suppress_warnings=True,
        with_intercept=cfg.get('with_intercept', True),
    )
    print(f"  Pre-break  SARIMA{pre_model.order}x{pre_model.seasonal_order}")

    # Post-break: test split carved from post_y (last forecast_horizon weeks)
    horizon       = cfg['forecast_horizon']
    train_post_y  = post_y.iloc[:-horizon]
    train_post_ex = post_exog.iloc[:-horizon]

    post_model = pm.auto_arima(
        train_post_y,
        exogenous=train_post_ex,
        d=cfg['d'],
        D=cfg.get('D', 1),
        m=cfg['m'],
        max_p=cfg.get('max_p', 3),
        max_q=cfg.get('max_q', 3),
        max_P=cfg.get('max_P', 2),
        max_Q=cfg.get('max_Q', 1),
        seasonal=True,
        stepwise=True,
        information_criterion='bic',
        error_action='ignore',
        suppress_warnings=True,
        with_intercept=cfg.get('with_intercept', True),
    )
    print(f"  Post-break SARIMA{post_model.order}x{post_model.seasonal_order}")

    return pre_model, post_model


# ── Residual Diagnostics ──────────────────────────────────────────────────────

def diagnose_residuals(model, label: str = '') -> dict:
    """
    Ljung-Box (lags=10) and Shapiro-Wilk normality test on model residuals.

    lb_pass=True  : no residual autocorrelation at alpha=0.05 (all lags p > 0.05)
    sw_pass=True  : residuals approximately normal at alpha=0.05

    lb_pass is the remediation trigger in train_sarima.py.
    sw_pass is informational — SARIMA is robust to mild non-normality.

    label is an optional string for print output (e.g. 'post-break', '[remediated]').
    No store id is printed.
    """
    resid  = model.resid()
    lb_res = acorr_ljungbox(resid, lags=10, return_df=True)
    lb_p   = lb_res['lb_pvalue'].min()
    lb_pass = bool(lb_p > 0.05)

    _, sw_p  = stats.shapiro(resid)
    sw_pass  = bool(sw_p > 0.05)

    tag = f' {label}' if label else ''
    print(f"  Diagnostics{tag}: LB_min_p={lb_p:.4f} ({'PASS' if lb_pass else 'FAIL'})  "
          f"SW_p={sw_p:.4f} ({'PASS' if sw_pass else 'FAIL'})") 

    return {
        'lb_pass':  lb_pass,
        'lb_min_p': round(float(lb_p), 4),
        'sw_pass':  sw_pass,
        'sw_p':     round(float(sw_p), 4),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(data: dict, model) -> dict:
    """
    Computes all evaluation metrics on the original (expm1) scale.

    For two_model stores: test split is within the post-break segment.
    The test set covers the last 12 weeks of the post-break data.
    test_exog is rebuilt without level_shift (consistent with fit_two_model).

    Metrics:
        wMAPE  — primary; weights errors by actual magnitude; robust to high CV
        SMAPE  — symmetric; bounded [0,2]; handles near-zero actuals
        MAPE   — reference only; usable here because CV <= 0.30
        MAE    — interpretable in original sales units ($)
        RMSE   — penalizes large errors; more sensitive to outliers than MAE
        DA     — Directional Accuracy; fraction of weeks where direction matches actual
    """
    cfg        = data['cfg']
    test_weeks = cfg['forecast_horizon']

    if cfg['model_type'] == 'two_model':
        break_idx  = data['break_idx']
        post_full  = data['log_sales'].iloc[break_idx:]
        test_y_log = post_full.iloc[-test_weeks:]
        test_exog  = build_exog_matrix(
            test_y_log.index,
            data['store_df'],
            data['scaler'],
            data['pca'],
        )
        # Reference point for directional accuracy: last post-break training week
        y_prev = float(np.expm1(post_full.iloc[-(test_weeks + 1)]))
    else:
        test_y_log = data['test_y']
        test_exog  = data['test_exog']
        y_prev     = float(np.expm1(data['train_y'].iloc[-1]))

    pred_log, ci = model.predict(
        n_periods=len(test_y_log),
        exogenous=test_exog,
        return_conf_int=True,
        alpha=0.05,
    )

    y_true = np.expm1(test_y_log.values)
    y_pred = np.expm1(pred_log)
    eps    = 1e-8

    wmape = float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))
    smape = float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)))
    mape  = float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))))
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    actual_dir = np.sign(np.diff(np.concatenate([[y_prev], y_true])))
    pred_dir   = np.sign(np.diff(np.concatenate([[y_prev], y_pred])))
    da         = float(np.mean(actual_dir == pred_dir))

    return {
        'wMAPE':               round(wmape, 4),
        'SMAPE':               round(smape, 4),
        'MAPE':                round(mape,  4),
        'MAE':                 round(mae,   2),
        'RMSE':                round(rmse,  2),
        'DirectionalAccuracy': round(da,    4),
    }


# ── Naive Benchmark ───────────────────────────────────────────────────────────

def naive_benchmark(data: dict) -> np.ndarray:
    """
    Seasonal naive benchmark: forecast = value at same week one year ago (lag-52).
    Computed on original scale. Used to compute naive_wMAPE in train_sarima.py.

    If the series is shorter than 52 + forecast_horizon, falls back to
    last observed value (non-seasonal naive).
    """
    cfg      = data['cfg']
    raw      = data['raw_sales']
    horizon  = cfg['forecast_horizon']
    n        = len(raw)

    if n >= 52 + horizon:
        naive = raw.iloc[-(52 + horizon):-52].values
    else:
        naive = np.full(horizon, raw.iloc[-horizon - 1])

    return naive


# ── Walk-Forward Cross-Validation ─────────────────────────────────────────────

def walk_forward_cv(data: dict, model, max_splits: int = 4) -> dict:
    """
    Expanding-window cross-validation on the training set.
    pmdarima.cross_val_score does not accept exogenous — manual loop used.

    Each fold:
        - Fit on train[:cutoff]
        - Predict on train[cutoff:cutoff+horizon]
        - Compute wMAPE on original scale

    Returns mean and std of wMAPE across folds.

    n_splits is computed dynamically: largest k such that the initial
    training window is at least one full seasonal cycle (m weeks), capped
    at max_splits. This prevents the gate from silently skipping CV on
    series that are long enough to fold but not long enough for 2*m.

    Gate: min_train >= m  (one seasonal cycle is the minimum for SARIMA fit).
    The old 2*m gate was overcautious and caused CV to skip on store 14.
    """
    import math
    cfg        = data['cfg']
    train_y    = data['train_y']
    train_exog = data['train_exog']
    horizon    = cfg['forecast_horizon']
    m          = cfg['m']

    n_train    = len(train_y)
    # Maximum folds such that the first training window >= m
    max_feasible = math.floor((n_train - m) / horizon)
    n_splits     = min(max_splits, max_feasible)

    if n_splits < 1:
        print(f"  CV skipped: insufficient history "
              f"(n_train={n_train}, m={m}, horizon={horizon})")
        return {'cv_wMAPE_mean': None, 'cv_wMAPE_std': None}

    min_train = n_train - n_splits * horizon

    fold_wmapes = []

    for i in range(n_splits):
        cutoff       = min_train + i * horizon
        fold_train_y = train_y.iloc[:cutoff]
        fold_train_x = train_exog.iloc[:cutoff]
        fold_test_y  = train_y.iloc[cutoff:cutoff + horizon]
        fold_test_x  = train_exog.iloc[cutoff:cutoff + horizon]

        if len(fold_test_y) == 0:
            continue

        try:
            fold_model = pm.auto_arima(
                fold_train_y,
                exogenous=fold_train_x,
                d=cfg['d'],
                D=cfg.get('D', 1),
                m=cfg['m'],
                max_p=cfg.get('max_p', 3),
                max_q=cfg.get('max_q', 3),
                max_P=cfg.get('max_P', 2),
                max_Q=cfg.get('max_Q', 1),
                seasonal=True,
                stepwise=True,
                information_criterion='bic',
                error_action='ignore',
                suppress_warnings=True,
                with_intercept=cfg.get('with_intercept', True),
            )
            log_pred = fold_model.predict(n_periods=len(fold_test_y),
                                          exogenous=fold_test_x)
            pred  = inverse_log_transform(log_pred)
            true  = inverse_log_transform(fold_test_y.values)
            wmape = float(np.sum(np.abs(true - pred)) / (np.sum(np.abs(true)) + 1e-8))
            fold_wmapes.append(wmape)
        except Exception as e:
            print(f"  CV fold {i+1} failed: {e}")

    if not fold_wmapes:
        return {'cv_wMAPE_mean': None, 'cv_wMAPE_std': None}

    mean_w = round(float(np.mean(fold_wmapes)), 4)
    std_w  = round(float(np.std(fold_wmapes)),  4)
    print(f"  CV wMAPE: {mean_w:.4f} ± {std_w:.4f} over {len(fold_wmapes)} folds")

    return {'cv_wMAPE_mean': mean_w, 'cv_wMAPE_std': std_w}


# ── Production Forecast ───────────────────────────────────────────────────────

def generate_forecast(model, data: dict) -> dict:
    """
    Refit the model on the full series (train + test), then predict 12 weeks ahead.
    copy.deepcopy applied before update() — model.update() mutates the cached object.

    For two_model stores: refit on the full post-break series.
    For single/level_shift: refit on the full log_sales series.

    Returns dict with keys:
        dates        : list of pd.Timestamp (12 future Fridays)
        forecast     : list[float] on original scale
        lower_95     : list[float] on original scale
        upper_95     : list[float] on original scale
        is_holiday   : list[int] binary
    """
    cfg      = data['cfg']
    horizon  = cfg['forecast_horizon']
    store_df = data['store_df']
    scaler   = data['scaler']
    pca      = data['pca']
    break_idx = data['break_idx']

    # Determine full series for refit
    if cfg['model_type'] == 'two_model':
        full_y = data['log_sales'].iloc[break_idx:]
    else:
        full_y = data['log_sales']

    # Rebuild full exog (no level_shift for two_model)
    include_shift = cfg['model_type'] != 'two_model'
    if include_shift:
        level_shift = data['level_shift']
        full_exog = build_exog_matrix(
            full_y.index, store_df, scaler, pca,
            level_shift_dummy=level_shift[:len(full_y)],
            is_imputed=data['is_imputed'][:len(full_y)]
        )
    else:
        full_exog = build_exog_matrix(full_y.index, store_df, scaler, pca)

    # Refit on full series via update (deepcopy to avoid mutating cached model)
    m = copy.deepcopy(model)
    m.update(full_y, exogenous=full_exog)

    # Build future exog
    last_date    = full_y.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(weeks=1),
        periods=horizon,
        freq='W-FRI'
    )

    # Holiday lookup from store_df (last known year, same week-of-year pattern)
    future_holiday = _build_future_holiday_flag(future_dates, store_df)

    # PC1 for future: repeat last observed macro values (no forward macro data)
    future_store_df = _build_future_store_df(future_dates, store_df, future_holiday)

    if include_shift:
        future_shift = np.ones(horizon, dtype=float) if break_idx is not None else \
                       np.zeros(horizon, dtype=float)
        future_exog = build_exog_matrix(
            future_dates, future_store_df, scaler, pca,
            level_shift_dummy=future_shift
        )
    else:
        future_exog = build_exog_matrix(future_dates, future_store_df, scaler, pca)

    # Predict
    log_pred, conf_int = m.predict(
        n_periods=horizon,
        exogenous=future_exog,
        return_conf_int=True,
        alpha=0.05
    )

    forecast  = inverse_log_transform(log_pred).tolist()
    lower_95  = inverse_log_transform(conf_int[:, 0]).tolist()
    upper_95  = inverse_log_transform(conf_int[:, 1]).tolist()

    return {
        'dates':      list(future_dates),
        'forecast':   [round(v, 2) for v in forecast],
        'lower_95':   [round(v, 2) for v in lower_95],
        'upper_95':   [round(v, 2) for v in upper_95],
        'is_holiday': future_holiday,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_future_holiday_flag(future_dates: pd.DatetimeIndex,
                               store_df: pd.DataFrame) -> list[int]:
    df_indexed = store_df.set_index('Date')
 
    week_holiday: dict[int, int] = {}
    for date, flag in zip(df_indexed.index, df_indexed['Holiday_Flag']):
        week = date.isocalendar()[1]
        week_holiday[week] = max(week_holiday.get(week, 0), int(flag))
 
    return [week_holiday.get(d.isocalendar()[1], 0) for d in future_dates]


def _build_future_store_df(future_dates: pd.DatetimeIndex,
                            store_df: pd.DataFrame,
                            future_holiday: list[int]) -> pd.DataFrame:
    """
    Build a synthetic store DataFrame for future weeks.
    Macro columns filled with last observed values (no forward macro data available).
    Store column is not used downstream — set to a dummy value.
    """
    last_row = store_df.sort_values('Date').iloc[-1]
    rows = []
    for date, hol in zip(future_dates, future_holiday):
        row = {col: last_row[col] for col in MACRO_COLS}
        row['Date']         = date
        row['Holiday_Flag'] = hol
        row['Store']        = 0          # dummy; not used downstream
        row['Weekly_Sales'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ── Artifact Loaders (used by main.py) ───────────────────────────────────────

def load_model(model_type: str = 'single'):
    """
    Load the pre-trained production model artifact.
    model_type is read from metrics.json by main.py — not from user input.

    For two_model: loads model_post.pkl (production model).
    For single/level_shift: loads model.pkl.
    """
    if model_type == 'two_model':
        model = joblib.load(_MODEL_POST_PATH)
        model_copy = copy.deepcopy(model)
        return model_copy
    model = joblib.load(_MODEL_PATH)
    model_copy = copy.deepcopy(model)
    return model_copy


def load_pca_artifacts() -> tuple:
    """Load train-fitted PCA scaler and PCA object for inference."""
    scaler = joblib.load(_SCALER_PATH)
    pca    = joblib.load(_PCA_PATH)
    return scaler, pca
