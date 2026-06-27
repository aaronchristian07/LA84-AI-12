"""
utils/sarima_model.py
Walmart SARIMA — Model Fitting, Evaluation, Forecasting

Functions:
    fit_single          — SARIMAX for model_type = single / level_shift
    fit_two_model       — two SARIMAX fits split at break_idx (model_type = two_model)
    diagnose_residuals  — Ljung-Box + Shapiro-Wilk on model residuals
    evaluate            — held-out test metrics (wMAPE, SMAPE, MAPE, MAE, RMSE, DA)
    naive_benchmark     — seasonal naive (lag-52) as accuracy floor
    walk_forward_cv     — expanding-window cross-validation
    generate_forecast   — production 12-week ahead forecast (refits on full series)
"""

import numpy as np
import pandas as pd
import holidays as hol_lib
from pmdarima import auto_arima
# RollingForecastCV / cross_val_score removed — exogenous not supported by
# pmdarima's cross_val_score. Walk-forward CV implemented manually in walk_forward_cv().
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

from utils.preprocessing import (
    build_exog_matrix,
    apply_pca,
    MACRO_COLS,
)


# ── Helper: auto_arima call with locked config ───────────────────────────────

def _fit_arima(y: pd.Series,
               exog: pd.DataFrame,
               cfg: dict,
               label: str,
               store_id: int,
               ic: str = 'aic') -> object:
    """
    Internal helper. Runs auto_arima with config-locked bounds.
    ic: 'aic' (default) or 'bic' (used in Priority 2 accuracy pass).
    """
    model = auto_arima(
        y,
        exogenous=exog,
        d=cfg['d'],
        D=0,                        # unconditional: D=1 at m=52 infeasible at n~143
        m=cfg['m'],
        max_p=cfg['max_p'],
        max_q=cfg['max_q'],
        max_P=cfg['max_P'],
        max_Q=cfg['max_Q'],
        start_p=0, start_q=0,
        start_P=0, start_Q=0,
        information_criterion=ic,
        seasonal=True,
        stepwise=True,
        error_action='ignore',
        suppress_warnings=True,
        out_of_sample_size=0,
    )
    print(f"  Store {store_id} [{label}] — "
          f"order={model.order}  seasonal={model.seasonal_order}  "
          f"AIC={model.aic():.2f}  BIC={model.bic():.2f}")
    return model


# ── 4.1 Fit Single / Level-Shift ─────────────────────────────────────────────

def fit_single(data: dict, ic: str = 'aic') -> object:
    """
    Used for model_type = 'single' and 'level_shift'.

    single stores: level_shift column in train_exog absorbs sub-threshold
    CUSUM breaks without requiring a series split.

    level_shift stores (Store 18): same logic; level_shift column carries the
    break signal. Store 18 was downgraded from two_model because post_n=62
    equals the feasibility boundary m+10=62 (strict inequality failed).

    d and m from config — EDA-corrected values. Do not override.
    D=0 unconditional for all stores.
    """
    return _fit_arima(
        data['train_y'],
        data['train_exog'],
        data['cfg'],
        data['cfg']['model_type'],
        data['store_id'],
        ic=ic,
    )


# ── 4.2 Fit Two-Model ────────────────────────────────────────────────────────

def fit_two_model(data: dict, ic: str = 'aic') -> tuple[object, object]:
    """
    Used for model_type = 'two_model'.
    Stores: 13, 15, 19, 21, 32, 34, 39, 41 (break_idx=45, post_n=98).

    Splits log_sales at break_idx. Fits independent SARIMAX on each segment.
    Returns (pre_model, post_model).
    Only post_model is used for forecasting — it reflects the current sales regime.
    pre_model is saved to disk for audit only.

    The level_shift column is NOT included in exog for two_model stores —
    the series split itself handles the regime change. Including it would
    introduce a constant=1 exog column across the entire post-break segment,
    which is collinear with the intercept.
    """
    cfg       = data['cfg']
    break_idx = data['break_idx']
    log_sales = data['log_sales']
    store_df  = data['store_df']
    scaler    = data['scaler']
    pca       = data['pca']
    test_weeks = cfg['forecast_horizon']

    # Pre-break segment: weeks [0, break_idx)
    pre_idx  = log_sales.index[:break_idx]
    pre_y    = log_sales.iloc[:break_idx]
    pre_exog = build_exog_matrix(pre_idx, store_df, scaler, pca)
    # No level_shift column for two_model

    # Post-break segment: weeks [break_idx, n)
    post_full_idx  = log_sales.index[break_idx:]
    post_y_full    = log_sales.iloc[break_idx:]
    post_exog_full = build_exog_matrix(post_full_idx, store_df, scaler, pca)

    # Training portion of post-break (hold out last test_weeks for evaluation)
    post_train_y    = post_y_full.iloc[:-test_weeks]
    post_train_exog = post_exog_full.iloc[:-test_weeks]

    pre_model  = _fit_arima(pre_y,  pre_exog,  cfg, 'pre-break',  data['store_id'], ic)
    post_model = _fit_arima(post_train_y, post_train_exog, cfg, 'post-break', data['store_id'], ic)

    return pre_model, post_model


# ── 4.3 Residual Diagnostics ─────────────────────────────────────────────────

def diagnose_residuals(model, store_id: int, label: str = '') -> dict:
    """
    Ljung-Box test (lag=10): H0 = residuals are white noise.
        p > 0.05 → PASS — no remaining autocorrelation structure
        p < 0.05 → FAIL — model has left extractable signal unfitted;
                   increase max_p by 1 and refit (handled in train_sarima.py)

    Shapiro-Wilk: H0 = residuals are normally distributed.
        p < 0.05 → WARN — non-normal residuals; 95% CI bands are approximate.
        Does not invalidate the model; SARIMA is robust to mild non-normality.
        Log1p transform reduces but does not always eliminate non-normality.

    resid_mean should be ~0. Systematic bias indicates a missing trend term.
    """
    resid  = model.resid()
    lb_p   = float(acorr_ljungbox(resid, lags=[10], return_df=True)['lb_pvalue'].iloc[0])
    sw_p   = float(shapiro(resid)[1])
    passed = lb_p > 0.05

    tag = f'[{label}]' if label else ''
    print(f"  Store {store_id} {tag} — "
          f"Ljung-Box p={lb_p:.4f} {'PASS' if passed else 'FAIL'}  "
          f"Shapiro p={sw_p:.4f}  "
          f"resid_mean={resid.mean():.5f}  resid_std={resid.std():.5f}")

    if not passed:
        print(f"    ACTION: increase max_p by 1 in config and refit.")

    return {
        'lb_p':       lb_p,
        'sw_p':       sw_p,
        'lb_pass':    passed,
        'resid_mean': float(resid.mean()),
        'resid_std':  float(resid.std()),
    }


# ── 4.4 Evaluate on Held-Out Test Set ────────────────────────────────────────

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


# ── 4.5 Seasonal Naive Benchmark ─────────────────────────────────────────────

def naive_benchmark(data: dict) -> np.ndarray:
    """
    Seasonal naive: predict next N weeks = same N weeks from prior year (lag-52).
    SARIMAX must beat this benchmark. If it does not, model adds no value above
    a trivial baseline — apply Priority 2 (BIC) or Priority 4 (ensemble).

    For two_model stores: use only the post-break raw series to avoid
    the pre-break regime contaminating the benchmark.

    Fallback to last-value naive if training segment < 52 weeks.
    """
    cfg        = data['cfg']
    test_weeks = cfg['forecast_horizon']

    if cfg['model_type'] == 'two_model':
        raw = data['raw_sales'].iloc[data['break_idx']:]
    else:
        raw = data['raw_sales']

    train_raw = raw.iloc[:-test_weeks]

    if len(train_raw) >= 52:
        return train_raw.iloc[-52: -52 + test_weeks].values
    # Fallback: last known value repeated
    return np.full(test_weeks, float(train_raw.iloc[-1]))


# ── 4.6 Walk-Forward Cross-Validation ────────────────────────────────────────

def walk_forward_cv(data: dict, model) -> dict:
    """
    Manual expanding-window walk-forward CV with exogenous support.

    pmdarima's cross_val_score does not accept an 'exogenous' argument —
    it is implemented here as a manual loop instead.

    Strategy:
        - Fit auto_arima on the initial window (80% of training segment)
        - At each step, call model.update() to extend the window by `step` weeks
        - Predict h=12 steps ahead using the corresponding exog slice
        - Record MAE for each fold
        - step=4 (monthly evaluation); h=12 (matches forecast horizon)

    For two_model stores: CV on post-break training segment only.
    """
    cfg        = data['cfg']
    test_weeks = cfg['forecast_horizon']
    h          = 12
    step       = 4

    if cfg['model_type'] == 'two_model':
        break_idx = data['break_idx']
        post_full = data['log_sales'].iloc[break_idx:]
        train_y   = post_full.iloc[:-test_weeks]
        train_x   = build_exog_matrix(
            train_y.index,
            data['store_df'],
            data['scaler'],
            data['pca'],
        )
    else:
        train_y = data['train_y']
        train_x = data['train_exog']

    n       = len(train_y)
    initial = int(0.80 * n)

    # Fit on initial window
    cv_model = auto_arima(
        train_y.iloc[:initial],
        exogenous=train_x.iloc[:initial],
        d=cfg['d'], D=0, m=cfg['m'],
        max_p=cfg['max_p'], max_q=cfg['max_q'],
        max_P=cfg['max_P'], max_Q=cfg['max_Q'],
        seasonal=True, stepwise=True,
        error_action='ignore', suppress_warnings=True,
    )

    mae_scores = []
    cursor     = initial

    while cursor + h <= n:
        # Predict h steps from current cursor using next h exog rows
        fold_exog = train_x.iloc[cursor: cursor + h]
        try:
            pred_log = cv_model.predict(n_periods=h, exogenous=fold_exog)
            actual   = train_y.iloc[cursor: cursor + h].values
            # Metrics on log scale (MAE on log scale is consistent across folds)
            mae = float(np.mean(np.abs(actual - pred_log)))
            mae_scores.append(mae)
        except Exception:
            mae_scores.append(np.nan)

        # Extend model window by `step` weeks
        update_y    = train_y.iloc[cursor: cursor + step]
        update_exog = train_x.iloc[cursor: cursor + step]
        try:
            cv_model.update(update_y, exogenous=update_exog)
        except Exception:
            pass

        cursor += step

    valid   = [s for s in mae_scores if not np.isnan(s)]
    cv_mean = float(np.mean(valid)) if valid else float('nan')
    cv_std  = float(np.std(valid))  if valid else float('nan')

    print(f"  Store {data['store_id']} CV — "
          f"log-MAE {cv_mean:.4f} ± {cv_std:.4f}  ({len(valid)} folds)")

    return {
        'cv_mae_mean': round(cv_mean, 4),
        'cv_mae_std':  round(cv_std,  4),
        'n_folds':     int(len(valid)),
    }


# ── 4.7 Generate Production Forecast ─────────────────────────────────────────

def generate_forecast(model, data: dict) -> dict:
    """
    Refits the model on the full available series, then forecasts n_periods ahead.

    Refit scope by model_type:
        single / level_shift : all 143 weeks
        two_model            : post-break weeks only (break_idx : 143)
                               Pre-break regime is not representative of current
                               sales behaviour and must not contaminate the refit.

    forecast_exog construction:
        PC1_macro    — forward-fill last known macro row for all future weeks.
                       Macro variables have low weekly volatility; 12-week ffill
                       is defensible. Do not extrapolate or model macro separately.
        Holiday_Flag — computed from `holidays` library on actual future dates.
                       Do not shift existing Holiday_Flag values — that would
                       assign last year's holiday pattern to future weeks.
        level_shift  — 1.0 for all future weeks for single/level_shift stores
                       (post-break regime is assumed to persist into forecast).
                       Excluded for two_model stores (no dummy used).

    Column order in forecast_exog must exactly match refit_exog.
    Mismatch causes ValueError at model.predict() with no informative message.
    """
    cfg        = data['cfg']
    n_periods  = cfg['forecast_horizon']
    model_type = cfg['model_type']
    store_id   = data['store_id']

    # ── Select refit series and exog ────────────────────────────────
    if model_type == 'two_model':
        refit_y    = data['log_sales'].iloc[data['break_idx']:]
        refit_exog = build_exog_matrix(
            refit_y.index,
            data['store_df'],
            data['scaler'],
            data['pca'],
        )
        # No level_shift column for two_model
    else:
        refit_y     = data['log_sales']
        refit_exog  = build_exog_matrix(
            refit_y.index,
            data['store_df'],
            data['scaler'],
            data['pca'],
            level_shift_dummy=data['level_shift'],
        )

    # ── Refit on full series ─────────────────────────────────────────
    model.update(refit_y, exogenous=refit_exog)

    # ── Build future date index ──────────────────────────────────────
    last_date  = refit_y.index[-1]
    future_idx = pd.date_range(
        last_date + pd.Timedelta(weeks=1),
        periods=n_periods,
        freq='W-FRI',
    )

    # ── Holiday flags for future weeks ───────────────────────────────
    us_hols = hol_lib.US(
        years=range(future_idx.min().year, future_idx.max().year + 1)
    )
    future_holiday = np.array([
        int(any((d + pd.Timedelta(days=k)) in us_hols for k in range(7)))
        for d in future_idx
    ], dtype=float)

    # ── PC1 for future weeks (forward-fill last known macro) ─────────
    last_macro = (
        data['store_df']
        .set_index('Date')[MACRO_COLS]
        .sort_index()
        .iloc[-1:]
    )
    future_macro = pd.DataFrame(
        np.tile(last_macro.values, (n_periods, 1)),
        columns=MACRO_COLS,
        index=future_idx,
    )
    future_pc1 = apply_pca(future_macro, data['scaler'], data['pca'])

    # ── Assemble forecast_exog (column order = refit_exog order) ────
    forecast_exog = pd.DataFrame(
        {'PC1_macro': future_pc1, 'Holiday_Flag': future_holiday},
        index=future_idx,
    )
    if model_type != 'two_model' and data['break_idx'] is not None:
        # Post-break regime continues into forecast window
        forecast_exog['level_shift'] = 1.0

    # ── Produce forecast ─────────────────────────────────────────────
    forecast_log, ci = model.predict(
        n_periods=n_periods,
        exogenous=forecast_exog,
        return_conf_int=True,
        alpha=0.05,
    )

    print(f"  Store {store_id} forecast — "
          f"weeks {future_idx[0].date()} → {future_idx[-1].date()}")

    return {
        'dates':      future_idx,
        'forecast':   np.expm1(forecast_log),
        'lower_95':   np.expm1(ci[:, 0]),
        'upper_95':   np.expm1(ci[:, 1]),
        'is_holiday': future_holiday.astype(int),
    }
