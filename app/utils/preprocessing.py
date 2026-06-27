"""
utils/preprocessing.py
Walmart SARIMA — Preprocessing Pipeline

Order of operations (strict — do not reorder):
    1. parse_and_sort
    2. fill_date_index       (defensive; gap_report confirmed zero gaps)
    3. winsorize_series      (k=3.0 for all stores; CV <= 0.30 confirmed)
    4. log_transform
    5. make_level_shift_dummy
    6. fit_pca               (train split only — no leakage)
    7. build_exog_matrix
    8. temporal_split
    9. preprocess_store      (orchestrates 1–8 for one store)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MACRO_COLS = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']


# ── 1. Parse and Sort ────────────────────────────────────────────────────────

def parse_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse dates (Walmart format: DD-MM-YYYY) and sort by Store, Date.
    Raises on any unparseable date — fix upstream, do not silently drop.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    n_nat = df['Date'].isna().sum()
    if n_nat > 0:
        raise ValueError(f"{n_nat} unparseable dates in Date column. Fix before preprocessing.")
    return df.sort_values(['Store', 'Date']).reset_index(drop=True)


# ── 2. Complete Weekly Index ─────────────────────────────────────────────────

def fill_date_index(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Reindex to a complete W-FRI DatetimeIndex.
    gap_report.csv confirmed zero missing weeks for all 45 stores.
    Retained as a defensive measure against future data updates.

    bfill is forbidden: at the train/test boundary it fills missing weeks
    with the next observation, which may be in the test set — data leakage.
    Strategy: ffill(limit=2) → linear interpolation(limit=4) → raise if gap remains.

    Returns:
        series       : complete series with no NaNs
        is_imputed   : binary array; 1 where original data was missing
    """
    full_idx   = pd.date_range(series.index.min(), series.index.max(), freq='W-FRI')
    s          = series.reindex(full_idx)
    is_imputed = s.isna().astype(int)

    s = s.ffill(limit=2)
    s = s.interpolate(method='linear', limit=4)

    if s.isna().any():
        n_remaining = s.isna().sum()
        raise ValueError(
            f"Unfillable gap: {n_remaining} weeks remain NaN after ffill+interpolate. "
            f"Max consecutive gap exceeds 4 weeks. Mark this store as exclude=True in model_config.json."
        )

    return s, is_imputed.reindex(full_idx).fillna(0).astype(int)


# ── 3. Winsorize ─────────────────────────────────────────────────────────────

def winsorize_series(series: pd.Series, k: float = 3.0) -> pd.Series:
    """
    IQR-based capping at k=3.0.
    k=3.0 confirmed: all 45 stores have CV <= 0.30. At low CV the IQR
    fences are tight; k=2.0 clips legitimate holiday peaks.
    Floor at 1.0 applied simultaneously in single clip() to prevent
    log1p(0)=0 ambiguity.

    Never drop rows — dropping creates date index gaps that corrupt
    ARIMA differencing (diff() produces NaN at gap positions).
    """
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    lo  = max(Q1 - k * IQR, 1.0)
    hi  = Q3 + k * IQR
    return series.clip(lower=lo, upper=hi)


# ── 4. Log1p Transform ───────────────────────────────────────────────────────

def log_transform(series: pd.Series) -> pd.Series:
    """
    Apply log1p.
    Justified by r(rolling_mean, rolling_std)=0.728 — multiplicative variance confirmed.
    Floor at 1.0 in winsorize_series guarantees log1p input >= 1.0.
    Inverse: np.expm1(). All evaluation metrics computed on original scale after inverse.
    """
    return np.log1p(series)


def inverse_log_transform(series) -> np.ndarray:
    """Inverse of log_transform. Accepts Series or ndarray."""
    return np.expm1(series)


# ── 5. Structural Break Dummy ────────────────────────────────────────────────

def make_level_shift_dummy(n: int, break_idx: int | None) -> np.ndarray:
    """
    Binary array: 0 before break_idx, 1 from break_idx onward.
    break_idx=None returns all zeros — zero effect on model.

    Usage by model_type:
        single       : included as exog column; absorbs sub-threshold CUSUM break
        level_shift  : included as exog column; absorbs break too severe for single
                       but infeasible for two_model split (Store 18: post_n=62)
        two_model    : NOT used; series is split at break_idx instead
    """
    dummy = np.zeros(n, dtype=float)
    if break_idx is not None and break_idx < n:
        dummy[break_idx:] = 1.0
    return dummy


# ── 6. PCA Fit and Apply ─────────────────────────────────────────────────────

def fit_pca(train_macro: pd.DataFrame, store_id: int) -> tuple:
    """
    Fit StandardScaler and PCA(n_components=1) on training macro data only.
    Saves fitted objects to outputs/ for use when building test/forecast exog.

    Fitting on test or future data is a leakage violation.
    PC1 explained 80.33% of macro variance on EDA run (>= 60% gate confirmed).

    PC1 interpretation (from pca_loadings.csv):
        High PC1 = high Fuel + high Temperature + low CPI + low Unemployment
                 = economic expansion proxy

    StandardScaler is required before PCA to prevent CPI (~210) from
    dominating PC1 due to magnitude, not correlation structure.
    StandardScaler is NOT applied to Holiday_Flag — binary regressors
    do not benefit from scaling in MLE-based models.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_macro[MACRO_COLS].values)

    pca = PCA(n_components=1)
    pca.fit(X_scaled)

    ev = pca.explained_variance_ratio_[0] * 100
    print(f"  Store {store_id} PCA — PC1 explains {ev:.2f}% of macro variance")
    if ev < 60:
        print(f"  WARNING: PC1 < 60% ({ev:.2f}%). Consider holiday-only exog path.")

    joblib.dump(scaler, f'app/outputs/pca_scaler_{store_id}.pkl')
    joblib.dump(pca,    f'app/outputs/pca_model_{store_id}.pkl')

    return scaler, pca


def apply_pca(macro_df: pd.DataFrame, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    """
    Transform macro variables to PC1 score using already-fitted scaler and PCA.
    Never refit here — always use the train-fitted objects.
    """
    X_scaled = scaler.transform(macro_df[MACRO_COLS].values)
    return pca.transform(X_scaled)[:, 0]


# ── 7. Build Exogenous Matrix ────────────────────────────────────────────────

def build_exog_matrix(date_index: pd.DatetimeIndex,
                      store_df: pd.DataFrame,
                      scaler: StandardScaler,
                      pca: PCA,
                      level_shift_dummy: np.ndarray | None = None,
                      is_imputed: np.ndarray | None = None) -> pd.DataFrame:
    """
    Build the exogenous regressor matrix for a given date range.

    Column order is fixed and must be identical across train, test,
    and forecast matrices for the same store. Column mismatch causes
    silent wrong predictions or ValueError at model.predict().

    Fixed column order:
        1. PC1_macro     — PCA score on [Temperature, Fuel_Price, CPI, Unemployment]
        2. Holiday_Flag  — raw binary; VIF=1.25; no scaling applied
        3. level_shift   — included only if level_shift_dummy is not None
        4. is_imputed    — included only if imputed weeks exist (defensive)

    Args:
        date_index        : DatetimeIndex of weeks to build matrix for
        store_df          : full raw store DataFrame (unfiltered)
        scaler, pca       : train-fitted PCA objects
        level_shift_dummy : precomputed binary array (from make_level_shift_dummy)
        is_imputed        : binary array from fill_date_index
    """
    slice_df = (
        store_df
        .set_index('Date')
        .reindex(date_index)
        .sort_index()
    )

    exog = pd.DataFrame(index=date_index)
    exog['PC1_macro']    = apply_pca(slice_df, scaler, pca)
    exog['Holiday_Flag'] = slice_df['Holiday_Flag'].values.astype(float)

    if level_shift_dummy is not None:
        exog['level_shift'] = level_shift_dummy

    if is_imputed is not None and int(is_imputed.sum()) > 0:
        exog['is_imputed'] = is_imputed.astype(float)

    return exog.astype(float)


# ── 8. Train/Test Split ──────────────────────────────────────────────────────

def temporal_split(series: pd.Series,
                   exog: pd.DataFrame,
                   test_weeks: int = 12) -> tuple:
    """
    Temporal holdout split by position. Never shuffle.

    Rules:
        - PCA/scaler must be fitted on train before this call (fit_pca uses train_idx)
        - test_exog is for held-out evaluation only
        - forecast_exog (built from future dates) is a separate object — not test_exog
        - test_weeks driven by config['forecast_horizon'] (12 for all stores)
    """
    train_y    = series.iloc[:-test_weeks]
    test_y     = series.iloc[-test_weeks:]
    train_exog = exog.iloc[:-test_weeks]
    test_exog  = exog.iloc[-test_weeks:]
    return train_y, test_y, train_exog, test_exog


# ── 9. Full Preprocessing Per Store ─────────────────────────────────────────

def preprocess_store(df: pd.DataFrame,
                     store_id: int,
                     config: dict) -> dict:
    """
    Orchestrates steps 1–8 for one store.
    Returns a data dict consumed directly by train_sarima.py and sarima_model.py.

    Note on level_shift for two_model stores:
        make_level_shift_dummy is still called and the dummy is stored in the
        returned dict, but build_exog_matrix for two_model stores is called
        WITHOUT passing the dummy (handled in fit_two_model and generate_forecast).
        The dummy is retained in the dict for reference only.
    """
    cfg      = config[str(store_id)]
    store_df = df[df['Store'] == store_id].copy()
    store_df['Date'] = pd.to_datetime(store_df['Date'], dayfirst=True)
    raw      = store_df.set_index('Date')['Weekly_Sales'].sort_index()

    # Steps 2–4
    sales, is_imputed = fill_date_index(raw)
    sales             = winsorize_series(sales, k=3.0)
    log_sales         = log_transform(sales)

    # Step 5 — dummy for all stores with a break index
    break_idx   = cfg.get('level_shift_idx')
    level_shift = make_level_shift_dummy(len(log_sales), break_idx)

    test_weeks  = cfg['forecast_horizon']
    train_idx   = log_sales.index[:-test_weeks]
    test_idx    = log_sales.index[-test_weeks:]

    # Step 6 — PCA fitted on train macro only (no leakage)
    train_macro      = store_df.set_index('Date').reindex(train_idx)
    scaler, pca      = fit_pca(train_macro, store_id)

    # Step 7 — exog matrices
    # single / level_shift: include level_shift column
    # two_model: no level_shift column (series split handles regime change)
    include_shift = cfg['model_type'] != 'two_model'

    train_exog = build_exog_matrix(
        train_idx, store_df, scaler, pca,
        level_shift_dummy=level_shift[:len(train_idx)] if include_shift else None,
        is_imputed=is_imputed.values[:len(train_idx)]
    )
    test_exog = build_exog_matrix(
        test_idx, store_df, scaler, pca,
        level_shift_dummy=level_shift[len(train_idx):] if include_shift else None,
        is_imputed=is_imputed.values[len(train_idx):]
    )

    return {
        'store_id':    store_id,
        'cfg':         cfg,
        'log_sales':   log_sales,
        'raw_sales':   sales,
        'store_df':    store_df,
        'train_y':     log_sales.iloc[:-test_weeks],
        'test_y':      log_sales.iloc[-test_weeks:],
        'train_exog':  train_exog,
        'test_exog':   test_exog,
        'level_shift': level_shift,
        'is_imputed':  is_imputed.values,
        'scaler':      scaler,
        'pca':         pca,
        'break_idx':   break_idx,
    }
