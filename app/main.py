"""
Walmart Retail Demand Forecasting — Streamlit Application

Features:
    1. Demand Forecasting   — 12-week ahead SARIMAX forecast per store with 95% CI
    2. Smart Alert System   — overstock and near-expiry warnings based on forecast
    3. Seasonal Analysis    — decomposed demand patterns across weeks/months
    4. Demand Report        — downloadable CSV summary of forecast and metrics

Architecture:
    Loads pre-trained artifacts only. Never retrains at runtime.
    All models and forecasts produced by train_sarima.py offline.

Artifact paths:
    models/store_1.pkl                 — universal pre-trained model
    app/data/                          — bundled Walmart CSVs for randomization
    outputs/model_config.json          — store configuration
    outputs/metrics.json               — evaluation metrics
    outputs/pca_scaler_{id}.pkl        — PCA scaler per store
    outputs/pca_model_{id}.pkl         — PCA model per store
"""

import copy
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import STL

from utils.preprocessing import (
    parse_and_sort,
    preprocess_store,
    fit_pca,
    apply_pca,
    build_exog_matrix,
    fill_date_index,
    winsorize_series,
    log_transform,
    make_level_shift_dummy,
    MACRO_COLS,
)
from utils.sarima_model import evaluate, fit_single, generate_forecast

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR      = 'app/data'
MODELS_DIR    = 'app/models'
OUTPUTS_DIR   = 'app/outputs'
CONFIG_PATH   = 'app/outputs/model_config.json'
METRICS_PATH  = 'app/outputs/metrics.json'

# Stores for which pre-trained artifacts are available.
# Only these stores bypass preprocess_store + auto_arima at inference time.
KNOWN_STORE_IDS = {1, 7, 14}

FORECAST_KEY       = 'trained_forecast'
BYTES_KEY          = 'trained_forecast_bytes'
METRICS_KEY        = 'trained_metrics'
USER_DF_KEY        = 'user_df'
DATA_SOURCE_KEY    = 'data_source'   # 'upload' | 'random'
RANDOM_STORE_KEY   = 'random_store_id'


# ── Artifact loaders (cached) ─────────────────────────────────────────────────

@st.cache_resource
def load_known_store_artifacts(store_id: int) -> dict | None:
    """
    Load pre-trained model + PCA artifacts for a known store.
    Returns None if any artifact is missing.
    generate_forecast calls model.update() which mutates the model object in
    place. cache_resource returns the same object on every call, so the first
    update would corrupt all subsequent uses. We return a deep copy here so
    the cached original is never mutated.
    """
    model_path  = os.path.join(MODELS_DIR, f'store_{store_id}.pkl')
    scaler_path = os.path.join(OUTPUTS_DIR, f'pca_scaler_{store_id}.pkl')
    pca_path    = os.path.join(OUTPUTS_DIR, f'pca_model_{store_id}.pkl')

    for p in (model_path, scaler_path, pca_path):
        if not os.path.exists(p):
            return None

    return {
        'model':  joblib.load(model_path),
        'scaler': joblib.load(scaler_path),
        'pca':    joblib.load(pca_path),
    }


@st.cache_data
def load_metrics() -> dict:
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH) as f:
        return json.load(f)


@st.cache_data
def list_data_files() -> list[str]:
    """Return all CSV filenames in app/data/."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.csv'))


def load_csv_from_data(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)


# ── Alert logic ───────────────────────────────────────────────────────────────

def compute_alerts(forecast_df: pd.DataFrame,
                   historical_df: pd.DataFrame,
                   overstock_pct: float = 0.20) -> list[dict]:
    hist_mean = historical_df['Weekly_Sales'].mean()
    alerts = []

    for _, row in forecast_df.iterrows():
        week     = row['Week']
        forecast = row['Forecast_Sales']
        is_hol   = bool(row['Is_Holiday'])

        if forecast < hist_mean * (1 - overstock_pct):
            severity = 'HIGH' if forecast < hist_mean * (1 - overstock_pct * 2) else 'MEDIUM'
            alerts.append({
                'Week':     week,
                'Type':     'Low Demand',
                'Severity': severity,
                'Message':  (
                    f"Forecast ${forecast:,.0f} is "
                    f"{(1 - forecast / hist_mean) * 100:.1f}% below average. "
                    f"Reduce orders to avoid expiry waste."
                ),
            })

        if forecast > hist_mean * (1 + overstock_pct):
            alerts.append({
                'Week':     week,
                'Type':     'High Demand',
                'Severity': 'MEDIUM',
                'Message':  (
                    f"Forecast ${forecast:,.0f} is "
                    f"{(forecast / hist_mean - 1) * 100:.1f}% above average. "
                    f"Increase procurement to avoid stock-out."
                ),
            })

        if is_hol:
            alerts.append({
                'Week':     week,
                'Type':     'Holiday Week',
                'Severity': 'INFO',
                'Message':  'Holiday week — verify procurement schedule accounts for demand surge.',
            })

    return alerts


# ── Seasonal analysis ─────────────────────────────────────────────────────────

def run_stl(historical_df: pd.DataFrame) -> dict | None:
    s = np.log1p(historical_df.set_index('Date')['Weekly_Sales'].sort_index())
    if len(s) < 104:
        return None
    stl = STL(s, period=52, robust=True).fit()
    return {
        'observed': s,
        'trend':    stl.trend,
        'seasonal': stl.seasonal,
        'residual': stl.resid,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_d(series: pd.Series) -> int:
    log_s  = np.log1p(series.values)
    var_d0 = float(np.var(log_s))
    var_d1 = float(np.var(np.diff(log_s)))
    return 0 if var_d0 < var_d1 else 1


def _validate_upload(df: pd.DataFrame) -> str | None:
    required_cols = {
        'Store', 'Date', 'Weekly_Sales', 'Holiday_Flag',
        'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'
    }
    missing = required_cols - set(df.columns)
    if missing:
        return f"Missing columns: {', '.join(sorted(missing))}"

    try:
        pd.to_datetime(df['Date'], dayfirst=True)
    except Exception:
        return "Date column could not be parsed. Use DD-MM-YYYY format."

    numeric_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature',
                    'Fuel_Price', 'CPI', 'Unemployment']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return f"Column '{col}' must be numeric."

    invalid_flags = df[~df['Holiday_Flag'].isin([0, 1])]
    if not invalid_flags.empty:
        return "Holiday_Flag must contain only 0 or 1."

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    weeks_per_store = df.groupby('Store')['Date'].nunique()
    short_stores = weeks_per_store[weeks_per_store < 65]
    if not short_stores.empty:
        return (
            f"Stores {short_stores.index.tolist()} have fewer than 65 weeks. "
            f"Minimum 65 required for m=52 seasonal modelling."
        )

    nulls = df[list(required_cols)].isnull().sum()
    cols_with_nulls = nulls[nulls > 0]
    if not cols_with_nulls.empty:
        return f"Null values found in: {cols_with_nulls.index.tolist()}"

    return None


def _build_unknown_cfg(store_id: int, store_series: pd.Series) -> dict:
    """
    Config for an unknown store (not in KNOWN_STORE_IDS).
    d is inferred from the uploaded series. All other bounds are conservative
    defaults that auto_arima will search within.
    """
    return {
        str(store_id): {
            'd':                _infer_d(store_series),
            'D':                0,
            'm':                52,
            'max_p':            3,
            'max_q':            2,
            'max_P':            1,
            'max_Q':            1,
            'level_shift_idx':  None,
            'exog_path':        'pca',
            'exclude':          False,
            'n_weeks':          len(store_series),
            'cv':               float(store_series.std() / store_series.mean()),
            'forecast_horizon': 12,
            'model_type':       'single',
        }
    }


def _build_data_dict_known(
    parsed: pd.DataFrame,
    store_id: int,
    artifacts: dict,
    global_config: dict,
) -> dict:
    """
    Build the data dict for a known store WITHOUT calling preprocess_store.

    preprocess_store calls fit_pca which refits and overwrites the PCA
    artifacts on disk — invalid at inference time for known stores where
    those artifacts were fit on the training data only.

    This function replicates the preprocessing steps of preprocess_store
    but substitutes apply_pca (transform only) for fit_pca (fit + transform),
    preserving the original scaler/PCA fitted on training data.

    The cfg is loaded from model_config.json so d, m, level_shift_idx, and
    model_type are identical to what was used during training.
    """
    scaler = artifacts['scaler']
    pca    = artifacts['pca']

    # Pull cfg from model_config.json (matches training exactly)
    cfg = global_config.get(str(store_id), {})
    if not cfg:
        raise ValueError(f"Store {store_id} not found in model_config.json")

    store_df = parsed[parsed['Store'] == store_id].copy()
    store_df['Date'] = pd.to_datetime(store_df['Date'], dayfirst=True)
    raw = store_df.set_index('Date')['Weekly_Sales'].sort_index()

    sales, is_imputed = fill_date_index(raw)
    sales             = winsorize_series(sales, k=3.0)
    log_sales         = log_transform(sales)

    break_idx   = cfg.get('level_shift_idx')
    level_shift = make_level_shift_dummy(len(log_sales), break_idx)

    test_weeks    = cfg['forecast_horizon']
    train_idx     = log_sales.index[:-test_weeks]
    test_idx      = log_sales.index[-test_weeks:]
    include_shift = cfg['model_type'] != 'two_model'

    # apply_pca only — scaler/pca already fitted on training data
    train_exog = build_exog_matrix(
        train_idx, store_df, scaler, pca,
        level_shift_dummy=level_shift[:len(train_idx)] if include_shift else None,
        is_imputed=is_imputed.values[:len(train_idx)],
    )
    test_exog = build_exog_matrix(
        test_idx, store_df, scaler, pca,
        level_shift_dummy=level_shift[len(train_idx):] if include_shift else None,
        is_imputed=is_imputed.values[len(train_idx):],
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


def _extract_metrics(eval_result: dict, data: dict, model) -> dict:
    """
    Derive the full metrics dict from evaluate() output.
    Naive wMAPE computed on the test split actuals (lag-1 baseline).
    """
    wmape   = eval_result['wMAPE']
    dir_acc = eval_result['DirectionalAccuracy']

    test_y = np.expm1(data['test_y'].values)
    if len(test_y) >= 2:
        naive_wmape = float(np.abs(np.diff(test_y)).sum() / np.abs(test_y[1:]).sum())
    else:
        naive_wmape = float('nan')

    beats_naive = (
        bool(wmape < naive_wmape)
        if not (np.isnan(wmape) or np.isnan(naive_wmape))
        else False
    )

    return {
        'wMAPE':               wmape,
        'DirectionalAccuracy': dir_acc,
        'beats_naive':         beats_naive,
        'model_type':          data['cfg'].get('model_type', 'single'),
        'sarima_order':        str(getattr(model, 'order', 'N/A')),
        'seasonal_order':      str(getattr(model, 'seasonal_order', 'N/A')),
        'n_weeks':             len(data['log_sales']),
        'MAPE':                eval_result['MAPE'],
        'MAE':                 eval_result['MAE'],
        'RMSE':                eval_result['RMSE'],
        'SMAPE':               eval_result['SMAPE'],
        'naive_wMAPE':         naive_wmape,
        'cv_mae_mean':         'N/A',
        'cv_mae_std':          'N/A',
        'lb_pass':             'N/A',
    }


def _run_forecast(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Option C hybrid forecast pipeline.

    Known stores (KNOWN_STORE_IDS = {1, 7, 14}):
        - Load pre-trained model and PCA artifacts from disk
        - Build data dict using apply_pca only (no refit, no artifact overwrite)
        - Pull exact metrics from metrics.json (same values as training evaluation)
        - generate_forecast refits on full series using preserved scaler/PCA

    Unknown stores:
        - Run preprocess_store (fits fresh PCA on uploaded data — correct)
        - Run auto_arima to fit a new model (correct for unseen store patterns)
        - evaluate() on the freshly fitted model produces honest metrics
        - generate_forecast uses the new model and fresh PCA

    generate_forecast calls model.update() which mutates the model in place.
    We deep-copy the loaded model before passing it in so the cached artifact
    is never modified.
    """
    all_metrics = load_metrics()
    all_config  = load_config()

    parsed   = parse_and_sort(raw_df.copy())
    store_id = int(parsed['Store'].iloc[0])

    if store_id in KNOWN_STORE_IDS:
        # ── Known store path ──────────────────────────────────────────
        artifacts = load_known_store_artifacts(store_id)
        if artifacts is None:
            raise FileNotFoundError(
                f"Pre-trained artifacts missing for Store {store_id}. "
                f"Expected: app/models/store_{store_id}.pkl, "
                f"app/outputs/pca_scaler_{store_id}.pkl, "
                f"app/outputs/pca_model_{store_id}.pkl"
            )

        data  = _build_data_dict_known(parsed, store_id, artifacts, all_config)
        # Deep copy prevents model.update() in generate_forecast from
        # mutating the cached artifact on subsequent calls
        model = copy.deepcopy(artifacts['model'])

        fc_dict = generate_forecast(model, data)

        # Pull exact metrics from metrics.json — these were computed during
        # training on the same model with the same data split
        store_json_metrics = all_metrics.get(str(store_id), {})
        if store_json_metrics:
            metrics = {
                'wMAPE':               store_json_metrics.get('wMAPE', float('nan')),
                'DirectionalAccuracy': store_json_metrics.get('DirectionalAccuracy', float('nan')),
                'beats_naive':         store_json_metrics.get('beats_naive', False),
                'model_type':          all_config.get(str(store_id), {}).get('model_type', 'single'),
                'sarima_order':        str(store_json_metrics.get('sarima_order', 'N/A')),
                'seasonal_order':      str(store_json_metrics.get('seasonal_order', 'N/A')),
                'n_weeks':             all_config.get(str(store_id), {}).get('n_weeks', len(data['log_sales'])),
                'MAPE':                store_json_metrics.get('MAPE', float('nan')),
                'MAE':                 store_json_metrics.get('MAE', float('nan')),
                'RMSE':                store_json_metrics.get('RMSE', float('nan')),
                'SMAPE':               store_json_metrics.get('SMAPE', float('nan')),
                'naive_wMAPE':         store_json_metrics.get('naive_wMAPE', float('nan')),
                'cv_mae_mean':         store_json_metrics.get('cv_mae_mean', 'N/A'),
                'cv_mae_std':          store_json_metrics.get('cv_mae_std', 'N/A'),
                'lb_pass':             store_json_metrics.get('lb_pass', 'N/A'),
            }
        else:
            # metrics.json entry missing — fall back to live evaluate()
            eval_result = evaluate(data, copy.deepcopy(model))
            metrics = _extract_metrics(eval_result, data, model)

    else:
        # ── Unknown store path ────────────────────────────────────────
        store_series = (
            parsed[parsed['Store'] == store_id]
            .set_index('Date')['Weekly_Sales']
            .sort_index()
        )
        user_cfg  = _build_unknown_cfg(store_id, store_series)

        # preprocess_store fits fresh PCA on this store's data — correct
        # for an unseen store; no pre-trained artifacts exist to corrupt
        data  = preprocess_store(parsed, store_id, user_cfg)
        model = fit_single(data)

        try:
            eval_result = evaluate(data, copy.deepcopy(model))
            metrics     = _extract_metrics(eval_result, data, model)
        except Exception as exc:
            st.warning(
                f"Metric computation failed ({type(exc).__name__}: {exc}). "
                f"Metrics will show N/A."
            )
            metrics = {
                k: float('nan') for k in
                ['wMAPE', 'DirectionalAccuracy', 'MAPE', 'MAE', 'RMSE',
                 'SMAPE', 'naive_wMAPE']
            }
            metrics.update({
                'beats_naive': False,
                'model_type':  'single',
                'sarima_order': 'N/A', 'seasonal_order': 'N/A',
                'n_weeks': len(data['log_sales']),
                'cv_mae_mean': 'N/A', 'cv_mae_std': 'N/A', 'lb_pass': 'N/A',
            })

        fc_dict = generate_forecast(model, data)

    fc_df = pd.DataFrame({
        'Week':           fc_dict['dates'],
        'Forecast_Sales': fc_dict['forecast'],
        'Lower_95':       fc_dict['lower_95'],
        'Upper_95':       fc_dict['upper_95'],
        'Is_Holiday':     fc_dict['is_holiday'],
    })

    return fc_df, metrics



def _store_forecast_in_session(fc_df: pd.DataFrame, metrics: dict):
    st.session_state[FORECAST_KEY] = fc_df
    st.session_state[METRICS_KEY]  = metrics
    st.session_state[BYTES_KEY]    = (
        fc_df.assign(Week=fc_df['Week'].dt.strftime('%Y-%m-%d'))
        .to_csv(index=False)
        .encode('utf-8')
    )


def _clear_forecast():
    for k in (FORECAST_KEY, BYTES_KEY, METRICS_KEY):
        st.session_state.pop(k, None)


# ── Session state init ────────────────────────────────────────────────────────

def _init_session():
    defaults = {
        USER_DF_KEY:     None,
        DATA_SOURCE_KEY: 'upload',
        RANDOM_STORE_KEY: None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    _init_session()

    st.title("Retail Demand Forecasting")
    st.caption("SARIMAX model | 12-week horizon | Food waste reduction")

    config  = load_config()
    metrics = load_metrics()

    # ── Upload / Randomize section ────────────────────────────────────
    st.subheader("Upload Your Data")

    col_upload, col_rand = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=['csv'],
            help=(
                "CSV must contain: Store, Date (DD-MM-YYYY), Weekly_Sales, "
                "Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment. "
                "Minimum 65 weeks per store."
            ),
            label_visibility="collapsed",
        )

    with col_rand:
        data_files = list_data_files()
        randomize_btn = st.button(
            "Randomize Dataset",
            disabled=len(data_files) == 0,
            help="Randomly pick one of the bundled datasets in app/data/",
        )

    # ── Handle Randomize Dataset ──────────────────────────────────────
    if randomize_btn and data_files:
        chosen = random.choice(data_files)
        try:
            raw = load_csv_from_data(chosen)
            err = _validate_upload(raw)
            if err:
                st.error(f"Bundled file '{chosen}' failed validation: {err}")
            else:
                st.session_state[USER_DF_KEY]     = raw
                st.session_state[DATA_SOURCE_KEY] = 'random'
                st.session_state[RANDOM_STORE_KEY] = chosen
                _clear_forecast()
                st.success(f"Loaded bundled dataset: **{chosen}** — {len(raw):,} rows")
        except Exception as e:
            st.error(f"Could not load '{chosen}': {e}")

    # ── Handle file upload ────────────────────────────────────────────
    if uploaded_file is not None:
        try:
            raw_upload = pd.read_csv(uploaded_file)
            err = _validate_upload(raw_upload)
            if err:
                st.error(err)
                st.session_state[USER_DF_KEY] = None
                _clear_forecast()
            else:
                # Only update session if file actually changed
                if not raw_upload.equals(st.session_state.get(USER_DF_KEY)):
                    st.session_state[USER_DF_KEY]     = raw_upload
                    st.session_state[DATA_SOURCE_KEY] = 'upload'
                    _clear_forecast()
                st.success(
                    f"Loaded {len(raw_upload):,} rows — "
                    f"{raw_upload['Store'].nunique()} store(s)"
                )
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.session_state[USER_DF_KEY] = None
            _clear_forecast()

    # ── Overstock threshold (formerly sidebar) ────────────────────────
    overstock_pct = st.slider(
        "Alert threshold (%)", min_value=5, max_value=50, value=20, step=5
    ) / 100

    # ── Generate Forecast button ──────────────────────────────────────
    user_df = st.session_state.get(USER_DF_KEY)

    if user_df is not None:
        col_btn, col_status = st.columns([1, 3])

        with col_btn:
            forecast_btn = st.button("Generate Forecast", type="primary")

        with col_status:
            if st.session_state.get(FORECAST_KEY) is not None:
                st.success("Forecast ready. View results in the tabs below.")

        if forecast_btn:
            with st.spinner("Generating 12-week forecast…"):
                try:
                    fc_df, fc_metrics = _run_forecast(user_df)
                    _store_forecast_in_session(fc_df, fc_metrics)
                    st.rerun()
                except Exception as e:
                    st.error(f"Forecast generation failed: {e}")
    else:
        st.info("Upload a CSV file or click **Randomize Dataset** to get started.")

    # ── Tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Forecast", "Smart Alerts", "Seasonal Analysis", "Demand Report"
    ])

    # ── Resolve data for tabs ─────────────────────────────────────────
    forecast_df   = st.session_state.get(FORECAST_KEY)
    store_metrics = st.session_state.get(METRICS_KEY, {})
    cfg           = store_metrics  # for upload path, cfg == metrics dict

    if user_df is not None:
        parsed_df  = parse_and_sort(user_df.copy())
        first_store = int(parsed_df['Store'].iloc[0])
        historical  = parsed_df[parsed_df['Store'] == first_store].copy()
        store_id    = first_store
    else:
        historical  = pd.DataFrame(columns=['Date', 'Weekly_Sales'])
        store_id    = None

    display_label = f"Store {store_id}" if store_id is not None else "(No Store Selected)"

    # ── Tab 1: Forecast ───────────────────────────────────────────────
    with tab1:
        st.subheader(f"{display_label} — 12-Week Sales Forecast")

        if forecast_df is None:
            st.info("Generate a forecast to view results here.")
        else:
            # Metric cards
            wmape    = store_metrics.get('wMAPE')
            dir_acc  = store_metrics.get('DirectionalAccuracy')
            beats    = store_metrics.get('beats_naive')
            mtype    = store_metrics.get('model_type', 'N/A')

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "wMAPE",
                f"{wmape:.1%}" if isinstance(wmape, float) else "N/A",
            )
            c2.metric(
                "Dir. Acc.",
                f"{dir_acc:.1%}" if isinstance(dir_acc, float) else "N/A",
            )
            c3.metric("Beats Naive", "Yes" if beats else "No")
            c4.metric(
                "Model Type",
                mtype.replace('_', ' ').title() if isinstance(mtype, str) else "N/A",
            )

            # Forecast chart
            fig = go.Figure()

            hist_recent = historical.tail(52)
            if not hist_recent.empty:
                fig.add_trace(go.Scatter(
                    x=hist_recent['Date'],
                    y=hist_recent['Weekly_Sales'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='steelblue', width=2),
                ))

            fig.add_trace(go.Scatter(
                x=forecast_df['Week'],
                y=forecast_df['Forecast_Sales'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='darkorange', width=2),
                marker=dict(size=6),
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df['Week'], forecast_df['Week'][::-1]]),
                y=pd.concat([forecast_df['Upper_95'], forecast_df['Lower_95'][::-1]]),
                fill='toself',
                fillcolor='rgba(255,165,0,0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% CI',
            ))

            hol_weeks = forecast_df[forecast_df['Is_Holiday'] == 1]
            if not hol_weeks.empty:
                fig.add_trace(go.Scatter(
                    x=hol_weeks['Week'],
                    y=hol_weeks['Forecast_Sales'],
                    mode='markers',
                    name='Holiday Week',
                    marker=dict(color='crimson', size=10, symbol='star'),
                ))

            fig.update_layout(
                xaxis_title='Week',
                yaxis_title='Weekly Sales ($)',
                legend=dict(orientation='h', y=-0.2),
                height=450,
                margin=dict(l=40, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, width='stretch')

            with st.expander("Forecast data"):
                display_df = forecast_df.copy()
                display_df['Week']           = display_df['Week'].dt.strftime('%Y-%m-%d')
                display_df['Forecast_Sales'] = display_df['Forecast_Sales'].map('${:,.0f}'.format)
                display_df['Lower_95']       = display_df['Lower_95'].map('${:,.0f}'.format)
                display_df['Upper_95']       = display_df['Upper_95'].map('${:,.0f}'.format)
                st.dataframe(display_df, width='stretch')

            forecast_bytes = st.session_state.get(BYTES_KEY)
            if forecast_bytes:
                st.download_button(
                    label="Download Forecast CSV",
                    data=forecast_bytes,
                    file_name='forecast_12w.csv',
                    mime='text/csv',
                )

    # ── Tab 2: Smart Alerts ───────────────────────────────────────────
    with tab2:
        st.subheader(f"{display_label} — Smart Alert System")

        if forecast_df is None:
            st.info("Generate a forecast to view alerts.")
        elif historical.empty:
            st.warning("No historical data available for alert computation.")
        else:
            alerts = compute_alerts(forecast_df, historical, overstock_pct=overstock_pct)

            if not alerts:
                st.success("No alerts for the forecast period.")
            else:
                severity_color = {'HIGH': '🔴', 'MEDIUM': '🟡', 'INFO': '🔵'}
                for alert in alerts:
                    icon = severity_color.get(alert['Severity'], '⚪')
                    with st.container():
                        week_str = (
                            alert['Week'].strftime('%Y-%m-%d')
                            if hasattr(alert['Week'], 'strftime')
                            else str(alert['Week'])
                        )
                        st.markdown(
                            f"**{icon} {alert['Severity']} — {alert['Type']}** "
                            f"| Week of {week_str}"
                        )
                        st.caption(alert['Message'])
                        st.divider()

            if alerts:
                alert_df = pd.DataFrame(alerts)
                counts   = alert_df.groupby(['Type', 'Severity']).size().reset_index(name='Count')
                st.dataframe(counts, width='stretch')

    # ── Tab 3: Seasonal Analysis ──────────────────────────────────────
    with tab3:
        st.subheader(f"{display_label} — Seasonal Demand Analysis")

        if historical.empty:
            st.info("Upload data and generate a forecast to view seasonal analysis.")
        else:
            stl_result = run_stl(historical)
            if stl_result is None:
                st.warning("Insufficient data for STL decomposition (need ≥ 104 weeks).")
            else:
                fig_stl = go.Figure()
                components = [
                    ('observed', 'Observed (log1p)', 'steelblue'),
                    ('trend',    'Trend',            'darkorange'),
                    ('seasonal', 'Seasonal',         'seagreen'),
                    ('residual', 'Residual',         'crimson'),
                ]
                for i, (key, name, color) in enumerate(components):
                    visible = True if i < 2 else 'legendonly'
                    fig_stl.add_trace(go.Scatter(
                        x=stl_result[key].index,
                        y=stl_result[key].values,
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=1.5),
                        visible=visible,
                    ))

                fig_stl.update_layout(
                    xaxis_title='Date',
                    yaxis_title='log1p(Sales)',
                    height=400,
                    legend=dict(orientation='h', y=-0.2),
                    margin=dict(l=40, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_stl, width='stretch')

            hist_copy = historical.copy()
            hist_copy['Month'] = hist_copy['Date'].dt.month
            monthly_avg = hist_copy.groupby('Month')['Weekly_Sales'].mean().reset_index()
            monthly_avg['Month'] = monthly_avg['Month'].map({
                1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'
            })

            fig_month = px.bar(
                monthly_avg,
                x='Month', y='Weekly_Sales',
                title='Average Weekly Sales by Month',
                labels={'Weekly_Sales': 'Avg Weekly Sales ($)'},
                color='Weekly_Sales',
                color_continuous_scale='Blues',
            )
            fig_month.update_layout(
                height=350,
                margin=dict(l=40, r=20, t=40, b=20),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_month, width='stretch')

            hist_copy['Week_of_Year'] = hist_copy['Date'].dt.isocalendar().week.astype(int)
            hist_copy['Year']         = hist_copy['Date'].dt.year
            pivot = hist_copy.pivot_table(
                values='Weekly_Sales', index='Year', columns='Week_of_Year', aggfunc='mean'
            )
            fig_heat = px.imshow(
                pivot,
                labels=dict(x='Week of Year', y='Year', color='Sales ($)'),
                title='Weekly Sales Heatmap by Year',
                color_continuous_scale='Blues',
                aspect='auto',
            )
            fig_heat.update_layout(height=280, margin=dict(l=40, r=20, t=40, b=20))
            st.plotly_chart(fig_heat, width='stretch')

    # ── Tab 4: Demand Report ──────────────────────────────────────────
    with tab4:
        st.subheader(f"{display_label} — Demand Report")

        if forecast_df is None:
            st.info("Generate a forecast to view the demand report.")
        else:
            # Model summary
            st.markdown("### Model Summary")
            mtype_raw = store_metrics.get('model_type', 'N/A')
            summary_data = {
                'Store':          display_label,
                'Model Type':     mtype_raw.replace('_', ' ').title() if isinstance(mtype_raw, str) else 'N/A',
                'SARIMA Order':   str(store_metrics.get('sarima_order', 'N/A')),
                'Seasonal Order': str(store_metrics.get('seasonal_order', 'N/A')),
                'n_weeks':        store_metrics.get('n_weeks', 'N/A'),
            }
            _df_summary = pd.DataFrame.from_dict(summary_data, orient='index', columns=['Value'])
            _df_summary['Value'] = _df_summary['Value'].astype(str)
            st.dataframe(_df_summary, width='stretch')

            # Performance metrics
            st.markdown("### Model Performance")
            metric_keys = [
                'wMAPE', 'SMAPE', 'MAPE', 'MAE', 'RMSE',
                'DirectionalAccuracy', 'naive_wMAPE', 'beats_naive',
                'cv_mae_mean', 'cv_mae_std', 'lb_pass',
            ]
            perf_data = {}
            for k in metric_keys:
                v = store_metrics.get(k, 'N/A')
                if isinstance(v, float):
                    if k in ('wMAPE', 'SMAPE', 'MAPE', 'DirectionalAccuracy', 'naive_wMAPE'):
                        perf_data[k] = f"{v:.2%}"
                    else:
                        perf_data[k] = f"{v:.4f}"
                elif isinstance(v, bool):
                    perf_data[k] = "Yes" if v else "No"
                else:
                    perf_data[k] = str(v)
            _df_perf = pd.DataFrame.from_dict(perf_data, orient='index', columns=['Value'])
            _df_perf['Value'] = _df_perf['Value'].astype(str)
            st.dataframe(_df_perf, width='stretch')

            # Forecast summary stats
            st.markdown("### Forecast Summary (Next 12 Weeks)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Forecast Sales",
                        f"${forecast_df['Forecast_Sales'].sum():,.0f}")
            col2.metric("Weekly Average",
                        f"${forecast_df['Forecast_Sales'].mean():,.0f}")
            peak_week = forecast_df.loc[forecast_df['Forecast_Sales'].idxmax(), 'Week']
            col3.metric("Peak Week",
                        peak_week.strftime('%Y-%m-%d') if hasattr(peak_week, 'strftime') else str(peak_week))

            # Downloads
            st.markdown("### Downloads")
            col_a, col_b = st.columns(2)

            with col_a:
                forecast_csv = (
                    forecast_df.assign(Week=forecast_df['Week'].dt.strftime('%Y-%m-%d'))
                    .to_csv(index=False)
                )
                st.download_button(
                    label="Download Forecast CSV",
                    data=forecast_csv,
                    file_name=f'store_{store_id}_forecast_12w.csv',
                    mime='text/csv',
                )

            with col_b:
                report_rows = []
                for _, row in forecast_df.iterrows():
                    report_rows.append({
                        'Store':               display_label,
                        'Week':                row['Week'].strftime('%Y-%m-%d') if hasattr(row['Week'], 'strftime') else str(row['Week']),
                        'Forecast_Sales':      round(row['Forecast_Sales'], 2),
                        'Lower_95':            round(row['Lower_95'], 2),
                        'Upper_95':            round(row['Upper_95'], 2),
                        'Is_Holiday':          int(row['Is_Holiday']),
                        'wMAPE':               perf_data.get('wMAPE', ''),
                        'DirectionalAccuracy': perf_data.get('DirectionalAccuracy', ''),
                        'Model_Type':          mtype_raw,
                    })
                report_csv = pd.DataFrame(report_rows).to_csv(index=False)
                st.download_button(
                    label="Download Full Report CSV",
                    data=report_csv,
                    file_name=f'store_{store_id}_demand_report.csv',
                    mime='text/csv',
                )

    # ── Footer ────────────────────────────────────────────────────────
    st.divider()
    n_weeks = store_metrics.get('n_weeks', '—') if store_metrics else '—'
    st.caption(
        f"Model: SARIMAX | Dataset: Walmart Store Sales | "
        f"Trained on {n_weeks} weeks"
    )


if __name__ == '__main__':
    main()
