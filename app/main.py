import copy
import io
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from statsmodels.tsa.seasonal import STL

from utils.preprocessing import (
    parse_and_sort,
    preprocess_store,
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

# Hide all Streamlit chrome — React owns 100% of the visible UI
st.markdown("""
<style>
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
.stDeployButton,
#MainMenu,
footer { visibility: hidden !important; }
[data-testid="stHeader"] { visibility: hidden !important; height: 0 !important; }
.stApp { background-color: #0e1117 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
iframe { width: 100% !important; border: none !important; display: block; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR     = 'app/data/random'
MODELS_DIR   = 'app/models'
OUTPUTS_DIR  = 'app/outputs'
CONFIG_PATH  = 'app/outputs/model_config.json'
METRICS_PATH = 'app/outputs/metrics.json'

KNOWN_STORE_IDS = {1, 7, 14}

# ── Declare React component ───────────────────────────────────────────────────
_FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'components', 'forecast_ui', 'frontend', 'dist')
)
_forecast_ui = components.declare_component('forecast_ui', path=_FRONTEND_DIR)

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_known_store_artifacts(store_id: int) -> dict | None:
    model_path  = os.path.join(MODELS_DIR,  f'store_{store_id}.pkl')
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
    path = os.path.isdir(DATA_DIR)
    if not path:
        return []
    return sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.csv'))


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_upload(df: pd.DataFrame) -> str | None:
    required = {'Store', 'Date', 'Weekly_Sales', 'Holiday_Flag',
                'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'}
    missing = required - set(df.columns)
    if missing:
        return f"Missing columns: {', '.join(sorted(missing))}"
    try:
        pd.to_datetime(df['Date'], dayfirst=True)
    except Exception:
        return "Date column could not be parsed. Use DD-MM-YYYY format."
    for col in ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return f"Column '{col}' must be numeric."
    if not df[~df['Holiday_Flag'].isin([0, 1])].empty:
        return "Holiday_Flag must contain only 0 or 1."
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    short = df.groupby('Store')['Date'].nunique()
    short = short[short < 65]
    if not short.empty:
        return f"Stores {short.index.tolist()} have fewer than 65 weeks."
    nulls = df[list(required)].isnull().sum()
    bad = nulls[nulls > 0]
    if not bad.empty:
        return f"Null values found in: {bad.index.tolist()}"
    return None


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _infer_d(series: pd.Series) -> int:
    log_s = np.log1p(series.values)
    return 0 if np.var(log_s) < np.var(np.diff(log_s)) else 1


def _build_unknown_cfg(store_id: int, store_series: pd.Series) -> dict:
    return {
        str(store_id): {
            'd': _infer_d(store_series), 'D': 0, 'm': 52,
            'max_p': 3, 'max_q': 2, 'max_P': 1, 'max_Q': 1,
            'level_shift_idx': None, 'exog_path': 'pca',
            'exclude': False, 'n_weeks': len(store_series),
            'cv': float(store_series.std() / store_series.mean()),
            'forecast_horizon': 12, 'model_type': 'single',
        }
    }


def _build_data_dict_known(parsed, store_id, artifacts, global_config) -> dict:
    scaler = artifacts['scaler']
    pca    = artifacts['pca']
    cfg    = global_config.get(str(store_id), {})
    if not cfg:
        raise ValueError(f"Store {store_id} not found in model_config.json")

    store_df = parsed[parsed['Store'] == store_id].copy()
    store_df['Date'] = pd.to_datetime(store_df['Date'], dayfirst=True)
    raw = store_df.set_index('Date')['Weekly_Sales'].sort_index()

    sales, is_imputed = fill_date_index(raw)
    sales             = winsorize_series(sales, k=3.0)
    log_sales         = log_transform(sales)

    break_idx     = cfg.get('level_shift_idx')
    level_shift   = make_level_shift_dummy(len(log_sales), break_idx)
    test_weeks    = cfg['forecast_horizon']
    train_idx     = log_sales.index[:-test_weeks]
    test_idx      = log_sales.index[-test_weeks:]
    include_shift = cfg['model_type'] != 'two_model'

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
        'store_id': store_id, 'cfg': cfg,
        'log_sales': log_sales, 'raw_sales': sales, 'store_df': store_df,
        'train_y': log_sales.iloc[:-test_weeks], 'test_y': log_sales.iloc[-test_weeks:],
        'train_exog': train_exog, 'test_exog': test_exog,
        'level_shift': level_shift, 'is_imputed': is_imputed.values,
        'scaler': scaler, 'pca': pca, 'break_idx': break_idx,
    }


def _extract_metrics(eval_result, data, model) -> dict:
    wmape   = eval_result['wMAPE']
    dir_acc = eval_result['DirectionalAccuracy']
    test_y  = np.expm1(data['test_y'].values)
    naive_wmape = (
        float(np.abs(np.diff(test_y)).sum() / np.abs(test_y[1:]).sum())
        if len(test_y) >= 2 else float('nan')
    )
    beats_naive = bool(wmape < naive_wmape) if not (np.isnan(wmape) or np.isnan(naive_wmape)) else False
    return {
        'wMAPE': wmape, 'DirectionalAccuracy': dir_acc,
        'beats_naive': beats_naive,
        'model_type': data['cfg'].get('model_type', 'single'),
        'sarima_order': str(getattr(model, 'order', 'N/A')),
        'seasonal_order': str(getattr(model, 'seasonal_order', 'N/A')),
        'n_weeks': len(data['log_sales']),
        'MAPE': eval_result['MAPE'], 'MAE': eval_result['MAE'],
        'RMSE': eval_result['RMSE'], 'SMAPE': eval_result['SMAPE'],
        'naive_wMAPE': naive_wmape,
        'cv_mae_mean': 'N/A', 'cv_mae_std': 'N/A', 'lb_pass': 'N/A',
    }


# ── Forecast pipeline ─────────────────────────────────────────────────────────

def _run_forecast(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    all_metrics = load_metrics()
    all_config  = load_config()
    parsed      = parse_and_sort(raw_df.copy())
    store_id    = int(parsed['Store'].iloc[0])

    if store_id in KNOWN_STORE_IDS:
        artifacts = load_known_store_artifacts(store_id)
        if artifacts is None:
            raise FileNotFoundError(
                f"Pre-trained artifacts missing for Store {store_id}."
            )
        data  = _build_data_dict_known(parsed, store_id, artifacts, all_config)
        model = copy.deepcopy(artifacts['model'])
        fc_dict = generate_forecast(model, data)

        store_json = all_metrics.get(str(store_id), {})
        if store_json:
            metrics = {
                'wMAPE':               store_json.get('wMAPE', float('nan')),
                'DirectionalAccuracy': store_json.get('DirectionalAccuracy', float('nan')),
                'beats_naive':         store_json.get('beats_naive', False),
                'model_type':          all_config.get(str(store_id), {}).get('model_type', 'single'),
                'sarima_order':        str(store_json.get('sarima_order', 'N/A')),
                'seasonal_order':      str(store_json.get('seasonal_order', 'N/A')),
                'n_weeks':             all_config.get(str(store_id), {}).get('n_weeks', len(data['log_sales'])),
                'MAPE':                store_json.get('MAPE', float('nan')),
                'MAE':                 store_json.get('MAE', float('nan')),
                'RMSE':                store_json.get('RMSE', float('nan')),
                'SMAPE':               store_json.get('SMAPE', float('nan')),
                'naive_wMAPE':         store_json.get('naive_wMAPE', float('nan')),
                'cv_mae_mean':         store_json.get('cv_mae_mean', 'N/A'),
                'cv_mae_std':          store_json.get('cv_mae_std', 'N/A'),
                'lb_pass':             store_json.get('lb_pass', 'N/A'),
            }
        else:
            eval_result = evaluate(data, copy.deepcopy(model))
            metrics     = _extract_metrics(eval_result, data, model)
    else:
        store_series = parsed[parsed['Store'] == store_id].set_index('Date')['Weekly_Sales'].sort_index()
        user_cfg     = _build_unknown_cfg(store_id, store_series)
        data         = preprocess_store(parsed, store_id, user_cfg)
        model        = fit_single(data)
        try:
            eval_result = evaluate(data, copy.deepcopy(model))
            metrics     = _extract_metrics(eval_result, data, model)
        except Exception:
            metrics = {k: float('nan') for k in ['wMAPE','DirectionalAccuracy','MAPE','MAE','RMSE','SMAPE','naive_wMAPE']}
            metrics.update({'beats_naive': False, 'model_type': 'single', 'sarima_order': 'N/A',
                            'seasonal_order': 'N/A', 'n_weeks': len(data['log_sales']),
                            'cv_mae_mean': 'N/A', 'cv_mae_std': 'N/A', 'lb_pass': 'N/A'})
        fc_dict = generate_forecast(model, data)

    fc_df = pd.DataFrame({
        'Week':           fc_dict['dates'],
        'Forecast_Sales': fc_dict['forecast'],
        'Lower_95':       fc_dict['lower_95'],
        'Upper_95':       fc_dict['upper_95'],
        'Is_Holiday':     fc_dict['is_holiday'],
    })
    return fc_df, metrics


# ── Alert computation ─────────────────────────────────────────────────────────

def _compute_alerts(forecast_df: pd.DataFrame, historical_df: pd.DataFrame, overstock_pct: float) -> list[dict]:
    hist_mean = historical_df['Weekly_Sales'].mean()
    alerts = []
    for _, row in forecast_df.iterrows():
        week     = row['Week'].strftime('%Y-%m-%d') if hasattr(row['Week'], 'strftime') else str(row['Week'])
        forecast = row['Forecast_Sales']
        is_hol   = bool(row['Is_Holiday'])

        if forecast < hist_mean * (1 - overstock_pct):
            sev = 'HIGH' if forecast < hist_mean * (1 - overstock_pct * 2) else 'MEDIUM'
            alerts.append({'week': week, 'type': 'Low Demand', 'severity': sev,
                'message': f"Forecast ${forecast:,.0f} is {(1 - forecast/hist_mean)*100:.1f}% below average. Reduce orders to avoid expiry waste."})
        if forecast > hist_mean * (1 + overstock_pct):
            alerts.append({'week': week, 'type': 'High Demand', 'severity': 'MEDIUM',
                'message': f"Forecast ${forecast:,.0f} is {(forecast/hist_mean - 1)*100:.1f}% above average. Increase procurement to avoid stock-out."})
        if is_hol:
            alerts.append({'week': week, 'type': 'Holiday Week', 'severity': 'INFO',
                'message': 'Holiday week — verify procurement schedule accounts for demand surge.'})
    return alerts


# ── Seasonal data ─────────────────────────────────────────────────────────────

def _build_seasonal(historical_df: pd.DataFrame) -> dict | None:
    if historical_df.empty:
        return None
    s = np.log1p(historical_df.set_index('Date')['Weekly_Sales'].sort_index())

    stl_result = None
    if len(s) >= 104:
        stl = STL(s, period=52, robust=True).fit()
        def to_pairs(series):
            return [[d.strftime('%Y-%m-%d'), float(v)] for d, v in series.items()]
        stl_result = {
            'observed': to_pairs(s),
            'trend':    to_pairs(stl.trend),
            'seasonal': to_pairs(stl.seasonal),
            'residual': to_pairs(stl.resid),
        }
    else:
        # Minimal placeholder so React doesn't crash
        pairs = [[d.strftime('%Y-%m-%d'), float(v)] for d, v in s.items()]
        stl_result = {'observed': pairs, 'trend': pairs, 'seasonal': pairs, 'residual': pairs}

    hist = historical_df.copy()
    hist['Month'] = hist['Date'].dt.month
    monthly = hist.groupby('Month')['Weekly_Sales'].mean()
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly_avg = [{'month': month_names[m], 'avg': float(v)} for m, v in monthly.items()]

    hist['week_of_year'] = hist['Date'].dt.isocalendar().week.astype(int)
    hist['year']         = hist['Date'].dt.year
    heatmap = [
        {'year': int(r['year']), 'week': int(r['week_of_year']), 'sales': float(r['Weekly_Sales'])}
        for _, r in hist[['year', 'week_of_year', 'Weekly_Sales']].iterrows()
    ]

    return {**stl_result, 'monthly_avg': monthly_avg, 'heatmap': heatmap}


# ── Serialise metrics for JSON (replace NaN/inf with None) ───────────────────

def _safe_metrics(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


# ── Session state ─────────────────────────────────────────────────────────────

def _init():
    defaults = {
        'app_state':   'idle',   # idle | ready | forecasting | done | error
        'user_df':     None,
        'store_id':    None,
        'forecast_df': None,
        'metrics':     None,
        'historical':  None,
        'alerts':      [],
        'seasonal':    None,
        'error':       None,
        'overstock':   0.20,
        'last_ts':     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init()
    ss = st.session_state

    # ── Build props to pass to React ──────────────────────────────────────────
    def _historical_rows() -> list[dict]:
        if ss['historical'] is None:
            return []
        h = ss['historical']
        return [
            {'date': r['Date'].strftime('%Y-%m-%d'), 'weekly_sales': float(r['Weekly_Sales'])}
            for _, r in h[['Date', 'Weekly_Sales']].iterrows()
        ]

    def _forecast_rows() -> list[dict] | None:
        if ss['forecast_df'] is None:
            return None
        fc = ss['forecast_df']
        return [
            {
                'week':           row['Week'].strftime('%Y-%m-%d') if hasattr(row['Week'], 'strftime') else str(row['Week']),
                'forecast_sales': float(row['Forecast_Sales']),
                'lower_95':       float(row['Lower_95']),
                'upper_95':       float(row['Upper_95']),
                'is_holiday':     int(row['Is_Holiday']),
            }
            for _, row in fc.iterrows()
        ]

    props = {
        'state':            ss['app_state'],
        'store_id':         ss['store_id'],
        'data_files_count': len(list_data_files()),
        'forecast':         _forecast_rows(),
        'metrics':          _safe_metrics(ss['metrics']) if ss['metrics'] else None,
        'historical':       _historical_rows(),
        'alerts':           ss['alerts'],
        'seasonal':         ss['seasonal'],
        'error':            ss['error'],
        'overstock_pct':    ss['overstock'],
    }

    # ── Render component and receive action ───────────────────────────────────
    action_value = _forecast_ui(
        **props,
        key='forecast_ui',
        default=None,
    )

    # ── Handle incoming action ────────────────────────────────────────────────
    if action_value is None:
        return

    ts     = action_value.get('timestamp')
    action = action_value.get('action')

    # Deduplicate: ignore repeated delivery of same timestamp
    if ts == ss['last_ts']:
        return
    ss['last_ts'] = ts

    if action == 'upload':
        csv_text = action_value.get('csvText', '')
        try:
            raw = pd.read_csv(io.StringIO(csv_text))
            err = _validate_upload(raw)
            if err:
                ss['error']     = err
                ss['app_state'] = 'idle'
                ss['user_df']   = None
            else:
                ss['user_df']   = raw
                parsed          = parse_and_sort(raw.copy())
                first_store     = int(parsed['Store'].iloc[0])
                historical      = parsed[parsed['Store'] == first_store].copy()
                ss['historical'] = historical
                ss['store_id']  = first_store
                ss['app_state'] = 'ready'
                ss['error']     = None
                ss['forecast_df'] = None
                ss['metrics']   = None
                ss['alerts']    = []
                ss['seasonal']  = _build_seasonal(historical)
        except Exception as e:
            ss['error']     = f"Could not read file: {e}"
            ss['app_state'] = 'idle'
        st.rerun()

    elif action == 'randomize':
        files = list_data_files()
        if not files:
            ss['error'] = "No bundled datasets found in app/data/random"
            st.rerun()
            return
        chosen = random.choice(files)
        try:
            raw = pd.read_csv(os.path.join(DATA_DIR, chosen))
            err = _validate_upload(raw)
            if err:
                ss['error'] = f"Bundled file '{chosen}' failed validation: {err}"
            else:
                ss['user_df']   = raw
                parsed          = parse_and_sort(raw.copy())
                first_store     = int(parsed['Store'].iloc[0])
                historical      = parsed[parsed['Store'] == first_store].copy()
                ss['historical'] = historical
                ss['store_id']  = first_store
                ss['app_state'] = 'ready'
                ss['error']     = None
                ss['forecast_df'] = None
                ss['metrics']   = None
                ss['alerts']    = []
                ss['seasonal']  = _build_seasonal(historical)
        except Exception as e:
            ss['error'] = f"Could not load '{chosen}': {e}"
        st.rerun()

    elif action == 'forecast':
        if ss['user_df'] is None:
            return
        overstock = float(action_value.get('overstock_pct', 0.20))
        ss['overstock']   = overstock
        ss['app_state']   = 'forecasting'
        ss['error']       = None
        # st.rerun()

    elif action == 'reset':
        for k in ('user_df', 'forecast_df', 'metrics', 'historical', 'seasonal'):
            ss[k] = None
        ss['alerts']    = []
        ss['app_state'] = 'idle'
        ss['store_id']  = None
        ss['error']     = None
        st.rerun()

    # ── Run forecast if state is 'forecasting' ────────────────────────────────
    if ss['app_state'] == 'forecasting' and ss['user_df'] is not None:
        try:
            fc_df, metrics = _run_forecast(ss['user_df'])
            # historical     = ss['historical'] or pd.DataFrame(columns=['Date', 'Weekly_Sales'])
            historical = ss['historical'] if ('historical' in ss and not ss['historical'].empty) else pd.DataFrame(columns=['Date', 'Weekly_Sales'])
            alerts         = _compute_alerts(fc_df, historical, ss['overstock'])
            
            ss['forecast_df'] = fc_df
            ss['metrics']     = metrics
            ss['alerts']      = alerts
            ss['app_state']   = 'done'
            ss['error']       = None
        except Exception as e:
            ss['error']     = f"Forecast generation failed: {e}"
            ss['app_state'] = 'ready'
        st.rerun()


if __name__ == '__main__':
    main()
