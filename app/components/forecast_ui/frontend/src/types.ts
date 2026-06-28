// ── Actions sent from React → Streamlit ─────────────────────────────────────

export type Action =
  | { action: 'upload';    csvText: string;          timestamp: number }
  | { action: 'randomize';                            timestamp: number }
  | { action: 'forecast';  overstock_pct: number;    timestamp: number }
  | { action: 'reset';                               timestamp: number }

// ── Data shapes received from Streamlit → React ──────────────────────────────

/** One row of the 12-week forecast table */
export interface ForecastRow {
  week:           string   // 'YYYY-MM-DD'
  forecast_sales: number
  lower_95:       number
  upper_95:       number
  is_holiday:     number   // 0 | 1
}

/** Flat metrics dict — floats for numeric, string 'N/A' for missing */
export interface Metrics {
  wMAPE:               number | 'N/A'
  DirectionalAccuracy: number | 'N/A'
  beats_naive:         boolean
  model_type:          string
  sarima_order:        string
  seasonal_order:      string
  n_weeks:             number | 'N/A'
  MAPE:                number | 'N/A'
  MAE:                 number | 'N/A'
  RMSE:                number | 'N/A'
  SMAPE:               number | 'N/A'
  naive_wMAPE:         number | 'N/A'
  cv_mae_mean:         number | 'N/A'
  cv_mae_std:          number | 'N/A'
  lb_pass:             boolean | 'N/A'
}

/** One row of historical weekly sales */
export interface HistoricalRow {
  date:         string   // 'YYYY-MM-DD'
  weekly_sales: number
}

/** One alert entry */
export interface Alert {
  week:     string
  type:     'Low Demand' | 'High Demand' | 'Holiday Week'
  severity: 'HIGH' | 'MEDIUM' | 'INFO'
  message:  string
}

/** STL decomposition component — list of [date, value] pairs */
export type STLSeries = [string, number][]

export interface SeasonalData {
  observed: STLSeries
  trend:    STLSeries
  seasonal: STLSeries
  residual: STLSeries
  monthly_avg: { month: string; avg: number }[]
  heatmap:     { year: number; week: number; sales: number }[]
}

/** Top-level props object Streamlit passes to the component */
export interface AppProps {
  /** 'idle' | 'ready' | 'forecasting' | 'done' | 'error' */
  state:            string
  store_id:         number | null
  data_files_count: number
  forecast:         ForecastRow[] | null
  metrics:          Metrics | null
  historical:       HistoricalRow[]
  alerts:           Alert[]
  seasonal:         SeasonalData | null
  error:            string | null
  overstock_pct:    number
}
