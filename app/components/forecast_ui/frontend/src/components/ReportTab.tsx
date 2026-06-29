import { Download } from 'react-feather'
import type { ForecastRow, Metrics } from '../types'

interface Props {
  forecast: ForecastRow[]
  metrics:  Metrics
  storeId:  number | null
}

function fmtMetric(key: string, v: number | boolean | string | 'N/A'): string {
  if (v === 'N/A' || v === null || v === undefined) return 'N/A'
  if (typeof v === 'boolean') return v ? 'Yes' : 'No'
  if (typeof v === 'number') {
    const pctKeys = ['wMAPE', 'SMAPE', 'MAPE', 'DirectionalAccuracy', 'naive_wMAPE']
    if (pctKeys.includes(key)) return `${(v * 100).toFixed(2)}%`
    if (key === 'MAE' || key === 'RMSE') return `$${v.toLocaleString('en-US', { maximumFractionDigits: 2 })}`
    return v.toFixed(4)
  }
  return String(v)
}

function fmtUSD(v: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency', currency: 'USD', maximumFractionDigits: 0,
  }).format(v)
}

function buildForecastCSV(forecast: ForecastRow[], storeId: number | null): string {
  const header = 'Store,Week,Forecast_Sales,Lower_95,Upper_95,Is_Holiday'
  const rows = forecast.map(r =>
    `Store ${storeId},${r.week},${r.forecast_sales.toFixed(2)},${r.lower_95.toFixed(2)},${r.upper_95.toFixed(2)},${r.is_holiday}`
  )
  return [header, ...rows].join('\n')
}

function buildReportCSV(forecast: ForecastRow[], metrics: Metrics, storeId: number | null): string {
  const wmape = fmtMetric('wMAPE', metrics.wMAPE)
  const da    = fmtMetric('DirectionalAccuracy', metrics.DirectionalAccuracy)
  const mtype = typeof metrics.model_type === 'string' ? metrics.model_type : 'N/A'
  const header = 'Store,Week,Forecast_Sales,Lower_95,Upper_95,Is_Holiday,wMAPE,DirectionalAccuracy,Model_Type'
  const rows = forecast.map(r =>
    `Store ${storeId},${r.week},${r.forecast_sales.toFixed(2)},${r.lower_95.toFixed(2)},${r.upper_95.toFixed(2)},${r.is_holiday},${wmape},${da},${mtype}`
  )
  return [header, ...rows].join('\n')
}

function downloadCSV(content: string, filename: string) {
  const blob = new Blob([content], { type: 'text/csv' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export default function ReportTab({ forecast, metrics, storeId }: Props) {
  const label   = storeId != null ? `Store ${storeId}` : '—'
  const mtype   = typeof metrics.model_type === 'string'
    ? metrics.model_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    : 'N/A'

  const totalSales = forecast.reduce((s, r) => s + r.forecast_sales, 0)
  const avgSales   = totalSales / (forecast.length || 1)
  const peakRow    = forecast.reduce((best, r) =>
    r.forecast_sales > best.forecast_sales ? r : best, forecast[0])

  const summaryRows: [string, string][] = [
    ['Model Type',     mtype],
    ['SARIMA Order',   metrics.sarima_order],
    ['Seasonal Order', metrics.seasonal_order],
    ['Training Weeks', String(metrics.n_weeks)],
  ]

  const perfKeys: (keyof Metrics)[] = [
    'wMAPE', 'SMAPE', 'MAPE', 'MAE', 'RMSE',
    'DirectionalAccuracy', 'naive_wMAPE', 'beats_naive',
    'cv_mae_mean', 'cv_mae_std', 'lb_pass',
  ]

  return (
    <div className="tab-content">
      <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 20 }}>
        Demand Report
      </h2>

      {/* Forecast summary */}
      <div className="report-section">
        <h3>Forecast Summary (Next 12 Weeks)</h3>
        <div className="summary-cards">
          <div className="summary-card">
            <div className="s-label">Total Forecast Sales</div>
            <div className="s-value">{fmtUSD(totalSales)}</div>
          </div>
          <div className="summary-card">
            <div className="s-label">Weekly Average</div>
            <div className="s-value">{fmtUSD(avgSales)}</div>
          </div>
          <div className="summary-card">
            <div className="s-label">Peak Week</div>
            <div className="s-value" style={{ fontSize: 16 }}>
              {peakRow ? peakRow.week : '—'}
            </div>
          </div>
        </div>
      </div>

      {/* Model summary */}
      <div className="report-section">
        <h3>Model Summary</h3>
        <table className="data-table">
          <tbody>
            {summaryRows.map(([k, v]) => (
              <tr key={k}>
                <td style={{ color: '#7c8099', width: '40%' }}>{k}</td>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Performance metrics */}
      <div className="report-section">
        <h3>Model Performance</h3>
        <table className="data-table">
          <thead>
            <tr><th>Metric</th><th>Value</th></tr>
          </thead>
          <tbody>
            {perfKeys.map(k => (
              <tr key={k}>
                <td style={{ color: '#7c8099' }}>{k}</td>
                <td>{fmtMetric(k, metrics[k] as any)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Downloads */}
      <div className="report-section">
        <h3>Downloads</h3>
        <div className="download-row">
          <button
            className="btn-download"
            onClick={() => downloadCSV(
              buildForecastCSV(forecast, storeId),
              `store_${storeId}_forecast_12w.csv`
            )}
          >
            <Download size={12} />
            Forecast CSV
          </button>
          <button
            className="btn-download"
            onClick={() => downloadCSV(
              buildReportCSV(forecast, metrics, storeId),
              `store_${storeId}_demand_report.csv`
            )}
          >
            <Download size={12} />
            Full Report CSV
          </button>
        </div>
      </div>
    </div>
  )
}
