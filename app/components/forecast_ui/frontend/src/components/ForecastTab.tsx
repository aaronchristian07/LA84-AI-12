import KPICards from './KPICards'
import ForecastChart from './ForecastChart'
import type { ForecastRow, HistoricalRow, Metrics } from '../types'

interface Props {
  forecast:   ForecastRow[]
  metrics:    Metrics
  historical: HistoricalRow[]
  storeId:    number | null
}

function fmtUSD(v: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(v)
}

export default function ForecastTab({ forecast, metrics, historical, storeId }: Props) {
  const label = storeId != null ? `Store ${storeId}` : '—'

  return (
    <div className="tab-content">
      <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 20 }}>
        {label} — 12-Week Sales Forecast
      </h2>

      <KPICards metrics={metrics} />

      <div className="chart-card">
        <h3>Weekly Sales Forecast vs Historical</h3>
        <ForecastChart forecast={forecast} historical={historical} />
      </div>

      <details className="table-expander chart-card" style={{ padding: '16px 20px' }}>
        <summary>Forecast data</summary>
        <table className="data-table" style={{ marginTop: 12 }}>
          <thead>
            <tr>
              <th>Week</th>
              <th>Forecast</th>
              <th>Lower 95%</th>
              <th>Upper 95%</th>
              <th>Holiday</th>
            </tr>
          </thead>
          <tbody>
            {forecast.map(r => (
              <tr key={r.week}>
                <td>{r.week}</td>
                <td>{fmtUSD(r.forecast_sales)}</td>
                <td style={{ color: '#7c8099' }}>{fmtUSD(r.lower_95)}</td>
                <td style={{ color: '#7c8099' }}>{fmtUSD(r.upper_95)}</td>
                <td>
                  {r.is_holiday === 1
                    ? <span className="holiday-badge">Holiday</span>
                    : <span style={{ color: '#7c8099' }}>—</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  )
}
