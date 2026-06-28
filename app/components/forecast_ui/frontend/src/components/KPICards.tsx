import type { Metrics } from '../types'

interface Props {
  metrics: Metrics
}

function fmt(v: number | 'N/A', type: 'pct' | 'num' | 'str' = 'str'): string {
  if (v === 'N/A' || v === null || v === undefined) return 'N/A'
  if (type === 'pct') return `${(v as number * 100).toFixed(1)}%`
  if (type === 'num') return (v as number).toFixed(4)
  return String(v)
}

function wMapeClass(v: number | 'N/A'): string {
  if (v === 'N/A') return ''
  const n = v as number
  if (n < 0.10) return 'good'
  if (n < 0.20) return 'warn'
  return 'bad'
}

function daClass(v: number | 'N/A'): string {
  if (v === 'N/A') return ''
  const n = v as number
  if (n >= 0.65) return 'good'
  if (n >= 0.50) return 'warn'
  return 'bad'
}

export default function KPICards({ metrics }: Props) {
  const mtype = typeof metrics.model_type === 'string'
    ? metrics.model_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    : 'N/A'

  return (
    <div className="kpi-grid">
      <div className="kpi-card">
        <div className="kpi-label">wMAPE</div>
        <div className={`kpi-value ${wMapeClass(metrics.wMAPE)}`}>
          {fmt(metrics.wMAPE, 'pct')}
        </div>
        <div className="kpi-sub">Weighted mean abs % error</div>
      </div>

      <div className="kpi-card">
        <div className="kpi-label">Dir. Acc.</div>
        <div className={`kpi-value ${daClass(metrics.DirectionalAccuracy)}`}>
          {fmt(metrics.DirectionalAccuracy, 'pct')}
        </div>
        <div className="kpi-sub">Directional accuracy</div>
      </div>

      <div className="kpi-card">
        <div className="kpi-label">Beats Naive</div>
        <div className={`kpi-value ${metrics.beats_naive ? 'good' : 'bad'}`}>
          {metrics.beats_naive ? 'Yes' : 'No'}
        </div>
        <div className="kpi-sub">vs lag-1 naive baseline</div>
      </div>

      <div className="kpi-card">
        <div className="kpi-label">Model Type</div>
        <div className="kpi-value" style={{ fontSize: '18px' }}>{mtype}</div>
        <div className="kpi-sub">{metrics.sarima_order} · {metrics.seasonal_order}</div>
      </div>
    </div>
  )
}
