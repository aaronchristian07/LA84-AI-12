import type { Alert } from '../types'

interface Props {
  alerts:  Alert[]
}

const SEVERITY_ORDER: Record<string, number> = { HIGH: 0, MEDIUM: 1, INFO: 2 }

export default function AlertsTab({ alerts }: Props) {
  const sorted = [...alerts].sort(
    (a, b) => (SEVERITY_ORDER[a.severity] ?? 9) - (SEVERITY_ORDER[b.severity] ?? 9)
  )

  // Summary counts
  const counts: Record<string, Record<string, number>> = {}
  for (const a of alerts) {
    counts[a.type] = counts[a.type] ?? {}
    counts[a.type][a.severity] = (counts[a.type][a.severity] ?? 0) + 1
  }

  return (
    <div className="tab-content">
      <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 20 }}>
        Smart Alert System
      </h2>

      {alerts.length === 0 ? (
        <div className="empty-state">
          <div style={{ fontSize: 28, marginBottom: 8 }}>✅</div>
          No alerts for the forecast period.
        </div>
      ) : (
        <>
          <div className="alerts-list">
            {sorted.map((alert, i) => (
              <div key={i} className={`alert-card ${alert.severity}`}>
                <div className="alert-dot" />
                <div className="alert-body">
                  <div className="alert-header">
                    <span className="alert-severity">{alert.severity}</span>
                    <span className="alert-type">{alert.type}</span>
                    <span className="alert-week">Week of {alert.week}</span>
                  </div>
                  <div className="alert-message">{alert.message}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="alert-summary-table chart-card">
            <h3>Alert Summary</h3>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Type</th>
                  <th>Severity</th>
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(counts).flatMap(([type, sevMap]) =>
                  Object.entries(sevMap).map(([sev, count]) => (
                    <tr key={`${type}-${sev}`}>
                      <td>{type}</td>
                      <td>
                        <span style={{
                          color: sev === 'HIGH' ? 'var(--red)' :
                                 sev === 'MEDIUM' ? 'var(--yellow)' : 'var(--blue-info)',
                          fontWeight: 600,
                          fontSize: 12,
                        }}>
                          {sev}
                        </span>
                      </td>
                      <td>{count}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
