import { useState } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, Cell,
} from 'recharts'
import type { SeasonalData, STLSeries } from '../types'

interface Props {
  seasonal: SeasonalData
  storeId:  number | null
}

type STLKey = 'observed' | 'trend' | 'seasonal' | 'residual'

const STL_COLORS: Record<STLKey, string> = {
  observed: '#4f8ef7',
  trend:    '#f59e0b',
  seasonal: '#22c55e',
  residual: '#ef4444',
}

const STL_LABELS: Record<STLKey, string> = {
  observed: 'Observed (log1p)',
  trend:    'Trend',
  seasonal: 'Seasonal',
  residual: 'Residual',
}

function stlToRows(series: STLSeries): { date: string; value: number }[] {
  return series.map(([date, value]) => ({ date, value }))
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#1a1d27', border: '1px solid #2d3147',
      borderRadius: 8, padding: '8px 12px', fontSize: 12,
    }}>
      <div style={{ color: '#7c8099', marginBottom: 4 }}>{label}</div>
      {payload.map((p: any) => (
        <div key={p.name} style={{ color: p.color }}>
          {p.name}: {Number(p.value).toFixed(3)}
        </div>
      ))}
    </div>
  )
}

const BarTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#1a1d27', border: '1px solid #2d3147',
      borderRadius: 8, padding: '8px 12px', fontSize: 12,
    }}>
      <div style={{ color: '#7c8099', marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#4f8ef7' }}>
        Avg Sales: ${Number(payload[0].value).toLocaleString('en-US', { maximumFractionDigits: 0 })}
      </div>
    </div>
  )
}

export default function SeasonalTab({ seasonal, storeId }: Props) {
  const label = storeId != null ? `Store ${storeId}` : '—'
  const [activeSTL, setActiveSTL] = useState<Set<STLKey>>(
    new Set(['observed', 'trend'])
  )

  const toggleSTL = (key: STLKey) => {
    setActiveSTL(prev => {
      const next = new Set(prev)
      if (next.has(key)) { if (next.size > 1) next.delete(key) }
      else next.add(key)
      return next
    })
  }

  // Merge all STL series by date for the composed chart
  const stlKeys: STLKey[] = ['observed', 'trend', 'seasonal', 'residual']
  const dateMap: Record<string, Record<string, number>> = {}
  for (const key of stlKeys) {
    for (const [date, value] of seasonal[key]) {
      dateMap[date] = dateMap[date] ?? {}
      dateMap[date][key] = value
    }
  }
  const stlRows = Object.entries(dateMap)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, vals]) => ({ date, ...vals }))

  // Heatmap: group by year
  const heatmapByYear: Record<number, Record<number, number>> = {}
  for (const { year, week, sales } of seasonal.heatmap) {
    heatmapByYear[year] = heatmapByYear[year] ?? {}
    heatmapByYear[year][week] = sales
  }
  const years = Object.keys(heatmapByYear).map(Number).sort()
  const allSales = seasonal.heatmap.map(h => h.sales)
  const minSales = Math.min(...allSales)
  const maxSales = Math.max(...allSales)

  function heatColor(sales: number): string {
    const t = maxSales === minSales ? 0.5 : (sales - minSales) / (maxSales - minSales)
    // dark blue → bright blue
    const r = Math.round(10 + t * 40)
    const g = Math.round(30 + t * 100)
    const b = Math.round(80 + t * 175)
    return `rgb(${r},${g},${b})`
  }

  return (
    <div className="tab-content">
      <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 20 }}>
        {label} — Seasonal Demand Analysis
      </h2>

      {/* STL decomposition */}
      <div className="chart-card">
        <h3>STL Decomposition</h3>
        <div className="toggle-row">
          {stlKeys.map(key => (
            <button
              key={key}
              className={`toggle-btn ${activeSTL.has(key) ? 'active' : ''}`}
              onClick={() => toggleSTL(key)}
              style={activeSTL.has(key) ? { borderColor: STL_COLORS[key], color: STL_COLORS[key], background: STL_COLORS[key] + '22' } : {}}
            >
              {STL_LABELS[key]}
            </button>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={stlRows} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2d3147" />
            <XAxis dataKey="date" tick={{ fill: '#7c8099', fontSize: 11 }} tickLine={false} axisLine={{ stroke: '#2d3147' }} interval="preserveStartEnd" tickCount={6} />
            <YAxis tick={{ fill: '#7c8099', fontSize: 11 }} tickLine={false} axisLine={false} width={56} tickFormatter={v => v.toFixed(2)} />
            <Tooltip content={<CustomTooltip />} />
            {stlKeys.filter(k => activeSTL.has(k)).map(key => (
              <Line
                key={key}
                dataKey={key}
                name={STL_LABELS[key]}
                stroke={STL_COLORS[key]}
                strokeWidth={1.5}
                dot={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Monthly average bar */}
      <div className="chart-card">
        <h3>Average Weekly Sales by Month</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={seasonal.monthly_avg} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2d3147" vertical={false} />
            <XAxis dataKey="month" tick={{ fill: '#7c8099', fontSize: 12 }} tickLine={false} axisLine={{ stroke: '#2d3147' }} />
            <YAxis tick={{ fill: '#7c8099', fontSize: 11 }} tickLine={false} axisLine={false} width={72}
              tickFormatter={v => `$${(v / 1000).toFixed(0)}K`} />
            <Tooltip content={<BarTooltip />} />
            <Bar dataKey="avg" name="Avg Sales" radius={[4, 4, 0, 0]}>
              {seasonal.monthly_avg.map((_, i) => (
                <Cell key={i} fill={`hsl(${210 + i * 4},70%,${40 + i * 2}%)`} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Heatmap */}
      <div className="chart-card">
        <h3>Weekly Sales Heatmap by Year</h3>
        <div style={{ overflowX: 'auto' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 700 }}>
            {/* Week labels */}
            <div style={{ display: 'flex', marginLeft: 48 }}>
              {Array.from({ length: 52 }, (_, i) => i + 1)
                .filter(w => w % 4 === 0)
                .map(w => (
                  <div key={w} style={{
                    width: `${100 / 52}%`,
                    marginLeft: `${((w - 1) % 4 === 0 ? (w - 4) : 0) / 52 * 100}%`,
                    fontSize: 10,
                    color: '#7c8099',
                    textAlign: 'center',
                    flexShrink: 0,
                  }}>
                    W{w}
                  </div>
                ))
              }
            </div>
            {years.map(year => (
              <div key={year} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                <div style={{ width: 44, fontSize: 11, color: '#7c8099', textAlign: 'right', flexShrink: 0 }}>
                  {year}
                </div>
                <div style={{ display: 'flex', flex: 1, gap: 2 }}>
                  {Array.from({ length: 52 }, (_, i) => {
                    const week = i + 1
                    const sales = heatmapByYear[year]?.[week]
                    return (
                      <div
                        key={week}
                        title={sales != null ? `W${week} ${year}: $${sales.toLocaleString()}` : `W${week} ${year}: no data`}
                        style={{
                          flex: 1,
                          height: 18,
                          borderRadius: 2,
                          background: sales != null ? heatColor(sales) : '#1a1d27',
                          border: '1px solid #0e1117',
                          cursor: 'default',
                        }}
                      />
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
          {/* Legend */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10, fontSize: 11, color: '#7c8099' }}>
            <span>Low</span>
            <div style={{
              width: 120, height: 8, borderRadius: 4,
              background: 'linear-gradient(to right, rgb(10,30,80), rgb(50,130,255))',
            }} />
            <span>High</span>
          </div>
        </div>
      </div>
    </div>
  )
}
