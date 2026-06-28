import {
  ComposedChart,
  Line,
  Area,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import type { ForecastRow, HistoricalRow } from '../types'

interface Props {
  forecast:   ForecastRow[]
  historical: HistoricalRow[]
}

function fmtDollar(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`
  if (v >= 1_000)     return `$${(v / 1_000).toFixed(1)}K`
  return `$${v.toFixed(0)}`
}

// Custom tooltip
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#1a1d27', border: '1px solid #2d3147',
      borderRadius: 8, padding: '10px 14px', fontSize: 12,
    }}>
      <div style={{ color: '#7c8099', marginBottom: 6 }}>{label}</div>
      {payload.map((p: any) => (
        p.value != null && (
          <div key={p.name} style={{ color: p.color ?? p.fill, marginBottom: 2 }}>
            {p.name}: {typeof p.value === 'number' ? fmtDollar(p.value) : p.value}
          </div>
        )
      ))}
    </div>
  )
}

export default function ForecastChart({ forecast, historical }: Props) {
  // Use last 52 weeks of historical
  const histSlice = historical.slice(-52).map(r => ({
    date:         r.date,
    historical:   r.weekly_sales,
  }))

  // Forecast rows with CI — recharts Area needs [lower, upper]
  const fcRows = forecast.map(r => ({
    date:         r.week,
    forecast:     r.forecast_sales,
    ci_band:      [r.lower_95, r.upper_95] as [number, number],
    lower:        r.lower_95,
    upper:        r.upper_95,
    is_holiday:   r.is_holiday,
  }))

  // Holiday reference lines
  const holidayWeeks = fcRows
    .filter(r => r.is_holiday === 1)
    .map(r => r.date)

  // Merge into one series for the x-axis (historical + forecast)
  const combined = [
    ...histSlice,
    ...fcRows.map(r => ({ date: r.date, forecast: r.forecast, lower: r.lower, upper: r.upper, is_holiday: r.is_holiday })),
  ]

  // Boundary date for the vertical split line
  const splitDate = histSlice.length > 0 ? histSlice[histSlice.length - 1].date : null

  return (
    <ResponsiveContainer width="100%" height={380}>
      <ComposedChart data={combined} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2d3147" />
        <XAxis
          dataKey="date"
          tick={{ fill: '#7c8099', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: '#2d3147' }}
          interval="preserveStartEnd"
          tickCount={8}
        />
        <YAxis
          tickFormatter={fmtDollar}
          tick={{ fill: '#7c8099', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          width={72}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 12, color: '#7c8099', paddingTop: 12 }}
        />

        {/* 95% CI band */}
        <Area
          dataKey="upper"
          stroke="none"
          fill="rgba(245,158,11,0.12)"
          legendType="none"
          name="95% CI upper"
          dot={false}
          activeDot={false}
          connectNulls
        />
        <Area
          dataKey="lower"
          stroke="none"
          fill="var(--bg)"
          legendType="none"
          name="95% CI lower"
          dot={false}
          activeDot={false}
          connectNulls
          fillOpacity={1}
        />

        {/* Historical */}
        <Line
          dataKey="historical"
          name="Historical"
          stroke="#4f8ef7"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: '#4f8ef7' }}
          connectNulls
        />

        {/* Forecast */}
        <Line
          dataKey="forecast"
          name="Forecast"
          stroke="#f59e0b"
          strokeWidth={2}
          dot={{ r: 3, fill: '#f59e0b', strokeWidth: 0 }}
          activeDot={{ r: 5 }}
          connectNulls
        />

        {/* Vertical split between history and forecast */}
        {splitDate && (
          <ReferenceLine
            x={splitDate}
            stroke="#2d3147"
            strokeDasharray="4 4"
            strokeWidth={1.5}
          />
        )}

        {/* Holiday markers */}
        {holidayWeeks.map(d => (
          <ReferenceLine
            key={d}
            x={d}
            stroke="#ef4444"
            strokeDasharray="3 3"
            strokeWidth={1}
            label={{ value: '🎄', position: 'top', fontSize: 10 }}
          />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
