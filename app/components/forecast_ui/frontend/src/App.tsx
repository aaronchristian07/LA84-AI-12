import { useState, useEffect, useRef } from 'react'
import { Streamlit } from './streamlit'
import type { AppProps } from './types'
import UploadPanel from './components/UploadPanel'
import ForecastTab from './components/ForecastTab'
import AlertsTab   from './components/AlertsTab'
import SeasonalTab from './components/SeasonalTab'
import ReportTab   from './components/ReportTab'

const TABS = ['Forecast', 'Smart Alerts', 'Seasonal Analysis', 'Demand Report'] as const
type Tab = typeof TABS[number]

const DEFAULT_PROPS: AppProps = {
  state:            'idle',
  store_id:         null,
  data_files_count: -1,
  forecast:         null,
  metrics:          null,
  historical:       [],
  alerts:           [],
  seasonal:         null,
  error:            null,
  overstock_pct:    0.20,
}

export default function App() {
  const [props, setProps]         = useState<AppProps>(DEFAULT_PROPS)
  const [activeTab, setActiveTab] = useState<Tab>('Forecast')
  const [statusMsg, setStatusMsg] = useState<string | null>(null)
  const rootRef                   = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // 1. Register render handler FIRST
    Streamlit.onRender((args) => {
      const incoming = args as unknown as AppProps
      setProps(incoming)

      if (incoming.state === 'done' && incoming.forecast) {
        setActiveTab('Forecast')
      }
      if (
        (incoming.state === 'ready' || incoming.state === 'done') &&
        incoming.store_id != null
      ) {
        setStatusMsg(`Store ${incoming.store_id} data loaded`)
      }
    })

    // 2. Signal ready AFTER handler is registered
    Streamlit.setComponentReady()
  }, [])

  // Keep iframe height in sync
  useEffect(() => {
    if (!rootRef.current) return
    const ro = new ResizeObserver(() => {
      Streamlit.setFrameHeight(document.documentElement.scrollHeight)
    })
    ro.observe(rootRef.current)
    return () => ro.disconnect()
  }, [])

  const hasForecast = props.forecast != null && props.forecast.length > 0
  const hasData     = props.state !== 'idle' && props.historical.length > 0

  return (
    <div className="app" ref={rootRef}>
      <header className="app-header">
        <h1>Retail Demand Forecasting</h1>
        <p>SARIMAX model · 12-week horizon · Food waste reduction</p>
      </header>

      <UploadPanel
        state={props.state}
        dataFilesCount={props.data_files_count}
        overstock_pct={props.overstock_pct}
        hasData={hasData}
        hasForecast={hasForecast}
        storeId={props.store_id}
        statusMessage={statusMsg}
        error={props.error}
      />

      <nav className="tabs">
        {TABS.map(tab => (
          <button
            key={tab}
            className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </nav>

      {activeTab === 'Forecast' && (
        hasForecast && props.metrics
          ? <ForecastTab
              forecast={props.forecast!}
              metrics={props.metrics}
              historical={props.historical}
              storeId={props.store_id}
            />
          : <div className="empty-state">
              Generate a forecast to view results here.
            </div>
      )}

      {activeTab === 'Smart Alerts' && (
        hasForecast
          ? <AlertsTab alerts={props.alerts} storeId={props.store_id} />
          : <div className="empty-state">Generate a forecast to view alerts.</div>
      )}

      {activeTab === 'Seasonal Analysis' && (
        props.seasonal && props.historical.length > 0
          ? <SeasonalTab seasonal={props.seasonal} storeId={props.store_id} />
          : <div className="empty-state">
              Upload data and generate a forecast to view seasonal analysis.
            </div>
      )}

      {activeTab === 'Demand Report' && (
        hasForecast && props.metrics
          ? <ReportTab
              forecast={props.forecast!}
              metrics={props.metrics}
              storeId={props.store_id}
            />
          : <div className="empty-state">
              Generate a forecast to view the demand report.
            </div>
      )}

      <footer className="app-footer">
        Model: SARIMAX · Dataset: Walmart Store Sales
        {props.metrics?.n_weeks && props.metrics.n_weeks !== 'N/A'
          ? ` · Trained on ${props.metrics.n_weeks} weeks`
          : ''}
      </footer>
    </div>
  )
}