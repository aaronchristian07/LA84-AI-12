import { useRef, useState, type DragEvent, type ChangeEvent } from 'react'
import { Streamlit } from '../streamlit'
import { Check, FolderPlus } from 'react-feather'

interface Props {
  state:           string
  dataFilesCount:  number
  overstock_pct:   number
  hasData:         boolean
  hasForecast:     boolean
  statusMessage:   string | null
  error:           string | null
}

export default function UploadPanel({
  state,
  dataFilesCount,
  overstock_pct,
  hasData,
  hasForecast,
  statusMessage,
  error,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null)

  // Local state: track whether this session has a pending/confirmed upload
  // without waiting for the Streamlit round-trip to update hasData.
  const [localReady, setLocalReady]   = useState(false)
  const [dragging, setDragging]       = useState(false)
  const [fileName, setFileName]       = useState<string | null>(null)
  const [localError, setLocalError]   = useState<string | null>(null)
  const [threshold, setThreshold]     = useState(Math.round(overstock_pct * 100))

  const isForecasting = state === 'forecasting'

  // Either Streamlit confirmed data is ready, or we locally sent an upload
  const canForecast = (hasData || localReady) && !isForecasting

  // Randomize is enabled as long as there are bundled files.
  // dataFilesCount comes from Streamlit; if the prop hasn't arrived yet
  // we default to enabling the button so it isn't permanently stuck disabled.
  const canRandomize = dataFilesCount !== 0 && !isForecasting

  function readAndSend(file: File) {
    if (!file.name.endsWith('.csv')) {
      setLocalError('Only .csv files are accepted.')
      return
    }
    setLocalError(null)
    setFileName(file.name)
    const reader = new FileReader()
    reader.onload = e => {
      const csvText = e.target?.result as string
      setLocalReady(true)
      Streamlit.setComponentValue({
        action:    'upload',
        csvText,
        timestamp: Date.now(),
      })
    }
    reader.onerror = () => {
      setLocalError('Failed to read file.')
      setLocalReady(false)
    }
    reader.readAsText(file)
  }

  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) readAndSend(f)
    // Reset so the same file can be re-selected if needed
    e.target.value = ''
  }

  function onDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) readAndSend(f)
  }

  function onRandomize() {
    setFileName(null)
    setLocalReady(true)   // optimistically enable forecast after randomize
    setLocalError(null)
    Streamlit.setComponentValue({ action: 'randomize', timestamp: Date.now() })
  }

  function onForecast() {
    Streamlit.setComponentValue({
      action:        'forecast',
      overstock_pct: threshold / 100,
      timestamp:     Date.now(),
    })
  }

  const dropClass = [
    'upload-drop-zone',
    dragging ? 'drag-over' : '',
    fileName || hasData ? 'has-file' : '',
  ].filter(Boolean).join(' ')

  const fileLabel = fileName
    ? fileName
    : hasData
      ? `Data loaded`
      : hasData
        ? 'Data loaded'
        : 'Drop CSV here or click to browse'

  // Show Streamlit's error above local error; local error shown if no server error
  const displayError = error || localError

  return (
    <div className="upload-panel">
      <h2>Upload Your Data</h2>

      <div className="upload-row">
        <div
          className={dropClass}
          onDragEnter={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDragOver={e => e.preventDefault()}
          onDrop={onDrop}
          onClick={() => fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            onChange={onFileChange}
            onClick={e => e.stopPropagation()}
          />
          <div className="drop-icon">
            <FolderPlus size={28} />
          </div>
          <div className="drop-label">
            {fileName || hasData
              ? <strong>{fileLabel}</strong>
              : <><strong>Choose a file</strong> or drag and drop</>}
          </div>
          <div className="drop-hint">
            CSV: Date (DD-MM-YYYY), Weekly_Sales, Holiday_Flag,
            Temperature, Fuel_Price, CPI, Unemployment · Min 65 weeks
          </div>
        </div>

        <button
          className="btn-randomize"
          onClick={onRandomize}
          disabled={!canRandomize}
          title={
            dataFilesCount === 0
              ? 'No bundled datasets found in app/data/random'
              : 'Pick a random bundled dataset'
          }
        >
          Randomize Dataset
        </button>
      </div>

      {statusMessage && !displayError && (
        <div className="upload-status success">{statusMessage}</div>
      )}
      {displayError && (
        <div className="upload-status error">{displayError}</div>
      )}

      {/* Threshold + forecast button */}
      <div className="controls-row" style={{ marginTop: 20 }}>
        <span className="threshold-label">Alert threshold</span>
        <input
          type="range"
          min={5} max={50} step={5}
          value={threshold}
          onChange={e => setThreshold(Number(e.target.value))}
          disabled={isForecasting}
        />
        <span className="threshold-value">{threshold}%</span>

        <button
          className="btn-forecast"
          onClick={onForecast}
          disabled={!canForecast}
        >
          {isForecasting ? 'Generating...' : 'Generate Forecast'}
        </button>

        {hasForecast && !isForecasting && (
          <div className="forecast-status">
            <Check size={16} />
            Forecast ready
          </div>
        )}
      </div>

      {isForecasting && (
        <div className="spinner-overlay">
          <div className="spinner" />
          Generating 12-week SARIMAX forecast...
        </div>
      )}
    </div>
  )
}
