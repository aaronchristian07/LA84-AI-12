/** Messages Component → Streamlit host */
const enum MsgType {
  COMPONENT_READY    = 'streamlit:componentReady',
  SET_COMPONENT_VALUE = 'streamlit:setComponentValue',
  SET_FRAME_HEIGHT   = 'streamlit:setFrameHeight',
}

const RENDER_EVENT = 'streamlit:render'

function sendBackMsg(type: string, data: Record<string, unknown>): void {
  window.parent.postMessage(
    { isStreamlitMessage: true, type, ...data },
    '*'
  )
}

let lastFrameHeight: number | undefined
let listenerRegistered = false

// Internal EventTarget — render events are dispatched here
const events = new EventTarget()

function onMessageEvent(event: MessageEvent): void {
  const type = event.data?.type
  if (type !== RENDER_EVENT) return

  const data = event.data
  const args = data.args ?? {}
  const disabled = Boolean(data.disabled)
  const theme = data.theme

  const detail = { args, disabled, theme }
  events.dispatchEvent(new CustomEvent(RENDER_EVENT, { detail }))
}

export const Streamlit = {
  /**
   * Register a callback for Streamlit render events.
   * Must be called before setComponentReady.
   */
  onRender(callback: (args: Record<string, unknown>) => void): void {
    events.addEventListener(RENDER_EVENT, (e) => {
      callback((e as CustomEvent).detail.args)
    })
  },

  /**
   * Signal to Streamlit that the component is ready.
   * Registers the window message listener and sends componentReady.
   */
  setComponentReady(): void {
    if (!listenerRegistered) {
      window.addEventListener('message', onMessageEvent)
      listenerRegistered = true
    }
    sendBackMsg(MsgType.COMPONENT_READY, { apiVersion: 1 })
  },

  /**
   * Send a value back to Python. Triggers a Streamlit rerun.
   */
  setComponentValue(value: unknown): void {
    sendBackMsg(MsgType.SET_COMPONENT_VALUE, {
      value,
      dataType: 'json',
    })
  },

  /**
   * Update the iframe height. Called automatically by ResizeObserver in App.tsx.
   */
  setFrameHeight(height?: number): void {
    if (height === undefined) {
      height = document.body.scrollHeight
    }
    if (height === lastFrameHeight) return
    lastFrameHeight = height
    sendBackMsg(MsgType.SET_FRAME_HEIGHT, { height })
  },
}