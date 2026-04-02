import { Component, StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error) {
    console.error("PaperMem UI crashed", error)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#f8fafc",
            color: "#0f172a",
            fontFamily: "Inter, SF Pro Text, system-ui, sans-serif",
            padding: "24px",
            boxSizing: "border-box",
          }}
        >
          <div
            style={{
              maxWidth: "460px",
              width: "100%",
              borderRadius: "24px",
              border: "1px solid rgba(148,163,184,0.25)",
              background: "#ffffff",
              boxShadow: "0 24px 60px rgba(15,23,42,0.08)",
              padding: "24px",
            }}
          >
            <div style={{ fontSize: "22px", fontWeight: 700 }}>PaperMem</div>
            <div style={{ marginTop: "10px", fontSize: "15px", lineHeight: 1.6, color: "#475569" }}>
              The UI hit a runtime error and was prevented from going fully blank.
              Reload the window once. If it happens again, the latest action likely triggered a frontend bug.
            </div>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
