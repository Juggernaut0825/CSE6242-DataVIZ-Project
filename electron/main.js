const {
  app,
  BrowserWindow,
  clipboard,
  ipcMain,
  screen,
  dialog,
  systemPreferences,
} = require("electron")
const fs = require("fs")
const path = require("path")
const { execFile, spawn } = require("child_process")
const { uIOhook } = require("uiohook-napi")

const isDev = !app.isPackaged
let backendProcess = null
let mainWindow = null
let overlayWindow = null
let dropzoneWindow = null
let edgeDropWindow = null
let activeProjectId = ""
let latestSelectionText = ""
let mouseDownPoint = null
let isQuitting = false

const OVERLAY_WIDTH = 70
const OVERLAY_HEIGHT = 32
const OVERLAY_OFFSET_X = 12
const OVERLAY_OFFSET_Y = 12

const DROPZONE_IDLE_SIZE = 32
const DROPZONE_ACTIVE_WIDTH = 300
const DROPZONE_ACTIVE_HEIGHT = 400
const DROPZONE_PADDING = 20
const EDGE_DROPZONE_SIZE = 200

function trimTrailingSlash(url) {
  return String(url || "").replace(/\/+$/, "")
}

function readPackagedApiBase() {
  try {
    const p = path.join(__dirname, "api-base.json")
    if (fs.existsSync(p)) {
      const j = JSON.parse(fs.readFileSync(p, "utf8"))
      if (j.apiBase) {
        return trimTrailingSlash(j.apiBase)
      }
    }
  } catch (err) {
    console.warn("[paperMem] api-base.json:", err.message)
  }
  return null
}

function getApiBase() {
  const env = process.env.PAPERMEM_API_BASE || process.env.VITE_API_BASE_URL
  if (env && String(env).trim()) {
    return trimTrailingSlash(env)
  }
  const fromFile = readPackagedApiBase()
  if (fromFile) {
    return fromFile
  }
  return "http://127.0.0.1:8000"
}

const API_BASE = getApiBase()
const BACKEND_URL = `${API_BASE}/health`

/** Per-upload AbortController for dropzone (cancel ×); keyed by renderer-provided uploadId */
const dropzoneUploadAborts = new Map()
const DROPZONE_UPLOAD_RESPONSE_MS = 15 * 60 * 1000

const sendCopyShortcut = () =>
  new Promise((resolve) => {
    if (process.platform === "darwin") {
      execFile(
        "osascript",
        ["-e", 'tell application "System Events" to keystroke "c" using command down'],
        () => resolve()
      )
      return
    }
    if (process.platform === "win32") {
      execFile(
        "powershell",
        [
          "-Command",
          "$wshell = New-Object -ComObject wscript.shell; $wshell.SendKeys('^c')",
        ],
        () => resolve()
      )
      return
    }
    resolve()
  })

const resolvePythonBin = (backendCwd) => {
  const candidates = [
    path.join(backendCwd, ".venv_local", "bin", "python"),
    path.join(backendCwd, ".venv", "bin", "python"),
    "python3",
    "python",
  ]
  return candidates.find((candidate) => candidate.startsWith("/") ? fs.existsSync(candidate) : true)
}

const isBackendHealthy = async () => {
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 1200)
    const response = await fetch(BACKEND_URL, {
      method: "GET",
      signal: controller.signal,
    })
    clearTimeout(timeout)
    return response.ok
  } catch {
    return false
  }
}

const startBackend = async () => {
  if (backendProcess) {
    return
  }
  if (await isBackendHealthy()) {
    return
  }

  const backendCwd = path.join(app.getAppPath(), "backend")
  const pythonBin = resolvePythonBin(backendCwd)

  const args = [
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
  ]

  if (isDev) {
    args.push("--reload")
  }

  backendProcess = spawn(pythonBin, args, {
    cwd: backendCwd,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    stdio: isDev ? "inherit" : "ignore",
  })

  backendProcess.on("exit", () => {
    backendProcess = null
  })
}

const createOverlayWindow = () => {
  overlayWindow = new BrowserWindow({
    width: OVERLAY_WIDTH,
    height: OVERLAY_HEIGHT,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    movable: false,
    show: false,
    skipTaskbar: true,
    hasShadow: false,
    webPreferences: {
      preload: path.join(__dirname, "overlay_preload.js"),
    },
  })

  overlayWindow.setAlwaysOnTop(true, "screen-saver")
  overlayWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true })
  overlayWindow.loadFile(path.join(__dirname, "overlay.html"))
}

const positionDropzone = (width, height) => {
  if (!dropzoneWindow) {
    return
  }
  const bounds = dropzoneWindow.getBounds()
  const display = screen.getDisplayMatching(bounds)
  const { x: dx, y: dy, width: dw, height: dh } = display.workArea
  const left = Math.round(dx + dw - width - DROPZONE_PADDING)
  const top = Math.round(dy + DROPZONE_PADDING)
  dropzoneWindow.setBounds({ x: left, y: top, width, height })
}

const createDropzoneWindow = () => {
  dropzoneWindow = new BrowserWindow({
    width: DROPZONE_IDLE_SIZE,
    height: DROPZONE_IDLE_SIZE,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    movable: false,
    focusable: true,
    focusOnShow: true,
    show: true,
    skipTaskbar: true,
    hasShadow: false,
    acceptFirstMouse: true,
    webPreferences: {
      preload: path.join(__dirname, "dropzone_preload.js"),
      backgroundThrottling: false,
    },
  })
  dropzoneWindow.setAlwaysOnTop(true, "floating")
  dropzoneWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true })
  dropzoneWindow.setIgnoreMouseEvents(false)
  dropzoneWindow.setFocusable(true)
  dropzoneWindow.loadFile(path.join(__dirname, "dropzone.html"))
  positionDropzone(DROPZONE_IDLE_SIZE, DROPZONE_IDLE_SIZE)
}

const positionEdgeDropzone = () => {
  if (!edgeDropWindow) {
    return
  }
  const display = screen.getPrimaryDisplay()
  const { x: dx, y: dy, width: dw } = display.workArea
  const left = Math.round(dx + dw - EDGE_DROPZONE_SIZE - DROPZONE_PADDING)
  const top = Math.round(dy + DROPZONE_PADDING)
  edgeDropWindow.setBounds({
    x: left,
    y: top,
    width: EDGE_DROPZONE_SIZE,
    height: EDGE_DROPZONE_SIZE,
  })
}

const createEdgeDropWindow = () => {
  edgeDropWindow = new BrowserWindow({
    width: EDGE_DROPZONE_SIZE,
    height: EDGE_DROPZONE_SIZE,
    frame: false,
    transparent: false,
    backgroundColor: "#ffffff",
    alwaysOnTop: true,
    resizable: false,
    movable: false,
    focusable: true,
    focusOnShow: true,
    show: true,
    skipTaskbar: true,
    hasShadow: false,
    acceptFirstMouse: true,
    webPreferences: {
      preload: path.join(__dirname, "edge_dropzone_preload.js"),
      backgroundThrottling: false,
    },
  })
  edgeDropWindow.setAlwaysOnTop(true, "screen-saver")
  edgeDropWindow.setVisibleOnAllWorkspaces(true, {
    visibleOnFullScreen: true,
  })
  edgeDropWindow.setIgnoreMouseEvents(false)
  edgeDropWindow.setFocusable(true)
  edgeDropWindow.focus()
  edgeDropWindow.setOpacity(0.12)
  edgeDropWindow.loadFile(path.join(__dirname, "edge_dropzone.html"))
  positionEdgeDropzone()
  const bounds = edgeDropWindow.getBounds()
  console.log("[edge-dropzone] bounds:", bounds)
}

const hideOverlay = () => {
  if (overlayWindow && overlayWindow.isVisible()) {
    overlayWindow.hide()
  }
}

const closeAuxiliaryWindows = () => {
  ;[overlayWindow, dropzoneWindow, edgeDropWindow].forEach((win) => {
    if (win && !win.isDestroyed()) {
      win.close()
    }
  })
  overlayWindow = null
  dropzoneWindow = null
  edgeDropWindow = null
}

const showOverlayAt = (x, y) => {
  if (!overlayWindow) {
    return
  }
  const display = screen.getDisplayNearestPoint({ x, y })
  const { x: dx, y: dy, width, height } = display.workArea

  const left = Math.min(
    Math.max(x + OVERLAY_OFFSET_X, dx),
    dx + width - OVERLAY_WIDTH
  )
  const top = Math.min(
    Math.max(y - OVERLAY_OFFSET_Y, dy),
    dy + height - OVERLAY_HEIGHT
  )

  overlayWindow.setBounds({
    x: Math.round(left),
    y: Math.round(top),
    width: OVERLAY_WIDTH,
    height: OVERLAY_HEIGHT,
  })
  overlayWindow.showInactive()
}

const handleMouseUp = (event) => {
  if (!mouseDownPoint) {
    return
  }
  const dx = event.x - mouseDownPoint.x
  const dy = event.y - mouseDownPoint.y
  const distance = Math.hypot(dx, dy)
  mouseDownPoint = null
  if (distance < 12) {
    return
  }

  sendCopyShortcut()

  setTimeout(() => {
    const text = clipboard.readText().trim()
    if (!text) {
      return
    }
    latestSelectionText = text
    showOverlayAt(event.x, event.y)
  }, 150)
}

const handleMouseDown = (event) => {
  mouseDownPoint = { x: event.x, y: event.y }
  if (overlayWindow && overlayWindow.isVisible()) {
    const bounds = overlayWindow.getBounds()
    const insideX = event.x >= bounds.x && event.x <= bounds.x + bounds.width
    const insideY = event.y >= bounds.y && event.y <= bounds.y + bounds.height
    if (!insideX || !insideY) {
      hideOverlay()
    }
  }
}

const startCapture = () => {
  uIOhook.on("mousedown", handleMouseDown)
  uIOhook.on("mouseup", handleMouseUp)
  uIOhook.start()
}

const ensureAccessibilityPermissions = () => {
  if (process.platform !== "darwin") {
    return
  }
  const trusted = systemPreferences.isTrustedAccessibilityClient(true)
  if (trusted) {
    return
  }
  dialog.showMessageBox({
    type: "warning",
    title: "需要辅助功能权限",
    message: "请在“系统设置 > 隐私与安全 > 辅助功能”允许本应用，以启用全局划词。",
  })
}

const remindWindowsAdminPermission = () => {
  if (process.platform !== "win32") {
    return
  }
  dialog.showMessageBox({
    type: "info",
    title: "需要管理员权限",
    message: "为启用全局划词功能，请以管理员权限运行本应用。",
  })
}

const createWindow = () => {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 840,
    backgroundColor: "#FFFFFF",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
    },
  })

  if (isDev) {
    mainWindow.loadURL("http://127.0.0.1:5173")
  } else {
    mainWindow.loadFile(path.join(app.getAppPath(), "frontend", "dist", "index.html"))
  }

  mainWindow.webContents.on("render-process-gone", (_event, details) => {
    console.error("[paperMem] renderer process gone", details)
  })

  mainWindow.webContents.on("unresponsive", () => {
    console.error("[paperMem] main window became unresponsive")
  })

  mainWindow.webContents.on("did-fail-load", (_event, errorCode, errorDescription, validatedURL) => {
    console.error("[paperMem] main window failed to load", {
      errorCode,
      errorDescription,
      validatedURL,
    })
  })

  mainWindow.webContents.on("console-message", (_event, level, message, line, sourceId) => {
    if (level < 2) {
      return
    }
    console.error("[paperMem] renderer console", { level, message, line, sourceId })
  })

  mainWindow.on("closed", () => {
    mainWindow = null
    closeAuxiliaryWindows()
    if (!isQuitting) {
      app.quit()
    }
  })
}

app.whenReady().then(async () => {
  if (isDev) {
    await startBackend()
  }
  ensureAccessibilityPermissions()
  remindWindowsAdminPermission()
  createOverlayWindow()
  createDropzoneWindow()
  startCapture()
  createWindow()

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on("before-quit", () => {
  isQuitting = true
  closeAuxiliaryWindows()
  uIOhook.stop()
  if (backendProcess) {
    backendProcess.kill()
    backendProcess = null
  }
})

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit()
  }
})

ipcMain.handle("set-active-project", (_event, projectId) => {
  activeProjectId = projectId || ""
  if (dropzoneWindow && dropzoneWindow.webContents) {
    dropzoneWindow.webContents.send("dropzone-project-changed", activeProjectId)
  }
  return { ok: true }
})

ipcMain.handle("overlay-save", async () => {
  if (!latestSelectionText) {
    return { ok: false, error: "empty" }
  }
  if (!activeProjectId) {
    return { ok: false, error: "no_project" }
  }

  try {
    const response = await fetch(`${API_BASE}/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: latestSelectionText,
        project_id: activeProjectId,
        source_name: "Quick Capture",
        content_type: "conversation",
        replace: false,
      }),
    })
    if (!response.ok) {
      return { ok: false, error: "bad_response" }
    }
    return { ok: true }
  } catch (error) {
    return { ok: false, error: "network" }
  }
})

ipcMain.handle("overlay-hide", () => {
  hideOverlay()
  return { ok: true }
})

ipcMain.handle("dropzone-get-project", () => {
  return { projectId: activeProjectId }
})

ipcMain.handle("dropzone-resize", (_event, payload) => {
  if (!payload || !dropzoneWindow) {
    return { ok: false }
  }
  const width = Math.max(1, Number(payload.width) || DROPZONE_IDLE_SIZE)
  const height = Math.max(1, Number(payload.height) || DROPZONE_IDLE_SIZE)
  positionDropzone(width, height)
  return { ok: true }
})


ipcMain.handle("dropzone-upload", async (_event, payload) => {
  if (!payload?.filePath || !payload?.projectId) {
    return { ok: false, error: "invalid_payload" }
  }
  const uploadId = payload.uploadId || `upload-${Date.now()}`
  const userController = new AbortController()
  dropzoneUploadAborts.set(uploadId, userController)
  let signal = userController.signal
  if (typeof AbortSignal !== "undefined" && AbortSignal.any && AbortSignal.timeout) {
    signal = AbortSignal.any([
      userController.signal,
      AbortSignal.timeout(DROPZONE_UPLOAD_RESPONSE_MS),
    ])
  }
  try {
    const buffer = await fs.promises.readFile(payload.filePath)
    const form = new FormData()
    form.append("project_id", payload.projectId)
    form.append("file", new Blob([buffer]), path.basename(payload.filePath))
    const response = await fetch(`${API_BASE}/files/ingest_upload`, {
      method: "POST",
      body: form,
      signal,
    })
    if (!response.ok) {
      const detail = await response.text()
      return { ok: false, error: detail || "ingest_failed" }
    }
    const rawText = await response.text()
    let result = {}
    try {
      result = rawText ? JSON.parse(rawText) : {}
    } catch (parseErr) {
      console.warn("[dropzone-upload] 200 OK but JSON parse failed; treating as success", {
        snippet: rawText?.slice?.(0, 240),
        message: parseErr?.message || parseErr,
      })
    }
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("memory-file-ingested", {
        projectId: payload.projectId,
        filePath: payload.filePath,
        filename: result?.filename || path.basename(payload.filePath),
        unitCount: result?.unit_count || 0,
      })
    }
    return { ok: true, result }
  } catch (error) {
    const name = error?.name || ""
    if (name === "AbortError" || name === "TimeoutError") {
      if (userController.signal.aborted) {
        return { ok: false, aborted: true }
      }
      return {
        ok: false,
        error: "timeout",
        detail: "Server took too long to respond (ingest can take several minutes for large PDFs).",
      }
    }
    console.error("[dropzone-upload] failed", {
      filePath: payload.filePath,
      projectId: payload.projectId,
      message: error?.message || error,
    })
    return { ok: false, error: "upload_failed" }
  } finally {
    dropzoneUploadAborts.delete(uploadId)
  }
})

ipcMain.handle("dropzone-abort-upload", (_event, uploadId) => {
  if (!uploadId) {
    return { ok: false }
  }
  const ctrl = dropzoneUploadAborts.get(uploadId)
  if (ctrl) {
    ctrl.abort()
    return { ok: true }
  }
  return { ok: false }
})

ipcMain.handle("dropzone-pick-files", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile", "multiSelections"],
    filters: [
      { name: "Documents", extensions: ["pdf", "md", "markdown", "txt"] },
      { name: "All Files", extensions: ["*"] },
    ],
  })
  if (result.canceled) {
    return { ok: true, files: [] }
  }
  return { ok: true, files: result.filePaths || [] }
})

ipcMain.on("dropzone-log-error", (_event, payload) => {
  console.error("[dropzone-error]", payload)
})
