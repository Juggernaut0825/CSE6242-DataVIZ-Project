const { contextBridge, ipcRenderer } = require("electron")

contextBridge.exposeInMainWorld("paperMem", {
  apiBase: "http://127.0.0.1:8000",
  setActiveProjectId: (projectId) =>
    ipcRenderer.invoke("set-active-project", projectId),
  onFileIngested: (callback) => {
    const handler = (_event, payload) => {
      callback(payload)
    }
    ipcRenderer.on("memory-file-ingested", handler)
    return () => ipcRenderer.removeListener("memory-file-ingested", handler)
  },
})
