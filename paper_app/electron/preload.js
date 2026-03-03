const { contextBridge, ipcRenderer } = require("electron")

contextBridge.exposeInMainWorld("paperMem", {
  apiBase: "http://127.0.0.1:8000",
  setActiveProjectId: (projectId) =>
    ipcRenderer.invoke("set-active-project", projectId),
})
