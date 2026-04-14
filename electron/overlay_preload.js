const { contextBridge, ipcRenderer } = require("electron")

contextBridge.exposeInMainWorld("overlayApi", {
  save: () => ipcRenderer.invoke("overlay-save"),
  hide: () => ipcRenderer.invoke("overlay-hide"),
})
