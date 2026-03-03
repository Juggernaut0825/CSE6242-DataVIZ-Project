const { contextBridge, ipcRenderer } = require("electron")

contextBridge.exposeInMainWorld("edgeDropzoneApi", {
  sendDragState: (payload) => ipcRenderer.send("edge-dropzone-drag", payload),
  sendDropFiles: (payload) => ipcRenderer.send("edge-dropzone-drop", payload),
})
