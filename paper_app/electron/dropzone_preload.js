const { contextBridge, ipcRenderer } = require("electron")

contextBridge.exposeInMainWorld("dropzoneApi", {
  resize: (size) => ipcRenderer.invoke("dropzone-resize", size),
  uploadFile: (payload) => ipcRenderer.invoke("dropzone-upload", payload),
  pickFiles: () => ipcRenderer.invoke("dropzone-pick-files"),
  getProjectId: () => ipcRenderer.invoke("dropzone-get-project"),
  logError: (payload) => ipcRenderer.send("dropzone-log-error", payload),
  onProjectChange: (callback) => {
    const handler = (_event, projectId) => {
      callback(projectId)
    }
    ipcRenderer.on("dropzone-project-changed", handler)
    return () => ipcRenderer.removeListener("dropzone-project-changed", handler)
  },
})
