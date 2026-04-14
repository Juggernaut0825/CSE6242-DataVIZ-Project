/**
 * Writes electron/api-base.json from VITE_API_BASE_URL or PAPERMEM_API_BASE
 * so the main process uses the same API as the Vite build (e.g. Railway).
 * If unset, removes api-base.json so dev defaults to http://127.0.0.1:8000.
 */
const fs = require("fs")
const path = require("path")

const root = path.join(__dirname, "..")
const target = path.join(root, "electron", "api-base.json")
const url = process.env.VITE_API_BASE_URL || process.env.PAPERMEM_API_BASE || ""

if (!String(url).trim()) {
  if (fs.existsSync(target)) {
    fs.unlinkSync(target)
  }
  console.log(
    "[write-electron-api-base] No VITE_API_BASE_URL / PAPERMEM_API_BASE; removed api-base.json (localhost)."
  )
  process.exit(0)
}

const apiBase = String(url).replace(/\/+$/, "")
fs.writeFileSync(target, JSON.stringify({ apiBase }, null, 2))
console.log("[write-electron-api-base] Wrote", target, "->", apiBase)
