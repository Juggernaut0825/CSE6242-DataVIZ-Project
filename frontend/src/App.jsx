import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { Network } from "vis-network/standalone/esm/vis-network"

function trimTrailingSlash(url) {
  return String(url || "").replace(/\/+$/, "")
}

function resolveApiBase() {
  const fromEnv = trimTrailingSlash(import.meta.env.VITE_API_BASE_URL)
  if (fromEnv) {
    return fromEnv
  }
  const fromElectron = trimTrailingSlash(window?.paperMem?.apiBase)
  if (fromElectron) {
    return fromElectron
  }
  return "http://127.0.0.1:8000"
}

const apiBase = resolveApiBase()

const makeId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`

const safeParse = (value) => {
  if (!value || typeof value !== "string") {
    return null
  }
  try {
    return JSON.parse(value)
  } catch {
    return null
  }
}

const readJsonResponse = async (response, fallback = null) => {
  const raw = await response.text()
  if (!raw.trim()) {
    return fallback
  }
  try {
    return JSON.parse(raw)
  } catch (error) {
    console.error("Failed to parse API response", {
      status: response.status,
      url: response.url,
      body: raw.slice(0, 500),
      error,
    })
    return fallback
  }
}

const truncate = (value, max = 40) => {
  if (!value) {
    return ""
  }
  return value.length > max ? `${value.slice(0, max)}...` : value
}

const sourceTypeLabel = (value) => {
  if (!value) {
    return "Memory"
  }
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

const citationDisplayText = (search, citation) => {
  if (!citation) {
    return ""
  }
  const matchedUnit = search?.retrieved_units?.find((item) => item.id === citation.unit_id)
  return matchedUnit?.text || citation.summary || ""
}

const RETRIEVAL_HIGHLIGHT_MIN_SCORE = 0.4
const RETRIEVAL_HIGHLIGHT_MAX_UNITS = 8
const RETRIEVAL_HIGHLIGHT_FALLBACK_UNITS = 3

const selectHighlightedEvidenceUnits = (search) => {
  const units = (search?.retrieved_units || []).filter((unit) => unit?.id)
  const scored = units.filter((unit) => Number(unit.score || 0) >= RETRIEVAL_HIGHLIGHT_MIN_SCORE)
  const selected = scored.length ? scored : units.slice(0, RETRIEVAL_HIGHLIGHT_FALLBACK_UNITS)
  return selected.slice(0, RETRIEVAL_HIGHLIGHT_MAX_UNITS)
}

/** Highlight only the highest-confidence retrieved evidence nodes, not every recalled unit. */
const computeRetrievalHighlight = (payload) => {
  const overlay = payload?.overlay
  if (!overlay?.nodes?.length) {
    return null
  }
  const unitIds = new Set(selectHighlightedEvidenceUnits(payload).map((u) => u.id))
  const queryNode = (overlay.nodes || []).find(
    (n) => n.kind === "query" || String(n.id).startsWith("query:")
  )
  const nodeIds = new Set(unitIds)
  if (queryNode?.id) {
    nodeIds.add(queryNode.id)
  }
  const edgeIds = new Set()
  for (const edge of overlay.edges || []) {
    if (nodeIds.has(edge.source) && nodeIds.has(edge.target)) {
      if (edge.id) {
        edgeIds.add(edge.id)
      }
    }
  }
  return { nodeIds: Array.from(nodeIds), edgeIds: Array.from(edgeIds) }
}

/** Node ids to zoom onto when expanding Evidence (query + highlighted evidence units). */
const computeEvidenceFitNodeIds = (search) => {
  if (!search?.overlay?.nodes) {
    return []
  }
  const unitIds = selectHighlightedEvidenceUnits(search).map((u) => u.id).filter(Boolean)
  const queryNode = (search.overlay.nodes || []).find(
    (n) => n.kind === "query" || String(n.id).startsWith("query:")
  )
  const ids = queryNode?.id ? [queryNode.id, ...unitIds] : [...unitIds]
  return [...new Set(ids)]
}

const tooltipElement = (value) => {
  if (!value || typeof document === "undefined") {
    return value || ""
  }
  const container = document.createElement("div")
  container.style.maxWidth = "320px"
  container.style.whiteSpace = "normal"
  container.style.lineHeight = "1.5"
  container.style.fontSize = "12px"
  container.textContent = value
  return container
}

const EMPTY_PINNED = []

const zoomBucketFromScale = (scale) => {
  if (scale < 0.45) {
    return 0
  }
  if (scale < 0.7) {
    return 1
  }
  if (scale < 1.0) {
    return 2
  }
  if (scale < 1.45) {
    return 3
  }
  return 4
}

const buildSemanticZoomGraph = (graph, zoomBucket, pinnedNodeIds = []) => {
  if (!graph?.nodes?.length) {
    return graph
  }

  const budgetByBucket = [10, 16, 28, 44, 72]
  const nodeBudget = budgetByBucket[Math.max(0, Math.min(zoomBucket, budgetByBucket.length - 1))]
  if (graph.nodes.length <= nodeBudget) {
    const localImportance = Object.fromEntries(
      graph.nodes.map((node) => [node.id, Number(node.score || 0)])
    )
    return {
      ...graph,
      nodes: graph.nodes.map((node) => ({
        ...node,
        local_importance: localImportance[node.id] || 0,
      })),
      meta: { ...(graph.meta || {}), zoom_bucket: zoomBucket, node_budget: nodeBudget },
    }
  }

  const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]))
  const pinnedInGraph = new Set()
  for (const id of pinnedNodeIds || []) {
    if (nodeMap.has(id)) {
      pinnedInGraph.add(id)
    }
  }
  const effectiveBudget = pinnedInGraph.size
    ? Math.max(nodeBudget, pinnedInGraph.size + 8)
    : nodeBudget

  const degreeMap = new Map(graph.nodes.map((node) => [node.id, 0]))
  for (const edge of graph.edges || []) {
    degreeMap.set(edge.source, (degreeMap.get(edge.source) || 0) + 1)
    degreeMap.set(edge.target, (degreeMap.get(edge.target) || 0) + 1)
  }

  const importanceScore = (node) => {
    const detail = node.detail || {}
    const betweenness = Number(detail.betweenness || 0)
    const degree = Number(degreeMap.get(node.id) || 0)
    const kindBonus =
      node.kind === "query"
        ? 100
        : node.kind === "claim"
        ? 3.4
        : node.kind === "entity"
        ? 2.4
        : node.kind === "concept"
        ? 2.1
        : node.kind === "conversation_turn" || node.kind === "quick_capture"
        ? 1.8
        : node.kind === "file_chunk"
        ? 1.5
        : 1
    return (Number(node.score || 0) * 4 + betweenness * 3 + degree * 0.18) * kindBonus
  }

  const rankedNodes = [...graph.nodes].sort((left, right) => {
    return importanceScore(right) - importanceScore(left)
  })

  const selected = new Set()
  for (const id of pinnedInGraph) {
    selected.add(id)
  }
  for (const node of rankedNodes) {
    if (node.kind === "query") {
      selected.add(node.id)
    }
  }

  const targetCore = Math.max(5, Math.floor(effectiveBudget * 0.5))
  for (const node of rankedNodes) {
    if (selected.size >= targetCore) {
      break
    }
    selected.add(node.id)
  }

  const bridgeCandidates = [...(graph.edges || [])]
    .map((edge) => {
      const source = nodeMap.get(edge.source)
      const target = nodeMap.get(edge.target)
      const sourceCommunity = source?.detail?.community
      const targetCommunity = target?.detail?.community
      const bridgeBonus =
        sourceCommunity !== undefined &&
        targetCommunity !== undefined &&
        sourceCommunity !== targetCommunity
          ? 3
          : 0
      const sourceBetweenness = Number(source?.detail?.betweenness || 0)
      const targetBetweenness = Number(target?.detail?.betweenness || 0)
      return {
        edge,
        score:
          Number(edge.weight || 0) +
          bridgeBonus +
          sourceBetweenness +
          targetBetweenness +
          (selected.has(edge.source) || selected.has(edge.target) ? 1.5 : 0),
      }
    })
    .sort((left, right) => right.score - left.score)

  for (const { edge } of bridgeCandidates) {
    if (selected.size >= effectiveBudget) {
      break
    }
    if (selected.has(edge.source) || selected.has(edge.target)) {
      selected.add(edge.source)
      if (selected.size < effectiveBudget) {
        selected.add(edge.target)
      }
    }
  }

  for (const node of rankedNodes) {
    if (selected.size >= effectiveBudget) {
      break
    }
    selected.add(node.id)
  }

  const filteredEdges = (graph.edges || []).filter(
    (edge) => selected.has(edge.source) && selected.has(edge.target)
  )

  const localDegreeMap = new Map([...selected].map((id) => [id, 0]))
  for (const edge of filteredEdges) {
    localDegreeMap.set(edge.source, (localDegreeMap.get(edge.source) || 0) + 1)
    localDegreeMap.set(edge.target, (localDegreeMap.get(edge.target) || 0) + 1)
  }

  const localImportanceMap = new Map()
  for (const id of selected) {
    const node = nodeMap.get(id)
    if (!node) {
      continue
    }
    const localImportance =
      Number(node.score || 0) * 3 +
      Number(node.detail?.betweenness || 0) * 2 +
      Number(localDegreeMap.get(id) || 0) * 0.35
    localImportanceMap.set(id, localImportance)
  }

  return {
    ...graph,
    nodes: graph.nodes
      .filter((node) => selected.has(node.id))
      .map((node) => ({
        ...node,
        local_importance: Number(localImportanceMap.get(node.id) || node.score || 0),
      })),
    edges: filteredEdges,
    meta: {
      ...(graph.meta || {}),
      zoom_bucket: zoomBucket,
      node_budget: nodeBudget,
      effective_node_budget: effectiveBudget,
      pinned_retrieval_nodes: pinnedInGraph.size,
    },
  }
}

function App() {
  const graphContainerRef = useRef(null)
  const graphNetworkRef = useRef(null)
  const graphViewportRef = useRef({ scale: 1, position: null })
  const zoomBucketRef = useRef(2)
  const chatScrollRef = useRef(null)
  const pendingEvidenceFitRef = useRef(null)

  const [projects, setProjects] = useState([])
  const [chatSessions, setChatSessions] = useState([])
  const [activeProjectId, setActiveProjectId] = useState("")
  const [activeSessionId, setActiveSessionId] = useState("")
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [sending, setSending] = useState(false)
  const [loadingProjects, setLoadingProjects] = useState(false)
  const [loadingSessions, setLoadingSessions] = useState(false)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const [graphMode, setGraphMode] = useState("global")
  const [graphView] = useState("macro")
  const [globalGraph, setGlobalGraph] = useState(null)
  const [liveGraph, setLiveGraph] = useState(null)
  const [graphLoading, setGraphLoading] = useState(false)
  const [graphError, setGraphError] = useState("")
  const [zoomBucket, setZoomBucket] = useState(2)
  const [selectedGraphNode, setSelectedGraphNode] = useState(null)
  const [isProjectModalOpen, setIsProjectModalOpen] = useState(false)
  const [newProjectName, setNewProjectName] = useState("")
  const [showLegend, setShowLegend] = useState(false)
  const [retrievalHighlight, setRetrievalHighlight] = useState(null)
  const [evidenceExpandedMessageId, setEvidenceExpandedMessageId] = useState(null)

  const activeSession = useMemo(
    () => chatSessions.find((session) => session.id === activeSessionId),
    [chatSessions, activeSessionId]
  )

  const visibleGraph = graphMode === "live" && liveGraph ? liveGraph : globalGraph
  const pinnedForZoom = retrievalHighlight?.nodeIds ?? EMPTY_PINNED
  const retrievalNodeSet = useMemo(
    () => new Set(retrievalHighlight?.nodeIds || []),
    [retrievalHighlight]
  )
  const retrievalEdgeSet = useMemo(
    () => new Set(retrievalHighlight?.edgeIds || []),
    [retrievalHighlight]
  )

  const zoomedGraph = useMemo(
    () => buildSemanticZoomGraph(visibleGraph, zoomBucket, pinnedForZoom),
    [visibleGraph, zoomBucket, pinnedForZoom]
  )

  const loadProjects = useCallback(async () => {
    setLoadingProjects(true)
    try {
      const response = await fetch(`${apiBase}/projects`)
      if (!response.ok) {
        return
      }
      const data = await readJsonResponse(response, [])
      setProjects(data)
      if (data.length && !activeProjectId) {
        setActiveProjectId(data[0].id)
      }
    } finally {
      setLoadingProjects(false)
    }
  }, [activeProjectId])

  const createChatSession = useCallback(async (projectId, title = "") => {
    if (!projectId) {
      return null
    }
    const response = await fetch(`${apiBase}/chat_sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: projectId, title: title || undefined }),
    })
    if (!response.ok) {
      return null
    }
    const session = await readJsonResponse(response)
    if (!session?.id) {
      return null
    }
    setChatSessions((prev) => [session, ...prev.map((item) => ({ ...item, is_current: false }))])
    setActiveSessionId(session.id)
    setMessages([])
    return session
  }, [])

  const loadChatSessions = useCallback(
    async (projectId) => {
      if (!projectId) {
        setChatSessions([])
        setActiveSessionId("")
        return
      }
      setLoadingSessions(true)
      try {
        const response = await fetch(`${apiBase}/projects/${projectId}/chat_sessions`)
        if (!response.ok) {
          return
        }
        const data = await readJsonResponse(response, [])
        if (!data.length) {
          const created = await createChatSession(projectId)
          if (created) {
            setChatSessions([created])
          }
          return
        }
        setChatSessions(data)
        setActiveSessionId((prev) => {
          if (prev && data.some((session) => session.id === prev)) {
            return prev
          }
          return data.find((session) => session.is_current)?.id || data[0]?.id || ""
        })
      } finally {
        setLoadingSessions(false)
      }
    },
    [createChatSession]
  )

  const activateChatSession = useCallback(async (sessionId) => {
    if (!sessionId) {
      return
    }
    const response = await fetch(`${apiBase}/chat_sessions/${sessionId}/activate`, {
      method: "POST",
    })
    if (!response.ok) {
      return
    }
    const activated = await readJsonResponse(response)
    if (!activated?.id) {
      return
    }
    setChatSessions((prev) =>
      prev.map((session) => ({
        ...session,
        is_current: session.id === activated.id,
      }))
    )
    setActiveSessionId(activated.id)
    setGraphMode("global")
    setLiveGraph(null)
    setRetrievalHighlight(null)
    setEvidenceExpandedMessageId(null)
    pendingEvidenceFitRef.current = null
  }, [])

  const loadMessages = useCallback(async (projectId, sessionId) => {
    if (!projectId) {
      setMessages([])
      return
    }
    setLoadingMessages(true)
    try {
      const search = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ""
      const response = await fetch(`${apiBase}/projects/${projectId}/messages${search}`)
      if (!response.ok) {
        return
      }
      const data = await readJsonResponse(response, [])
      setMessages(
        data.map((message) => ({
          ...message,
          search: safeParse(message.search_results),
          reasoning: safeParse(message.reasoning_trace) || message.reasoning_trace || "",
        }))
      )
    } finally {
      setLoadingMessages(false)
    }
  }, [])

  const loadGraph = useCallback(
    async (projectId, nextView = graphView) => {
      if (!projectId) {
        return
      }
      setGraphLoading(true)
      setGraphError("")
      try {
        const response = await fetch(
          `${apiBase}/graph/${projectId}?view=${encodeURIComponent(nextView)}&limit=90`
        )
        if (!response.ok) {
          setGraphError("Graph load failed")
          return
        }
        const data = await readJsonResponse(response)
        if (!data) {
          setGraphError("Graph load failed")
          return
        }
        setGlobalGraph(data)
      } catch {
        setGraphError("Graph load failed")
      } finally {
        setGraphLoading(false)
      }
    },
    [graphView]
  )

  const deleteProject = useCallback(
    async (projectId, projectName) => {
      if (!projectId) {
        return
      }
      const confirmed = window.confirm(`Delete project "${projectName || "Untitled"}"?`)
      if (!confirmed) {
        return
      }
      const response = await fetch(`${apiBase}/projects/${projectId}`, { method: "DELETE" })
      if (!response.ok) {
        return
      }
      setProjects((prev) => {
        const nextProjects = prev.filter((project) => project.id !== projectId)
        if (activeProjectId === projectId) {
          setActiveProjectId(nextProjects[0]?.id || "")
          setChatSessions([])
          setActiveSessionId("")
          setMessages([])
          setGlobalGraph(null)
          setLiveGraph(null)
          setRetrievalHighlight(null)
          setEvidenceExpandedMessageId(null)
          pendingEvidenceFitRef.current = null
        }
        return nextProjects
      })
    },
    [activeProjectId]
  )

  const deleteChatSession = useCallback(
    async (sessionId, sessionTitle) => {
      if (!sessionId) {
        return
      }
      const confirmed = window.confirm(`Delete chat "${sessionTitle || "Untitled chat"}"?`)
      if (!confirmed) {
        return
      }
      const response = await fetch(`${apiBase}/chat_sessions/${sessionId}`, { method: "DELETE" })
      if (!response.ok) {
        return
      }
      const payload = await readJsonResponse(response, {})
      setChatSessions((prev) => prev.filter((session) => session.id !== sessionId))
      if (activeSessionId === sessionId) {
        setActiveSessionId(payload.next_session_id || "")
        setMessages([])
        setLiveGraph(null)
        setRetrievalHighlight(null)
        setEvidenceExpandedMessageId(null)
        pendingEvidenceFitRef.current = null
      }
      await Promise.all([
        loadChatSessions(activeProjectId),
        loadMessages(activeProjectId, payload.next_session_id || ""),
        loadGraph(activeProjectId, graphView),
      ])
    },
    [activeProjectId, activeSessionId, graphView, loadChatSessions, loadGraph, loadMessages]
  )

  const refreshGraph = useCallback(async () => {
    if (!activeProjectId) {
      return
    }
    await loadGraph(activeProjectId, graphView)
  }, [activeProjectId, graphView, loadGraph])

  useEffect(() => {
    loadProjects()
  }, [loadProjects])

  useLayoutEffect(() => {
    if (!activeProjectId) {
      return
    }
    setActiveSessionId("")
    setChatSessions([])
  }, [activeProjectId])

  useEffect(() => {
    if (!activeProjectId) {
      return
    }
    if (window.paperMem?.setActiveProjectId) {
      window.paperMem.setActiveProjectId(activeProjectId)
    }
    setGraphMode("global")
    setLiveGraph(null)
    setRetrievalHighlight(null)
    setEvidenceExpandedMessageId(null)
    pendingEvidenceFitRef.current = null
    loadChatSessions(activeProjectId)
    loadGraph(activeProjectId, graphView)
  }, [activeProjectId, graphView, loadChatSessions, loadGraph])

  useEffect(() => {
    if (!activeProjectId) {
      setMessages([])
      return
    }
    if (loadingSessions) {
      return
    }
    if (
      activeSessionId &&
      chatSessions.length > 0 &&
      !chatSessions.some((session) => session.id === activeSessionId)
    ) {
      return
    }
    loadMessages(activeProjectId, activeSessionId)
  }, [
    activeProjectId,
    activeSessionId,
    chatSessions,
    loadingSessions,
    loadMessages,
  ])

  useEffect(() => {
    if (graphMode === "global") {
      loadGraph(activeProjectId, graphView)
    }
  }, [activeProjectId, graphMode, graphView, loadGraph])

  useEffect(() => {
    if (!chatScrollRef.current) {
      return
    }
    const frame = window.requestAnimationFrame(() => {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight
    })
    return () => window.cancelAnimationFrame(frame)
  }, [messages, loadingMessages, sending, activeSessionId])

  useEffect(() => {
    if (!window.paperMem?.onFileIngested) {
      return undefined
    }
    const unsubscribe = window.paperMem.onFileIngested((payload) => {
      if (!payload?.projectId || payload.projectId !== activeProjectId) {
        return
      }
      setGraphMode("global")
      setLiveGraph(null)
      setRetrievalHighlight(null)
      setEvidenceExpandedMessageId(null)
      pendingEvidenceFitRef.current = null
      loadGraph(payload.projectId, graphView)
    })
    return () => {
      unsubscribe?.()
    }
  }, [activeProjectId, graphView, loadGraph])

  useEffect(() => {
    zoomBucketRef.current = zoomBucket
  }, [zoomBucket])

  useEffect(() => {
    if (!evidenceExpandedMessageId) {
      return
    }
    if (messages.some((m) => m.id === evidenceExpandedMessageId)) {
      return
    }
    const withOverlay = messages.filter((m) => m.role === "assistant" && m.search?.overlay)
    if (withOverlay.length === 1) {
      setEvidenceExpandedMessageId(withOverlay[0].id)
      return
    }
    setEvidenceExpandedMessageId(null)
    setGraphMode("global")
    setLiveGraph(null)
    setRetrievalHighlight(null)
    pendingEvidenceFitRef.current = null
  }, [messages, evidenceExpandedMessageId])

  useEffect(() => {
    if (evidenceExpandedMessageId) {
      return
    }
    graphViewportRef.current = { scale: 1, position: null }
    setZoomBucket(2)
    setSelectedGraphNode(null)
  }, [activeProjectId, activeSessionId, graphMode, evidenceExpandedMessageId])

  useEffect(() => {
    if (!graphContainerRef.current || !zoomedGraph) {
      return
    }

    if (graphNetworkRef.current) {
      graphNetworkRef.current.destroy()
      graphNetworkRef.current = null
    }

    const palette = {
      query: "#8b5cf6",
      concept: "#60a5fa",
      entity: "#34d399",
      claim: "#f59e0b",
      conversation_turn: "#f97316",
      quick_capture: "#f97316",
      file_chunk: "#ec4899",
    }

    const evidenceDimActive = Boolean(retrievalHighlight?.nodeIds?.length)

    const nodes = (zoomedGraph.nodes || []).map((node) => {
      const color = palette[node.kind] || "#94a3b8"
      const localImportance = Number(node.local_importance || node.score || 0)
      const fullText =
        node.detail?.full_text || node.detail?.text || node.detail?.query || node.label || node.id
      const retrievalHl = retrievalNodeSet.has(node.id)
      const dimmed = evidenceDimActive && !retrievalHl
      if (dimmed) {
        return {
          id: node.id,
          label: truncate(node.label || node.id, 28),
          title: tooltipElement(fullText),
          shape: node.kind === "query" ? "star" : "dot",
          size: Math.max(8, (node.kind === "query" ? 28 : 14 + Math.min(20, localImportance * 8)) * 0.75),
          opacity: 0.28,
          borderWidth: 1,
          color: {
            background: "#e2e8f0",
            border: "#cbd5e1",
            highlight: {
              background: "#e2e8f0",
              border: "#94a3b8",
            },
          },
          font: {
            color: "#94a3b8",
            size: 11,
            face: "Inter, SF Pro Text, system-ui, sans-serif",
          },
        }
      }
      return {
        id: node.id,
        label: truncate(node.label || node.id, 28),
        title: tooltipElement(fullText),
        shape: node.kind === "query" ? "star" : "dot",
        size: node.kind === "query" ? 28 : 14 + Math.min(20, localImportance * 8),
        borderWidth: retrievalHl ? 3 : 1,
        color: {
          background: color,
          border: retrievalHl ? "#eab308" : color,
          highlight: {
            background: color,
            border: retrievalHl ? "#ca8a04" : "#0f172a",
          },
        },
        font: {
          color: "#0f172a",
          size: 12,
          face: "Inter, SF Pro Text, system-ui, sans-serif",
        },
      }
    })

    const edges = (zoomedGraph.edges || []).map((edge) => {
      const baseWidth = Math.max(1.2, edge.weight || 1)
      const retrievalHl = retrievalEdgeSet.has(edge.id)
      const dimmed = evidenceDimActive && !retrievalHl
      return {
        id: edge.id,
        from: edge.source,
        to: edge.target,
        label: zoomBucket >= 3 ? edge.type : "",
        arrows: "to",
        width: dimmed ? Math.max(0.6, baseWidth * 0.45) : retrievalHl ? Math.max(3, baseWidth + 1.2) : baseWidth,
        color: dimmed
          ? { color: "#e2e8f0", highlight: "#cbd5e1" }
          : retrievalHl
          ? { color: "#eab308", highlight: "#ca8a04" }
          : { color: "#cbd5e1", highlight: "#475569" },
        font: { color: dimmed ? "#cbd5e1" : "#64748b", size: 10, align: "middle" },
        smooth: { enabled: true, type: "dynamic" },
        opacity: dimmed ? 0.35 : 1,
      }
    })

    let network = null
    try {
      network = new Network(
        graphContainerRef.current,
        { nodes, edges },
        {
          autoResize: true,
          interaction: {
            hover: true,
            navigationButtons: true,
            keyboard: true,
            zoomSpeed: 0.35,
          },
          physics: {
            stabilization: false,
            barnesHut: {
              gravitationalConstant: -4800,
              springLength: 140,
              springConstant: 0.03,
            },
          },
        }
      )
      setGraphError("")
    } catch {
      setGraphError("Graph render failed")
      return
    }

    const fitTargetIds = pendingEvidenceFitRef.current
    pendingEvidenceFitRef.current = null
    const visNodeIdSet = new Set(nodes.map((n) => n.id))
    const fitPresent =
      fitTargetIds?.length > 0 ? fitTargetIds.filter((id) => visNodeIdSet.has(id)) : []

    if (fitPresent.length) {
      window.setTimeout(() => {
        try {
          network.fit({
            nodes: fitPresent,
            animation: { duration: 480 },
          })
          graphViewportRef.current = {
            scale: network.getScale(),
            position: network.getViewPosition(),
          }
        } catch {
          /* ignore */
        }
      }, 160)
    } else {
      const currentViewport = graphViewportRef.current
      if (currentViewport.position || currentViewport.scale !== 1) {
        network.moveTo({
          position: currentViewport.position || undefined,
          scale: currentViewport.scale || 1,
          animation: false,
        })
      }
    }

    network.on("zoom", (params) => {
      graphViewportRef.current = {
        scale: params.scale,
        position: network.getViewPosition(),
      }
      const nextBucket = zoomBucketFromScale(params.scale)
      if (nextBucket !== zoomBucketRef.current) {
        setZoomBucket(nextBucket)
      }
    })

    network.on("dragEnd", () => {
      graphViewportRef.current = {
        scale: network.getScale(),
        position: network.getViewPosition(),
      }
    })

    network.on("click", (params) => {
      if (!params.nodes?.length) {
        setSelectedGraphNode(null)
        return
      }
      const clickedNode = (zoomedGraph.nodes || []).find((node) => node.id === params.nodes[0])
      if (!clickedNode) {
        setSelectedGraphNode(null)
        return
      }
      setSelectedGraphNode({
        id: clickedNode.id,
        label: clickedNode.label || clickedNode.id,
        kind: clickedNode.kind || "semantic",
        fullText:
          clickedNode.detail?.full_text ||
          clickedNode.detail?.text ||
          clickedNode.detail?.query ||
          clickedNode.label ||
          clickedNode.id,
      })
    })

    graphNetworkRef.current = network
    return () => {
      network?.destroy()
      graphNetworkRef.current = null
    }
  }, [zoomBucket, zoomedGraph, retrievalNodeSet, retrievalEdgeSet, retrievalHighlight])

  const createProject = async (name) => {
    const trimmedName = (name || "").trim()
    if (!trimmedName) {
      return
    }
    const response = await fetch(`${apiBase}/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: trimmedName }),
    })
    if (!response.ok) {
      return
    }
    const project = await readJsonResponse(response)
    if (!project?.id) {
      return
    }
    setProjects((prev) => [project, ...prev])
    setActiveProjectId(project.id)
    setIsProjectModalOpen(false)
    setNewProjectName("")
  }

  const updateAssistantMessage = (assistantId, patch) => {
    setMessages((prev) =>
      prev.map((message) =>
        message.id === assistantId ? { ...message, ...patch } : message
      )
    )
  }

  const handleNewChat = async () => {
    const created = await createChatSession(activeProjectId)
    if (created) {
      setMessages([])
      setGraphMode("global")
      setLiveGraph(null)
      setRetrievalHighlight(null)
      setEvidenceExpandedMessageId(null)
      pendingEvidenceFitRef.current = null
    }
  }

  const handleCreateProject = async () => {
    await createProject(newProjectName)
  }

  const handleSend = async () => {
    if (!input.trim() || !activeProjectId || sending) {
      return
    }
    const session =
      activeSessionId
        ? { id: activeSessionId }
        : await createChatSession(activeProjectId, `Chat ${chatSessions.length + 1}`)
    if (!session?.id) {
      return
    }

    const query = input.trim()
    const assistantId = makeId()
    setInput("")
    setSending(true)
    setEvidenceExpandedMessageId(null)
    setGraphMode("global")
    setLiveGraph(null)
    setRetrievalHighlight(null)
    pendingEvidenceFitRef.current = null
    setMessages((prev) => [
      ...prev,
      { id: makeId(), role: "user", content: query },
      { id: assistantId, role: "assistant", content: "", reasoning: "", search: null },
    ])

    try {
      const response = await fetch(`${apiBase}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          project_id: activeProjectId,
          session_id: session.id,
          top_k: 6,
        }),
      })
      if (!response.ok || !response.body) {
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder("utf-8")
      let buffer = ""

      while (true) {
        const { value, done } = await reader.read()
        if (done) {
          break
        }
        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split("\n\n")
        buffer = events.pop() || ""

        for (const event of events) {
          const payload = event.replace(/^data:\s*/, "")
          if (!payload) {
            continue
          }
          let parsed = null
          try {
            parsed = JSON.parse(payload)
          } catch {
            parsed = null
          }
          if (!parsed) {
            continue
          }

          if (parsed.type === "search") {
            updateAssistantMessage(assistantId, { search: parsed.payload })
          } else if (parsed.type === "reasoning") {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      reasoning: `${message.reasoning || ""}${parsed.content || ""}`,
                    }
                  : message
              )
            )
          } else if (parsed.type === "content_chunk") {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: `${message.content || ""}${parsed.content || ""}`,
                    }
                  : message
              )
            )
          } else if (parsed.type === "error") {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: parsed.message || "LLM request failed.",
                    }
                  : message
              )
            )
          }
        }
      }

      await Promise.all([
        loadMessages(activeProjectId, session.id),
        loadChatSessions(activeProjectId),
      ])
    } finally {
      setSending(false)
    }
  }

  const renderMessage = (message) => {
    const isUser = message.role === "user"
    const focusTerms = message.search?.focus_terms || []
    const reasoningText =
      !isUser && message.reasoning
        ? typeof message.reasoning === "string"
          ? message.reasoning
          : JSON.stringify(message.reasoning, null, 2)
        : ""
    return (
      <div
        key={message.id}
        className={`rounded-3xl border px-4 py-4 shadow-sm ${
          isUser ? "border-violet-200 bg-violet-50" : "border-slate-200 bg-white"
        }`}
      >
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
          {isUser ? "User" : "PaperMem"}
        </div>
        <div className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-800">
          {message.content ||
            reasoningText ||
            (sending && !isUser ? "Thinking..." : "")}
        </div>
        {!isUser && focusTerms.length ? (
          <div className="mt-3 rounded-2xl bg-violet-50 p-3">
            <div className="text-xs font-medium text-violet-700">Focus</div>
            <div className="scrollbar-hidden mt-2 flex gap-2 overflow-x-auto pb-1">
              {focusTerms.map((term) => (
                <div
                  key={term}
                  className="shrink-0 rounded-full border border-violet-200 bg-white px-3 py-1 text-xs text-violet-700"
                >
                  {term}
                </div>
              ))}
            </div>
          </div>
        ) : null}
        {!isUser && message.search?.citations?.length ? (
          <details
            className="mt-3 rounded-2xl bg-emerald-50 p-3"
            open={evidenceExpandedMessageId === message.id}
            onToggle={(e) => {
              const nextOpen = e.target.open
              if (nextOpen) {
                setEvidenceExpandedMessageId(message.id)
                if (message.search?.overlay) {
                  setGraphMode("live")
                  setLiveGraph(message.search.overlay)
                  setRetrievalHighlight(computeRetrievalHighlight(message.search))
                  pendingEvidenceFitRef.current = computeEvidenceFitNodeIds(message.search)
                } else {
                  setGraphMode("global")
                  setLiveGraph(null)
                  setRetrievalHighlight(null)
                  pendingEvidenceFitRef.current = null
                }
              } else if (evidenceExpandedMessageId === message.id) {
                setEvidenceExpandedMessageId(null)
                if (message.search?.overlay) {
                  setGraphMode("global")
                  setLiveGraph(null)
                  setRetrievalHighlight(null)
                  pendingEvidenceFitRef.current = null
                }
              }
            }}
          >
            <summary className="cursor-pointer text-xs font-medium text-emerald-700">
              {`Evidence (${message.search.citations.length})`}
            </summary>
            <div className="mt-2 space-y-2 text-xs text-emerald-900">
              {message.search.citations.map((citation) => (
                <div
                  key={citation.unit_id}
                  className="rounded-xl bg-white/70 p-2"
                >
                  <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-emerald-600">
                    {sourceTypeLabel(citation.source_type)}
                  </div>
                  <div className="scrollbar-hidden mt-1 overflow-x-auto whitespace-nowrap text-xs leading-5 text-emerald-900">
                    {citationDisplayText(message.search, citation)}
                  </div>
                </div>
              ))}
            </div>
          </details>
        ) : null}
      </div>
    )
  }

  return (
    <div className="relative h-screen overflow-hidden bg-slate-100 text-slate-900">
      {isProjectModalOpen ? (
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-slate-900/30 px-4">
          <div className="w-full max-w-sm rounded-3xl border border-slate-200 bg-white p-5 shadow-2xl">
            <div className="text-base font-semibold text-slate-900">New project</div>
            <div className="mt-1 text-sm text-slate-500">
              Create a project to group files, chats, and graph memory.
            </div>
            <input
              autoFocus
              value={newProjectName}
              onChange={(event) => setNewProjectName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault()
                  handleCreateProject()
                }
                if (event.key === "Escape") {
                  setIsProjectModalOpen(false)
                  setNewProjectName("")
                }
              }}
              placeholder="Project name"
              className="mt-4 w-full rounded-2xl border border-slate-200 px-4 py-3 text-sm text-slate-800 outline-none transition focus:border-violet-300"
            />
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  setIsProjectModalOpen(false)
                  setNewProjectName("")
                }}
                className="rounded-full px-4 py-2 text-sm font-medium text-slate-500 transition hover:bg-slate-100"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleCreateProject}
                disabled={!newProjectName.trim()}
                className="rounded-full bg-violet-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      ) : null}
      <div className="flex h-full overflow-hidden">
        <aside className="h-full w-64 overflow-y-auto border-r border-slate-200 bg-white px-5 py-6">
          <div className="text-xl font-semibold">PaperMem</div>
          <div className="mt-8 flex items-center justify-between">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
              Projects
            </div>
            <button
              type="button"
              onClick={() => setIsProjectModalOpen(true)}
              className="rounded-full bg-violet-600 px-3 py-1 text-xs font-medium text-white"
            >
              New
            </button>
          </div>
          <div className="mt-4 space-y-2">
            {loadingProjects ? (
              <div className="text-sm text-slate-400">Loading...</div>
            ) : null}
            {projects.map((project) => (
              <div key={project.id} className="group relative">
                <button
                  type="button"
                  onClick={() => setActiveProjectId(project.id)}
                  className={`w-full rounded-2xl border px-4 py-3 pr-10 text-left transition ${
                    activeProjectId === project.id
                      ? "border-violet-300 bg-violet-50"
                      : "border-slate-200 bg-white hover:border-slate-300"
                  }`}
                >
                  <div className="text-sm font-medium">{project.name}</div>
                  <div className="mt-1 text-xs text-slate-400">
                    {project.type || "General"}
                  </div>
                </button>
                <button
                  type="button"
                  aria-label={`Delete ${project.name}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    deleteProject(project.id, project.name)
                  }}
                  className="absolute right-3 top-3 hidden h-6 w-6 items-center justify-center rounded-full text-sm text-slate-400 transition hover:bg-white hover:text-rose-500 group-hover:flex"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </aside>

        <main className="flex min-h-0 flex-1 overflow-hidden">
          <section className="flex min-h-0 min-w-0 flex-[0_0_68%] flex-col border-r border-slate-200 bg-white">
            <header className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 px-6 py-4">
              <div className="flex items-center gap-2">
                <div className="text-lg font-semibold">PaperMem</div>
                <button
                  type="button"
                  onClick={() => setShowLegend((prev) => !prev)}
                  className="flex h-6 w-6 items-center justify-center rounded-full border border-slate-200 bg-white text-xs font-semibold text-slate-500 transition hover:border-slate-300 hover:bg-slate-50"
                  aria-label="Toggle graph legend"
                  title="Toggle graph legend"
                >
                  i
                </button>
              </div>
              <button
                type="button"
                onClick={refreshGraph}
                disabled={!activeProjectId || graphLoading}
                className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50 disabled:opacity-50"
              >
                {graphLoading ? "Refreshing..." : "Refresh"}
              </button>
            </header>

            <div className="relative min-h-0 flex-1 overflow-hidden">
              {graphLoading ? (
                <div className="flex h-full items-center justify-center text-sm text-slate-400">
                  Loading graph...
                </div>
              ) : graphError ? (
                <div className="flex h-full items-center justify-center text-sm text-rose-500">
                  {graphError}
                </div>
              ) : (
                <div ref={graphContainerRef} className="h-full w-full" />
              )}

              {showLegend ? (
                <div className="pointer-events-auto absolute right-4 top-4 z-10 w-[240px] rounded-2xl border border-slate-200 bg-white/95 p-4 shadow-xl backdrop-blur">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-slate-900">Legend</div>
                    <button
                      type="button"
                      onClick={() => setShowLegend(false)}
                      className="rounded-full px-2 py-1 text-xs text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
                    >
                      Close
                    </button>
                  </div>
                  <div className="mt-3 space-y-2 text-xs text-slate-600">
                    <div className="flex items-center gap-2">
                      <span className="inline-flex h-4 w-4 items-center justify-center text-[10px] text-violet-600">
                        ★
                      </span>
                      <span>`query` node</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="h-3 w-3 rounded-full bg-amber-500" />
                      <span>`claim` node</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="h-3 w-3 rounded-full bg-emerald-400" />
                      <span>`entity` node</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="h-3 w-3 rounded-full bg-sky-400" />
                      <span>`concept` node</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="h-3 w-3 rounded-full bg-pink-500" />
                      <span>`file_chunk` node</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="h-3 w-3 rounded-full bg-orange-500" />
                      <span>`conversation` / `capture` node</span>
                    </div>
                    <div className="pt-2 text-[11px] text-slate-400">
                      Arrows show relation direction. Larger nodes indicate higher local importance in the current zoomed view.
                    </div>
                  </div>
                </div>
              ) : null}

              {selectedGraphNode ? (
                <div className="pointer-events-auto absolute bottom-4 left-4 z-10 w-[340px] rounded-2xl border border-slate-200 bg-white/95 p-4 shadow-xl backdrop-blur">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                        {selectedGraphNode.kind}
                      </div>
                      <div className="mt-1 text-sm font-semibold text-slate-900">
                        {selectedGraphNode.label}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => setSelectedGraphNode(null)}
                      className="rounded-full px-2 py-1 text-xs text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
                    >
                      Close
                    </button>
                  </div>
                  <div className="mt-3 max-h-36 overflow-y-auto whitespace-pre-wrap text-xs leading-5 text-slate-600">
                    {selectedGraphNode.fullText}
                  </div>
                </div>
              ) : null}

            </div>
          </section>

          <section className="flex h-full min-h-0 min-w-[360px] max-w-[420px] flex-[0_0_32%] flex-col overflow-hidden bg-slate-50">
            <header className="border-b border-slate-200 px-5 py-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold">Agent Chat</div>
                  <div className="mt-1 text-xs text-slate-400">
                    Quick capture writes into the active project and active chat.
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleNewChat}
                  disabled={!activeProjectId}
                  className="rounded-full bg-violet-600 px-3 py-1.5 text-xs font-medium text-white shadow-sm disabled:opacity-50"
                >
                  New chat
                </button>
              </div>
              <div className="mt-4 flex gap-2 overflow-x-auto pb-1">
                {loadingSessions ? (
                  <div className="text-xs text-slate-400">Loading chats...</div>
                ) : (
                  chatSessions.map((session) => (
                    <div key={session.id} className="group relative min-w-[116px]">
                      <button
                        type="button"
                        onClick={() => activateChatSession(session.id)}
                        className={`w-full rounded-2xl border px-3 py-2 pr-8 text-left text-xs transition ${
                          activeSessionId === session.id
                            ? "border-violet-300 bg-violet-50"
                            : "border-slate-200 bg-white"
                        }`}
                      >
                        <div className="font-medium text-slate-700">
                          {truncate(session.title || "Untitled chat", 20)}
                        </div>
                        <div className="mt-1 text-slate-400">
                          {session.message_count || 0} messages
                        </div>
                      </button>
                      <button
                        type="button"
                        aria-label={`Delete ${session.title || "chat"}`}
                        onClick={(event) => {
                          event.stopPropagation()
                          deleteChatSession(session.id, session.title)
                        }}
                        className="absolute right-2 top-2 hidden h-5 w-5 items-center justify-center rounded-full text-sm text-slate-400 transition hover:bg-white hover:text-rose-500 group-hover:flex"
                      >
                        ×
                      </button>
                    </div>
                  ))
                )}
              </div>
            </header>

            <div ref={chatScrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4">
              {loadingMessages ? (
                <div className="text-sm text-slate-400">Loading chat history...</div>
              ) : messages.length ? (
                messages.map(renderMessage)
              ) : (
                <div className="rounded-3xl border border-dashed border-slate-300 bg-white px-4 py-6 text-sm text-slate-500">
                  {activeSession
                    ? "This chat is empty. Ask a new question to start a fresh reasoning trace."
                    : "Start chatting or create a new chat for this project."}
                </div>
              )}
            </div>

            <div className="border-t border-slate-200 px-4 py-4">
              <div className="rounded-3xl border border-slate-200 bg-white p-3 shadow-sm">
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  disabled={!activeProjectId || sending}
                  rows={3}
                  placeholder={
                    activeProjectId
                      ? "Ask the memory graph..."
                      : "Create or select a project first"
                  }
                  className="w-full resize-none border-0 text-sm leading-6 text-slate-800 outline-none"
                />
                <div className="mt-3 flex items-center justify-between gap-3">
                  <div className="text-xs text-slate-400">Semantic zoom updates from retrieval and refresh.</div>
                  <button
                    type="button"
                    onClick={handleSend}
                    disabled={!activeProjectId || sending || !input.trim()}
                    className="rounded-full bg-violet-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
                  >
                    {sending ? "Sending..." : "Send"}
                  </button>
                </div>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  )
}

export default App
