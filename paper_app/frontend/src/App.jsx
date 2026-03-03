import { useEffect, useMemo, useRef, useState } from "react"
import { Network } from "vis-network/standalone/esm/vis-network"

const apiBase = window?.paperMem?.apiBase || "http://127.0.0.1:8000"

const makeId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`

function App() {
  const [projects, setProjects] = useState([])
  const [activeProjectId, setActiveProjectId] = useState("")
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [sending, setSending] = useState(false)
  const [loadingProjects, setLoadingProjects] = useState(false)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [showGraph, setShowGraph] = useState(false)
  const [graphLoading, setGraphLoading] = useState(false)
  const [graphError, setGraphError] = useState("")
  const [graphData, setGraphData] = useState(null)
  const [graphQuery, setGraphQuery] = useState("")
  const [graphSearchResult, setGraphSearchResult] = useState(null)
  const [graphSearchLoading, setGraphSearchLoading] = useState(false)
  const [graphSearchError, setGraphSearchError] = useState("")
  const [graphTopK, setGraphTopK] = useState(5)
  const [showAllGraph, setShowAllGraph] = useState(true)
  const [selectedNodeDetail, setSelectedNodeDetail] = useState(null)
  const graphContainerRef = useRef(null)
  const graphNetworkRef = useRef(null)

  const activeProject = useMemo(
    () => projects.find((p) => p.id === activeProjectId),
    [projects, activeProjectId]
  )

  const loadProjects = async () => {
    setLoadingProjects(true)
    const response = await fetch(`${apiBase}/projects`)
    if (!response.ok) {
      setLoadingProjects(false)
      return
    }
    const data = await response.json()
    setProjects(data)
    if (data.length && !activeProjectId) {
      setActiveProjectId(data[0].id)
    }
    setLoadingProjects(false)
  }

  useEffect(() => {
    loadProjects()
  }, [])

  useEffect(() => {
    if (activeProjectId && window.paperMem?.setActiveProjectId) {
      window.paperMem.setActiveProjectId(activeProjectId)
    }
    loadMessages(activeProjectId)
    setGraphQuery("")
    setGraphSearchResult(null)
    setGraphSearchError("")
    setSelectedNodeDetail(null)
    setShowAllGraph(true)
  }, [activeProjectId])

  const fetchGraph = async (projectId) => {
    if (!projectId) {
      return
    }
    setGraphLoading(true)
    setGraphError("")
    const response = await fetch(`${apiBase}/graph/${projectId}?limit=200`)
    if (!response.ok) {
      setGraphError("图谱加载失败")
      setGraphLoading(false)
      return
    }
    const data = await response.json()
    setGraphData(data)
    setGraphLoading(false)
  }

  useEffect(() => {
    if (!showGraph || !activeProjectId) {
      return
    }
    let cancelled = false
    const loadGraph = async () => {
      if (cancelled) {
        return
      }
      await fetchGraph(activeProjectId)
    }
    loadGraph()
    return () => {
      cancelled = true
    }
  }, [showGraph, activeProjectId])

  const searchBundles = useMemo(() => {
    const bundles =
      graphSearchResult?.search_results?.bundles ||
      graphSearchResult?.bundles ||
      []
    return Array.isArray(bundles) ? bundles : []
  }, [graphSearchResult])

  const recentTurns = useMemo(() => {
    const conversations =
      graphSearchResult?.search_results?.recent_turns?.conversations || []
    return Array.isArray(conversations) ? conversations : []
  }, [graphSearchResult])

  const topicColorMap = useMemo(() => {
    if (!graphSearchResult) {
      return null
    }
    const palette = [
      "#8B5CF6",
      "#06B6D4",
      "#F97316",
      "#22C55E",
      "#EC4899",
      "#F59E0B",
      "#3B82F6",
      "#14B8A6",
      "#A855F7",
      "#EF4444",
    ]
    const map = new Map()
    let cursor = 0
    searchBundles.forEach((bundle) => {
      const topics = Array.isArray(bundle?.topics) ? bundle.topics : []
      topics.forEach((topic) => {
        if (!topic?.topic_id) {
          return
        }
        const key = String(topic.topic_id)
        if (!map.has(key)) {
          map.set(key, palette[cursor % palette.length])
          cursor += 1
        }
      })
    })
    return map.size ? map : null
  }, [graphSearchResult, searchBundles])

  const topicInfoMap = useMemo(() => {
    if (!graphSearchResult) {
      return null
    }
    const map = new Map()
    searchBundles.forEach((bundle) => {
      const topics = Array.isArray(bundle?.topics) ? bundle.topics : []
      topics.forEach((topic) => {
        if (!topic?.topic_id) {
          return
        }
        map.set(String(topic.topic_id), topic.title || `Topic ${topic.topic_id}`)
      })
    })
    return map.size ? map : null
  }, [graphSearchResult, searchBundles])

  const bundleSummary = useMemo(() => {
    if (!graphSearchResult) {
      return null
    }
    const list = searchBundles.map((bundle) => ({
      bundleId: bundle?.bundle_id,
      factCount: Array.isArray(bundle?.facts) ? bundle.facts.length : 0,
      topicCount: Array.isArray(bundle?.topics) ? bundle.topics.length : 0,
      conversationCount: Array.isArray(bundle?.conversations)
        ? bundle.conversations.length
        : 0,
    }))
    return list.length ? list : null
  }, [graphSearchResult, searchBundles])

  const topicLegend = useMemo(() => {
    if (!topicColorMap) {
      return null
    }
    const entries = Array.from(topicColorMap.entries()).map(([id, color]) => ({
      id,
      color,
      title: topicInfoMap?.get(id) || `Topic ${id}`,
    }))
    return entries.length ? entries : null
  }, [topicColorMap, topicInfoMap])

  const factIdToGraphNodeId = useMemo(() => {
    const map = new Map()
    if (!graphData?.nodes) {
      return map
    }
    graphData.nodes.forEach((node) => {
      if (node?.fact_id === null || node?.fact_id === undefined) {
        return
      }
      map.set(String(node.fact_id), String(node.id))
    })
    return map
  }, [graphData])

  const truncateText = (value, maxLength) => {
    if (typeof value !== "string") {
      return ""
    }
    if (value.length <= maxLength) {
      return value
    }
    return value.slice(0, maxLength)
  }

  useEffect(() => {
    if (!showGraph || !graphContainerRef.current) {
      return
    }
    if (!graphData && !graphSearchResult) {
      return
    }
    if (graphNetworkRef.current) {
      graphNetworkRef.current.destroy()
      graphNetworkRef.current = null
    }

    const rawNodes = Array.isArray(graphData?.nodes) ? graphData.nodes : []
    const rawEdges = Array.isArray(graphData?.edges) ? graphData.edges : []
    const hasSearch = Boolean(graphSearchResult)

    const nodeMap = new Map()
    const detailMap = new Map()
    const edgeMap = new Map()

    const upsertNode = (id, patch) => {
      const current = nodeMap.get(id) || { id }
      nodeMap.set(id, { ...current, ...patch })
    }

    const upsertDetail = (id, detail) => {
      detailMap.set(id, detail)
    }

    const addEdge = (edge) => {
      if (edgeMap.has(edge.id)) {
        return
      }
      edgeMap.set(edge.id, edge)
    }

    if (!hasSearch || showAllGraph) {
      rawNodes.forEach((node) => {
        if (node?.id === null || node?.id === undefined) {
          return
        }
        const id = String(node.id)
        upsertNode(id, {
          label: truncateText(node.label, 50) || id,
          size: 10,
          shape: "dot",
          color: hasSearch
            ? { background: "#E5E7EB", border: "#CBD5F5" }
            : { background: "#8B5CF6", border: "#6D28D9" },
          font: { color: "#1F2937", size: 12 },
        })
        upsertDetail(id, {
          type: "fact",
          title: node.label || id,
          content: typeof node.label === "string" ? node.label : "",
          score: null,
          meta: { fact_id: node.fact_id },
        })
      })
    }

    if (hasSearch) {
      searchBundles.forEach((bundle) => {
        const bundleId = bundle?.bundle_id
        const bundleKey = `bundle-${bundleId}`
        const facts = Array.isArray(bundle?.facts) ? bundle.facts : []
        const topics = Array.isArray(bundle?.topics) ? bundle.topics : []
        const conversations = Array.isArray(bundle?.conversations)
          ? bundle.conversations
          : []
        const topTopic = topics.reduce((acc, topic) => {
          if (!topic) {
            return acc
          }
          if (!acc || (topic.score || 0) > (acc.score || 0)) {
            return topic
          }
          return acc
        }, null)
        const topicId = topTopic?.topic_id ?? null
        const topicColor = topicId
          ? topicColorMap?.get(String(topicId)) || "#8B5CF6"
          : "#8B5CF6"
        const conversationScore = conversations.reduce((max, convo) => {
          const score = Number.isFinite(convo?.score) ? convo.score : 0
          return Math.max(max, score)
        }, 0)
        const bundleSize = 40 + (facts.length + topics.length + conversations.length) * 6

        if (bundleId !== null && bundleId !== undefined) {
          upsertNode(bundleKey, {
            label: `Bundle ${bundleId}`,
            size: bundleSize,
            shape: "dot",
            color: {
              background: "rgba(148, 163, 184, 0.15)",
              border: "rgba(148, 163, 184, 0.4)",
            },
            font: { color: "#64748B", size: 11 },
          })
          upsertDetail(bundleKey, {
            type: "bundle",
            title: `Bundle ${bundleId}`,
            content: "",
            score: null,
            meta: {
              facts: facts.length,
              topics: topics.length,
              conversations: conversations.length,
            },
          })
        }

        facts.forEach((fact) => {
          if (fact?.fact_id === null || fact?.fact_id === undefined) {
            return
          }
          const factId = String(fact.fact_id)
          const graphNodeId = factIdToGraphNodeId.get(factId)
          const nodeId = graphNodeId || `fact-${factId}`
          const score = Number.isFinite(fact.score) ? fact.score : 0
          const size = Math.max(6, 6 + score * 24)
          const borderWidth = conversationScore > 0 ? 1 + Math.min(conversationScore, 1) * 2 : 1
          const factLabel = truncateText(fact.content, 50) || `Fact ${factId}`
          upsertNode(nodeId, {
            label: factLabel,
            size,
            shape: "dot",
            color: {
              background: topicColor,
              border: topicColor,
            },
            borderWidth,
            font: { color: "#1F2937", size: 12 },
          })
          upsertDetail(nodeId, {
            type: "fact",
            title: `Fact ${factId}`,
            content: typeof fact.content === "string" ? fact.content : "",
            score,
            meta: fact.metadata || {},
          })
          if (bundleId !== null && bundleId !== undefined) {
            addEdge({
              id: `${bundleKey}-${nodeId}`,
              from: bundleKey,
              to: nodeId,
              color: { color: "#E2E8F0" },
              dashes: true,
            })
          }
        })

        topics.forEach((topic) => {
          if (!topic?.topic_id) {
            return
          }
          const topicKey = `topic-${topic.topic_id}`
          const color = topicColorMap?.get(String(topic.topic_id)) || "#8B5CF6"
          const topicLabel =
            truncateText(topic.title, 30) || `Topic ${topic.topic_id}`
          upsertNode(topicKey, {
            label: topicLabel,
            size: 14,
            shape: "square",
            color: {
              background: color,
              border: color,
            },
            font: { color: "#1F2937", size: 11 },
          })
          upsertDetail(topicKey, {
            type: "topic",
            title: topic.title || `Topic ${topic.topic_id}`,
            content: typeof topic.summary === "string" ? topic.summary : "",
            score: Number.isFinite(topic.score) ? topic.score : null,
            meta: {},
          })
          if (bundleId !== null && bundleId !== undefined) {
            addEdge({
              id: `${bundleKey}-${topicKey}`,
              from: bundleKey,
              to: topicKey,
              color: { color: "#E2E8F0" },
              dashes: true,
            })
          }
        })

        conversations.forEach((conversation) => {
          if (!conversation?.conversation_id) {
            return
          }
          const convoKey = `conv-${conversation.conversation_id}`
          const convoLabel =
            truncateText(conversation.text, 30) ||
            `Conversation ${conversation.conversation_id}`
          upsertNode(convoKey, {
            label: convoLabel,
            size: 12,
            shape: "triangle",
            color: {
              background: "#94A3B8",
              border: "#64748B",
            },
            font: { color: "#1F2937", size: 11 },
          })
          upsertDetail(convoKey, {
            type: "conversation",
            title: `Conversation ${conversation.conversation_id}`,
            content:
              typeof conversation.text === "string" ? conversation.text : "",
            score: Number.isFinite(conversation.score) ? conversation.score : null,
            meta: conversation.metadata || {},
          })
          if (bundleId !== null && bundleId !== undefined) {
            addEdge({
              id: `${bundleKey}-${convoKey}`,
              from: bundleKey,
              to: convoKey,
              color: { color: "#E2E8F0" },
              dashes: true,
            })
          }
        })
      })

      rawEdges.forEach((edge) => {
        if (edge?.source === null || edge?.source === undefined) {
          return
        }
        if (edge?.target === null || edge?.target === undefined) {
          return
        }
        const edgeId = edge.id ? String(edge.id) : `${edge.source}-${edge.target}`
        addEdge({
          id: edgeId,
          from: String(edge.source),
          to: String(edge.target),
          label: edge.type || "",
          color: { color: "#CBD5F5" },
        })
      })
    } else {
      rawEdges.forEach((edge) => {
        if (edge?.source === null || edge?.source === undefined) {
          return
        }
        if (edge?.target === null || edge?.target === undefined) {
          return
        }
        const edgeId = edge.id ? String(edge.id) : `${edge.source}-${edge.target}`
        addEdge({
          id: edgeId,
          from: String(edge.source),
          to: String(edge.target),
          label: edge.type || "",
          color: { color: "#CBD5F5" },
        })
      })
    }

    const nodes = Array.from(nodeMap.values())
    const nodeIdSet = new Set(nodes.map((node) => node.id))
    const edges = Array.from(edgeMap.values()).filter(
      (edge) => nodeIdSet.has(edge.from) && nodeIdSet.has(edge.to)
    )

    if (nodes.length === 0) {
      setGraphError(hasSearch ? "检索结果暂无图谱数据" : "暂无图谱数据")
      return
    }

    try {
      setGraphError("")
      graphNetworkRef.current = new Network(
        graphContainerRef.current,
        { nodes, edges },
        {
          nodes: {
            shape: "dot",
            size: 10,
            color: { background: "#8B5CF6", border: "#6D28D9" },
            font: { color: "#1F2937", size: 12 },
          },
          edges: {
            color: { color: "#CBD5F5" },
            smooth: true,
          },
          physics: {
            stabilization: true,
          },
        }
      )
      graphNetworkRef.current.on("click", (params) => {
        if (!params?.nodes?.length) {
          setSelectedNodeDetail(null)
          return
        }
        const nodeId = params.nodes[0]
        setSelectedNodeDetail(detailMap.get(nodeId) || null)
      })
    } catch (error) {
      const message =
        error && error.message ? `图谱渲染失败：${error.message}` : "图谱渲染失败"
      setGraphError(message)
    }
  }, [
    showGraph,
    graphData,
    graphSearchResult,
    showAllGraph,
    searchBundles,
    topicColorMap,
    factIdToGraphNodeId,
  ])

  useEffect(() => {
    if (showGraph) {
      return
    }
    if (graphNetworkRef.current) {
      graphNetworkRef.current.destroy()
      graphNetworkRef.current = null
    }
    setSelectedNodeDetail(null)
  }, [showGraph])

  const createProject = async () => {
    const name = `Project ${new Date().toLocaleString()}`
    const response = await fetch(`${apiBase}/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    })
    if (!response.ok) {
      return
    }
    const project = await response.json()
    setProjects((prev) => [project, ...prev])
    setActiveProjectId(project.id)
  }

  const loadMessages = async (projectId) => {
    if (!projectId) {
      setMessages([])
      return
    }
    setLoadingMessages(true)
    const response = await fetch(`${apiBase}/projects/${projectId}/messages`)
    if (!response.ok) {
      setLoadingMessages(false)
      return
    }
    const data = await response.json()
    const mapped = data.map((msg) => {
      let searchPayload = null
      if (msg.search_results) {
        try {
          searchPayload = JSON.parse(msg.search_results)
        } catch (error) {
          searchPayload = msg.search_results
        }
      }
      return {
        id: msg.id,
        role: msg.role,
        content: msg.content,
        reasoning: msg.reasoning_trace || "",
        search: searchPayload,
      }
    })
    setMessages(mapped)
    setLoadingMessages(false)
  }

  const confirmDeleteProject = async () => {
    if (!deleteTarget) {
      return
    }
    const response = await fetch(`${apiBase}/projects/${deleteTarget.id}`, {
      method: "DELETE",
    })
    if (response.ok) {
      setProjects((prev) => prev.filter((p) => p.id !== deleteTarget.id))
      if (activeProjectId === deleteTarget.id) {
        const next = projects.find((p) => p.id !== deleteTarget.id)
        setActiveProjectId(next ? next.id : "")
        setMessages([])
      }
    }
    setDeleteTarget(null)
  }

  const appendMessage = (message) => {
    setMessages((prev) => [...prev, message])
  }

  const updateMessage = (id, updater) => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.id !== id) {
          return msg
        }
        const patch = typeof updater === "function" ? updater(msg) : updater
        return { ...msg, ...patch }
      })
    )
  }

  const handleSend = async () => {
    if (!input.trim() || !activeProjectId || sending || loadingMessages) {
      return
    }
    setSending(true)

    const userMessage = { id: makeId(), role: "user", content: input.trim() }
    appendMessage(userMessage)
    setInput("")

    const assistantId = makeId()
    appendMessage({
      id: assistantId,
      role: "assistant",
      content: "",
      reasoning: "",
      search: null,
    })

    const response = await fetch(`${apiBase}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: userMessage.content,
        project_id: activeProjectId,
        top_k: 5,
      }),
    })

    if (!response.body) {
      setSending(false)
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder("utf-8")
    let buffer = ""

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      let boundaryIndex = buffer.indexOf("\n\n")
      while (boundaryIndex !== -1) {
        const rawEvent = buffer.slice(0, boundaryIndex).trim()
        buffer = buffer.slice(boundaryIndex + 2)
        boundaryIndex = buffer.indexOf("\n\n")

        if (!rawEvent.startsWith("data:")) {
          continue
        }

        const payload = rawEvent.replace(/^data:\s*/, "")
        if (!payload) {
          continue
        }

        let parsed = null
        try {
          parsed = JSON.parse(payload)
        } catch (error) {
          parsed = null
        }

        if (!parsed) {
          continue
        }

        if (parsed.type === "search") {
          updateMessage(assistantId, { search: parsed.payload })
        } else if (parsed.type === "content_chunk") {
          updateMessage(assistantId, (prev) => ({
            content: prev.content + parsed.content,
          }))
        } else if (parsed.type === "reasoning") {
          updateMessage(assistantId, (prev) => ({
            reasoning: prev.reasoning + parsed.content,
          }))
        }
      }
    }

    setSending(false)
  }

  const handleGraphSearch = async () => {
    if (!graphQuery.trim() || !activeProjectId || graphSearchLoading) {
      return
    }
    setGraphSearchLoading(true)
    setGraphSearchError("")
    setGraphError("")
    setSelectedNodeDetail(null)
    const topKValue = Math.max(1, Math.min(50, Number(graphTopK) || 1))
    const response = await fetch(`${apiBase}/agenticSearch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: graphQuery.trim(),
        project_id: activeProjectId,
        top_k: topKValue,
      }),
    })
    if (!response.ok) {
      setGraphSearchError("检索失败")
      setGraphSearchLoading(false)
      return
    }
    const data = await response.json()
    setGraphSearchResult(data)
    setGraphSearchLoading(false)
  }

  const handleGraphRefresh = async () => {
    setGraphQuery("")
    setGraphSearchResult(null)
    setGraphSearchError("")
    setGraphError("")
    setSelectedNodeDetail(null)
    setShowAllGraph(true)
    await fetchGraph(activeProjectId)
  }

  const renderMessage = (message) => {
    if (message.role === "assistant") {
      const searchText = message.search
        ? JSON.stringify(message.search, null, 2)
        : ""
      return (
        <div className="rounded-2xl border border-gray-100 bg-white px-6 py-5 shadow-sm">
          <div className="text-xs uppercase tracking-wider text-gray-400">
            Assistant
          </div>
          {message.search ? (
            <details className="mt-2 text-xs text-gray-400">
              <summary className="cursor-pointer">检索到的记忆</summary>
              <div className="mt-2 whitespace-pre-wrap text-xs leading-6 text-gray-400">
                {searchText}
              </div>
            </details>
          ) : null}
          {message.reasoning ? (
            <details className="mt-2 text-xs text-gray-400">
              <summary className="cursor-pointer">思考过程</summary>
              <div className="mt-2 whitespace-pre-wrap text-xs leading-6 text-gray-400">
                {message.reasoning}
              </div>
            </details>
          ) : null}
          <div className="mt-3 whitespace-pre-wrap text-sm leading-6 text-ink">
            {message.content || "生成中..."}
          </div>
        </div>
      )
    }

    return (
      <div className="rounded-2xl border border-gray-100 bg-white px-6 py-5 shadow-sm">
        <div className="text-xs uppercase tracking-wider text-gray-400">User</div>
        <div className="mt-2 whitespace-pre-wrap text-sm leading-6 text-ink">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-paper text-ink">
      <div className="flex min-h-screen">
        <aside className="w-64 border-r border-gray-100 bg-mist px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="text-sm font-semibold tracking-wide text-ink">
              PaperMem Copilot
            </div>
            <button
              type="button"
              onClick={createProject}
              className="rounded-md border border-transparent bg-ether px-2 py-1 text-xs font-medium text-white"
            >
              New
            </button>
          </div>
          <div className="mt-8 space-y-2">
            <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">
              Projects
            </div>
            <div className="space-y-2">
              {loadingProjects ? (
                <div className="flex items-center gap-2 rounded-lg border border-dashed border-gray-200 bg-white px-3 py-3 text-xs text-gray-400">
                  <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-gray-200 border-t-ether" />
                  Loading projects
                </div>
              ) : null}
              {projects.map((project) => (
                <div
                  key={project.id}
                  className={`group relative rounded-lg border px-3 py-2 text-sm ${project.id === activeProjectId
                    ? "border-ether bg-white text-ink"
                    : "border-gray-100 bg-white text-gray-600"
                    }`}
                >
                  <button
                    type="button"
                    disabled={loadingProjects}
                    onClick={() => setActiveProjectId(project.id)}
                    className="w-full text-left"
                  >
                    {project.name}
                  </button>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation()
                      setDeleteTarget(project)
                    }}
                    className="absolute right-2 top-2 hidden h-5 w-5 items-center justify-center rounded-full text-xs text-red-500 hover:bg-red-50 group-hover:flex"
                  >
                    ×
                  </button>
                </div>
              ))}
              <button
                type="button"
                onClick={createProject}
                disabled={loadingProjects}
                className="w-full rounded-lg border border-dashed border-gray-200 bg-white px-3 py-2 text-left text-sm text-gray-400"
              >
                Add Project
              </button>
            </div>
          </div>
        </aside>
        <main className="flex flex-1 flex-col">
          <header className="flex items-center justify-between border-b border-gray-100 px-8 py-6">
            <div>
              <div className="text-lg font-semibold text-ink">
                {activeProject?.name || "Main Chat"}
              </div>
              <div className="text-xs text-gray-400">
                Query your memory with long-term context
              </div>
            </div>
            <button
              type="button"
              onClick={() => setShowGraph(true)}
              disabled={!activeProjectId}
              className="flex h-8 w-8 items-center justify-center rounded-full bg-lavender text-white disabled:opacity-50"
              title="Project Graph"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="9" />
                <path d="M3 12h18" />
                <path d="M12 3a15 15 0 0 1 0 18" />
                <path d="M12 3a15 15 0 0 0 0 18" />
              </svg>
            </button>
          </header>
          <section className="flex-1 px-8 py-8">
            <div className="mx-auto max-w-3xl space-y-6">
              {loadingMessages ? (
                <div className="flex items-center gap-2 rounded-2xl border border-gray-100 bg-white px-6 py-5 text-xs text-gray-400 shadow-sm">
                  <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-gray-200 border-t-ether" />
                  Loading messages
                </div>
              ) : messages.length === 0 ? (
                <div className="rounded-2xl border border-gray-100 bg-white px-6 py-5 shadow-sm">
                  <div className="text-xs uppercase tracking-wider text-gray-400">
                    Assistant
                  </div>
                  <div className="mt-2 text-sm leading-6 text-ink">
                    在这里进行检索式对话，回答会融合长期记忆内容。
                  </div>
                </div>
              ) : (
                messages.map((message) => (
                  <div key={message.id}>{renderMessage(message)}</div>
                ))
              )}
            </div>
          </section>
          <footer className="border-t border-gray-100 px-8 py-6">
            <div className="mx-auto flex max-w-3xl items-center gap-3 rounded-2xl border border-gray-200 bg-white px-4 py-3 shadow-sm">
              <input
                className="flex-1 text-sm text-ink outline-none placeholder:text-gray-300"
                placeholder={
                  activeProjectId
                    ? loadingMessages
                      ? "加载中..."
                      : "输入问题，Enter 发送"
                    : "先创建项目"
                }
                value={input}
                disabled={!activeProjectId || sending || loadingMessages}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    handleSend()
                  }
                }}
              />
              <button
                type="button"
                onClick={handleSend}
                disabled={!activeProjectId || sending || loadingMessages}
                className="rounded-lg bg-ether px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
              >
                Send
              </button>
            </div>
          </footer>
        </main>
      </div>
      {showGraph ? (
        <div className="fixed inset-0 z-40 flex bg-white">
          <aside className="flex w-64 flex-col border-r border-gray-100 bg-mist px-4 py-6">
            <div className="text-sm font-semibold text-ink">调试面板</div>
            <label className="mt-4 text-xs font-medium text-gray-500">
              Query
            </label>
            <textarea
              className="mt-2 h-24 w-full resize-none rounded-lg border border-gray-200 bg-white p-2 text-xs text-ink outline-none"
              placeholder="输入 query 后点击检索"
              value={graphQuery}
              onChange={(event) => setGraphQuery(event.target.value)}
              disabled={!activeProjectId || graphSearchLoading}
            />
            <label className="mt-4 text-xs font-medium text-gray-500">
              Top K
            </label>
            <input
              type="number"
              min="1"
              max="50"
              value={graphTopK}
              onChange={(event) => {
                const value = Number(event.target.value)
                setGraphTopK(Number.isNaN(value) ? 1 : value)
              }}
              className="mt-2 w-full rounded-lg border border-gray-200 bg-white px-2 py-1 text-xs text-ink outline-none"
              disabled={!activeProjectId || graphSearchLoading}
            />
            <button
              type="button"
              onClick={handleGraphSearch}
              disabled={
                !activeProjectId || graphSearchLoading || !graphQuery.trim()
              }
              className="mt-3 rounded-lg bg-ether px-3 py-2 text-xs font-medium text-white disabled:opacity-50"
            >
              {graphSearchLoading ? "检索中..." : "检索"}
            </button>
            {graphSearchError ? (
              <div className="mt-2 text-xs text-red-500">{graphSearchError}</div>
            ) : null}
            <div className="mt-5 text-xs font-medium text-gray-500">
              agentsearch 返回
            </div>
            <div className="mt-2 flex-1 overflow-hidden">
              <div className="h-full w-full overflow-y-auto rounded-lg border border-gray-200 bg-white p-2 text-[11px] text-gray-500">
                {graphSearchResult
                  ? JSON.stringify(graphSearchResult, null, 2)
                  : "等待检索结果"}
              </div>
            </div>
          </aside>
          <div className="flex flex-1 flex-col">
            <div className="flex items-center justify-between border-b border-gray-100 px-6 py-4">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setShowGraph(false)}
                  className="flex h-8 w-8 items-center justify-center rounded-full bg-lavender text-white"
                  title="返回主页"
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M3 10.5 12 3l9 7.5" />
                    <path d="M5 9.5V21h14V9.5" />
                    <path d="M9 21v-6h6v6" />
                  </svg>
                </button>
                <div className="text-sm font-semibold text-ink">
                  {activeProject?.name || "Project"} 图谱
                </div>
              </div>
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 text-xs text-gray-500">
                  <input
                    type="checkbox"
                    checked={showAllGraph}
                    onChange={(event) => setShowAllGraph(event.target.checked)}
                    disabled={!graphSearchResult}
                    className="h-3 w-3 rounded border-gray-300"
                  />
                  显示全部图谱
                </label>
                <button
                  type="button"
                  onClick={handleGraphRefresh}
                  className="flex h-8 w-8 items-center justify-center rounded-full border border-gray-200 text-gray-600 hover:bg-gray-50"
                  title="刷新"
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M20 12a8 8 0 0 1-8 8 8 8 0 0 1-7.2-4.5" />
                    <path d="M4 12a8 8 0 0 1 8-8 8 8 0 0 1 7.2 4.5" />
                    <path d="M4 7v5h5" />
                    <path d="M20 17v-5h-5" />
                  </svg>
                </button>
              </div>
            </div>
            <div className="relative flex-1">
              {graphLoading ? (
                <div className="flex h-full items-center justify-center text-xs text-gray-400">
                  <span className="mr-2 inline-block h-3 w-3 animate-spin rounded-full border-2 border-gray-200 border-t-ether" />
                  图谱加载中
                </div>
              ) : graphError ? (
                <div className="flex h-full items-center justify-center text-xs text-red-500">
                  {graphError}
                </div>
              ) : (
                <div className="h-full w-full">
                  <div ref={graphContainerRef} className="h-full w-full" />
                  <div className="absolute right-4 top-4 w-72 max-h-[80vh] overflow-hidden rounded-lg border border-gray-200 bg-white/95 p-3 text-[11px] text-gray-600 shadow-sm">
                    <div className="text-xs font-semibold text-ink">图例与详情</div>
                    <div className="mt-2 max-h-[72vh] overflow-y-auto pr-1">
                      <div>
                        <div className="text-[11px] font-medium text-gray-500">
                          节点类型
                        </div>
                        <div className="mt-1 space-y-1 text-[10px] text-gray-500">
                          <div>圆点：Fact</div>
                          <div>方块：Topic</div>
                          <div>三角：Conversation</div>
                          <div>大圆：Bundle</div>
                        </div>
                      </div>
                      <div className="mt-3 text-[11px] text-gray-500">
                        节点大小：fact score
                      </div>
                      <div className="mt-1 text-[11px] text-gray-500">
                        边框粗细：bundle 内对话命中强度
                      </div>
                      <div className="mt-3">
                        <div className="text-[11px] font-medium text-gray-500">
                          Topic 颜色
                        </div>
                        <div className="mt-1 space-y-1">
                          {topicLegend && topicLegend.length ? (
                            topicLegend.map((item) => (
                              <div
                                key={item.id}
                                className="flex items-center gap-2"
                              >
                                <span
                                  className="inline-block h-3 w-3 rounded-full"
                                  style={{ backgroundColor: item.color }}
                                />
                                <span className="truncate">{item.title}</span>
                              </div>
                            ))
                          ) : (
                            <div className="text-gray-400">暂无</div>
                          )}
                        </div>
                      </div>
                      <div className="mt-3">
                        <div className="text-[11px] font-medium text-gray-500">
                          Bundle 列表
                        </div>
                        <div className="mt-1 space-y-1 text-[10px] text-gray-500">
                          {bundleSummary && bundleSummary.length ? (
                            bundleSummary.map((bundle) => (
                              <div
                                key={bundle.bundleId}
                                className="flex items-center justify-between rounded border border-gray-200 px-2 py-1"
                              >
                                <span>Bundle {bundle.bundleId}</span>
                                <span>
                                  F{bundle.factCount} T{bundle.topicCount} C
                                  {bundle.conversationCount}
                                </span>
                              </div>
                            ))
                          ) : (
                            <div className="text-gray-400">暂无</div>
                          )}
                        </div>
                      </div>
                      <div className="mt-3">
                        <div className="text-[11px] font-medium text-gray-500">
                          Recent Turns
                        </div>
                        <div className="mt-1 space-y-1">
                          {recentTurns && recentTurns.length ? (
                            recentTurns.map((turn) => (
                              <div
                                key={turn.conversation_id}
                                className="rounded border border-gray-200 px-2 py-1 text-[10px]"
                              >
                                <div className="truncate text-gray-500">
                                  {turn.text || ""}
                                </div>
                                <div className="text-[10px] text-gray-400">
                                  score {turn.score ?? "-"}
                                </div>
                              </div>
                            ))
                          ) : (
                            <div className="text-gray-400">暂无</div>
                          )}
                        </div>
                      </div>
                      <div className="mt-3">
                        <div className="text-[11px] font-medium text-gray-500">
                          节点详情
                        </div>
                        {selectedNodeDetail ? (
                          <div className="mt-1 rounded border border-gray-200 bg-white p-2">
                            <div className="text-[11px] font-medium text-ink">
                              {selectedNodeDetail.title}
                            </div>
                            {selectedNodeDetail.score !== null &&
                              selectedNodeDetail.score !== undefined ? (
                              <div className="mt-1 text-[10px] text-gray-400">
                                score {selectedNodeDetail.score}
                              </div>
                            ) : null}
                            <div className="mt-1 max-h-32 overflow-y-auto whitespace-pre-wrap text-[11px] text-gray-500">
                              {selectedNodeDetail.content || "暂无内容"}
                            </div>
                          </div>
                        ) : (
                          <div className="mt-1 text-gray-400">
                            点击节点查看详情
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}
      {deleteTarget ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
          <div className="w-[320px] rounded-2xl bg-white px-6 py-5 shadow-xl">
            <div className="text-sm font-semibold text-ink">删除项目</div>
            <div className="mt-2 text-xs text-gray-500">
              确认删除 “{deleteTarget.name}” 吗？此操作不可撤销。
            </div>
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setDeleteTarget(null)}
                className="rounded-md border border-gray-200 px-3 py-1 text-xs text-gray-600"
              >
                取消
              </button>
              <button
                type="button"
                onClick={confirmDeleteProject}
                className="rounded-md bg-red-500 px-3 py-1 text-xs font-medium text-white"
              >
                删除
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default App
