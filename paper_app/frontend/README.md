# PaperMem Copilot Frontend

本目录为 Electron 渲染进程 UI（React + Vite + Tailwind）。下述为系统架构与数据流概览。

## 架构概览

分层结构：

- UI 层（Renderer）：React 单列对话 + 左侧 Project 列表
- 桌面壳（Main）：Electron 主进程，负责窗口与系统能力
- 本地服务（Backend）：FastAPI，提供数据持久化与 LLM 接口
- 外部能力：Memory API、Gemini(OpenRouter)

```
Renderer (React)
   |
   | HTTP(S) / SSE
   v
FastAPI (localhost:8000)
   |            \
   |             \--> OpenRouter (Gemini)
   |
   \--> Memory API (extract / agenticSearch)

Electron Main
   |
   | 启动/管理 FastAPI 子进程
   \--> 未来: 全局划词与 Overlay
```

## 关键模块

- Electron 主进程：`/Users/wangzijian/Desktop/gauz/paper_app/electron/main.js`
  - 启动渲染进程与 FastAPI 子进程
- 渲染进程入口：`/Users/wangzijian/Desktop/gauz/paper_app/frontend/src/main.jsx`
  - React 应用启动与路由占位
- FastAPI 入口：`/Users/wangzijian/Desktop/gauz/paper_app/backend/app/main.py`
  - `/extract`、`/agenticSearch` 代理
  - `/chat/stream` SSE 流式输出
- ORM 模型：`/Users/wangzijian/Desktop/gauz/paper_app/backend/app/models.py`
  - `projects`、`chat_sessions` 表结构

## 数据流说明

1. 用户在 UI 提问
2. Renderer 调用 `POST /chat/stream`
3. FastAPI 先走 Memory API `/agenticSearch` 做检索
4. 拼接上下文后请求 OpenRouter Gemini 流式输出
5. SSE 返回给前端渲染（reasoning 可折叠，content 为最终回复）

## 图谱调试视图

- 全屏图谱视图，左侧调试面板支持输入 query 与 Top K
- 支持同时显示全量图谱与检索结果（可切换“显示全部图谱”）
- 检索结果按 bundle 组织，并以节点形状/颜色/大小区分 fact、topic、conversation
- 右侧面板展示图例、recent turns、bundle 汇总及节点详情

## 未来模块（规划）

- 全局划词：
  - Main 使用 `uiohook-napi` 监听鼠标
  - 使用剪贴板读取选区文本
  - Overlay 悬浮窗提供 `Mem` 快速入库
