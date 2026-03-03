# PaperMem Copilot

A memory-augmented AI assistant with visual knowledge graph visualization, built for Georgia Tech CSE6242 Data and Visual Analytics.

## Overview

PaperMem Copilot is an intelligent conversation system that remembers and learns from your conversations. It combines a **GraphRAG-based memory system** with an **Electron desktop application** to provide context-aware responses with full transparency into its knowledge structure.

### Key Features

- **Dual-Layer Memory Architecture**: Short-term (immediate) + Long-term (semantic knowledge graphs)
- **Multi-Dimensional Search**: Search by time, topics, entities, and semantic relationships
- **Visual Knowledge Graph**: Interactive visualization of facts, topics, and their connections
- **Memory-Augmented Chat**: AI responses incorporate relevant past conversations
- **Graph Expansion**: Multi-hop retrieval along semantic relationships

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PaperMem Copilot App                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │  Electron   │───▶│   FastAPI   │───▶│  GauzRag Memory │  │
│  │  Frontend   │    │   Backend   │    │     System      │  │
│  │  (React)    │◀───│  (Python)   │◀───│   (Python)      │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌─────────┐    ┌─────────┐    ┌─────────┐
         │  Neo4j  │    │  MySQL  │    │ Qdrant  │
         │ (Graph) │    │  (SQL)  │    │ (Vector)│
         └─────────┘    └─────────┘    └─────────┘
```

## Project Structure

```
.
├── Mem_System1/              # GauzRag Memory System
│   ├── GauzRag/              # Core RAG implementation
│   │   ├── api.py            # FastAPI endpoints
│   │   ├── pipeline.py       # Memory processing pipeline
│   │   ├── entity_extractor.py
│   │   ├── fact_extractor.py
│   │   ├── leiden_community_detector.py
│   │   ├── vector_store.py   # Qdrant integration
│   │   ├── neo4j_storage.py  # Graph database
│   │   └── ...
│   ├── run_api.py            # API server entry point
│   └── requirements.txt
│
└── paper_app/                # Desktop Application
    ├── frontend/             # React + Vite + Tailwind
    │   └── src/App.jsx       # Main UI with graph visualization
    ├── backend/              # FastAPI backend
    │   └── app/
    │       ├── main.py       # API endpoints
    │       ├── memory_client.py  # GauzRag integration
    │       ├── reasoner_agent.py # LLM reasoning
    │       └── openrouter_client.py
    └── electron/             # Desktop shell
        └── main.js           # Window management
```

## Components

### Mem_System1 - GauzRag Memory System

A sophisticated RAG system implementing GraphRAG principles:

| Component | Description |
|-----------|-------------|
| **Fact Extractor** | Extracts atomic facts from conversations |
| **Entity Extractor** | Identifies entities and their types |
| **Relation Builder** | Discovers relationships between facts |
| **Community Detector** | Groups facts by topics using Leiden algorithm |
| **Vector Store** | Semantic embeddings via Qdrant |
| **Neo4j Storage** | Knowledge graph persistence |

**API Endpoints:**
- `POST /extract` - Process conversations into memory
- `POST /search` - Multi-dimensional search
- `POST /agenticSearch` - Natural language to structured queries
- `GET /search/time_dimension` - Temporal retrieval
- `GET /fact/{id}/relations` - Relationship queries

### paper_app - Desktop Application

An Electron app providing the user interface:

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | React + Vite + Tailwind | Chat UI, graph visualization |
| **Backend** | FastAPI | API gateway, session management |
| **Shell** | Electron | Cross-platform desktop wrapper |

**Features:**
- Real-time streaming chat responses
- Interactive knowledge graph visualization
- Project-based memory isolation
- Global hotkeys and clipboard integration
- Overlay mode for quick capture

## Data Flow

```
1. User sends message
       ↓
2. FastAPI receives at /chat/stream
       ↓
3. Query GauzRag /agenticSearch for relevant memories
       ↓
4. Construct context from retrieved facts & conversations
       ↓
5. Stream LLM response via OpenRouter
       ↓
6. Auto-extract conversation to GauzRag /extract
       ↓
7. Update knowledge graph visualization
```

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- Neo4j (for graph storage)
- MySQL (for structured data)
- Qdrant (for vector search)

### Mem_System1 Setup

```bash
cd Mem_System1
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database credentials

# Start the API server
python run_api.py
```

### Paper App Setup

```bash
cd paper_app

# Install dependencies
npm install

# Configure backend
cd backend
pip install -r requirements.txt
cp env.example .env
# Edit .env with GauzRag API URL

# Start the app
cd ..
npm start
```

## Configuration

### Environment Variables

**Mem_System1/.env:**
```env
OPENAI_API_KEY=your_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=password
QDRANT_URL=http://localhost:6333
```

**paper_app/backend/.env:**
```env
MEMORY_API_BASE=http://localhost:1235
OPENROUTER_API_KEY=your_key
```

## Technology Stack

| Category | Technologies |
|----------|--------------|
| **Frontend** | React, Vite, Tailwind CSS, D3.js |
| **Backend** | FastAPI, Python |
| **Desktop** | Electron |
| **Databases** | Neo4j, MySQL, Qdrant |
| **AI/ML** | OpenAI Embeddings, Leiden Algorithm |
| **LLM** | OpenRouter (Gemini) |

## Authors

Georgia Tech CSE6242 Team - Spring 2026

## License

MIT
