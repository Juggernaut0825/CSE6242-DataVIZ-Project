# PaperMem

PaperMem is a local-first memory system and explainable dashboard for AI agents and chatbots, built for Georgia Tech CSE6242 Data and Visual Analytics. It combines an Electron desktop shell, a React graph-and-chat interface, a FastAPI memory backend, `Postgres + pgvector` for retrieval, and `Neo4j` for logical relation graphs.

The current implementation is optimized for two memory sources:

- conversation history generated inside the app
- uploaded documents such as PDF, Markdown, and plain text files

The system is designed to make retrieval visible. Every chat response is backed by retrieved memory units, graph expansion, and evidence cards that the user can inspect directly in the UI.

## What The System Does

- stores chat turns and uploaded file chunks as unified memory units
- embeds all memory units into `pgvector` for semantic retrieval
- extracts lightweight semantic structure with an LLM-first pipeline plus local fallback
- builds a logical knowledge graph in Neo4j using claims, concepts, entities, and relations
- streams answers from the backend to the chat UI
- renders a large interactive memory graph with semantic zoom and retrieval-driven overlays
- supports a desktop quick-capture workflow through Electron

## Current Architecture

```text
User
  |
  v
Electron desktop shell
  |- main window: PaperMem dashboard
  |- floating upload capsule
  |- desktop integrations / IPC bridge
  |
  v
React frontend (Vite + Tailwind + vis-network)
  |- project sidebar
  |- chat sessions and streaming messages
  |- evidence / focus display
  |- semantic zoom graph
  |
  v
FastAPI backend
  |- project + chat session APIs
  |- file ingestion APIs
  |- chat streaming API
  |- graph retrieval APIs
  |
  +--> Postgres + pgvector
  |     |- projects
  |     |- chat sessions / messages
  |     |- source files
  |     |- memory units + embeddings
  |     |- retrieval / reasoning records
  |
  +--> Neo4j
        |- semantic nodes
        |- logical relations
        |- overlay graph for retrieval visualization
```

## End-To-End Flow

### 1. File Ingestion

1. A user uploads a file from the Electron capsule or desktop UI.
2. The backend parses the file locally.
3. The file is chunked into memory-sized segments.
4. Each chunk is embedded and stored in `Postgres`.
5. The semantic extraction service produces claims, concepts, entities, and display labels.
6. The graph service writes semantic nodes and logical edges into `Neo4j`.
7. The Electron shell sends an event back to the renderer so the graph can refresh automatically.

### 2. Conversation Ingestion

1. A user sends a chat message in the dashboard.
2. The backend embeds the query and retrieves nearby memory units from `pgvector`.
3. The backend expands the retrieved set using logical graph relations in `Neo4j`.
4. The backend streams an answer from the configured LLM.
5. The user turn and assistant turn are persisted as memory units.
6. The conversation is folded back into the same memory + graph pipeline used for files.

### 3. Visualization

1. The frontend requests a graph snapshot for the active project.
2. The backend computes graph scores and returns graph nodes and edges.
3. The frontend applies client-side semantic zoom.
4. Important nodes remain visible first, while lower-priority nodes appear as the user zooms in.
5. Retrieval-related nodes and evidence appear in the chat area for the corresponding assistant response.

## Frontend

The frontend lives in `frontend` and is a single-page React application served by Vite.

### Main Responsibilities

- render the split layout: graph area + chat area
- manage project selection and chat sessions
- stream assistant responses in real time
- display evidence and retrieval focus for each assistant message
- render the graph with `vis-network`
- apply semantic zoom behavior client-side
- refresh the graph after uploads or manual user action

### Important Files

- `frontend/src/App.jsx`
  - main application state
  - chat message rendering
  - evidence / focus UI
  - project and session management
  - graph refresh and semantic zoom logic
- `frontend/src/index.css`
  - global layout rules
  - Tailwind entrypoint
  - shared utility styles such as hidden scrollbars
- `frontend/src/main.jsx`
  - app bootstrapping and error boundary mounting

### Frontend Graph Model

The graph is not rendered as a raw dump of every memory unit. Instead, the frontend applies a layered view:

- high-importance nodes stay visible at low zoom
- bridge / gatekeeper nodes are preserved to keep communities connected
- node size reflects local importance rather than all nodes shrinking uniformly
- the graph can be refreshed after new ingestion or further conversation

The UI also distinguishes between:

- the global project memory map
- retrieval-driven overlays relevant to the current conversation

## Electron Shell

The Electron shell lives in `electron`.

### Responsibilities

- launch the desktop window that hosts the PaperMem dashboard
- expose a preload bridge so the renderer can call trusted native features
- manage the floating upload capsule
- notify the frontend when file ingestion completes
- keep the desktop experience available outside a plain browser workflow

### Important Files

- `electron/main.js`
  - window lifecycle
  - IPC handlers
  - upload completion notifications
- `electron/preload.js`
  - safe bridge exposed on `window.paperMem`
- `electron/dropzone.html`
  - floating upload capsule UI

## Backend

The backend lives in `backend` and is a FastAPI service that owns memory ingestion, retrieval, graph updates, and chat streaming.

### Main Responsibilities

- manage projects, sessions, messages, and files
- parse uploaded files locally
- chunk and embed text
- persist memory units into `Postgres`
- build and query logical graphs in `Neo4j`
- retrieve evidence for chat queries
- stream assistant responses and persist retrieval metadata

### Important Files

- `backend/app/main.py`
  - FastAPI application
  - API routes for projects, sessions, files, graphs, and chat
- `backend/app/config.py`
  - central settings model
  - all required environment configuration
- `backend/app/database.py`
  - SQLAlchemy engine/session setup
  - `pgvector` extension initialization
- `backend/app/models.py`
  - SQLAlchemy models for messages, memory units, files, and retrieval events
- `backend/app/schemas.py`
  - request / response schema definitions
- `backend/app/openrouter_client.py`
  - LLM client wrapper
- `backend/app/reasoner_agent.py`
  - answer generation layer

### Backend Services

- `backend/app/services/file_parser.py`
  - parses PDF / Markdown / TXT locally
  - sanitizes text for storage
- `backend/app/services/embedding_service.py`
  - generates embeddings
  - normalizes vector dimensions to the configured size
- `backend/app/services/semantic_service.py`
  - extracts claims, concepts, entities, and display labels
  - uses an LLM-first approach with local fallback
- `backend/app/services/graph_service.py`
  - upserts semantic nodes and logical relations in Neo4j
  - builds graph payloads for the frontend
- `backend/app/services/memory_service.py`
  - orchestrates ingestion, persistence, retrieval, evidence creation, and trace metadata

## Storage Layer

### Postgres + pgvector

`Postgres` is the system of record for structured application data and semantic retrieval.

It stores:

- projects
- chat sessions
- chat messages
- source file records
- memory units
- embeddings
- retrieval events
- reasoning trace metadata

The key design choice is to unify file chunks and conversation turns under the same `MemoryUnit` abstraction, so the same retrieval and graph-building pipeline can operate on both.

### Neo4j

`Neo4j` stores the semantic graph used for explainable reasoning and large-scale graph visualization.

The graph contains:

- semantic nodes such as claims, concepts, and entities
- links from memory units to semantic nodes
- logical relations such as support, causality, contradiction, elaboration, and association
- retrieval overlays used to visualize the current reasoning context

This graph is intentionally more logical than provenance-heavy so that the UI feels closer to a mind map than a file tree.

## Memory Pipeline

The current memory pipeline is intentionally lighter than the original GauzRag pipeline.

### Ingestion Pipeline

1. parse or receive text
2. sanitize text
3. chunk text
4. embed chunks
5. select a deterministic subset of chunks for LLM semantic labeling
6. extract semantic bundles with either LLM labels or local fallback labels
7. store memory units in `Postgres`
8. write semantic nodes and relations to `Neo4j`

### Retrieval Pipeline

1. embed query
2. semantic nearest-neighbor search in `pgvector`
3. expand through graph relations
4. collect evidence anchors
5. stream answer generation
6. persist retrieval metadata and reasoning records

This keeps the system generalizable and fast enough for interactive use, while still surfacing semantic structure for visualization.

## Semantic Extraction

The semantic extraction stage produces:

- `claims`
- `concepts`
- `entities`
- relation candidates
- `display_label` values for graph-friendly rendering

PaperMem uses a hybrid semantic-labeling strategy so ingestion remains practical for long papers, reports, notes, contracts, and deployed environments.

Every uploaded file chunk becomes a `MemoryUnit`, receives an embedding, is stored in `Postgres`, and is written into the `Neo4j` graph. The system does **not** drop unsampled chunks from graph retrieval. Instead, chunks differ only in how their semantic labels are produced:

- roughly one third of file chunks receive higher-quality LLM semantic labels
- the remaining chunks receive fast local fallback labels
- all chunks still participate in vector retrieval, graph rendering, graph expansion, and evidence selection

The LLM-labeled subset is selected with deterministic MMR-style sampling rather than pure randomness or a paper-specific section heuristic. PaperMem first keeps the first and last chunk, then scores candidate chunks using local salience signals such as length, numeric values, acronyms, named entities, heading-like text, and words such as summary, result, method, limitation, risk, requirement, decision, experiment, and evaluation. It then uses chunk embeddings to prefer candidates that cover different topics from chunks already selected. A small position-spread term prevents the selected chunks from collapsing into one nearby part of the document when embeddings are similar.

This is useful because user uploads are not always papers. A position-only strategy assumes structures like intro, method, results, and conclusion, which can fail for meeting notes, product specs, legal documents, or mixed research material. The MMR strategy instead tries to spend LLM calls on chunks that are both information-dense and semantically diverse, while local fallback keeps the rest of the graph complete.

The local fallback path is rule-based and runs without remote LLM calls. It splits text into candidate claims, extracts entities from uppercase/title-like spans, extracts concepts from frequent non-stopword tokens, looks for relation cues such as because, therefore, however, and for example, and creates compact display labels. These labels are rougher than LLM labels but are fast enough to apply to every chunk.

The main controls are:

- `SEMANTIC_LLM_FILE_SAMPLE_RATIO`, default `0.34`, controls the target fraction of file chunks that receive LLM labels
- `SEMANTIC_LLM_FILE_SAMPLE_MIN`, default `8`, keeps small files from being under-labeled
- `SEMANTIC_LLM_FILE_SAMPLE_MAX`, default `0`, means no maximum cap
- `SEMANTIC_LLM_SELECTION_STRATEGY=mmr` enables salience plus embedding-diversity selection
- `SEMANTIC_LLM_CONCURRENCY` and `SEMANTIC_LLM_TIMEOUT_SECONDS` bound ingestion latency
- `RELATION_LINK_TOP_K` bounds cross-chunk relation linking work

## Graph Visualization Strategy

PaperMem is built to visualize more data than a naive full-graph rendering can handle.

### Semantic Zoom

The graph uses semantic zoom rather than simple geometric scaling:

- zoomed-out views show a selective subgraph
- node retention is driven by graph importance signals
- more nodes appear as the user zooms in
- bridge nodes are preserved so communities do not visually disconnect too early

### Importance Heuristics

The frontend and backend rely on graph-theoretic signals including:

- centrality-style scores
- betweenness-like bridge behavior
- local connectivity / degree
- retrieval relevance

This is inspired by the CSE6242 graph algorithms material and is used to keep the large graph readable.

## API Surface

The most important backend capabilities exposed to the app are:

- project creation / listing / deletion
- chat session creation / activation / deletion
- message retrieval for a session
- file ingestion
- graph retrieval for a project
- streaming chat responses

The chat stream returns not only text tokens, but also retrieval metadata that the frontend can attach to the assistant reply, such as:

- evidence items
- retrieval focus terms
- graph overlay data

## Local Development

### Prerequisites

- **Python 3.11+** (3.13 recommended; matches production). On macOS, use **Homebrew** Python—do **not** create the venv with Xcode’s bundled Python or `python3` may not land on your `PATH` after `activate`.
- Node.js 18+
- Docker Desktop

### 1. Start Databases

```bash
docker compose up -d
```

This starts:

- `Postgres` on `127.0.0.1:5432`
- `Neo4j` on `127.0.0.1:7474` and `127.0.0.1:7687`

### 2. Configure Backend

Create the virtualenv with a **known** interpreter (adjust the path if your Homebrew prefix differs):

```bash
cd backend
/opt/homebrew/bin/python3.13 -m venv .venv_local
source .venv_local/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp env.example .env
```

After `source .venv_local/bin/activate`, use **`python`** and **`pip`** (not bare `python3` from Xcode) so `which python` resolves to `.venv_local/bin/python`.

If you ever see `python: not found` after activating, open a new terminal and run `source .venv_local/bin/activate` again, or invoke the venv explicitly: `./.venv_local/bin/python -m uvicorn ...`.

Fill in the required keys in `backend/.env`:

```env
LLM_API_KEY=
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini

EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=256

NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=papermemneo4j

POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DATABASE=papermem
```

### 3. Start Backend

```bash
cd backend
source .venv_local/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Install Desktop / Frontend Dependencies

```bash
npm install
```

### 5. Start Electron + Frontend

```bash
npm run dev
```

This script does both:

- starts the Vite renderer on `127.0.0.1:5173`
- waits for the renderer, then launches Electron

## Repository Layout

```text
.
├── backend/
├── electron/
├── frontend/
├── package.json
├── package-lock.json
├── README.md
├── docker-compose.yml
└── ...
```

## Desktop installers (DMG / EXE)

You do **not** need a separate public website for the Electron app: `npm run build:renderer` produces `frontend/dist`, which `electron-builder` packs into the desktop installer. Point the UI at your deployed API with the same URL for both the renderer and the main process:

```bash
export VITE_API_BASE_URL="https://YOUR-APP.up.railway.app"
npm run dist:mac    # macOS → release/*.dmg
npm run dist:win    # Windows → release/*.exe (run on Windows or use CI)
```

`scripts/write-electron-api-base.js` writes `electron/api-base.json` from that URL so Electron IPC calls (`/extract`, file ingest) hit the same host as the React app.

To publish builds on GitHub: push a version tag (`v1.0.0`). The workflow `.github/workflows/release-desktop.yml` builds macOS and Windows artifacts and attaches them to a Release. Set variable **`PAPERMEM_API_BASE_URL`** on the **`production`** environment (repository **Settings → Environments → production**), or the workflow falls back to a default Railway URL.

**Note:** Chat and graph work against a remote API. The Electron dropzone uploads file bytes with **`POST /files/ingest_upload`** (multipart), so ingestion works against a **deployed** backend. The JSON-only **`POST /files/ingest`** (local path on the API machine) remains useful for server-side debugging or when the API runs on the same machine as the files.

## Git Notes

The repository intentionally ignores local/demo-heavy assets such as:

- generated virtual environments
- local `.env` files
- untracked demo directories like `papermem-demo/`
- large local demo media such as `demo.gif`
- local PDF test assets unless already tracked in git history

## Team

Georgia Tech CSE6242 Team, Spring 2026
