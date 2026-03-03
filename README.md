# CSE6242 Data Visualization Project

Georgia Tech Spring 2026 - CSE6242 Data and Visual Analytics

## Project Structure

```
.
├── Mem_System1/     # Memory System with RAG (GauzRag)
├── paper_app/       # Electron Application (Backend + Frontend)
└── mem0.pdf         # Project Documentation
```

## Components

### Mem_System1
A memory system built with RAG (Retrieval-Augmented Generation) using GauzRag and Qdrant vector database.

### Paper App
An Electron-based desktop application with:
- Backend service
- Frontend interface
- Electron wrapper

## Requirements

- Python 3.x (for Mem_System1)
- Node.js (for paper_app)

## Getting Started

### Mem_System1
```bash
cd Mem_System1
pip install -r requirements.txt
python run_api.py
```

### Paper App
```bash
cd paper_app
npm install
npm start
```

## Authors

Georgia Tech CSE6242 Team

## License

MIT
