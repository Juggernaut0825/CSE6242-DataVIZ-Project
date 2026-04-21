from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from .config import EvalSettings
from .papers import PaperRecord, iter_pdf_paths, load_metadata


class PaperMemService:
    """Client for the local PaperMem FastAPI backend."""

    def __init__(
        self,
        settings: EvalSettings,
        papers_dir: Path,
        state_path: Path,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.papers_dir = papers_dir
        self.state_path = state_path
        self.project_id = project_id
        self.session_id = session_id
        self.records_by_id: dict[str, PaperRecord] = {}

    async def prepare(self, ingest: bool = True, reuse_state: bool = True) -> None:
        records = load_metadata(self.papers_dir)
        self.records_by_id = {record.paper_id: record for record in records}
        if reuse_state and self.state_path.exists() and not self.project_id:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.project_id = state.get("project_id")
            self.session_id = state.get("session_id")
        async with httpx.AsyncClient(base_url=self.settings.papermem_api_base, timeout=None) as client:
            health = await client.get("/health")
            health.raise_for_status()
            if not self.project_id:
                response = await client.post(
                    "/projects",
                    json={
                        "name": f"PaperMem LLM Judge Eval {int(time.time())}",
                        "type": "evaluation",
                        "description": "Auto-created project for eval_paper2 LLM-as-judge benchmark.",
                    },
                )
                response.raise_for_status()
                self.project_id = response.json()["id"]
            if not self.session_id:
                response = await client.post(
                    "/chat_sessions",
                    json={"project_id": self.project_id, "title": "LLM-as-judge evaluation"},
                )
                response.raise_for_status()
                self.session_id = response.json()["id"]
            if ingest:
                for pdf_path in iter_pdf_paths(records):
                    response = await client.post(
                        "/files/ingest",
                        json={"project_id": self.project_id, "file_path": str(pdf_path)},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    print(f"PaperMem ingested {payload.get('filename')} with {payload.get('unit_count')} units")
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(
                    {
                        "project_id": self.project_id,
                        "session_id": self.session_id,
                        "api_base": self.settings.papermem_api_base,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    async def answer(self, question: dict[str, Any]) -> dict[str, Any]:
        if not self.project_id:
            raise RuntimeError("PaperMem project is not prepared.")
        started = time.perf_counter()
        answer_parts: list[str] = []
        search_payload: dict[str, Any] = {}
        record = self.records_by_id.get(question.get("paper_id", ""))
        source_name = Path(record.pdf_path).name if record and self.settings.papermem_use_source_hint else None
        async with httpx.AsyncClient(base_url=self.settings.papermem_api_base, timeout=None) as client:
            async with client.stream(
                "POST",
                "/chat/stream",
                json={
                    "query": question["question"],
                    "project_id": self.project_id,
                    "session_id": self.session_id,
                    "top_k": self.settings.papermem_top_k,
                    "source_name": source_name,
                    "context_budget": self.settings.papermem_context_budget,
                    "evidence_chars_per_unit": self.settings.papermem_evidence_chars_per_unit,
                    "skip_memory_ingest": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[len("data: ") :].strip()
                    if not raw:
                        continue
                    event = json.loads(raw)
                    if event.get("type") == "content_chunk":
                        answer_parts.append(event.get("content", ""))
                    elif event.get("type") == "search":
                        search_payload = event.get("payload", {})
                    elif event.get("type") == "error":
                        answer_parts.append(f"[PaperMem error] {event.get('message', '')}")
        elapsed = time.perf_counter() - started
        retrieved_units = search_payload.get("retrieved_units", []) if search_payload else []
        return {
            "system": "papermem",
            "question_id": question["id"],
            "answer": "".join(answer_parts).strip(),
            "latency_seconds": elapsed,
            "usage": {},
            "retrieved_context_chars": min(
                self.settings.papermem_context_budget,
                sum(len(item.get("text", "")) for item in retrieved_units),
            ),
            "retrieved_chunks": [
                {
                    "unit_id": item.get("id"),
                    "source_type": item.get("source_type"),
                    "summary": item.get("summary"),
                    "score": item.get("score"),
                    "retrieval_source": item.get("retrieval_source"),
                    "metadata": item.get("metadata"),
                }
                for item in retrieved_units
            ],
        }
