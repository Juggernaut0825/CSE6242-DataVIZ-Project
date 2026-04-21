from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import EvalSettings
from .llm_client import EvalLLMClient
from .papers import PaperRecord, load_metadata, load_paper_text


@dataclass
class RagChunk:
    chunk_id: str
    paper_id: str
    title: str
    text: str
    embedding: list[float]


class RagAgentService:
    """Small vector RAG baseline powered by GPT-4o mini."""

    def __init__(self, settings: EvalSettings, papers_dir: Path, index_path: Path) -> None:
        self.settings = settings
        self.papers_dir = papers_dir
        self.index_path = index_path
        self.client = EvalLLMClient(settings)
        self.chunks: list[RagChunk] = []

    async def prepare(self, force_rebuild: bool = False) -> None:
        if self.index_path.exists() and not force_rebuild:
            print(f"Loading RAG index from {self.index_path}", flush=True)
            self.chunks = [
                RagChunk(**item)
                for item in json.loads(self.index_path.read_text(encoding="utf-8"))
            ]
            print(f"Loaded {len(self.chunks)} RAG chunks", flush=True)
            return
        records = load_metadata(self.papers_dir)
        chunk_payloads = self._build_chunk_payloads(records)
        print(f"Embedding {len(chunk_payloads)} RAG chunks", flush=True)
        vectors = await self.client.embed_texts([item["text"] for item in chunk_payloads])
        self.chunks = [
            RagChunk(
                chunk_id=item["chunk_id"],
                paper_id=item["paper_id"],
                title=item["title"],
                text=item["text"],
                embedding=vectors[index],
            )
            for index, item in enumerate(chunk_payloads)
        ]
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote RAG index with {len(self.chunks)} chunks to {self.index_path}", flush=True)

    async def answer(self, question: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        query_vector = (await self.client.embed_texts([question["question"]]))[0]
        hits = self._top_chunks(query_vector, self.settings.rag_top_k)
        context = self._format_context(hits)
        prompt = f"""Answer the question using only the retrieved paper excerpts.
If the excerpts are insufficient, say what is missing instead of guessing.
Cite paper ids in brackets when useful.

Question:
{question["question"]}

Retrieved excerpts:
{context}
"""
        answer, usage = await self.client.chat_text(
            model=self.settings.rag_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful RAG paper QA agent. Stay grounded in the retrieved excerpts.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_completion_tokens=900,
        )
        elapsed = time.perf_counter() - started
        return {
            "system": "rag_gpt4o_mini",
            "question_id": question["id"],
            "answer": answer,
            "latency_seconds": elapsed,
            "usage": usage,
            "retrieved_context_chars": len(context),
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "title": chunk.title,
                    "score": round(score, 4),
                }
                for chunk, score in hits
            ],
        }

    def _build_chunk_payloads(self, records: list[PaperRecord]) -> list[dict[str, str]]:
        payloads = []
        for record in records:
            text = load_paper_text(record)
            chunks = self._chunk_text(text)
            for index, chunk in enumerate(chunks):
                payloads.append(
                    {
                        "chunk_id": f"{record.paper_id}::chunk_{index:04d}",
                        "paper_id": record.paper_id,
                        "title": record.title,
                        "text": chunk,
                    }
                )
        return payloads

    def _chunk_text(self, text: str) -> list[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []
        chunks = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + self.settings.rag_chunk_chars)
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = max(end - self.settings.rag_chunk_overlap, start + 1)
        return chunks

    def _top_chunks(self, query_vector: list[float], top_k: int) -> list[tuple[RagChunk, float]]:
        scored = [(chunk, cosine_similarity(query_vector, chunk.embedding)) for chunk in self.chunks]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _format_context(self, hits: list[tuple[RagChunk, float]]) -> str:
        parts = []
        budget = self.settings.rag_max_context_chars
        used = 0
        for chunk, score in hits:
            header = f"[{chunk.paper_id} | {chunk.chunk_id} | score={score:.3f} | {chunk.title}]\n"
            body = chunk.text.strip()
            available = budget - used - len(header) - 2
            if available <= 0:
                break
            if len(body) > available:
                body = body[:available]
            parts.append(header + body)
            used += len(header) + len(body) + 2
        return "\n\n".join(parts)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
