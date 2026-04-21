from __future__ import annotations

import asyncio
import logging
import math
import re
from time import perf_counter
import uuid
from typing import Any, Dict, List, Sequence

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.models import MemoryUnit, Project, RetrievalEvent, SourceFile, beijing_now
from app.services.embedding_service import EmbeddingService
from app.services.file_parser import parse_file_bytes, parse_local_file, sanitize_text
from app.services.graph_service import GraphService
from app.services.semantic_service import SemanticService

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MemoryService:
    def __init__(self, graph_service: GraphService) -> None:
        self.graph_service = graph_service
        self.embedding_service = EmbeddingService()
        self.semantic_service = SemanticService()

    async def ingest_capture(
        self,
        db: Session,
        project_id: str,
        text: str,
        source_type: str,
        source_name: str | None = None,
        session_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
        file_id: str | None = None,
    ) -> Dict[str, Any]:
        started_at = perf_counter()
        chunks = self._chunk_capture_text(text)
        if not chunks:
            return {"project_id": project_id, "source_type": source_type, "unit_count": 0, "units": []}
        chunked_at = perf_counter()
        embeddings = await self.embedding_service.embed_texts([item["text"] for item in chunks])
        embedded_at = perf_counter()
        labels_by_chunk = await self._extract_labels_for_chunks(
            [item["text"] for item in chunks],
            source_type=source_type,
            embeddings=embeddings,
        )
        labeled_at = perf_counter()
        created_units = await self._persist_units(
            db=db,
            project_id=project_id,
            source_type=source_type,
            source_name=source_name,
            chunks=chunks,
            embeddings=embeddings,
            labels_by_chunk=labels_by_chunk,
            metadata=metadata,
            session_id=session_id,
            file_id=file_id,
        )
        persisted_at = perf_counter()
        self._refresh_project_stats(db, project_id)
        finished_at = perf_counter()
        logger.info(
            "ingest_capture project=%s source_type=%s chunks=%s chunk=%.3fs embed=%.3fs labels=%.3fs persist=%.3fs refresh=%.3fs total=%.3fs",
            project_id,
            source_type,
            len(chunks),
            chunked_at - started_at,
            embedded_at - chunked_at,
            labeled_at - embedded_at,
            persisted_at - labeled_at,
            finished_at - persisted_at,
            finished_at - started_at,
        )
        return {
            "project_id": project_id,
            "source_type": source_type,
            "unit_count": len(created_units),
            "units": [self._serialize_unit(unit) for unit in created_units],
        }

    async def ingest_file(self, db: Session, project_id: str, file_path: str) -> Dict[str, Any]:
        started_at = perf_counter()
        parsed = parse_local_file(file_path)
        parsed_at = perf_counter()
        return await self._ingest_parsed_file(db, project_id, parsed, started_at, parsed_at)

    async def ingest_file_bytes(
        self, db: Session, project_id: str, filename: str, data: bytes
    ) -> Dict[str, Any]:
        started_at = perf_counter()
        parsed = parse_file_bytes(filename, data)
        parsed_at = perf_counter()
        return await self._ingest_parsed_file(db, project_id, parsed, started_at, parsed_at)

    async def _ingest_parsed_file(
        self,
        db: Session,
        project_id: str,
        parsed: Dict[str, Any],
        started_at: float,
        parsed_at: float,
    ) -> Dict[str, Any]:
        source_file = SourceFile(
            project_id=project_id,
            filename=parsed["filename"],
            file_path=parsed["file_path"],
            file_hash=parsed["file_hash"],
            media_type=parsed["media_type"],
            parser=parsed["parser"],
            page_count=parsed["page_count"],
            status="processed",
            metadata_json={"chunk_count": len(parsed["chunks"])},
        )
        db.add(source_file)
        db.flush()

        for chunk in parsed["chunks"]:
            chunk.setdefault("metadata", {})
            chunk["metadata"]["file_hash"] = parsed["file_hash"]
            chunk["metadata"]["filename"] = parsed["filename"]
        embeddings = await self.embedding_service.embed_texts([item["text"] for item in parsed["chunks"]])
        embedded_at = perf_counter()
        labels_by_chunk = await self._extract_labels_for_chunks(
            [item["text"] for item in parsed["chunks"]],
            source_type="file_chunk",
            embeddings=embeddings,
        )
        labeled_at = perf_counter()
        created_units = await self._persist_units(
            db=db,
            project_id=project_id,
            source_type="file_chunk",
            source_name=parsed["filename"],
            chunks=parsed["chunks"],
            embeddings=embeddings,
            labels_by_chunk=labels_by_chunk,
            metadata={"filename": parsed["filename"]},
            file_id=source_file.id,
        )
        persisted_at = perf_counter()
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.file_count = (project.file_count or 0) + 1
            project.last_active_at = beijing_now()
            db.commit()
        finished_at = perf_counter()
        logger.info(
            "ingest_file project=%s file=%s chunks=%s pages=%s parse=%.3fs embed=%.3fs labels=%.3fs persist=%.3fs finalize=%.3fs total=%.3fs",
            project_id,
            parsed["filename"],
            len(parsed["chunks"]),
            parsed["page_count"],
            parsed_at - started_at,
            embedded_at - parsed_at,
            labeled_at - embedded_at,
            persisted_at - labeled_at,
            finished_at - persisted_at,
            finished_at - started_at,
        )

        return {
            "project_id": project_id,
            "file_id": source_file.id,
            "filename": parsed["filename"],
            "unit_count": len(created_units),
            "units": [self._serialize_unit(unit) for unit in created_units],
        }

    async def search(
        self,
        db: Session,
        project_id: str,
        query: str,
        top_k: int,
        session_id: str | None = None,
        source_name: str | None = None,
    ) -> Dict[str, Any]:
        query_embedding = (await self.embedding_service.embed_texts([query]))[0]
        source_file_ids = self._source_file_ids(db, project_id, source_name)
        search_query = (
            db.query(
                MemoryUnit,
                MemoryUnit.embedding.cosine_distance(query_embedding).label("distance"),
            )
            .filter(MemoryUnit.project_id == project_id)
        )
        if source_file_ids:
            search_query = search_query.filter(MemoryUnit.file_id.in_(source_file_ids))
        rows = search_query.order_by("distance").limit(top_k).all()
        lexical_units = self._search_lexical_units(
            db=db,
            project_id=project_id,
            query=query,
            source_file_ids=source_file_ids,
            exclude_ids=[],
            top_k=max(0, settings.retrieval_lexical_top_k),
        )
        lexical_ids = {unit.id for unit, _ in lexical_units}
        retrieved_units = []
        seen_unit_ids = set()
        for unit, lexical_score in lexical_units:
            retrieved_units.append(
                {
                    **self._serialize_unit(unit),
                    "score": round(lexical_score, 4),
                    "retrieval_source": "lexical",
                }
            )
            seen_unit_ids.add(unit.id)
        for unit, distance in rows:
            if unit.id in seen_unit_ids:
                continue
            retrieved_units.append(
                {
                    **self._serialize_unit(unit),
                    "score": round(max(0.0, 1.0 - float(distance)), 4),
                    "retrieval_source": "vector",
                }
            )
            seen_unit_ids.add(unit.id)
        seed_units = [unit for unit, _ in lexical_units] + [unit for unit, _ in rows if unit.id not in lexical_ids]
        expanded_units = self._expand_retrieved_units(
            db=db,
            project_id=project_id,
            seed_units=seed_units,
            source_file_ids=source_file_ids,
            existing_ids=[item["id"] for item in retrieved_units],
        )
        for unit, retrieval_source in expanded_units:
            retrieved_units.append(
                {
                    **self._serialize_unit(unit),
                    "score": 0.0,
                    "retrieval_source": retrieval_source,
                }
            )
        trace_steps = [
            {
                "title": "Recall",
                "detail": (
                    f"Retrieved {len(rows)} memory units from pgvector and {len(lexical_units)} lexical exact-match units"
                    + (f" within source file '{source_name}'." if source_file_ids else ".")
                ),
            },
            {
                "title": "Expand",
                "detail": (
                    f"Expanded to {len(retrieved_units)} evidence units using graph-related units "
                    "and adjacent source chunks."
                ),
            },
            {
                "title": "Select",
                "detail": f"Selected {min(3, len(retrieved_units))} evidence anchors for answer generation.",
            },
        ]

        graph_overlay = self.graph_service.build_reasoning_overlay(project_id, query, retrieved_units)
        retrieval_event = RetrievalEvent(
            project_id=project_id,
            session_id=session_id,
            query_text=query,
            selected_unit_ids=[item["id"] for item in retrieved_units],
            metadata_json={"top_k": top_k},
        )
        db.add(retrieval_event)
        db.commit()

        return {
            "query": query,
            "retrieved_units": retrieved_units,
            "citations": [
                {
                    "unit_id": item["id"],
                    "source_type": item["source_type"],
                    "summary": item["summary"],
                    "metadata": item.get("metadata"),
                }
                for item in retrieved_units
            ],
            "semantic_nodes": graph_overlay["nodes"],
            "reasoning_edges": graph_overlay["edges"],
            "trace_steps": trace_steps,
            "overlay": graph_overlay,
            "focus_terms": self.semantic_service.summarize_focus_nodes(
                [item.get("semantic_labels", {}) for item in retrieved_units]
            ),
        }

    def _source_file_ids(self, db: Session, project_id: str, source_name: str | None) -> List[str]:
        normalized = (source_name or "").strip()
        if not normalized:
            return []
        rows = (
            db.query(SourceFile.id)
            .filter(SourceFile.project_id == project_id, SourceFile.filename == normalized)
            .all()
        )
        if not rows:
            rows = (
                db.query(SourceFile.id)
                .filter(SourceFile.project_id == project_id, SourceFile.filename.ilike(f"%{normalized}%"))
                .all()
            )
        return [row[0] for row in rows]

    def _expand_retrieved_units(
        self,
        db: Session,
        project_id: str,
        seed_units: Sequence[MemoryUnit],
        source_file_ids: Sequence[str],
        existing_ids: Sequence[str],
    ) -> List[tuple[MemoryUnit, str]]:
        seen = set(existing_ids)
        expanded: List[tuple[MemoryUnit, str]] = []
        radius = max(0, settings.retrieval_adjacent_chunk_radius)
        if radius:
            for unit in seed_units:
                chunk_index = (unit.metadata_json or {}).get("chunk_index")
                page = (unit.metadata_json or {}).get("page")
                if unit.file_id and isinstance(chunk_index, int):
                    neighbor_query = (
                        db.query(MemoryUnit)
                        .filter(MemoryUnit.project_id == project_id, MemoryUnit.file_id == unit.file_id)
                    )
                    if isinstance(page, int):
                        neighbor_query = neighbor_query.filter(
                            MemoryUnit.metadata_json["page"].as_integer().between(page - radius, page + radius)
                        )
                    neighbor_query = neighbor_query.filter(
                        MemoryUnit.metadata_json["chunk_index"].as_integer().between(chunk_index - radius, chunk_index + radius)
                    ).order_by(
                        MemoryUnit.metadata_json["page"].as_integer(),
                        MemoryUnit.metadata_json["chunk_index"].as_integer(),
                    )
                    neighbors = neighbor_query.all()
                    for neighbor in neighbors:
                        if neighbor.id in seen:
                            continue
                        seen.add(neighbor.id)
                        expanded.append((neighbor, "adjacent_chunk"))

        graph_limit = max(0, settings.retrieval_graph_unit_limit)
        graph_ids = self.graph_service.get_related_unit_ids(project_id, [unit.id for unit in seed_units], graph_limit)
        if graph_ids:
            graph_query = db.query(MemoryUnit).filter(MemoryUnit.project_id == project_id, MemoryUnit.id.in_(graph_ids))
            if source_file_ids:
                graph_query = graph_query.filter(MemoryUnit.file_id.in_(list(source_file_ids)))
            graph_units = graph_query.all()
            graph_lookup = {unit.id: unit for unit in graph_units}
            for unit_id in graph_ids:
                unit = graph_lookup.get(unit_id)
                if not unit or unit.id in seen:
                    continue
                seen.add(unit.id)
                expanded.append((unit, "graph"))
        return expanded

    def _search_lexical_units(
        self,
        db: Session,
        project_id: str,
        query: str,
        source_file_ids: Sequence[str],
        exclude_ids: Sequence[str],
        top_k: int,
    ) -> List[tuple[MemoryUnit, float]]:
        if top_k <= 0:
            return []
        terms = self._query_terms(query)
        if not terms:
            return []
        candidates_query = db.query(MemoryUnit).filter(MemoryUnit.project_id == project_id)
        if source_file_ids:
            candidates_query = candidates_query.filter(MemoryUnit.file_id.in_(list(source_file_ids)))
        if exclude_ids:
            candidates_query = candidates_query.filter(~MemoryUnit.id.in_(list(exclude_ids)))
        candidates = candidates_query.all()
        scored: List[tuple[float, MemoryUnit]] = []
        for unit in candidates:
            haystack = f"{unit.summary or ''} {unit.text or ''}".lower()
            score = 0.0
            for term in terms:
                if term in haystack:
                    score += 1.0
                    if term.isdigit() or any(char.isdigit() for char in term):
                        score += 0.5
            if score:
                score += min(len(set(terms)) / 20.0, 0.5)
                scored.append((score, unit))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(unit, score) for score, unit in scored[:top_k]]

    def _query_terms(self, query: str) -> List[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "what",
            "does",
            "how",
            "this",
            "that",
            "from",
            "paper",
            "according",
            "compare",
            "compared",
            "under",
            "same",
            "which",
            "when",
        }
        terms = []
        seen = set()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.%+-]*", query.lower()):
            if len(token) < 3 and not any(char.isdigit() for char in token):
                continue
            if token in stopwords or token in seen:
                continue
            seen.add(token)
            terms.append(token)
        return terms[:24]

    def _chunk_capture_text(self, text: str) -> List[Dict[str, Any]]:
        stripped = sanitize_text(text).strip()
        if not stripped:
            return []
        if len(stripped) <= 1200:
            return [{"text": stripped, "metadata": {}}]
        chunks = []
        step = 900
        overlap = 180
        start = 0
        index = 0
        while start < len(stripped):
            end = min(len(stripped), start + step)
            chunks.append({"text": stripped[start:end], "metadata": {"chunk_index": index}})
            if end >= len(stripped):
                break
            start = max(end - overlap, start + 1)
            index += 1
        return chunks

    def _build_source_id(self, source_type: str, source_name: str | None, index: int) -> str:
        prefix = source_name or source_type
        return f"{prefix}:{index}:{uuid.uuid4()}"

    async def _persist_units(
        self,
        db: Session,
        project_id: str,
        source_type: str,
        source_name: str | None,
        chunks: Sequence[Dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
        labels_by_chunk: Sequence[Dict[str, Any]],
        metadata: Dict[str, Any] | None = None,
        session_id: str | None = None,
        file_id: str | None = None,
    ) -> List[MemoryUnit]:
        started_at = perf_counter()
        created_units: List[MemoryUnit] = []
        for index, chunk in enumerate(chunks):
            labels = labels_by_chunk[index]
            source_id = (
                f"{file_id}:{chunk.get('metadata', {}).get('chunk_index', index)}"
                if file_id
                else self._build_source_id(source_type, source_name, index)
            )
            primary_claim = labels["claims"][0] if labels["claims"] else chunk["text"][:180]
            primary_label = labels.get("display_labels", {}).get("claims", {}).get(primary_claim, primary_claim)
            unit = MemoryUnit(
                project_id=project_id,
                source_type=source_type,
                source_id=source_id,
                session_id=session_id,
                file_id=file_id,
                text=chunk["text"],
                summary=primary_label,
                metadata_json={**(metadata or {}), **chunk.get("metadata", {})},
                semantic_labels=labels,
                embedding=list(embeddings[index]),
            )
            db.add(unit)
            created_units.append(unit)

        db.flush()
        logger.info(
            "persist_units project=%s source_type=%s flush_complete units=%s",
            project_id,
            source_type,
            len(created_units),
        )
        flushed_at = perf_counter()
        for unit_index, unit in enumerate(created_units, start=1):
            self.graph_service.upsert_memory_unit(project_id, self._unit_to_graph_payload(unit), unit.semantic_labels or {})
            logger.info(
                "persist_units project=%s source_type=%s graph_upserted=%s/%s unit_id=%s",
                project_id,
                source_type,
                unit_index,
                len(created_units),
                unit.id,
            )
        db.commit()
        logger.info(
            "persist_units project=%s source_type=%s relational_db_commit_complete",
            project_id,
            source_type,
        )
        persisted_at = perf_counter()

        for unit_index, unit in enumerate(created_units, start=1):
            relation_top_k = max(0, settings.relation_link_top_k)
            related = (
                self._search_related_units(
                    db,
                    project_id,
                    unit.embedding,
                    exclude_ids=[unit.id],
                    top_k=relation_top_k,
                )
                if relation_top_k
                else []
            )
            relation_hits = 0
            for neighbor in related:
                relation = self.semantic_service.classify_unit_relation(
                    unit.text,
                    unit.semantic_labels or {},
                    neighbor.text,
                    neighbor.semantic_labels or {},
                )
                if relation:
                    self.graph_service.link_units(project_id, unit.id, neighbor.id, relation)
                    relation_hits += 1
            logger.info(
                "persist_units project=%s source_type=%s relation_links=%s unit=%s/%s unit_id=%s",
                project_id,
                source_type,
                relation_hits,
                unit_index,
                len(created_units),
                unit.id,
            )
        finished_at = perf_counter()
        logger.info(
            "persist_units project=%s source_type=%s units=%s flush=%.3fs graph_write=%.3fs relation_link=%.3fs total=%.3fs",
            project_id,
            source_type,
            len(created_units),
            flushed_at - started_at,
            persisted_at - flushed_at,
            finished_at - persisted_at,
            finished_at - started_at,
        )
        return created_units

    async def _extract_labels_for_chunks(
        self,
        texts: Sequence[str],
        source_type: str,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> List[Dict[str, Any]]:
        started_at = perf_counter()
        semaphore = asyncio.Semaphore(max(1, settings.semantic_llm_concurrency))
        llm_indexes = self._llm_indexes_for_source(
            source_type,
            len(texts),
            texts=texts,
            embeddings=embeddings,
        )

        async def run(index: int, text: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.semantic_service.extract_bundle(text, allow_llm=index in llm_indexes)

        labels = await asyncio.gather(*(run(index, text) for index, text in enumerate(texts)))
        finished_at = perf_counter()
        logger.info(
            "extract_labels source_type=%s chunks=%s llm_chunks=%s total=%.3fs",
            source_type,
            len(texts),
            len(llm_indexes),
            finished_at - started_at,
        )
        return labels

    def _llm_indexes_for_source(
        self,
        source_type: str,
        count: int,
        texts: Sequence[str] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> set[int]:
        if count <= 0 or not settings.semantic_llm_enabled:
            return set()
        if source_type == "file_chunk":
            target = self._file_llm_target_count(count)
            if target <= 0:
                return set()
            if target >= count:
                return set(range(count))
            if (
                settings.semantic_llm_selection_strategy.lower() == "mmr"
                and texts is not None
                and embeddings is not None
                and len(embeddings) == count
            ):
                return self._select_mmr_indexes(texts, embeddings, target)
            return self._select_evenly_spaced_indexes(count, target)
        limit = max(0, settings.semantic_llm_conversation_sample_limit)
        return set(range(min(count, limit)))

    def _file_llm_target_count(self, count: int) -> int:
        ratio = max(0.0, settings.semantic_llm_file_sample_ratio)
        target = math.ceil(count * ratio)
        target = max(target, min(count, max(0, settings.semantic_llm_file_sample_min)))
        max_count = max(0, settings.semantic_llm_file_sample_max)
        if max_count:
            target = min(target, max_count)
        return min(count, target)

    def _select_evenly_spaced_indexes(self, count: int, target: int) -> set[int]:
        if target <= 0:
            return set()
        if target >= count:
            return set(range(count))
        if target == 1:
            return {0}
        step = (count - 1) / float(target - 1)
        return {round(index * step) for index in range(target)}

    def _select_mmr_indexes(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        target: int,
    ) -> set[int]:
        count = len(texts)
        selected = {0, count - 1}
        salience = [self._chunk_salience(text) for text in texts]
        while len(selected) < target:
            best_index = None
            best_score = float("-inf")
            for index in range(count):
                if index in selected:
                    continue
                max_similarity = max(
                    self._cosine_similarity(embeddings[index], embeddings[selected_index])
                    for selected_index in selected
                )
                position_spread = min(abs(index - selected_index) for selected_index in selected) / max(1, count - 1)
                score = 0.50 * salience[index] - 0.30 * max_similarity + 0.20 * position_spread
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is None:
                break
            selected.add(best_index)
        return selected

    def _chunk_salience(self, text: str) -> float:
        snippet = (text or "")[:1600]
        length_score = min(len(snippet) / 900.0, 1.0)
        number_score = min(len(re.findall(r"\b\d+(?:\.\d+)?%?\b", snippet)) / 8.0, 1.0)
        acronym_score = min(len(re.findall(r"\b[A-Z]{2,}\b", snippet)) / 8.0, 1.0)
        entity_score = min(len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", snippet)) / 8.0, 1.0)
        heading_score = 1.0 if re.match(r"^\s*(?:[#*\-]+\s*)?[A-Z][A-Za-z0-9 ,:/()_-]{3,90}(?:\n|:)", text or "") else 0.0
        signal_score = 1.0 if re.search(
            r"\b(summary|conclusion|result|finding|method|approach|limitation|risk|decision|requirement|definition|experiment|evaluation)\b",
            snippet,
            flags=re.IGNORECASE,
        ) else 0.0
        return (
            0.30 * length_score
            + 0.18 * number_score
            + 0.14 * acronym_score
            + 0.14 * entity_score
            + 0.12 * heading_score
            + 0.12 * signal_score
        )

    def _cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        size = min(len(left), len(right))
        if size == 0:
            return 0.0
        dot = sum(float(left[index]) * float(right[index]) for index in range(size))
        left_norm = math.sqrt(sum(float(value) * float(value) for value in left[:size]))
        right_norm = math.sqrt(sum(float(value) * float(value) for value in right[:size]))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _search_related_units(
        self,
        db: Session,
        project_id: str,
        embedding: Sequence[float],
        exclude_ids: Sequence[str],
        top_k: int,
    ) -> List[MemoryUnit]:
        query = (
            db.query(MemoryUnit)
            .filter(MemoryUnit.project_id == project_id)
            .filter(~MemoryUnit.id.in_(list(exclude_ids)))
            .order_by(MemoryUnit.embedding.cosine_distance(list(embedding)))
            .limit(top_k)
        )
        return list(query.all())

    def _serialize_unit(self, unit: MemoryUnit) -> Dict[str, Any]:
        return {
            "id": unit.id,
            "project_id": unit.project_id,
            "source_type": unit.source_type,
            "source_id": unit.source_id,
            "text": unit.text,
            "summary": unit.summary,
            "metadata": unit.metadata_json or {},
            "semantic_labels": unit.semantic_labels or {},
            "created_at": unit.created_at.isoformat() if unit.created_at else None,
        }

    def _unit_to_graph_payload(self, unit: MemoryUnit) -> Dict[str, Any]:
        return {
            "id": unit.id,
            "source_type": unit.source_type,
            "source_id": unit.source_id,
            "text": unit.text,
            "metadata_json": unit.metadata_json or {},
            "created_at": unit.created_at or beijing_now(),
        }

    def _refresh_project_stats(self, db: Session, project_id: str) -> None:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return
        message_units = (
            db.query(func.count(MemoryUnit.id))
            .filter(MemoryUnit.project_id == project_id, MemoryUnit.source_type != "file_chunk")
            .scalar()
            or 0
        )
        project.message_count = int(message_units)
        project.last_active_at = beijing_now()
        db.commit()
