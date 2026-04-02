from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

import networkx as nx
from neo4j import GraphDatabase

from app.config import settings


RELATION_TYPES = {
    "SUPPORTS",
    "CONTRADICTS",
    "CAUSES",
    "TEMPORAL_BEFORE",
    "ELABORATES",
    "ASSOCIATED_WITH",
    "SAME_TOPIC",
}


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "node"


def compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def truncate_words(value: str, max_words: int, max_chars: int) -> str:
    cleaned = compact_whitespace(value)
    if not cleaned:
        return ""
    words = cleaned.split()
    clipped = " ".join(words[:max_words])
    if len(clipped) > max_chars:
        clipped = clipped[: max_chars - 3].rstrip() + "..."
    elif len(words) > max_words or len(cleaned) > len(clipped):
        clipped = clipped.rstrip(" ,.;:") + "..."
    return clipped


def display_label_for_kind(kind: str, text: str) -> str:
    cleaned = compact_whitespace(text)
    if not cleaned:
        return ""
    if kind == "claim":
        return truncate_words(cleaned, max_words=6, max_chars=36)
    if kind == "concept":
        return truncate_words(cleaned, max_words=3, max_chars=24)
    if kind == "entity":
        return truncate_words(cleaned, max_words=3, max_chars=24)
    if kind in {"file_chunk", "conversation_turn", "quick_capture", "memory_unit"}:
        return truncate_words(cleaned, max_words=6, max_chars=36)
    if kind == "query":
        return truncate_words(cleaned, max_words=6, max_chars=36)
    return truncate_words(cleaned, max_words=6, max_chars=36)


class GraphService:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT memory_unit_id_unique IF NOT EXISTS FOR (u:MemoryUnit) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT semantic_node_id_unique IF NOT EXISTS FOR (n:SemanticNode) REQUIRE n.id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for statement in statements:
                session.run(statement)

    def close(self) -> None:
        self.driver.close()

    def delete_project_graph(self, project_id: str) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MATCH (n {project_id: $project_id})
                DETACH DELETE n
                """,
                project_id=project_id,
            )

    def delete_session_graph(self, project_id: str, session_id: str) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MATCH (u:MemoryUnit {project_id: $project_id, session_id: $session_id})
                DETACH DELETE u
                """,
                project_id=project_id,
                session_id=session_id,
            )
            session.run(
                """
                MATCH (s:SemanticNode {project_id: $project_id})
                WHERE NOT EXISTS {
                    MATCH (:MemoryUnit {project_id: $project_id})-[:DERIVED_FROM]->(s)
                }
                DETACH DELETE s
                """,
                project_id=project_id,
            )
            session.run(
                """
                MATCH (p:Project {id: $project_id})
                WHERE NOT EXISTS {
                    MATCH (p)-[:HAS_UNIT]->(:MemoryUnit {project_id: $project_id})
                }
                DETACH DELETE p
                """,
                project_id=project_id,
            )

    def upsert_memory_unit(
        self,
        project_id: str,
        unit: Dict[str, Any],
        labels: Dict[str, Any],
    ) -> None:
        display_labels = labels.get("display_labels", {})
        concept_labels = display_labels.get("concepts", {})
        entity_labels = display_labels.get("entities", {})
        claim_labels = display_labels.get("claims", {})
        with self.driver.session() as session:
            session.run(
                """
                MERGE (p:Project {id: $project_id})
                MERGE (u:MemoryUnit {id: $unit_id})
                SET u.project_id = $project_id,
                    u.source_type = $source_type,
                    u.source_id = $source_id,
                    u.text = $text,
                    u.created_at = $created_at,
                    u.metadata = $metadata
                MERGE (p)-[:HAS_UNIT]->(u)
                """,
                project_id=project_id,
                unit_id=unit["id"],
                source_type=unit["source_type"],
                source_id=unit["source_id"],
                text=unit["text"],
                created_at=unit["created_at"].isoformat(),
                metadata=json.dumps(unit.get("metadata_json") or {}, ensure_ascii=False),
            )

            for concept in labels.get("concepts", []):
                self._merge_semantic_node(
                    session,
                    project_id,
                    "concept",
                    concept,
                    unit["id"],
                    display_label=concept_labels.get(concept),
                )
            for entity in labels.get("entities", []):
                self._merge_semantic_node(
                    session,
                    project_id,
                    "entity",
                    entity,
                    unit["id"],
                    display_label=entity_labels.get(entity),
                )
            claim_ids: List[str] = []
            claim_id_lookup: Dict[str, str] = {}
            for claim in labels.get("claims", []):
                claim_id = self._merge_semantic_node(
                    session,
                    project_id,
                    "claim",
                    claim,
                    unit["id"],
                    display_label=claim_labels.get(claim),
                )
                claim_ids.append(claim_id)
                claim_id_lookup[claim] = claim_id

            for relation in labels.get("relations", []):
                source_claim = relation.get("source_claim")
                target_claim = relation.get("target_claim")
                if not source_claim or not target_claim:
                    continue
                source_id = self._semantic_node_id(project_id, "claim", source_claim)
                target_id = self._semantic_node_id(project_id, "claim", target_claim)
                self._merge_relation(session, source_id, target_id, relation.get("type", "ELABORATES"), 1.0)

            claim_details = labels.get("claim_details", [])
            if claim_details:
                for claim_detail in claim_details:
                    claim_text = claim_detail.get("text")
                    claim_id = claim_id_lookup.get(claim_text)
                    if not claim_id:
                        continue
                    for concept in claim_detail.get("concepts", [])[:4]:
                        self._merge_relation(
                            session,
                            claim_id,
                            self._semantic_node_id(project_id, "concept", concept),
                            "ASSOCIATED_WITH",
                            0.5,
                        )
                    for entity in claim_detail.get("entities", [])[:4]:
                        self._merge_relation(
                            session,
                            claim_id,
                            self._semantic_node_id(project_id, "entity", entity),
                            "ASSOCIATED_WITH",
                            0.65,
                        )
            else:
                for claim_id in claim_ids:
                    for concept in labels.get("concepts", [])[:3]:
                        self._merge_relation(
                            session,
                            claim_id,
                            self._semantic_node_id(project_id, "concept", concept),
                            "ASSOCIATED_WITH",
                            0.5,
                        )
                    for entity in labels.get("entities", [])[:3]:
                        self._merge_relation(
                            session,
                            claim_id,
                            self._semantic_node_id(project_id, "entity", entity),
                            "ASSOCIATED_WITH",
                            0.65,
                        )

    def link_units(self, project_id: str, source_unit_id: str, target_unit_id: str, relation_type: str) -> None:
        relation_type = relation_type if relation_type in RELATION_TYPES else "ASSOCIATED_WITH"
        with self.driver.session() as session:
            self._merge_relation(
                session,
                source_unit_id,
                target_unit_id,
                relation_type,
                0.8,
                source_label="MemoryUnit",
                target_label="MemoryUnit",
            )

    def get_graph(
        self,
        project_id: str,
        view: str = "macro",
        limit: int = 80,
        focus_id: str | None = None,
        seed_unit_ids: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        semantic_nodes, semantic_edges = self._load_semantic_graph(project_id)
        graph = nx.Graph()
        for node in semantic_nodes:
            graph.add_node(node["id"], **node)
        for edge in semantic_edges:
            graph.add_edge(edge["source"], edge["target"], **edge)

        if graph.number_of_nodes() == 0:
            return {"nodes": [], "edges": [], "meta": {"view": view, "communities": []}}

        pagerank_scores = (
            self._compute_pagerank(graph, alpha=0.9)
            if graph.number_of_nodes() > 1
            else {next(iter(graph.nodes)): 1.0}
        )
        betweenness_scores = (
            nx.betweenness_centrality(graph, k=min(20, graph.number_of_nodes()), normalized=True)
            if graph.number_of_nodes() > 2
            else {node_id: 0.0 for node_id in graph.nodes}
        )
        communities = list(nx.community.greedy_modularity_communities(graph)) if graph.number_of_edges() else [set(graph.nodes)]
        community_lookup = {}
        for index, community in enumerate(communities):
            for node_id in community:
                community_lookup[node_id] = index

        if view == "community":
            focus_id = focus_id or max(pagerank_scores, key=pagerank_scores.get)
            selected = self._select_community_nodes(graph, community_lookup, focus_id, pagerank_scores, limit)
        elif view == "evidence":
            selected = self._select_evidence_nodes(project_id, graph, seed_unit_ids, limit)
        elif view == "reasoning":
            selected = self._select_reasoning_nodes(project_id, graph, seed_unit_ids, pagerank_scores, limit)
        else:
            selected = self._select_macro_nodes(graph, pagerank_scores, betweenness_scores, limit)

        nodes, edges = self._materialize_subgraph(
            project_id,
            graph,
            selected,
            pagerank_scores,
            betweenness_scores,
            community_lookup,
            view,
            seed_unit_ids or [],
        )
        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "view": view,
                "selected_count": len(nodes),
                "communities": self._community_meta(graph, communities, pagerank_scores),
            },
        }

    def build_reasoning_overlay(
        self,
        project_id: str,
        query: str,
        retrieved_units: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        seed_unit_ids = [unit["id"] for unit in retrieved_units]
        graph_payload = self.get_graph(
            project_id=project_id,
            view="reasoning",
            limit=max(40, len(seed_unit_ids) * 8),
            seed_unit_ids=seed_unit_ids,
        )
        query_node_id = f"query:{normalize_key(query[:64])}"
        query_node = {
            "id": query_node_id,
            "label": display_label_for_kind("query", query),
            "kind": "query",
            "score": 1.0,
            "detail": {"query": query, "full_text": query},
        }
        nodes = [query_node] + graph_payload["nodes"]
        edges = list(graph_payload["edges"])
        for rank, unit in enumerate(retrieved_units, start=1):
            edges.append(
                {
                    "id": f"retrieved:{query_node_id}:{unit['id']}",
                    "source": query_node_id,
                    "target": unit["id"],
                    "type": "retrieved_by",
                    "weight": max(0.1, 1.0 - (rank - 1) * 0.12),
                }
            )
        graph_payload["nodes"] = nodes
        graph_payload["edges"] = edges
        graph_payload["meta"]["query"] = query
        return graph_payload

    def _load_semantic_graph(self, project_id: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        with self.driver.session() as session:
            node_rows = session.run(
                """
                MATCH (n:SemanticNode {project_id: $project_id})
                RETURN n.id AS id,
                       n.name AS name,
                       n.display_label AS display_label,
                       n.kind AS kind
                """,
                project_id=project_id,
            )
            nodes = [
                {
                    "id": row["id"],
                    "label": row["display_label"] or display_label_for_kind(row["kind"], row["name"]),
                    "kind": row["kind"],
                    "full_text": row["name"],
                }
                for row in node_rows
            ]
            edge_rows = session.run(
                """
                MATCH (a:SemanticNode {project_id: $project_id})-[r]->(b:SemanticNode {project_id: $project_id})
                WHERE type(r) IN $types
                RETURN a.id AS source, b.id AS target, type(r) AS type, coalesce(r.weight, 1.0) AS weight
                """,
                project_id=project_id,
                types=list(RELATION_TYPES),
            )
            edges = [
                {
                    "id": f"{row['type']}:{row['source']}:{row['target']}",
                    "source": row["source"],
                    "target": row["target"],
                    "type": row["type"],
                    "weight": float(row["weight"] or 1.0),
                }
                for row in edge_rows
            ]
        return nodes, edges

    def _select_macro_nodes(
        self,
        graph: nx.Graph,
        pagerank_scores: Dict[str, float],
        betweenness_scores: Dict[str, float],
        limit: int,
    ) -> List[str]:
        scored = []
        for node_id in graph.nodes:
            score = pagerank_scores.get(node_id, 0.0) + 0.5 * betweenness_scores.get(node_id, 0.0)
            scored.append((score, node_id))
        scored.sort(reverse=True)
        return [node_id for _, node_id in scored[:limit]]

    def _select_community_nodes(
        self,
        graph: nx.Graph,
        community_lookup: Dict[str, int],
        focus_id: str,
        pagerank_scores: Dict[str, float],
        limit: int,
    ) -> List[str]:
        community_id = community_lookup.get(focus_id)
        if community_id is None:
            return self._select_macro_nodes(graph, pagerank_scores, defaultdict(float), limit)
        selected = [node_id for node_id, cid in community_lookup.items() if cid == community_id]
        selected.sort(key=lambda node_id: pagerank_scores.get(node_id, 0.0), reverse=True)
        return selected[:limit]

    def _select_reasoning_nodes(
        self,
        project_id: str,
        graph: nx.Graph,
        seed_unit_ids: Sequence[str] | None,
        pagerank_scores: Dict[str, float],
        limit: int,
    ) -> List[str]:
        if not seed_unit_ids:
            return self._select_macro_nodes(graph, pagerank_scores, defaultdict(float), limit)
        seed_semantic_ids = self._select_evidence_nodes(project_id, graph, seed_unit_ids, max(limit, 12))
        personalization = {node_id: 0.0 for node_id in graph.nodes}
        for seed in seed_semantic_ids:
            if seed in graph:
                personalization[seed] = 1.0
        if any(personalization.values()) and graph.number_of_edges():
            ppr = self._compute_pagerank(graph, alpha=0.85, personalization=personalization)
            ordered = sorted(ppr.items(), key=lambda item: item[1], reverse=True)
            return [node_id for node_id, _ in ordered[:limit]]
        return self._select_macro_nodes(graph, pagerank_scores, defaultdict(float), limit)

    def _select_evidence_nodes(
        self,
        project_id: str,
        graph: nx.Graph,
        seed_unit_ids: Sequence[str] | None,
        limit: int,
    ) -> List[str]:
        if not seed_unit_ids:
            return list(graph.nodes)[:limit]
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (u:MemoryUnit {project_id: $project_id})-[r]->(s:SemanticNode {project_id: $project_id})
                WHERE u.id IN $seed_ids
                RETURN u.id AS unit_id, s.id AS semantic_id
                """,
                project_id=project_id,
                seed_ids=list(seed_unit_ids),
            )
            selected = set()
            for row in rows:
                selected.add(row["semantic_id"])
            return list(selected)[:limit]

    def _materialize_subgraph(
        self,
        project_id: str,
        graph: nx.Graph,
        selected: Sequence[str],
        pagerank_scores: Dict[str, float],
        betweenness_scores: Dict[str, float],
        community_lookup: Dict[str, int],
        view: str,
        seed_unit_ids: Sequence[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        selected_set = set(selected)
        nodes: List[Dict[str, Any]] = []
        for node_id in selected_set:
            data = graph.nodes[node_id]
            nodes.append(
                {
                    "id": node_id,
                    "label": data.get("label") or display_label_for_kind(data.get("kind", "semantic"), node_id),
                    "kind": data.get("kind", "semantic"),
                    "score": round(pagerank_scores.get(node_id, 0.0), 4),
                    "detail": {
                        "community": community_lookup.get(node_id),
                        "betweenness": round(betweenness_scores.get(node_id, 0.0), 4),
                        "view": view,
                        "full_text": data.get("full_text") or data.get("label", node_id),
                    },
                }
            )
        edges: List[Dict[str, Any]] = []
        for source, target, edge_data in graph.edges(data=True):
            if source not in selected_set or target not in selected_set:
                continue
            edges.append(
                {
                    "id": edge_data.get("id", f"{source}:{target}"),
                    "source": source,
                    "target": target,
                    "type": edge_data.get("type", "ASSOCIATED_WITH"),
                    "weight": float(edge_data.get("weight", 1.0)),
                }
            )

        if view in {"reasoning", "evidence"} and seed_unit_ids:
            unit_nodes, unit_edges = self._load_unit_nodes(project_id, seed_unit_ids)
            nodes.extend(unit_nodes)
            edges.extend(unit_edges)
        return nodes, edges

    def _load_unit_nodes(
        self, project_id: str, seed_unit_ids: Sequence[str]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (u:MemoryUnit {project_id: $project_id})
                WHERE u.id IN $seed_ids
                OPTIONAL MATCH (u)-[:DERIVED_FROM]-(s:SemanticNode {project_id: $project_id})
                OPTIONAL MATCH (u)-[r]->(v:MemoryUnit {project_id: $project_id})
                WHERE type(r) IN $types
                RETURN u.id AS unit_id,
                       u.text AS text,
                       u.source_type AS source_type,
                       collect(distinct s.id) AS semantic_ids,
                       collect(distinct {
                           target: v.id,
                           target_text: v.text,
                           target_kind: v.source_type,
                           type: type(r),
                           weight: coalesce(r.weight, 0.8)
                       }) AS unit_relations
                """,
                project_id=project_id,
                seed_ids=list(seed_unit_ids),
                types=list(RELATION_TYPES),
            )
            nodes: List[Dict[str, Any]] = []
            edges: List[Dict[str, Any]] = []
            seen_nodes = set()
            for row in rows:
                unit_id = row["unit_id"]
                if unit_id not in seen_nodes:
                    nodes.append(
                        {
                            "id": unit_id,
                            "label": display_label_for_kind(row["source_type"] or "memory_unit", row["text"] or ""),
                            "kind": row["source_type"] or "memory_unit",
                            "score": 1.0,
                            "detail": {"text": row["text"], "full_text": row["text"]},
                        }
                    )
                    seen_nodes.add(unit_id)
                for semantic_id in row["semantic_ids"] or []:
                    if semantic_id:
                        edges.append(
                            {
                                "id": f"derived:{unit_id}:{semantic_id}",
                                "source": unit_id,
                                "target": semantic_id,
                                "type": "derived_from",
                                "weight": 0.8,
                            }
                        )
                for relation in row["unit_relations"] or []:
                    if relation.get("target"):
                        if relation["target"] not in seen_nodes:
                            nodes.append(
                                {
                                    "id": relation["target"],
                                    "label": display_label_for_kind(
                                        relation.get("target_kind") or "memory_unit",
                                        relation.get("target_text") or "",
                                    ),
                                    "kind": relation.get("target_kind") or "memory_unit",
                                    "score": 0.8,
                                    "detail": {
                                        "text": relation.get("target_text"),
                                        "full_text": relation.get("target_text"),
                                    },
                                }
                            )
                            seen_nodes.add(relation["target"])
                        edges.append(
                            {
                                "id": f"unit:{unit_id}:{relation['target']}:{relation['type']}",
                                "source": unit_id,
                                "target": relation["target"],
                                "type": relation["type"].lower(),
                                "weight": float(relation.get("weight") or 0.8),
                            }
                        )
        return nodes, edges

    def _community_meta(
        self,
        graph: nx.Graph,
        communities: Sequence[Iterable[str]],
        pagerank_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        for index, community in enumerate(communities[:8]):
            members = list(community)
            if not members:
                continue
            ranked = sorted(members, key=lambda node_id: pagerank_scores.get(node_id, 0.0), reverse=True)
            labels = [graph.nodes[node_id].get("label", "") for node_id in ranked[:3]]
            meta.append(
                {
                    "id": f"community:{index}",
                    "size": len(members),
                    "label": " / ".join(label for label in labels if label) or f"Community {index + 1}",
                    "top_nodes": ranked[:6],
                }
            )
        return meta

    def _compute_pagerank(
        self,
        graph: nx.Graph,
        alpha: float,
        personalization: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        try:
            return nx.pagerank(graph, alpha=alpha, personalization=personalization)
        except Exception:
            return self._fallback_rank(graph, personalization)

    def _fallback_rank(
        self,
        graph: nx.Graph,
        personalization: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        if graph.number_of_nodes() == 0:
            return {}
        scores: Dict[str, float] = {}
        for node_id in graph.nodes:
            score = 1.0 + float(graph.degree(node_id))
            if personalization and personalization.get(node_id):
                score += 2.0 * float(personalization[node_id])
            scores[node_id] = score
        total = sum(scores.values()) or 1.0
        return {node_id: value / total for node_id, value in scores.items()}

    def _merge_semantic_node(
        self,
        session,
        project_id: str,
        kind: str,
        name: str,
        unit_id: str,
        display_label: str | None = None,
    ) -> str:
        node_id = self._semantic_node_id(project_id, kind, name)
        session.run(
            """
            MERGE (n:SemanticNode {id: $node_id})
            SET n.project_id = $project_id,
                n.kind = $kind,
                n.name = $name,
                n.display_label = $display_label
            WITH n
            MATCH (u:MemoryUnit {id: $unit_id})
            MERGE (u)-[:DERIVED_FROM]->(n)
            """,
            node_id=node_id,
            project_id=project_id,
            kind=kind,
            name=name[:400],
            display_label=display_label or display_label_for_kind(kind, name),
            unit_id=unit_id,
        )
        return node_id

    def _merge_relation(
        self,
        session,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float,
        source_label: str = "SemanticNode",
        target_label: str = "SemanticNode",
    ) -> None:
        relation_type = relation_type if relation_type in RELATION_TYPES else "ASSOCIATED_WITH"
        query = f"""
        MATCH (a:{source_label} {{id: $source_id}})
        MATCH (b:{target_label} {{id: $target_id}})
        MERGE (a)-[r:{relation_type}]->(b)
        SET r.weight = $weight
        """
        session.run(query, source_id=source_id, target_id=target_id, weight=weight)

    def _semantic_node_id(self, project_id: str, kind: str, value: str) -> str:
        return f"{project_id}:{kind}:{normalize_key(value)}"
