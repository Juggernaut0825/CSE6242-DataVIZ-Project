from typing import Any, Dict, List

from neo4j import GraphDatabase

from app.config import settings


def get_project_graph(project_id: str, limit: int = 200) -> Dict[str, List[Dict[str, Any]]]:
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    cypher = """
    MATCH (n:Fact {project_id: $project_id})-[r]-(m:Fact {project_id: $project_id})
    RETURN n, r, m
    LIMIT $limit
    """

    nodes: Dict[int, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    with driver.session() as session:
        result = session.run(cypher, project_id=project_id, limit=limit)
        for record in result:
            n = record["n"]
            m = record["m"]
            r = record["r"]

            if n.id not in nodes:
                nodes[n.id] = {
                    "id": str(n.id),
                    "label": (n.get("content", "") or "")[:50],
                    "fact_id": n.get("fact_id"),
                    "project_id": n.get("project_id"),
                }
            if m.id not in nodes:
                nodes[m.id] = {
                    "id": str(m.id),
                    "label": (m.get("content", "") or "")[:50],
                    "fact_id": m.get("fact_id"),
                    "project_id": m.get("project_id"),
                }

            edges.append(
                {
                    "id": f"{r.id}",
                    "source": str(n.id),
                    "target": str(m.id),
                    "type": r.type,
                    "confidence": r.get("confidence"),
                }
            )

    driver.close()
    return {"nodes": list(nodes.values()), "edges": edges}
