"""
InMemoryGraphStore — in-process stub graph store using plain Python dicts.

Useful for unit tests and early development before Neo4j is available.
"""

from __future__ import annotations

from typing import Any

from app.core.graph_store import BaseGraphStore


class InMemoryGraphStore(BaseGraphStore):
    """
    Graph store backed by Python dicts — no external dependencies.

    Data is lost when the process exits.
    """

    def __init__(self) -> None:
        # node_id -> {labels: [...], **properties}
        self._nodes: dict[str, dict[str, Any]] = {}
        # List of edge dicts: {src, dst, relation, **properties}
        self._edges: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """No-op — nothing to connect."""

    async def close(self) -> None:
        """No-op."""

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def add_node(self, node_id: str, labels: list[str], data: dict[str, Any]) -> None:
        self._nodes[node_id] = {"labels": labels, **data}

    async def add_edge(
        self,
        src_id: str,
        dst_id: str,
        relation: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self._edges.append({"src": src_id, "dst": dst_id, "relation": relation, **(data or {})})

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self._nodes.get(node_id)

    async def get_subgraph(self, node_id: str, depth: int = 1) -> dict[str, Any]:
        """BFS over in-memory edges up to `depth` hops."""
        visited_nodes: set[str] = set()
        frontier = {node_id}
        result_edges: list[dict] = []

        for _ in range(depth):
            next_frontier: set[str] = set()
            for edge in self._edges:
                if edge["src"] in frontier and edge["dst"] not in visited_nodes:
                    result_edges.append(edge)
                    next_frontier.add(edge["dst"])
            visited_nodes |= frontier
            frontier = next_frontier

        all_node_ids = visited_nodes | frontier
        nodes = [{"id": nid, **self._nodes[nid]} for nid in all_node_ids if nid in self._nodes]
        return {"nodes": nodes, "edges": result_edges}

    async def get_relations(self, entity_id: str) -> list[str]:
        relations = (
            edge["relation"]
            for edge in self._edges
            if edge["src"] == entity_id or edge["dst"] == entity_id
        )
        return list(dict.fromkeys(relations))

    async def get_tail_entities(self, entity_id: str, relation: str) -> list[dict[str, Any]]:
        """Return entities reachable FROM entity_id via outgoing relation."""
        results = []
        for edge in self._edges:
            if edge["src"] == entity_id and edge["relation"] == relation:
                node = self._nodes.get(edge["dst"], {})
                results.append({
                    "id": edge["dst"],
                    "name": node.get("name", edge["dst"]),
                    "source_chunk_id": node.get("source_chunk_id", ""),
                    "edge_chunk_id": edge.get("source_chunk_id", ""),
                })
        return results

    async def get_head_entities(self, entity_id: str, relation: str) -> list[dict[str, Any]]:
        """Return entities pointing TO entity_id via incoming relation."""
        results = []
        for edge in self._edges:
            if edge["dst"] == entity_id and edge["relation"] == relation:
                node = self._nodes.get(edge["src"], {})
                results.append({
                    "id": edge["src"],
                    "name": node.get("name", edge["src"]),
                    "source_chunk_id": node.get("source_chunk_id", ""),
                    "edge_chunk_id": edge.get("source_chunk_id", ""),
                })
        return results

    async def search_nodes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Substring search over name and description of Entity nodes only."""
        results = []
        query_lower = query.lower()
        for node_id, props in self._nodes.items():
            if "Entity" not in props.get("labels", []):
                continue
            if (query_lower in str(props.get("name", "")).lower()
                    or query_lower in str(props.get("description", "")).lower()):
                results.append({"id": node_id, **props})
                if len(results) >= top_k:
                    break
        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        self._edges = [e for e in self._edges if e["src"] != node_id and e["dst"] != node_id]

    async def delete_nodes_by_doc_id(self, doc_id: str) -> None:
        """Delete all nodes with the given doc_id property and their edges."""
        to_delete = {
            nid for nid, props in self._nodes.items()
            if props.get("doc_id") == doc_id
        }
        for nid in to_delete:
            self._nodes.pop(nid, None)
        self._edges = [
            e for e in self._edges
            if e["src"] not in to_delete and e["dst"] not in to_delete
        ]
