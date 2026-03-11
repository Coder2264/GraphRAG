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

    async def search_nodes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Naive substring search over node property values."""
        results = []
        query_lower = query.lower()
        for node_id, props in self._nodes.items():
            if any(query_lower in str(v).lower() for v in props.values()):
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
