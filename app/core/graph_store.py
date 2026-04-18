"""
Abstract interface for graph storage implementations.

Targets Neo4j as the primary real backend while keeping the contract
generic enough for other property-graph systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGraphStore(ABC):
    """
    Contract for any graph database backend.

    Examples: InMemoryGraphStore, Neo4jGraphStore, ...
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection / session to the graph store."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release connection / session."""
        ...

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def add_node(self, node_id: str, labels: list[str], data: dict[str, Any]) -> None:
        """
        Create or update a node.

        Args:
            node_id: Unique identifier for the node.
            labels:  List of label strings (e.g. ["Entity", "Person"]).
            data:    Property key-value pairs.
        """
        ...

    @abstractmethod
    async def add_edge(
        self,
        src_id: str,
        dst_id: str,
        relation: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Create or update a directed edge between two nodes.

        Args:
            src_id:   Source node ID.
            dst_id:   Destination node ID.
            relation: Relationship type string (e.g. "MENTIONS", "RELATED_TO").
            data:     Optional properties on the edge.
        """
        ...

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return a node's properties, or None if not found."""
        ...

    @abstractmethod
    async def get_subgraph(self, node_id: str, depth: int = 1) -> dict[str, Any]:
        """
        Retrieve the neighbourhood subgraph around a node.

        Args:
            node_id: Root node ID.
            depth:   Number of hops to traverse.

        Returns:
            Dict with keys "nodes" (list) and "edges" (list).
        """
        ...

    @abstractmethod
    async def search_nodes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Full-text or semantic search over node properties.

        Args:
            query: Search string.
            top_k: Maximum number of results.

        Returns:
            List of matching node dicts.
        """
        ...

    @abstractmethod
    async def get_relations(self, entity_id: str) -> list[str]:
        """
        Return all unique relation labels connected to this entity
        (both inbound and outbound).

        Args:
            entity_id: The node's id property in the graph.

        Returns:
            Deduplicated list of relation label strings.
            Returns an empty list if the entity does not exist.
        """
        ...

    @abstractmethod
    async def get_tail_entities(self, entity_id: str, relation: str) -> list[dict[str, Any]]:
        """
        Return entities reachable FROM entity_id via an outgoing relation.

        Args:
            entity_id: The node's id property (e.g. "{doc_id}__slug").
            relation:  Relationship type label (e.g. "BORN_IN").

        Returns:
            List of dicts, each with keys:
              - id:              The target node's id property.
              - name:            The target node's name property.
              - source_chunk_id: ID of the chunk that produced the entity node
                                 (empty string when absent).
              - edge_chunk_id:   ID of the chunk that implied this edge
                                 (empty string when absent).
        """
        ...

    @abstractmethod
    async def get_head_entities(self, entity_id: str, relation: str) -> list[dict[str, Any]]:
        """
        Return entities pointing TO entity_id via an incoming relation.

        Args:
            entity_id: The node's id property (e.g. "{doc_id}__slug").
            relation:  Relationship type label (e.g. "BORN_IN").

        Returns:
            Same schema as get_tail_entities.
        """
        ...

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges."""
        ...

    @abstractmethod
    async def delete_nodes_by_doc_id(self, doc_id: str) -> None:
        """Delete all nodes whose 'doc_id' property equals doc_id, and their edges."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Delete ALL nodes and relationships — use only for testing cleanup."""
        ...
