"""
Neo4jGraphStore — BaseGraphStore backed by Neo4j.

Requires:
  - Neo4j >= 5.x running (local or cloud)
  - `neo4j` Python package (in requirements.txt)

Uses the official async Neo4j Python driver.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from app.core.graph_store import BaseGraphStore

logger = logging.getLogger(__name__)


class Neo4jGraphStore(BaseGraphStore):
    """
    Production graph store backed by Neo4j.

    Swap in by registering "neo4j" in registry.py.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "graphRAG") -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver: AsyncDriver | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialise the async Neo4j driver and ensure schema constraints exist."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        # Verify connectivity
        await self._driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s (database=%s)", self._uri, self._database)
        # Ensure constraints + indexes exist (idempotent)
        await self.setup_schema()

    async def setup_schema(self) -> None:
        """
        Create Neo4j schema objects required by GraphRAG (idempotent).

        Objects created:
          - Uniqueness constraint on Entity.id  (prevents duplicate nodes)
          - Uniqueness constraint on Document.id
          - Fulltext index `entitySearch` on Entity.name + Entity.description
            (used by search_nodes for fast keyword lookup)
        """
        assert self._driver, "Call connect() first."
        async with self._driver.session(database=self._database) as session:
            # Uniqueness constraints — safe to re-run (IF NOT EXISTS)
            await session.run(
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )
            await session.run(
                "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.id IS UNIQUE"
            )
            # Fulltext index for keyword search over entity names/descriptions
            await session.run(
                "CREATE FULLTEXT INDEX entitySearch IF NOT EXISTS "
                "FOR (n:Entity) ON EACH [n.name, n.description]"
            )

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def add_node(self, node_id: str, labels: list[str], data: dict[str, Any]) -> None:
        """MERGE a node by its id property, set labels and properties."""
        assert self._driver, "Call connect() first."
        escaped = [f"`{lbl.replace('`', '')}`" for lbl in labels] if labels else ["`Node`"]
        label_str = ":".join(escaped)
        async with self._driver.session(database=self._database) as session:
            await session.run(
                f"""
                MERGE (n:{label_str} {{id: $node_id}})
                SET n += $props
                """,
                node_id=node_id,
                props=data,
            )

    async def add_edge(
        self,
        src_id: str,
        dst_id: str,
        relation: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """MERGE a directed relationship between two nodes."""
        assert self._driver, "Call connect() first."
        escaped_rel = f"`{relation.replace('`', '')}`"
        async with self._driver.session(database=self._database) as session:
            await session.run(
                f"""
                MATCH (a {{id: $src_id}}), (b {{id: $dst_id}})
                MERGE (a)-[r:{escaped_rel}]->(b)
                SET r += $props
                """,
                src_id=src_id,
                dst_id=dst_id,
                props=data or {},
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        assert self._driver, "Call connect() first."
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (n {id: $node_id}) RETURN properties(n) AS props LIMIT 1",
                node_id=node_id,
            )
            record = await result.single()
        return dict(record["props"]) if record else None

    async def get_subgraph(self, node_id: str, depth: int = 1) -> dict[str, Any]:
        """Return nodes and relationships within `depth` hops of `node_id`."""
        assert self._driver, "Call connect() first."
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MATCH path = (root {{id: $node_id}})-[*0..{depth}]-(neighbour)
                WITH nodes(path) AS ns, relationships(path) AS rs
                UNWIND ns AS n
                WITH COLLECT(DISTINCT properties(n)) AS nodes, rs
                UNWIND rs AS r
                RETURN nodes,
                       COLLECT(DISTINCT {{
                           src:      startNode(r).id,
                           dst:      endNode(r).id,
                           relation: type(r),
                           raw_text: r.raw_text
                       }}) AS edges
                """,
                node_id=node_id,
            )
            record = await result.single()

        if not record:
            return {"nodes": [], "edges": []}
        return {"nodes": record["nodes"], "edges": record["edges"]}

    async def get_relations(self, entity_id: str) -> list[str]:
        if not self._driver:
            return []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (n {id: $entity_id})-[r]-() RETURN DISTINCT type(r) AS relation",
                entity_id=entity_id,
            )
            records = await result.data()
        return [r["relation"] for r in records]

    async def search_nodes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Full-text search over Entity nodes using the entitySearch fulltext index."""
        if not self._driver:
            return []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                CALL db.index.fulltext.queryNodes("entitySearch", $query)
                YIELD node, score
                RETURN properties(node) AS props
                ORDER BY score DESC
                LIMIT $top_k
                """,
                query=query,
                top_k=top_k,
            )
            records = await result.data()
        return [dict(r["props"]) for r in records]

    async def get_tail_entities(self, entity_id: str, relation: str) -> list[dict[str, Any]]:
        """Return entities reachable FROM entity_id via outgoing relation."""
        if not self._driver:
            return []
        escaped_rel = f"`{relation.replace('`', '')}`"
        query = f"""
        MATCH (n {{id: $entity_id}})-[r:{escaped_rel}]->(m:Entity)
        RETURN m.id AS id, m.name AS name,
               COALESCE(m.source_chunk_id, '') AS source_chunk_id,
               COALESCE(r.source_chunk_id, '') AS edge_chunk_id
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, entity_id=entity_id)
            records = await result.data()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "source_chunk_id": r["source_chunk_id"],
                "edge_chunk_id": r["edge_chunk_id"],
            }
            for r in records
        ]

    async def get_head_entities(self, entity_id: str, relation: str) -> list[dict[str, Any]]:
        """Return entities pointing TO entity_id via incoming relation."""
        if not self._driver:
            return []
        escaped_rel = f"`{relation.replace('`', '')}`"
        query = f"""
        MATCH (m:Entity)-[r:{escaped_rel}]->(n {{id: $entity_id}})
        RETURN m.id AS id, m.name AS name,
               COALESCE(m.source_chunk_id, '') AS source_chunk_id,
               COALESCE(r.source_chunk_id, '') AS edge_chunk_id
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, entity_id=entity_id)
            records = await result.data()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "source_chunk_id": r["source_chunk_id"],
                "edge_chunk_id": r["edge_chunk_id"],
            }
            for r in records
        ]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its relationships."""
        assert self._driver, "Call connect() first."
        async with self._driver.session(database=self._database) as session:
            await session.run(
                "MATCH (n {id: $node_id}) DETACH DELETE n",
                node_id=node_id,
            )

    async def delete_nodes_by_doc_id(self, doc_id: str) -> None:
        """Delete all nodes with the given doc_id property and their relationships."""
        assert self._driver, "Call connect() first."
        async with self._driver.session(database=self._database) as session:
            await session.run(
                "MATCH (n {doc_id: $doc_id}) DETACH DELETE n",
                doc_id=doc_id,
            )
