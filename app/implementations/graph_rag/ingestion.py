"""
GraphRAGIngestionPipeline — BaseIngestionPipeline for the GraphRAG path.

Pipeline:
  1. Extract plain text from the document (same as RAG).
     The FULL document text is passed to the entity extractor as a single
     piece — it is never split into chunks.  Chunking is a RAG-only concern;
     entity extraction needs global document context to produce a coherent,
     connected knowledge graph.
  2. Call BaseEntityExtractor to extract entities + relations from the text.
  3. Filter entities and relations to the largest connected component
     (isolated entities are dropped so the persisted graph is always connected).
  4. Persist each entity as a Neo4j node (:Entity + type label).
  5. Persist each relation as a Neo4j edge.

No Document root node is created; entities are linked only through their
own extracted relationships.  The doc_id is stored as a property on every
entity node so they can be found and deleted together.

SRP: Only owns the graph ingestion concern.
DIP: Depends on BaseEntityExtractor, BaseDocumentProcessor, BaseGraphStore.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict, deque
from typing import Any

from app.core.document_processor import BaseDocumentProcessor
from app.core.entity_extractor import BaseEntityExtractor
from app.core.graph_store import BaseGraphStore
from app.core.ingestion import BaseIngestionPipeline

logger = logging.getLogger(__name__)


class GraphRAGIngestionPipeline(BaseIngestionPipeline):
    """
    Ingestion pipeline that extracts a knowledge graph from documents.

    Entities and their relationships are stored in Neo4j.  The same
    `doc_id` (UUID) used by the RAG pipeline is re-used here so both
    stores stay in sync on the same document identity.

    Only entities that form part of the largest connected component are
    persisted.  No Document root node is created; connectivity is
    determined solely by extracted entity-to-entity relationships.

    Usage:
        pipeline = GraphRAGIngestionPipeline(
            document_processor=...,
            entity_extractor=...,
            graph_store=...,
        )
        doc_id = await pipeline.ingest(
            content="", metadata={},
            file_bytes=b"...", filename="report.pdf",
            processing_instruction="Extract people, companies, and funding relationships.",
        )
    """

    def __init__(
        self,
        document_processor: BaseDocumentProcessor,
        entity_extractor: BaseEntityExtractor,
        graph_store: BaseGraphStore,
    ) -> None:
        self._processor = document_processor
        self._extractor = entity_extractor
        self._graph_store = graph_store

        # Set after ingest() for the service to read
        self._last_entities_count: int = 0

    # ------------------------------------------------------------------
    # BaseIngestionPipeline interface
    # ------------------------------------------------------------------

    async def ingest(
        self,
        content: str,
        metadata: dict[str, Any],
        *,
        file_bytes: bytes | None = None,
        filename: str | None = None,
        doc_id: str | None = None,
        processing_instruction: str = "",
    ) -> str:
        """
        Extract entities/relations and store them in Neo4j.

        The entire document text is passed to the entity extractor as a
        single unit — no chunking is performed.  Chunking is a RAG-only
        concern; splitting the text here would fragment entity context and
        produce disconnected graphs.

        Args:
            content:    Raw text (used when file_bytes is not provided).
            metadata:   Metadata; at minimum should contain "source".
            file_bytes: Raw file bytes (optional, triggers extraction).
            filename:   Original filename (determines extraction strategy).
            doc_id:     Optional shared doc_id so RAG + GraphRAG are linked.

        Returns:
            The doc_id (UUID string).
        """
        doc_id = doc_id or str(uuid.uuid4())

        # 1. Extract full document text — never chunked for GraphRAG.
        # chunk_text() is intentionally NOT called here; the extractor needs
        # the complete document to reason about entity relationships globally.
        if file_bytes is not None and filename is not None:
            text = await self._processor.extract_text(file_bytes, filename)
        else:
            text = content

        if not text.strip():
            logger.warning("GraphRAGIngestionPipeline: document is empty, skipping.")
            self._last_entities_count = 0
            return doc_id

        # 2. Extract entities + relations via LLM
        logger.info("GraphRAG [%s]: starting entity extraction (text length=%d chars)", doc_id, len(text))
        extraction = await self._extractor.extract(text, processing_instruction)
        entities: list[dict] = extraction.get("entities", [])
        relations: list[dict] = extraction.get("relations", [])

        logger.info(
            "GraphRAG [%s]: extracted %d entities, %d relations",
            doc_id, len(entities), len(relations),
        )

        # 3. Filter to largest connected component
        entities, relations = self._largest_connected_component(entities, relations)

        logger.info(
            "GraphRAG [%s]: after connectivity filter — %d entities, %d relations",
            doc_id, len(entities), len(relations),
        )
        for e in entities:
            logger.info(
                "  entity: name=%r  type=%s  id=%s",
                e.get("name"), e.get("type"), e.get("id"),
            )
        for r in relations:
            logger.info(
                "  relation: %s -[%s]-> %s",
                r.get("src_id"), r.get("relation"), r.get("dst_id"),
            )

        # 4. Persist :Entity nodes
        entity_id_map: dict[str, str] = {}   # local_id → global neo4j id
        for entity in entities:
            local_id = entity.get("id", "")
            if not local_id:
                continue
            # Make globally unique by prefixing with doc_id
            global_id = f"{doc_id}__{local_id}"
            entity_id_map[local_id] = global_id

            entity_type = entity.get("type", "Entity")
            await self._graph_store.add_node(
                node_id=global_id,
                labels=["Entity", entity_type],
                data={
                    "id": global_id,
                    "name": entity.get("name", local_id),
                    "type": entity_type,
                    "description": entity.get("description", ""),
                    "doc_id": doc_id,
                },
            )

        # 5. Persist relationships between entities
        for rel in relations:
            src_local = rel.get("src_id", "")
            dst_local = rel.get("dst_id", "")
            # Guard against malformed LLM output (e.g. list instead of string)
            if not isinstance(src_local, str) or not isinstance(dst_local, str):
                logger.warning(
                    "GraphRAG [%s]: skipping relation with non-string src_id/dst_id: %r",
                    doc_id, rel,
                )
                continue
            relation_type = rel.get("relation", "RELATED_TO")
            if src_local not in entity_id_map or dst_local not in entity_id_map:
                continue
            await self._graph_store.add_edge(
                src_id=entity_id_map[src_local],
                dst_id=entity_id_map[dst_local],
                relation=relation_type,
                data=rel.get("properties", {}),
            )

        self._last_entities_count = len(entities)
        logger.info(
            "GraphRAG [%s]: ingestion complete — %d entities and %d relations written to Neo4j",
            doc_id, len(entities), len(relations),
        )
        return doc_id

    async def delete(self, doc_id: str) -> None:
        """Delete all entity nodes belonging to this document."""
        await self._graph_store.delete_nodes_by_doc_id(doc_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _largest_connected_component(
        entities: list[dict],
        relations: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Return only the entities and relations that form the largest connected
        component, treating the graph as undirected.

        Entities with no valid relations are isolated nodes and are excluded
        when a larger connected component exists.  If all entities are
        isolated (no relations at all) the largest single-node component is
        returned — i.e. only one entity survives.

        Args:
            entities:  List of entity dicts, each must have an "id" key.
            relations: List of relation dicts with "src_id" and "dst_id".

        Returns:
            (filtered_entities, filtered_relations)
        """
        entity_ids = {e["id"] for e in entities if e.get("id")}

        # Only keep relations where both endpoints are valid and distinct
        valid_relations = [
            r for r in relations
            if (
                r.get("src_id") in entity_ids
                and r.get("dst_id") in entity_ids
                and r.get("src_id") != r.get("dst_id")
            )
        ]

        # Build undirected adjacency list
        adj: dict[str, set[str]] = defaultdict(set)
        for rel in valid_relations:
            src, dst = rel["src_id"], rel["dst_id"]
            adj[src].add(dst)
            adj[dst].add(src)

        # BFS to find all connected components
        visited: set[str] = set()
        components: list[set[str]] = []

        for eid in entity_ids:
            if eid in visited:
                continue
            component: set[str] = set()
            queue: deque[str] = deque([eid])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbour in adj.get(node, set()):
                    if neighbour not in visited:
                        queue.append(neighbour)
            components.append(component)

        if not components:
            return [], []

        largest = max(components, key=len)

        dropped = len(entity_ids) - len(largest)
        if dropped:
            logger.info(
                "GraphRAGIngestionPipeline: dropped %d disconnected entities "
                "not reachable from the main component.",
                dropped,
            )

        filtered_entities = [e for e in entities if e.get("id") in largest]
        filtered_relations = [
            r for r in valid_relations
            if r.get("src_id") in largest and r.get("dst_id") in largest
        ]
        return filtered_entities, filtered_relations
