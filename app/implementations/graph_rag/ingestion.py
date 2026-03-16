"""
GraphRAGIngestionPipeline — BaseIngestionPipeline for the GraphRAG path.

Pipeline:
  1. Extract plain text from the document (same as RAG)
  2. Load entity and relation type catalogs from Postgres
  3. Call BaseEntityExtractor to extract entities + relations from the text
  4. Persist each entity as a Neo4j node (:Entity + type label)
  5. Persist each relation as a Neo4j edge
  6. Create a :Document root node and link it to all extracted entities

SRP: Only owns the graph ingestion concern.
DIP: Depends on BaseEntityExtractor, BaseDocumentProcessor, BaseGraphStore.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import asyncpg

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

    Usage:
        pipeline = GraphRAGIngestionPipeline(
            document_processor=...,
            entity_extractor=...,
            graph_store=...,
            postgres_dsn=settings.postgres_dsn,
        )
        doc_id = await pipeline.ingest(content="", metadata={}, file_bytes=b"...", filename="report.pdf")
    """

    def __init__(
        self,
        document_processor: BaseDocumentProcessor,
        entity_extractor: BaseEntityExtractor,
        graph_store: BaseGraphStore,
        postgres_dsn: str,
    ) -> None:
        self._processor = document_processor
        self._extractor = entity_extractor
        self._graph_store = graph_store
        self._postgres_dsn = postgres_dsn

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
        doc_id: str | None = None,     # accept shared doc_id from parallel call
    ) -> str:
        """
        Extract entities/relations and store them in Neo4j.

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

        # 1. Extract text
        if file_bytes is not None and filename is not None:
            text = await self._processor.extract_text(file_bytes, filename)
        else:
            text = content

        if not text.strip():
            logger.warning("GraphRAGIngestionPipeline: document is empty, skipping.")
            self._last_entities_count = 0
            return doc_id

        # 2. Load type catalogs from Postgres
        entity_types, relation_types = await self._load_type_catalogs()

        # 3. Extract entities + relations via LLM
        extraction = await self._extractor.extract(text, entity_types, relation_types)
        entities: list[dict] = extraction.get("entities", [])
        relations: list[dict] = extraction.get("relations", [])

        logger.info(
            "GraphRAG extraction: %d entities, %d relations for doc %s",
            len(entities), len(relations), doc_id,
        )

        # 4. Persist :Document root node
        await self._graph_store.add_node(
            node_id=doc_id,
            labels=["Document"],
            data={
                "id": doc_id,
                "source": metadata.get("source", ""),
                "content_preview": text[:500],
            },
        )

        # 5. Persist :Entity nodes
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
            # Link entity to parent document
            await self._graph_store.add_edge(
                src_id=doc_id,
                dst_id=global_id,
                relation="MENTIONS",
                data={"doc_id": doc_id},
            )

        # 6. Persist relationships between entities
        for rel in relations:
            src_local = rel.get("src_id", "")
            dst_local = rel.get("dst_id", "")
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
        return doc_id

    async def delete(self, doc_id: str) -> None:
        """Delete the Document node and all entities linked to it."""
        await self._graph_store.delete_node(doc_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_type_catalogs(
        self,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """
        Fetch active entity and relation types from Postgres.

        Falls back to empty lists if the table doesn't exist yet or the DB
        is unreachable — the extractor will still run with generic types.
        """
        try:
            conn = await asyncpg.connect(self._postgres_dsn)
            try:
                entity_rows = await conn.fetch(
                    "SELECT name, description FROM graph_entity_types "
                    "WHERE is_active = TRUE ORDER BY id;"
                )
                relation_rows = await conn.fetch(
                    "SELECT name, description FROM graph_relation_types "
                    "WHERE is_active = TRUE ORDER BY id;"
                )
                entity_types = [dict(r) for r in entity_rows]
                relation_types = [dict(r) for r in relation_rows]
            finally:
                await conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load type catalogs from Postgres: %s", exc)
            entity_types = []
            relation_types = []

        return entity_types, relation_types
