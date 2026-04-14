"""
GraphRAGIngestionPipeline — BaseIngestionPipeline for the GraphRAG path.

Pipeline:
  1. Extract plain text from the document (same as RAG).
  2. Split the text into overlapping chunks and run the entity extractor on
     each chunk independently (multi-pass).  For short documents that fit
     inside a single chunk the extraction is a single LLM call.
  3. Merge results across chunks: entities are deduplicated by normalised name
     so the same real-world entity mentioned in multiple chunks maps to one
     canonical node; relation IDs are remapped to canonical entity IDs and
     exact duplicates are removed.
  4. Persist each entity as a Neo4j node (:Entity + type label).
  5. Persist each relation as a Neo4j edge.

ALL extracted entities are persisted — including isolated ones (no relations).
The old "largest connected component" filter has been removed so that no valid
entity is silently dropped.

SRP: Only owns the graph ingestion concern.
DIP: Depends on BaseEntityExtractor, BaseDocumentProcessor, BaseGraphStore.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict, deque
from typing import Any

from app.config import settings
from app.core.document_processor import BaseDocumentProcessor
from app.core.entity_extractor import BaseEntityExtractor
from app.core.graph_store import BaseGraphStore
from app.core.ingestion import BaseIngestionPipeline
from app.core.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class GraphRAGIngestionPipeline(BaseIngestionPipeline):
    """
    Ingestion pipeline that extracts a knowledge graph from documents.

    Entities and their relationships are stored in Neo4j.  The same
    `doc_id` (UUID) used by the RAG pipeline is re-used here so both
    stores stay in sync on the same document identity.

    Multi-pass chunked extraction is used so that long documents are
    processed in overlapping segments and results merged.  All extracted
    entities are persisted regardless of connectivity.

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
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self._processor = document_processor
        self._extractor = entity_extractor
        self._graph_store = graph_store
        self._vector_store = vector_store

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

        The document is split into overlapping chunks and each chunk is
        processed by the entity extractor independently.  Results are merged
        by normalised entity name before being persisted so the same
        real-world entity is stored only once regardless of how many chunks
        it appears in.

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

        # 1. Extract full document text.
        if file_bytes is not None and filename is not None:
            text = await self._processor.extract_text(file_bytes, filename)
        else:
            text = content

        if not text.strip():
            logger.warning("GraphRAGIngestionPipeline: document is empty, skipping.")
            self._last_entities_count = 0
            return doc_id

        # 2. Extract entities + relations via multi-pass chunked extraction.
        logger.info(
            "GraphRAG [%s]: starting entity extraction (text length=%d chars)",
            doc_id, len(text),
        )
        entities, relations = await self._extract_multi_pass(text, processing_instruction, doc_id)

        logger.info(
            "GraphRAG [%s]: extracted %d entities, %d relations",
            doc_id, len(entities), len(relations),
        )

        # 3. Drop relations referencing unknown entity IDs (defensive clean-up
        #    in case the LLM produced a relation whose endpoint was later
        #    deduped away or never extracted).
        entity_id_set = {e["id"] for e in entities if e.get("id")}
        relations = [
            r for r in relations
            if r.get("src_id") in entity_id_set
            and r.get("dst_id") in entity_id_set
            and r.get("src_id") != r.get("dst_id")
        ]

        logger.info(
            "GraphRAG [%s]: after validation — %d entities, %d relations",
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
                    "source_chunk_id": entity.get("source_chunk_id", ""),
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
            edge_data = {**rel.get("properties", {}), "source_chunk_id": rel.get("source_chunk_id", "")}
            await self._graph_store.add_edge(
                src_id=entity_id_map[src_local],
                dst_id=entity_id_map[dst_local],
                relation=relation_type,
                data=edge_data,
            )

        self._last_entities_count = len(entities)
        logger.info(
            "GraphRAG [%s]: ingestion complete — %d entities and %d relations written to Neo4j",
            doc_id, len(entities), len(relations),
        )
        return doc_id

    async def delete(self, doc_id: str) -> None:
        """Delete all entity nodes and graph extraction chunks for this document."""
        await self._graph_store.delete_nodes_by_doc_id(doc_id)
        if self._vector_store and hasattr(self._vector_store, "delete_by_doc_id"):
            # Graph extraction chunks are stored under "{doc_id}__graph" parent doc_id
            await self._vector_store.delete_by_doc_id(f"{doc_id}__graph")  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _extract_multi_pass(
        self,
        text: str,
        processing_instruction: str,
        doc_id: str,
    ) -> tuple[list[dict], list[dict]]:
        """
        Extract entities and relations using overlapping chunks.

        For short documents (≤ graph_extraction_chunk_size chars) a single
        extraction call is made.  For longer documents the text is split into
        overlapping chunks; each chunk is extracted independently and the
        results are merged by normalised entity name so cross-chunk duplicates
        collapse into a single canonical entity node.

        Each chunk is also stored in PostgreSQL (with NULL embedding) via
        the optional vector_store so that the ToG retriever can later fetch
        source text by chunk ID.

        Args:
            text:                   Full document text.
            processing_instruction: Optional extraction hint passed to the LLM.
            doc_id:                 Parent document ID; used to build chunk IDs.

        Returns:
            (merged_entities, merged_relations)
        """
        chunk_size = settings.graph_extraction_chunk_size
        overlap = settings.graph_extraction_chunk_overlap
        chunks = self._split_text(text, chunk_size, overlap)

        if len(chunks) <= 1:
            chunk_id = f"{doc_id}__graph_chunk_0"
            if self._vector_store:
                await self._vector_store.upsert(
                    doc_id=chunk_id,
                    vector=None,
                    metadata={"doc_id": f"{doc_id}__graph", "chunk_index": 0, "total_chunks": 1, "source": ""},
                    content=text,
                )
            result = await self._extractor.extract(text, processing_instruction)
            entities: list[dict] = result.get("entities", [])
            relations: list[dict] = result.get("relations", [])
            for e in entities:
                e["source_chunk_id"] = chunk_id
            for r in relations:
                r["source_chunk_id"] = chunk_id
            return entities, relations

        logger.info(
            "GraphRAGIngestionPipeline: multi-pass extraction over %d chunks", len(chunks)
        )

        # normalised_name → first-seen entity dict (canonical record)
        all_entities: dict[str, dict] = {}
        # normalised_name → canonical slug id chosen on first encounter
        name_to_canonical_id: dict[str, str] = {}
        all_relations: list[dict] = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}__graph_chunk_{i}"
            logger.info(
                "GraphRAGIngestionPipeline: extracting chunk %d/%d (id=%s)",
                i + 1, len(chunks), chunk_id,
            )
            if self._vector_store:
                await self._vector_store.upsert(
                    doc_id=chunk_id,
                    vector=None,
                    metadata={"doc_id": f"{doc_id}__graph", "chunk_index": i, "total_chunks": len(chunks), "source": ""},
                    content=chunk,
                )
            result = await self._extractor.extract(chunk, processing_instruction)

            chunk_entities: list[dict] = result.get("entities", [])
            chunk_relations: list[dict] = result.get("relations", [])

            # Map this chunk's local entity IDs → the canonical global IDs
            local_to_canonical: dict[str, str] = {}

            for entity in chunk_entities:
                local_id = entity.get("id", "")
                name = entity.get("name", "")
                if not local_id or not name:
                    continue
                key = name.strip().lower()
                if key not in all_entities:
                    entity["source_chunk_id"] = chunk_id  # first-seen chunk owns the entity
                    all_entities[key] = entity
                    name_to_canonical_id[key] = local_id
                canonical_id = name_to_canonical_id[key]
                local_to_canonical[local_id] = canonical_id

            for rel in chunk_relations:
                src = local_to_canonical.get(rel.get("src_id", ""))
                dst = local_to_canonical.get(rel.get("dst_id", ""))
                if src and dst:
                    all_relations.append({**rel, "src_id": src, "dst_id": dst, "source_chunk_id": chunk_id})

        # Ensure every entity record carries the canonical id
        merged_entities = list(all_entities.values())
        for entity in merged_entities:
            key = entity.get("name", "").strip().lower()
            entity["id"] = name_to_canonical_id.get(key, entity["id"])

        # Deduplicate relations with the same (src, dst, relation) triple
        seen_rels: set[tuple] = set()
        deduped_relations: list[dict] = []
        for rel in all_relations:
            key = (rel.get("src_id"), rel.get("dst_id"), rel.get("relation"))
            if key not in seen_rels:
                seen_rels.add(key)
                deduped_relations.append(rel)

        logger.info(
            "GraphRAGIngestionPipeline: merged extraction — %d entities, %d relations",
            len(merged_entities), len(deduped_relations),
        )
        return merged_entities, deduped_relations

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """
        Split text into overlapping character-level chunks.

        Returns the original text as a single-element list when chunk_size
        is not positive or the text fits in one chunk.

        Args:
            text:       Full document text.
            chunk_size: Maximum characters per chunk.
            overlap:    Characters shared between adjacent chunks.

        Returns:
            List of non-empty chunk strings in document order.
        """
        if chunk_size <= 0 or len(text) <= chunk_size:
            return [text]
        step = max(1, chunk_size - overlap)
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunk = text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    @staticmethod
    def _largest_connected_component(
        entities: list[dict],
        relations: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Return only the entities and relations that form the largest connected
        component, treating the graph as undirected.

        NOTE: This method is no longer called by ingest().  It is retained for
        debugging / future use only.  The ingestion pipeline now persists ALL
        extracted entities rather than filtering to the largest cluster.

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
