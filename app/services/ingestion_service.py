"""
IngestionService — business logic for document ingestion.

SRP:  This class owns the ingestion use case only.
DIP:  Depends on BaseIngestionPipeline, not on any concrete pipeline.

Parallel ingestion:
  When a graph_rag_pipeline is also injected the service runs both the
  RAG pipeline (chunks → vector store) and the GraphRAG pipeline
  (entities → Neo4j) concurrently using asyncio.gather.  The shared
  doc_id is generated here and passed to both so both stores are keyed
  on the same document identity.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from app.core.ingestion import BaseIngestionPipeline
from app.models.ingest import IngestRequest, IngestResponse


class IngestionService:
    """
    Orchestrates the ingestion use case.

    Receives a raw IngestRequest (or file bytes from an upload), delegates
    to one or two pipelines, and returns a structured IngestResponse.

    When `graph_rag_pipeline` is provided both pipelines run in parallel
    via asyncio.gather and the response includes graph_entities_count.
    """

    def __init__(
        self,
        pipeline: BaseIngestionPipeline,
        graph_rag_pipeline: BaseIngestionPipeline | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._graph_rag_pipeline = graph_rag_pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_document(self, request: IngestRequest) -> IngestResponse:
        """
        Ingest a document from a plain-text JSON request.

        Args:
            request: Parsed IngestRequest containing content and metadata.

        Returns:
            IngestResponse with doc_id, chunks_count, graph_entities_count.
        """
        metadata: dict[str, Any] = {**request.metadata}
        if request.source:
            metadata["source"] = request.source

        doc_id, chunks_count, entities_count = await self._run_parallel(
            content=request.content,
            metadata=metadata,
            processing_instruction=request.processing_instruction,
        )

        return IngestResponse(
            doc_id=doc_id,
            message="Document ingested successfully.",
            chunks_count=chunks_count,
            graph_entities_count=entities_count,
            metadata=metadata,
        )

    async def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: dict[str, Any] | None = None,
        processing_instruction: str = "",
    ) -> IngestResponse:
        """
        Ingest a document from an uploaded file (PDF or TXT).

        Args:
            file_bytes:             Raw bytes of the uploaded file.
            filename:               Original filename (determines extraction strategy).
            metadata:               Optional caller-supplied metadata.
            processing_instruction: Optional free-text hint guiding entity extraction.

        Returns:
            IngestResponse with doc_id, chunks_count, graph_entities_count.
        """
        meta: dict[str, Any] = {**(metadata or {}), "source": filename}

        doc_id, chunks_count, entities_count = await self._run_parallel(
            content="",
            metadata=meta,
            file_bytes=file_bytes,
            filename=filename,
            processing_instruction=processing_instruction,
        )

        return IngestResponse(
            doc_id=doc_id,
            message=f"File '{filename}' ingested successfully.",
            chunks_count=chunks_count,
            graph_entities_count=entities_count,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_parallel(
        self,
        content: str,
        metadata: dict[str, Any],
        file_bytes: bytes | None = None,
        filename: str | None = None,
        processing_instruction: str = "",
    ) -> tuple[str, int, int]:
        """
        Run RAG and GraphRAG pipelines (if present) in parallel.

        A single doc_id is generated here and shared with both pipelines so
        both stores (Postgres + Neo4j) are keyed on the same document UUID.

        Returns:
            (doc_id, chunks_count, graph_entities_count)
        """
        import inspect

        shared_doc_id = str(uuid.uuid4())

        # --- RAG pipeline call ---
        sig = inspect.signature(self._pipeline.ingest)
        has_file_kwargs = "file_bytes" in sig.parameters

        if has_file_kwargs:
            rag_coro = self._pipeline.ingest(  # type: ignore[call-arg]
                content=content,
                metadata=metadata,
                file_bytes=file_bytes,
                filename=filename,
                doc_id=shared_doc_id,
            )
        else:
            rag_coro = self._pipeline.ingest(content=content, metadata=metadata)

        # --- GraphRAG pipeline call (optional) ---
        if self._graph_rag_pipeline is not None:
            graph_coro = self._graph_rag_pipeline.ingest(  # type: ignore[call-arg]
                content=content,
                metadata=metadata,
                file_bytes=file_bytes,
                filename=filename,
                doc_id=shared_doc_id,
                processing_instruction=processing_instruction,
            )
            rag_result, _ = await asyncio.gather(rag_coro, graph_coro)
            entities_count = getattr(
                self._graph_rag_pipeline, "_last_entities_count", 0
            )
        else:
            rag_result = await rag_coro
            entities_count = 0

        chunks_count = getattr(self._pipeline, "_last_chunks_count", 1)
        return shared_doc_id, chunks_count, entities_count
