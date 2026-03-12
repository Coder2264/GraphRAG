"""
IngestionService — business logic for document ingestion.

SRP:  This class owns the ingestion use case only.
DIP:  Depends on BaseIngestionPipeline, not on any concrete pipeline.
"""

from __future__ import annotations

from typing import Any

from app.core.ingestion import BaseIngestionPipeline
from app.models.ingest import IngestRequest, IngestResponse


class IngestionService:
    """
    Orchestrates the ingestion use case.

    Receives a raw IngestRequest (or file bytes from an upload), delegates
    to the pipeline, and returns a structured IngestResponse.

    Both routes share the same service — the file vs. text distinction is
    handled by passing optional `file_bytes` / `filename` kwargs that the
    RAGIngestionPipeline understands.
    """

    def __init__(self, pipeline: BaseIngestionPipeline) -> None:
        self._pipeline = pipeline

    async def ingest_document(self, request: IngestRequest) -> IngestResponse:
        """
        Ingest a document from a plain-text JSON request.

        Args:
            request: Parsed IngestRequest containing content and metadata.

        Returns:
            IngestResponse with doc_id, chunks_count, and status message.
        """
        metadata: dict[str, Any] = {**request.metadata}
        if request.source:
            metadata["source"] = request.source

        doc_id, chunks_count = await self._run_pipeline(
            content=request.content,
            metadata=metadata,
        )

        return IngestResponse(
            doc_id=doc_id,
            message="Document ingested successfully.",
            chunks_count=chunks_count,
            metadata=metadata,
        )

    async def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResponse:
        """
        Ingest a document from an uploaded file (PDF or TXT).

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename:   Original filename (determines extraction strategy).
            metadata:   Optional caller-supplied metadata.

        Returns:
            IngestResponse with doc_id, chunks_count, and status message.
        """
        meta: dict[str, Any] = {**(metadata or {}), "source": filename}

        doc_id, chunks_count = await self._run_pipeline(
            content="",
            metadata=meta,
            file_bytes=file_bytes,
            filename=filename,
        )

        return IngestResponse(
            doc_id=doc_id,
            message=f"File '{filename}' ingested successfully.",
            chunks_count=chunks_count,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    async def _run_pipeline(
        self,
        content: str,
        metadata: dict[str, Any],
        file_bytes: bytes | None = None,
        filename: str | None = None,
    ) -> tuple[str, int]:
        """
        Call the pipeline and return (doc_id, chunks_count).

        RAGIngestionPipeline.ingest accepts the extra kwargs for file-based
        ingestion; InMemoryIngestionPipeline ignores them gracefully.
        """
        import inspect

        sig = inspect.signature(self._pipeline.ingest)
        has_file_kwargs = "file_bytes" in sig.parameters

        if has_file_kwargs:
            doc_id = await self._pipeline.ingest(  # type: ignore[call-arg]
                content=content,
                metadata=metadata,
                file_bytes=file_bytes,
                filename=filename,
            )
            # Estimate chunks_count from vector store if pipeline doesn't return it
            # For RAGIngestionPipeline, we infer from metadata stored per chunk
            # Simple proxy: return at least 1 if doc_id is valid
            chunks_count = getattr(self._pipeline, "_last_chunks_count", 1)
        else:
            doc_id = await self._pipeline.ingest(content=content, metadata=metadata)
            chunks_count = 1

        return doc_id, chunks_count
