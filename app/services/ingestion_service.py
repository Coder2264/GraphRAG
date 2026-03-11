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

    Receives a raw IngestRequest, delegates to the pipeline, and returns
    a structured IngestResponse.
    """

    def __init__(self, pipeline: BaseIngestionPipeline) -> None:
        self._pipeline = pipeline

    async def ingest_document(self, request: IngestRequest) -> IngestResponse:
        """
        Ingest a document and return its assigned doc_id.

        Args:
            request: Parsed IngestRequest containing content and metadata.

        Returns:
            IngestResponse with doc_id and status message.
        """
        metadata: dict[str, Any] = {**request.metadata}
        if request.source:
            metadata["source"] = request.source

        doc_id = await self._pipeline.ingest(
            content=request.content,
            metadata=metadata,
        )

        return IngestResponse(
            doc_id=doc_id,
            message="Document ingested successfully.",
            metadata=metadata,
        )
