"""
Pydantic models for ingestion endpoints.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Payload for document ingestion (JSON body route)."""

    content: str = Field(..., description="Raw text / document content to ingest.")
    metadata: dict = Field(default_factory=dict, description="Arbitrary key-value metadata.")
    source: Optional[str] = Field(None, description="Optional source identifier (URL, file path, etc.)")


class IngestResponse(BaseModel):
    """Response returned after successful ingestion."""

    doc_id: str = Field(..., description="Unique identifier assigned to the ingested document.")
    message: str = Field("Document ingested successfully.", description="Human-readable status.")
    chunks_count: int = Field(0, description="Number of chunks stored in the vector store.")
    metadata: dict = Field(default_factory=dict)
