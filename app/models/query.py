"""
Pydantic models for query endpoints.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QueryMode(str, Enum):
    """Supported query strategies."""

    GRAPHRAG = "graphrag"
    RAG = "rag"
    NONE = "none"
    TOG = "tog"
    TOG_R = "tog_r"


class QueryRequest(BaseModel):
    """Payload for all query routes."""

    question: str = Field(..., description="The user's question.")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to retrieve (RAG/GraphRAG).")
    metadata_filter: dict = Field(default_factory=dict, description="Optional metadata filters.")


class QueryResponse(BaseModel):
    """Standard response for all query modes."""

    question: str
    mode: QueryMode
    answer: str = Field(..., description="LLM-generated answer.")
    context: str = Field("", description="Retrieved context used to generate the answer.")
    sources: list[str] = Field(default_factory=list, description="Source doc IDs used.")
    elapsed_seconds: float = Field(..., description="Wall-clock seconds from request receipt to response.")
