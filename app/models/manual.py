"""
Pydantic request/response models for the human-in-the-loop manual query flow.

Provides session management models (start, resume, inspect) and stateless
tool endpoint models (vector search, graph queries, chunk lookup).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ManualQueryMode(str, Enum):
    RAG = "rag"
    GRAPHRAG = "graphrag"
    TOG = "tog"


class SessionStatus(str, Enum):
    NEEDS_LLM = "needs_llm"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# Session request/response models
# ---------------------------------------------------------------------------


class StartSessionRequest(BaseModel):
    """Start a new manual session for the given query and mode."""

    query: str
    mode: ManualQueryMode
    top_k: int = 5


class ResumeSessionRequest(BaseModel):
    """Submit the LLM response for the current step to advance the session."""

    llm_response: str


class LLMPromptPayload(BaseModel):
    """The exact prompts to paste into ChatGPT / Gemini for the current step."""

    step_name: str
    system_prompt: str
    user_prompt: str
    response_format: str


class SessionResponse(BaseModel):
    """
    Returned by POST /sessions (start) and POST /sessions/{id}/resume.

    When status == "needs_llm": copy llm_prompt into your LLM webapp, then
    POST the response to resume_endpoint.
    When status == "complete": final_answer contains the answer.
    """

    session_id: str
    status: SessionStatus
    step_name: str
    step_number: int
    llm_prompt: LLMPromptPayload | None
    resume_endpoint: str | None
    accumulated_context: dict[str, Any]
    final_answer: str | None


class SessionInfoResponse(BaseModel):
    """Returned by GET /sessions/{id} — current state snapshot, no LLM prompt."""

    session_id: str
    mode: str
    query: str
    status: str
    step: str
    step_number: int
    accumulated_context: dict[str, Any]
    created_at: str


# ---------------------------------------------------------------------------
# Tool endpoint models
# ---------------------------------------------------------------------------


class VectorSearchRequest(BaseModel):
    """Embed query and run pgvector ANN search."""

    query: str
    top_k: int = 5


class NodeSearchRequest(BaseModel):
    """Neo4j fulltext search over entity names/descriptions."""

    keyword: str
    top_k: int = 5


class SubgraphRequest(BaseModel):
    """Expand the neighbourhood subgraph around a node."""

    node_id: str
    depth: int = 1


class RelationsRequest(BaseModel):
    """Get all unique relation labels connected to an entity."""

    entity_id: str


class TailEntitiesRequest(BaseModel):
    """Get entities reachable FROM entity_id via an outgoing relation."""

    entity_id: str
    relation: str


class HeadEntitiesRequest(BaseModel):
    """Get entities pointing TO entity_id via an incoming relation."""

    entity_id: str
    relation: str


class ChunkLookupRequest(BaseModel):
    """Fetch raw chunk text by chunk_id from the vector store."""

    chunk_id: str
