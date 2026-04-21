"""
Pydantic request/response models for the human-in-the-loop manual extraction flow.

Mirrors the structure of app/models/manual.py so the two flows feel consistent
to API consumers.  Reuses LLMPromptPayload and SessionStatus from that module.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.models.manual import LLMPromptPayload, SessionStatus


# ---------------------------------------------------------------------------
# Session request/response models
# ---------------------------------------------------------------------------


class StartExtractionRequest(BaseModel):
    """Start a new manual extraction session for a PDF document."""

    source: str
    processing_instruction: str = ""
    doc_id: str | None = None


class ResumeExtractionRequest(BaseModel):
    """Submit the LLM response for the current extraction step."""

    llm_response: str


class ExtractionSessionResponse(BaseModel):
    """
    Returned by POST /sessions (start) and POST /sessions/{id}/resume.

    When status == "needs_llm": copy llm_prompt into your LLM webapp (with the
    PDF already uploaded), then POST the response to resume_endpoint.
    When status == "complete": final_summary contains ingestion counts.
    """

    session_id: str
    doc_id: str
    status: SessionStatus
    step_name: str
    step_number: int
    llm_prompt: LLMPromptPayload | None
    resume_endpoint: str | None
    accumulated_context: dict[str, Any]
    final_summary: dict[str, Any] | None


class ExtractionSessionInfoResponse(BaseModel):
    """Returned by GET /sessions/{id} — current state snapshot, no LLM prompt."""

    session_id: str
    doc_id: str
    source: str
    processing_instruction: str
    status: str
    step: str
    step_number: int
    accumulated_context: dict[str, Any]
    created_at: str
