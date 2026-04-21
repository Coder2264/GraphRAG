"""
Human-in-the-loop manual extraction router.

Exposes session endpoints (start / resume / inspect) so that the LLM calls
during document ingestion — chunk extraction and entity/relation extraction —
can be performed manually via ChatGPT or Gemini with a PDF uploaded, instead
of requiring a configured LLM API key on the server.

OCP: Adding new extraction steps requires only changes to ManualExtractService,
     not here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.embedder import BaseEmbedder
from app.core.graph_store import BaseGraphStore
from app.core.vector_store import BaseVectorStore
from app.dependencies import get_embedder, get_graph_store, get_vector_store
from app.models.manual_extract import (
    ExtractionSessionInfoResponse,
    ExtractionSessionResponse,
    ResumeExtractionRequest,
    StartExtractionRequest,
)
from app.services.manual_extract_service import (
    ManualExtractService,
    get_manual_extract_service,
)

router = APIRouter(prefix="/manual/extract", tags=["manual-extract"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def _get_svc() -> ManualExtractService:
    return get_manual_extract_service()


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/sessions",
    response_model=ExtractionSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Start a new manual extraction session",
    description=(
        "Creates a session for the given document source. "
        "Returns the first LLM prompt to paste into ChatGPT / Gemini "
        "(with the PDF already uploaded). "
        "Copy system_prompt + user_prompt into your LLM webapp, then POST the "
        "response to the resume endpoint."
    ),
)
async def start_extraction_session(
    request: StartExtractionRequest,
    svc: Annotated[ManualExtractService, Depends(_get_svc)],
    graph_store: Annotated[BaseGraphStore, Depends(get_graph_store)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
    embedder: Annotated[BaseEmbedder, Depends(get_embedder)],
) -> ExtractionSessionResponse:
    session = svc.create_session(
        request.source,
        request.processing_instruction,
        request.doc_id,
    )
    try:
        return await svc.advance(
            session.session_id, None, vector_store, graph_store, embedder
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.post(
    "/sessions/{session_id}/resume",
    response_model=ExtractionSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit LLM response and advance extraction to the next step",
    description=(
        "Paste the text you got from ChatGPT / Gemini into llm_response. "
        "After the chunk extraction step the server embeds and stores the chunks. "
        "After the entity extraction step the server writes entities and relations "
        "to Neo4j and returns status=complete with a final_summary."
    ),
)
async def resume_extraction_session(
    session_id: str,
    request: ResumeExtractionRequest,
    svc: Annotated[ManualExtractService, Depends(_get_svc)],
    graph_store: Annotated[BaseGraphStore, Depends(get_graph_store)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
    embedder: Annotated[BaseEmbedder, Depends(get_embedder)],
) -> ExtractionSessionResponse:
    try:
        return await svc.advance(
            session_id, request.llm_response, vector_store, graph_store, embedder
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id!r} not found or expired (TTL = 1 hour).",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during extraction step: {exc}",
        )


@router.get(
    "/sessions/{session_id}",
    response_model=ExtractionSessionInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect current extraction session state",
    description="Returns session metadata and accumulated context. Does not advance the session.",
)
async def get_extraction_session(
    session_id: str,
    svc: Annotated[ManualExtractService, Depends(_get_svc)],
) -> ExtractionSessionInfoResponse:
    try:
        return svc.session_info(session_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id!r} not found or expired (TTL = 1 hour).",
        )
