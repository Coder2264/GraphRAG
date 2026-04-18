"""
Human-in-the-loop manual query router.

Exposes session endpoints (start / resume / inspect) and stateless tool
endpoints (vector search, Neo4j queries, chunk lookup) so that all LLM
calls in RAG, GraphRAG, and ToG can be performed manually via ChatGPT or
Gemini instead of hitting the Ollama API.

OCP: Adding new modes requires only changes to ManualSessionService, not here.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.embedder import BaseEmbedder
from app.core.graph_store import BaseGraphStore
from app.core.vector_store import BaseVectorStore
from app.dependencies import get_embedder, get_graph_store, get_vector_store
from app.models.manual import (
    ChunkLookupRequest,
    HeadEntitiesRequest,
    NodeSearchRequest,
    RelationsRequest,
    SessionInfoResponse,
    SessionResponse,
    StartSessionRequest,
    SubgraphRequest,
    TailEntitiesRequest,
    ResumeSessionRequest,
    VectorSearchRequest,
)
from app.services.manual_session_service import (
    ManualSessionService,
    get_manual_session_service,
)

router = APIRouter(prefix="/manual", tags=["manual"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def _get_svc() -> ManualSessionService:
    return get_manual_session_service()


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Start a new manual session",
    description=(
        "Creates a session for the given query and mode, runs all automatic steps "
        "(embedding, Neo4j searches), then returns the first LLM prompt. "
        "Copy system_prompt + user_prompt into ChatGPT / Gemini, then POST the response "
        "to the resume endpoint."
    ),
)
async def start_session(
    request: StartSessionRequest,
    svc: Annotated[ManualSessionService, Depends(_get_svc)],
    graph_store: Annotated[BaseGraphStore, Depends(get_graph_store)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
    embedder: Annotated[BaseEmbedder, Depends(get_embedder)],
) -> SessionResponse:
    session = svc.create_session(request.query, request.mode.value, request.top_k)
    try:
        return await svc.advance(
            session.session_id, None, graph_store, vector_store, embedder
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.post(
    "/sessions/{session_id}/resume",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit LLM response and advance to the next step",
    description=(
        "Paste the text you got from ChatGPT / Gemini into llm_response. "
        "The session will run the next automatic steps (Neo4j / vector searches) "
        "and return the next LLM prompt, or status=complete with the final_answer."
    ),
)
async def resume_session(
    session_id: str,
    request: ResumeSessionRequest,
    svc: Annotated[ManualSessionService, Depends(_get_svc)],
    graph_store: Annotated[BaseGraphStore, Depends(get_graph_store)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
    embedder: Annotated[BaseEmbedder, Depends(get_embedder)],
) -> SessionResponse:
    try:
        return await svc.advance(
            session_id, request.llm_response, graph_store, vector_store, embedder
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


@router.get(
    "/sessions/{session_id}",
    response_model=SessionInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect current session state",
    description="Returns session metadata and accumulated context. Does not advance the session.",
)
async def get_session(
    session_id: str,
    svc: Annotated[ManualSessionService, Depends(_get_svc)],
) -> SessionInfoResponse:
    try:
        return svc.session_info(session_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id!r} not found or expired (TTL = 1 hour).",
        )


# ---------------------------------------------------------------------------
# Tool endpoints (stateless — no session required)
# ---------------------------------------------------------------------------


@router.post(
    "/tools/vector-search",
    status_code=status.HTTP_200_OK,
    summary="Embed a query and run pgvector ANN search",
)
async def tool_vector_search(
    request: VectorSearchRequest,
    embedder: Annotated[BaseEmbedder, Depends(get_embedder)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
) -> dict[str, Any]:
    vector = await embedder.embed(request.query)
    results = await vector_store.search(vector, top_k=request.top_k)
    return {
        "chunks": [
            {
                "content": r.get("content", ""),
                "score": float(r.get("score", 0)),
                "doc_id": r.get("doc_id", ""),
            }
            for r in results
        ]
    }


@router.post(
    "/tools/node-search",
    status_code=status.HTTP_200_OK,
    summary="Neo4j fulltext entity search",
)
async def tool_node_search(
    request: NodeSearchRequest,
    graph_store: Annotated[BaseGraphStore, Depends(get_graph_store)],
) -> dict[str, Any]:
    nodes = await graph_store.search_nodes(request.keyword, top_k=request.top_k)
    return {"nodes": nodes}


@router.post(
    "/tools/subgraph",
    status_code=status.HTTP_200_OK,
    summary="Expand the neighbourhood subgraph around a node",
)
async def tool_subgraph(
    request: SubgraphRequest,
    graph_store: Annotated[BaseGraphStore, Depends(get_graph_store)],
) -> dict[str, Any]:
    return await graph_store.get_subgraph(request.node_id, depth=request.depth)


@router.post(
    "/tools/relations",
    status_code=status.HTTP_200_OK,
    summary="Get all unique relation labels connected to an entity",
)
async def tool_relations(
    request: RelationsRequest,
    graph_store: Annotated[BaseGraphStore, Depends(get_graph_store)],
) -> dict[str, Any]:
    relations = await graph_store.get_relations(request.entity_id)
    return {"relations": relations}


@router.post(
    "/tools/tail-entities",
    status_code=status.HTTP_200_OK,
    summary="Get entities reachable FROM entity_id via an outgoing relation",
)
async def tool_tail_entities(
    request: TailEntitiesRequest,
    graph_store: Annotated[BaseGraphStore, Depends(get_graph_store)],
) -> dict[str, Any]:
    entities = await graph_store.get_tail_entities(request.entity_id, request.relation)
    return {"entities": entities}


@router.post(
    "/tools/head-entities",
    status_code=status.HTTP_200_OK,
    summary="Get entities pointing TO entity_id via an incoming relation",
)
async def tool_head_entities(
    request: HeadEntitiesRequest,
    graph_store: Annotated[BaseGraphStore, Depends(get_graph_store)],
) -> dict[str, Any]:
    entities = await graph_store.get_head_entities(request.entity_id, request.relation)
    return {"entities": entities}


@router.post(
    "/tools/chunk-lookup",
    status_code=status.HTTP_200_OK,
    summary="Fetch raw chunk text by chunk_id from the vector store",
)
async def tool_chunk_lookup(
    request: ChunkLookupRequest,
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
) -> dict[str, Any]:
    result = await vector_store.get(request.chunk_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk {request.chunk_id!r} not found.",
        )
    return {
        "content": result.get("content", ""),
        "metadata": result.get("metadata", {}),
    }
