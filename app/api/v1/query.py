"""
POST /api/v1/query/{mode} — Unified query endpoint for all three modes.

Route parameter `mode` drives which retriever is injected (GraphRAG / RAG / None).
No conditional logic in the handler — pure DI.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.dependencies import get_query_service_for_mode
from app.models.query import QueryMode, QueryRequest, QueryResponse
from app.services.query_service import QueryService

router = APIRouter(prefix="/query", tags=["query"])

# ------------------------------------------------------------------
# One route per mode — each binds its own QueryService via DI.
# This avoids a runtime if/elif dispatch and keeps each endpoint
# independently testable and documentable in Swagger.
# ------------------------------------------------------------------

@router.post(
    "/graphrag",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query using GraphRAG",
    description=(
        "Retrieves context by traversing the knowledge graph, then generates "
        "an answer grounded in graph-derived context."
    ),
)
async def query_graphrag(
    request: QueryRequest,
    service: Annotated[QueryService, Depends(get_query_service_for_mode(QueryMode.GRAPHRAG))],
) -> QueryResponse:
    return await service.answer(request, mode=QueryMode.GRAPHRAG)


@router.post(
    "/rag",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query using RAG",
    description=(
        "Performs ANN search over the vector store to retrieve relevant chunks, "
        "then generates an answer grounded in those chunks."
    ),
)
async def query_rag(
    request: QueryRequest,
    service: Annotated[QueryService, Depends(get_query_service_for_mode(QueryMode.RAG))],
) -> QueryResponse:
    return await service.answer(request, mode=QueryMode.RAG)


@router.post(
    "/none",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query without retrieval (LLM only)",
    description=(
        "Sends the question directly to the LLM with no retrieved context. "
        "Useful as a baseline against RAG and GraphRAG."
    ),
)
async def query_none(
    request: QueryRequest,
    service: Annotated[QueryService, Depends(get_query_service_for_mode(QueryMode.NONE))],
) -> QueryResponse:
    return await service.answer(request, mode=QueryMode.NONE)


@router.post(
    "/tog",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query using Think-on-Graph (ToG)",
    description=(
        "Runs the Think-on-Graph algorithm (Sun et al., ICLR 2024): iteratively "
        "prunes relations and entities on the knowledge graph via LLM scoring, "
        "then generates an answer from the accumulated reasoning paths."
    ),
)
async def query_tog(
    request: QueryRequest,
    service: Annotated[QueryService, Depends(get_query_service_for_mode(QueryMode.TOG))],
) -> QueryResponse:
    return await service.answer(request, mode=QueryMode.TOG)


@router.post(
    "/tog_r",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query using Think-on-Graph with Random pruning (ToG-R)",
    description=(
        "Variant of ToG where entity pruning (Step D) uses random sampling instead "
        "of LLM scoring, halving the number of LLM calls while preserving the "
        "relation-pruning and reasoning-check steps."
    ),
)
async def query_tog_r(
    request: QueryRequest,
    service: Annotated[QueryService, Depends(get_query_service_for_mode(QueryMode.TOG_R))],
) -> QueryResponse:
    return await service.answer(request, mode=QueryMode.TOG_R)
