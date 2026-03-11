"""
POST /api/v1/ingest — Document ingestion endpoint.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.dependencies import get_ingestion_service
from app.models.ingest import IngestRequest, IngestResponse
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a document",
    description=(
        "Chunk, embed, and store a document in both the vector store (for RAG) "
        "and the graph store (for GraphRAG)."
    ),
)
async def ingest_document(
    request: IngestRequest,
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> IngestResponse:
    return await service.ingest_document(request)
