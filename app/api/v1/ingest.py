"""
/api/v1/ingest — Document ingestion endpoints.

Two routes:
  POST /api/v1/ingest          — JSON body (raw text content)
  POST /api/v1/ingest/upload   — multipart file upload (PDF or TXT)
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.dependencies import get_ingestion_service
from app.models.ingest import IngestRequest, IngestResponse
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# ---------------------------------------------------------------------------
# Route 1: JSON body — raw text ingestion
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a document (raw text)",
    description=(
        "Chunk, embed, and store a document from a raw text payload. "
        "Use the /upload endpoint instead for PDF or TXT file uploads."
    ),
)
async def ingest_document(
    request: IngestRequest,
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> IngestResponse:
    response = await service.ingest_document(request)
    logger.info(
        "Ingest complete — doc_id=%s  chunks=%d  graph_entities=%d",
        response.doc_id, response.chunks_count, response.graph_entities_count,
    )
    return response


# ---------------------------------------------------------------------------
# Route 2: File upload — PDF or TXT
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a file (PDF or TXT)",
    description=(
        "Upload a PDF or plain-text file. The server extracts text, chunks it, "
        "embeds each chunk via Ollama, and stores vectors in PostgreSQL. "
        "Returns the root doc_id and the number of chunks stored."
    ),
)
async def ingest_file(
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    file: UploadFile = File(..., description="PDF or TXT file to ingest."),
    source: Optional[str] = Form(None, description="Optional source label (URL, path, etc.)."),
) -> IngestResponse:
    allowed = {"pdf", "txt"}
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '.{ext}'. Allowed: {allowed}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    metadata = {"source": source or file.filename}
    response = await service.ingest_file(
        file_bytes=file_bytes,
        filename=file.filename or "upload",
        metadata=metadata,
    )
    logger.info(
        "Ingest complete — file=%s  doc_id=%s  chunks=%d  graph_entities=%d",
        file.filename, response.doc_id, response.chunks_count, response.graph_entities_count,
    )
    return response

