"""
RAGIngestionPipeline — production ingestion pipeline for the RAG path.

Pipeline:
  1. Extract plain text from the raw document bytes (PDF / TXT)
  2. Chunk the text with overlapping sliding window
  3. Embed each chunk via BaseEmbedder
  4. Upsert every chunk into BaseVectorStore

SRP: Each step is delegated to a focused collaborator.
DIP: Depends on base interfaces, not concrete classes.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.core.document_processor import BaseDocumentProcessor
from app.core.embedder import BaseEmbedder
from app.core.ingestion import BaseIngestionPipeline
from app.core.vector_store import BaseVectorStore


class RAGIngestionPipeline(BaseIngestionPipeline):
    """
    Ingestion pipeline that chunks, embeds, and stores documents for RAG.

    Each chunk is stored as a separate vector with metadata that includes
    the parent `doc_id`, `chunk_index`, and any caller-supplied metadata.

    The returned doc_id is the root document UUID; individual chunk IDs
    follow the pattern `{doc_id}__chunk_{i}`.
    """

    def __init__(
        self,
        document_processor: BaseDocumentProcessor,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ) -> None:
        self._processor = document_processor
        self._embedder = embedder
        self._vector_store = vector_store

    async def ingest(
        self,
        content: str,
        metadata: dict[str, Any],
        *,
        file_bytes: bytes | None = None,
        filename: str | None = None,
        doc_id: str | None = None,          # shared UUID from IngestionService
    ) -> str:
        """
        Ingest a document and return its root doc_id.

        If `file_bytes` and `filename` are provided, text is extracted from
        the binary file (PDF or TXT). Otherwise `content` is used directly.

        Args:
            content:    Raw text (used when file_bytes is not provided).
            metadata:   Metadata to attach to every chunk.
            file_bytes: Raw file bytes (optional, triggers extraction).
            filename:   Original filename (determines extraction strategy).
            doc_id:     Optional pre-generated UUID (from IngestionService).
                        When omitted a fresh UUID is generated here.

        Returns:
            Root doc_id (UUID string).
        """
        doc_id = doc_id or str(uuid.uuid4())

        # 1. Extract text (file upload path) or use content directly
        if file_bytes is not None and filename is not None:
            text = await self._processor.extract_text(file_bytes, filename)
        else:
            text = content

        if not text.strip():
            raise ValueError("Document is empty after text extraction.")

        # 2. Chunk
        chunks = self._processor.chunk_text(text)
        if not chunks:
            chunks = [text]  # edge case: text shorter than chunk_size

        # 3. Embed + 4. Upsert each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}__chunk_{i}"
            vector = await self._embedder.embed(chunk)
            chunk_metadata = {
                **metadata,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            await self._vector_store.upsert(
                doc_id=chunk_id,
                vector=vector,
                metadata=chunk_metadata,
                content=chunk,
            )

        # Track for IngestionService to report in the response
        self._last_chunks_count = len(chunks)

        return doc_id

    async def delete(self, doc_id: str) -> None:
        """
        Delete all chunks that belong to the given root doc_id.

        For PostgresVectorStore this uses the efficient `delete_by_doc_id`
        method (a single DELETE WHERE doc_id = $1).
        For other stores (e.g. InMemoryVectorStore) we fall back gracefully.
        """
        if hasattr(self._vector_store, "delete_by_doc_id"):
            await self._vector_store.delete_by_doc_id(doc_id)  # type: ignore[attr-defined]
        else:
            # Fallback for stores that don't know about the parent/chunk split
            await self._vector_store.delete(doc_id)

