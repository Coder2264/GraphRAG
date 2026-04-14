"""
Abstract interface for vector storage implementations.

Targets PostgreSQL + pgvector as the primary real backend while keeping
the contract compatible with other ANN stores (Pinecone, Qdrant, Chroma, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """
    Contract for any vector / embedding store.

    Examples: InMemoryVectorStore, PostgresVectorStore, ...
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the vector store."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release connection."""
        ...

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def upsert(
        self,
        doc_id: str,
        vector: list[float] | None,
        metadata: dict[str, Any],
        content: str = "",
    ) -> None:
        """
        Insert or update a document vector.

        Args:
            doc_id:   Unique document identifier.
            vector:   Embedding vector. Pass None to store a text-only row
                      (e.g. graph extraction chunks that don't need ANN search).
            metadata: Arbitrary metadata to store alongside the vector.
            content:  Optional raw text chunk (useful for retrieval display).
        """
        ...

    @abstractmethod
    async def delete(self, doc_id: str) -> None:
        """Remove a document vector by ID."""
        ...

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Nearest-neighbour search.

        Args:
            vector:          Query embedding.
            top_k:           Maximum number of results.
            metadata_filter: Optional key-value filter applied before ranking.

        Returns:
            List of dicts with keys: doc_id, score, content, metadata.
        """
        ...

    @abstractmethod
    async def get(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch a stored document by exact ID, or None."""
        ...
