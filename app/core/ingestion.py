"""
Abstract interface for document ingestion pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseIngestionPipeline(ABC):
    """
    Contract for end-to-end document ingestion.

    An ingestion pipeline is responsible for:
      1. Chunking the raw content.
      2. Embedding each chunk.
      3. Persisting chunks to the vector store.
      4. Optionally extracting entities/relations and persisting them to the graph store.

    Implementations: InMemoryIngestionPipeline, ProductionIngestionPipeline, ...
    """

    @abstractmethod
    async def ingest(self, content: str, metadata: dict[str, Any]) -> str:
        """
        Ingest a document and return its assigned doc_id.

        Args:
            content:  Raw document text.
            metadata: Descriptive metadata (source, title, author, etc.).

        Returns:
            doc_id — unique identifier for the stored document.
        """
        ...

    @abstractmethod
    async def delete(self, doc_id: str) -> None:
        """
        Remove all stored artifacts for a previously ingested document.

        Args:
            doc_id: ID returned by a prior `ingest` call.
        """
        ...
