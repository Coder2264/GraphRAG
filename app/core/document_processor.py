"""
Abstract interface for document processing (text extraction + chunking).

SRP: Separates document I/O concerns from embedding and storage.
OCP: New formats (DOCX, HTML) are added by subclassing, not modifying.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDocumentProcessor(ABC):
    """
    Contract for extracting text from raw documents and splitting into chunks.

    Supported file types are implementation-defined; callers pass `filename`
    to let the processor dispatch on extension.
    """

    @abstractmethod
    async def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """
        Extract plain text from the raw file bytes.

        Args:
            file_bytes: Raw binary content of the file.
            filename:   Original filename (used to determine file type via extension).

        Returns:
            Plain text string, with leading/trailing whitespace stripped.

        Raises:
            ValueError: If the file format is unsupported.
        """
        ...

    @abstractmethod
    def chunk_text(self, text: str) -> list[str]:
        """
        Split plain text into overlapping character-level chunks.

        Args:
            text: The full document text.

        Returns:
            Ordered list of non-empty text chunks.
        """
        ...
