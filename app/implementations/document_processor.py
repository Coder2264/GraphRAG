"""
DefaultDocumentProcessor — concrete document processor for PDF and TXT files.

Handles:
  - PDF: text extracted page-by-page using pypdf (no OCR)
  - TXT: raw UTF-8 / latin-1 decode

Chunking: character-level sliding window driven by config settings.
"""

from __future__ import annotations

import io

from app.config import settings
from app.core.document_processor import BaseDocumentProcessor


class DefaultDocumentProcessor(BaseDocumentProcessor):
    """
    Extracts plain text and splits into overlapping character chunks.

    Supported extensions: .pdf, .txt
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    async def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """
        Dispatch to the correct extractor based on file extension.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename:   Original filename (e.g. "report.pdf").

        Returns:
            Cleaned plain-text string.

        Raises:
            ValueError: For unsupported file types.
        """
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext == "pdf":
            return self._extract_pdf(file_bytes)
        elif ext == "txt":
            return self._extract_txt(file_bytes)
        else:
            raise ValueError(
                f"Unsupported file type '.{ext}'. Supported: .pdf, .txt"
            )

    def _extract_pdf(self, file_bytes: bytes) -> str:
        """Extract text from a PDF using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for PDF extraction. "
                "Run: pip install pypdf"
            ) from exc

        reader = PdfReader(io.BytesIO(file_bytes))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)

    def _extract_txt(self, file_bytes: bytes) -> str:
        """Decode a plain-text file (UTF-8 with latin-1 fallback)."""
        try:
            return file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1").strip()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping character-level chunks.

        Uses a sliding window of size `chunk_size` with `chunk_overlap`
        characters of overlap between adjacent chunks.

        Args:
            text: Full document text.

        Returns:
            List of non-empty chunk strings, in document order.
        """
        if not text:
            return []

        size = self._chunk_size
        overlap = self._chunk_overlap
        step = size - overlap

        if step <= 0:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be less than chunk_size ({size})."
            )

        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunk = text[start : start + size].strip()
            if chunk:
                chunks.append(chunk)
            start += step

        return chunks
