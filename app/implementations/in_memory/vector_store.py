"""
InMemoryVectorStore — in-process stub vector store.

Uses brute-force cosine similarity. Useful for tests and local dev
before PostgreSQL + pgvector is available.
"""

from __future__ import annotations

import math
from typing import Any

from app.core.vector_store import BaseVectorStore


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class InMemoryVectorStore(BaseVectorStore):
    """
    Vector store backed by a Python list — no external dependencies.

    Performs brute-force ANN search; suitable for small datasets only.
    """

    def __init__(self) -> None:
        # doc_id -> {vector, content, metadata}
        self._store: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """No-op."""

    async def close(self) -> None:
        """No-op."""

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert(
        self,
        doc_id: str,
        vector: list[float] | None,
        metadata: dict[str, Any],
        content: str = "",
    ) -> None:
        self._store[doc_id] = {"vector": vector, "content": content, "metadata": metadata}

    async def delete(self, doc_id: str) -> None:
        self._store.pop(doc_id, None)

    async def clear(self) -> None:
        self._store.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search(
        self,
        vector: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Brute-force cosine similarity search with optional metadata filtering."""
        candidates = []
        for doc_id, entry in self._store.items():
            # Skip text-only rows (no embedding vector)
            if entry["vector"] is None:
                continue
            if metadata_filter:
                if not all(entry["metadata"].get(k) == v for k, v in metadata_filter.items()):
                    continue
            score = _cosine_similarity(vector, entry["vector"])
            candidates.append(
                {
                    "doc_id": doc_id,
                    "score": score,
                    "content": entry["content"],
                    "metadata": entry["metadata"],
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        entry = self._store.get(doc_id)
        if entry is None:
            return None
        return {"doc_id": doc_id, **entry}
