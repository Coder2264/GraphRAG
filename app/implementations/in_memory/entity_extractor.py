"""
InMemoryEntityExtractor — no-op BaseEntityExtractor for local/stub use.

Returns empty entities and relations so the GraphRAG ingestion path
degrades gracefully when no real LLM-backed extractor is configured.
"""

from __future__ import annotations

from app.core.entity_extractor import BaseEntityExtractor


class InMemoryEntityExtractor(BaseEntityExtractor):
    """No-op extractor used when the in_memory graph store is active."""

    async def extract(
        self,
        text: str,
        entity_types: list[dict],
        relation_types: list[dict],
    ) -> dict:
        return {"entities": [], "relations": []}
