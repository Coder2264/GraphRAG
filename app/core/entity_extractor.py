"""
Abstract interface for entity extraction from text.

Implementors receive a chunk of text along with the domain-specific entity
and relation types (loaded from Postgres at runtime) and return a structured
dict of extracted entities and relations ready to be persisted to Neo4j.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEntityExtractor(ABC):
    """
    Contract for any LLM-backed entity/relationship extractor.

    The caller supplies the allowed entity and relation type catalogs so the
    implementation can embed them into its prompt.  This keeps the extractor
    decoupled from the database layer.
    """

    @abstractmethod
    async def extract(
        self,
        text: str,
        entity_types: list[dict],   # [{"name": str, "description": str}, ...]
        relation_types: list[dict], # [{"name": str, "description": str}, ...]
    ) -> dict:
        """
        Extract entities and relationships from *text*.

        Args:
            text:           The document text (or chunk) to analyse.
            entity_types:   Allowed entity type descriptors from Postgres.
            relation_types: Allowed relation type descriptors from Postgres.

        Returns:
            A dict with two keys:
              "entities"  — list of {id, name, type, description}
              "relations" — list of {src_id, dst_id, relation, properties}

        Implementations should return empty lists on parse errors rather than
        raising, so the ingestion pipeline can degrade gracefully.
        """
        ...
