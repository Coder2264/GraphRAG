"""
Abstract interface for entity extraction from text.

Implementors receive a chunk of text and an optional free-text
processing_instruction, and return a structured dict of extracted entities
and relations ready to be persisted to Neo4j.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEntityExtractor(ABC):
    """
    Contract for any LLM-backed entity/relationship extractor.

    The caller supplies a free-text processing_instruction that guides what
    to extract (domain, entity focus, etc.).  This keeps the extractor
    decoupled from any database catalog layer.
    """

    @abstractmethod
    async def extract(
        self,
        text: str,
        processing_instruction: str = "",
    ) -> dict:
        """
        Extract entities and relationships from *text*.

        Args:
            text:                   The document text (or chunk) to analyse.
            processing_instruction: Optional free-text hint guiding extraction
                                    (e.g. domain, entity types to focus on).

        Returns:
            A dict with two keys:
              "entities"  — list of {id, name, type, description}
              "relations" — list of {src_id, dst_id, relation, properties}

        Implementations should return empty lists on parse errors rather than
        raising, so the ingestion pipeline can degrade gracefully.
        """
        ...
