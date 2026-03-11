"""
Abstract interface for retrieval strategies.

DIP: QueryService depends on BaseRetriever, not on any concrete retriever.
OCP: New retrieval approaches (HybridRetriever, MultiHopRetriever, ...) are
     added by subclassing — no existing code changes needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """Structured output from any retriever."""

    context: str = ""
    sources: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseRetriever(ABC):
    """
    Contract for retrieval strategies used in query pipelines.

    Subclasses:
      - GraphRAGRetriever   — traverses the knowledge graph
      - RAGRetriever        — ANN search over the vector store
      - NoneRetriever       — no retrieval; passes query directly to LLM
    """

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve relevant context for the given query.

        Args:
            query: User's natural-language question.
            top_k: Maximum number of results to fetch (ANN / graph hops).

        Returns:
            RetrievalResult with a context string and source doc IDs.
        """
        ...
