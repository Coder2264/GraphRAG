"""
Provider Registry — maps string keys to concrete implementation classes.

OCP: Add a new provider by inserting an entry here — no other files change.
DIP: ServiceFactory and routes depend on these registries, not on concrete classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.implementations.in_memory.embedder import InMemoryEmbedder
from app.implementations.in_memory.graph_store import InMemoryGraphStore
from app.implementations.in_memory.llm import InMemoryLLM
from app.implementations.in_memory.vector_store import InMemoryVectorStore
from app.implementations.neo4j.graph_store import Neo4jGraphStore
from app.implementations.postgres.vector_store import PostgresVectorStore

if TYPE_CHECKING:
    from app.core.embedder import BaseEmbedder
    from app.core.graph_store import BaseGraphStore
    from app.core.llm import BaseLLM
    from app.core.vector_store import BaseVectorStore


# ------------------------------------------------------------------
# LLM registry
# ------------------------------------------------------------------
LLM_REGISTRY: dict[str, type[BaseLLM]] = {
    "in_memory": InMemoryLLM,
    # "openai":     OpenAILLM,       ← uncomment when ready
    # "anthropic":  AnthropicLLM,
    # "ollama":     OllamaLLM,
}

# ------------------------------------------------------------------
# Embedder registry
# ------------------------------------------------------------------
EMBEDDER_REGISTRY: dict[str, type[BaseEmbedder]] = {
    "in_memory": InMemoryEmbedder,
    # "openai":   OpenAIEmbedder,
    # "cohere":   CohereEmbedder,
}

# ------------------------------------------------------------------
# Graph store registry
# ------------------------------------------------------------------
GRAPH_STORE_REGISTRY: dict[str, type[BaseGraphStore]] = {
    "in_memory": InMemoryGraphStore,
    "neo4j":     Neo4jGraphStore,
}

# ------------------------------------------------------------------
# Vector store registry
# ------------------------------------------------------------------
VECTOR_STORE_REGISTRY: dict[str, type[BaseVectorStore]] = {
    "in_memory": InMemoryVectorStore,
    "postgres":  PostgresVectorStore,
}
