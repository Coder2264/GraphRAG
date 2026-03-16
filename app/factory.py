"""
ServiceFactory — constructs and wires concrete implementations.

DIP: Services receive dependencies through the factory; they never
     instantiate concrete classes themselves.
OCP: New providers are supported by updating registry.py, not this file.
"""

from __future__ import annotations

from typing import Optional

from app.config import settings
from app.core.embedder import BaseEmbedder
from app.core.entity_extractor import BaseEntityExtractor
from app.core.graph_store import BaseGraphStore
from app.core.ingestion import BaseIngestionPipeline
from app.core.llm import BaseLLM
from app.core.retriever import BaseRetriever
from app.core.vector_store import BaseVectorStore
from app.implementations.document_processor import DefaultDocumentProcessor
from app.implementations.graph_rag.ingestion import GraphRAGIngestionPipeline
from app.implementations.in_memory.ingestion import InMemoryIngestionPipeline
from app.implementations.in_memory.retrievers import (
    GraphRAGRetriever,
    NoneRetriever,
    RAGRetriever,
)
from app.implementations.rag.ingestion import RAGIngestionPipeline
from app.models.query import QueryMode
from app.registry import (
    EMBEDDER_REGISTRY,
    ENTITY_EXTRACTOR_REGISTRY,
    GRAPH_STORE_REGISTRY,
    LLM_REGISTRY,
    VECTOR_STORE_REGISTRY,
)


class ServiceFactory:
    """
    Wires up the dependency graph and exposes factory methods for services.

    Usage:
        factory = ServiceFactory()
        await factory.startup()        # call on app lifespan start
        await factory.shutdown()       # call on app lifespan end
    """

    def __init__(
        self,
        llm_key: Optional[str] = None,
        embedder_key: Optional[str] = None,
        graph_store_key: Optional[str] = None,
        vector_store_key: Optional[str] = None,
        entity_extractor_key: Optional[str] = None,
    ) -> None:
        # Fall back to settings defaults if not explicitly overridden
        self._llm_key = llm_key or settings.default_llm
        self._embedder_key = embedder_key or settings.default_embedder
        self._graph_store_key = graph_store_key or settings.default_graph_store
        self._vector_store_key = vector_store_key or settings.default_vector_store
        self._entity_extractor_key = entity_extractor_key or settings.default_entity_extractor

        # Shared singleton instances (avoids re-creating pools on every request)
        self._llm: BaseLLM | None = None
        self._embedder: BaseEmbedder | None = None
        self._graph_store: BaseGraphStore | None = None
        self._vector_store: BaseVectorStore | None = None
        self._entity_extractor: BaseEntityExtractor | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Build all singletons and open any persistent connections."""
        self._llm = self._build_llm()
        self._embedder = self._build_embedder()
        self._graph_store = self._build_graph_store()
        self._vector_store = self._build_vector_store()
        self._entity_extractor = self._build_entity_extractor()

        await self._graph_store.connect()
        await self._vector_store.connect()

    async def shutdown(self) -> None:
        """Gracefully close connections."""
        if self._graph_store:
            await self._graph_store.close()
        if self._vector_store:
            await self._vector_store.close()

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_llm(self) -> BaseLLM:
        cls = LLM_REGISTRY[self._llm_key]
        if self._llm_key == "ollama":
            return cls(  # type: ignore[call-arg]
                model_name=settings.ollama_llm_model,
                base_url=settings.ollama_base_url,
            )
        return cls()

    def _build_embedder(self) -> BaseEmbedder:
        cls = EMBEDDER_REGISTRY[self._embedder_key]
        if self._embedder_key == "ollama":
            return cls(  # type: ignore[call-arg]
                model_name=settings.ollama_embed_model,
                base_url=settings.ollama_base_url,
            )
        return cls()

    def _build_graph_store(self) -> BaseGraphStore:
        cls = GRAPH_STORE_REGISTRY[self._graph_store_key]
        if self._graph_store_key == "neo4j":
            return cls(  # type: ignore[call-arg]
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password,
            )
        return cls()

    def _build_vector_store(self) -> BaseVectorStore:
        cls = VECTOR_STORE_REGISTRY[self._vector_store_key]
        if self._vector_store_key == "postgres":
            return cls(  # type: ignore[call-arg]
                dsn=settings.postgres_dsn,
                embedding_dim=settings.embedding_dim,
            )
        return cls()

    def _build_entity_extractor(self) -> BaseEntityExtractor:
        """
        Build the entity extractor from the registry.

        Args:
            None — reads self._entity_extractor_key set during __init__.

        Returns:
            A fully constructed BaseEntityExtractor instance.
        """
        cls = ENTITY_EXTRACTOR_REGISTRY[self._entity_extractor_key]
        if self._entity_extractor_key == "ollama":
            return cls(  # type: ignore[call-arg]
                model_name=settings.ollama_llm_model,
                base_url=settings.ollama_base_url,
            )
        return cls()

    # ------------------------------------------------------------------
    # Service getters (called by FastAPI Depends)
    # ------------------------------------------------------------------

    def get_ingestion_pipeline(self) -> BaseIngestionPipeline:
        assert self._embedder and self._vector_store and self._graph_store
        # Use the real RAG pipeline when postgres vector store is active
        if self._vector_store_key == "postgres":
            return RAGIngestionPipeline(
                document_processor=DefaultDocumentProcessor(),
                embedder=self._embedder,
                vector_store=self._vector_store,
            )
        # Fallback: in-memory stub (also wires graph store for GraphRAG path)
        return InMemoryIngestionPipeline(
            embedder=self._embedder,
            vector_store=self._vector_store,
            graph_store=self._graph_store,
        )

    def get_graph_rag_ingestion_pipeline(self) -> BaseIngestionPipeline | None:
        """
        Build the GraphRAG ingestion pipeline if a real graph store is active.

        Returns None when using the in-memory graph store so that
        IngestionService simply skips the graph branch.
        """
        assert self._entity_extractor and self._graph_store
        if self._graph_store_key == "neo4j":
            return GraphRAGIngestionPipeline(
                document_processor=DefaultDocumentProcessor(),
                entity_extractor=self._entity_extractor,
                graph_store=self._graph_store,
                postgres_dsn=settings.postgres_dsn,
            )
        # in_memory graph store — skip real extraction
        return None

    def get_retriever(self, mode: QueryMode) -> BaseRetriever:
        assert self._embedder and self._vector_store and self._graph_store
        if mode == QueryMode.GRAPHRAG:
            return GraphRAGRetriever(
                graph_store=self._graph_store,
                embedder=self._embedder,
            )
        if mode == QueryMode.RAG:
            return RAGRetriever(
                vector_store=self._vector_store,
                embedder=self._embedder,
            )
        return NoneRetriever()

    def get_llm(self) -> BaseLLM:
        assert self._llm
        return self._llm
