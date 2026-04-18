"""
FastAPI dependency injection helpers.

Uses a module-level factory singleton so that connection pools are shared
across requests rather than re-created per request.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.core.graph_store import BaseGraphStore
from app.core.vector_store import BaseVectorStore
from app.factory import ServiceFactory
from app.models.query import QueryMode
from app.services.ingestion_service import IngestionService
from app.services.query_service import QueryService

# Module-level singleton — populated at app startup via lifespan
_factory: ServiceFactory | None = None


def get_factory() -> ServiceFactory:
    assert _factory is not None, "ServiceFactory not initialised. Was startup() called?"
    return _factory


def set_factory(factory: ServiceFactory) -> None:
    """Called during app lifespan to register the singleton."""
    global _factory
    _factory = factory


# ------------------------------------------------------------------
# Typed dependencies for route handlers
# ------------------------------------------------------------------

def get_ingestion_service(
    factory: Annotated[ServiceFactory, Depends(get_factory)],
) -> IngestionService:
    return IngestionService(
        pipeline=factory.get_ingestion_pipeline(),
        graph_rag_pipeline=factory.get_graph_rag_ingestion_pipeline(),
    )


def get_vector_store(
    factory: Annotated[ServiceFactory, Depends(get_factory)],
) -> BaseVectorStore:
    return factory.get_vector_store()


def get_graph_store(
    factory: Annotated[ServiceFactory, Depends(get_factory)],
) -> BaseGraphStore:
    return factory.get_graph_store()


def get_query_service_for_mode(mode: QueryMode):
    """Returns a FastAPI dependency that builds a QueryService for `mode`."""

    def _dep(factory: Annotated[ServiceFactory, Depends(get_factory)]) -> QueryService:
        return QueryService(
            retriever=factory.get_retriever(mode),
            llm=factory.get_llm(),
        )

    return _dep
