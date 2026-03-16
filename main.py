"""
Application entrypoint — creates the FastAPI app, manages lifespan.

Run with:
    uvicorn main:app --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as v1_router
from app.dependencies import set_factory
from app.factory import ServiceFactory
from app.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Startup / shutdown lifecycle.

    On startup: build the ServiceFactory, open DB connections.
    On shutdown: gracefully close them.
    """
    setup_logging()
    factory = ServiceFactory()
    await factory.startup()
    set_factory(factory)

    yield  # Application runs here

    await factory.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Thinking on Graph",
        description=(
            "GraphRAG-powered question answering API.\n\n"
            "Supports three query strategies: **GraphRAG** (knowledge graph traversal), "
            "**RAG** (vector similarity), and **None** (LLM only)."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(v1_router)

    @app.get("/health", tags=["health"])
    async def health_check() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
