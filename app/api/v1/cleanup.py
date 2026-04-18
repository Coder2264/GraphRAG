"""
Cleanup route — wipes all data from Postgres and Neo4j for independent test runs.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.graph_store import BaseGraphStore
from app.core.vector_store import BaseVectorStore
from app.dependencies import get_graph_store, get_vector_store

router = APIRouter(tags=["cleanup"])


@router.get("/cleanup")
async def cleanup(
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
    graph_store: Annotated[BaseGraphStore, Depends(get_graph_store)],
) -> dict[str, str]:
    """Delete all vectors from Postgres and all nodes/edges from Neo4j."""
    await asyncio.gather(vector_store.clear(), graph_store.clear())
    return {"status": "ok", "message": "All data cleared from vector store and graph store."}
