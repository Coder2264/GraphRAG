"""
InMemoryIngestionPipeline — stub ingestion pipeline that wires together
InMemory embedder, vector store, and graph store.

SRP: This class orchestrates chunking → embedding → storage.
     Each step is delegated to a focused collaborator.
"""

import uuid
from typing import Any

from app.core.embedder import BaseEmbedder
from app.core.graph_store import BaseGraphStore
from app.core.ingestion import BaseIngestionPipeline
from app.core.vector_store import BaseVectorStore


class InMemoryIngestionPipeline(BaseIngestionPipeline):
    """
    Stub ingestion pipeline.

    Chunking strategy: single chunk (the whole document).
    Embedding: delegated to BaseEmbedder.
    Storage: delegated to BaseVectorStore + BaseGraphStore.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._graph_store = graph_store

    async def ingest(self, content: str, metadata: dict[str, Any]) -> str:
        doc_id = str(uuid.uuid4())

        # 1. Embed (stub: returns zero vector)
        vector = await self._embedder.embed(content)

        # 2. Store in vector store
        await self._vector_store.upsert(
            doc_id=doc_id,
            vector=vector,
            metadata=metadata,
            content=content,
        )

        # 3. Represent document as a node in the graph store
        #    (real pipeline would extract entities/relations here)
        await self._graph_store.add_node(
            node_id=doc_id,
            labels=["Document"],
            data={"content_preview": content[:200], **metadata},
        )

        return doc_id

    async def delete(self, doc_id: str) -> None:
        await self._vector_store.delete(doc_id)
        await self._graph_store.delete_node(doc_id)
