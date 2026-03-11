"""
In-memory stub retrievers for all three query modes.

Each retriever is a separate class (SRP), all extending BaseRetriever (LSP).
New retrieval strategies are added by subclassing BaseRetriever (OCP).
"""

from app.core.embedder import BaseEmbedder
from app.core.graph_store import BaseGraphStore
from app.core.retriever import BaseRetriever, RetrievalResult
from app.core.vector_store import BaseVectorStore


class GraphRAGRetriever(BaseRetriever):
    """
    Retriever that uses the knowledge graph (GraphRAG).

    Strategy:
      1. Find seed nodes matching the query via graph full-text search.
      2. Expand each seed node into a subgraph.
      3. Serialise the subgraph into a context string for the LLM.

    Uses BaseGraphStore — works with InMemoryGraphStore or Neo4jGraphStore.
    """

    def __init__(self, graph_store: BaseGraphStore, embedder: BaseEmbedder) -> None:
        self._graph_store = graph_store
        self._embedder = embedder

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        # 1. Find seed nodes
        seed_nodes = await self._graph_store.search_nodes(query, top_k=top_k)

        context_parts: list[str] = []
        sources: list[str] = []

        for node in seed_nodes:
            node_id = node.get("id", "unknown")
            sources.append(node_id)

            # 2. Expand one hop
            subgraph = await self._graph_store.get_subgraph(node_id, depth=1)
            node_summaries = [
                f"Node({n.get('id', '?')}): {n}"
                for n in subgraph.get("nodes", [])
            ]
            edge_summaries = [
                f"{e.get('src', '?')} --[{e.get('relation', '?')}]--> {e.get('dst', '?')}"
                for e in subgraph.get("edges", [])
            ]
            context_parts.append(
                f"--- Subgraph for '{node_id}' ---\n"
                + "\n".join(node_summaries + edge_summaries)
            )

        return RetrievalResult(
            context="\n\n".join(context_parts) or "No relevant graph context found.",
            sources=sources,
        )


class RAGRetriever(BaseRetriever):
    """
    Standard vector-similarity RAG retriever.

    Uses BaseVectorStore — works with InMemoryVectorStore or PostgresVectorStore.
    """

    def __init__(self, vector_store: BaseVectorStore, embedder: BaseEmbedder) -> None:
        self._vector_store = vector_store
        self._embedder = embedder

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        # 1. Embed the query
        query_vector = await self._embedder.embed(query)

        # 2. ANN search
        results = await self._vector_store.search(query_vector, top_k=top_k)

        context_parts = [
            f"[score={r['score']:.4f}] {r['content']}"
            for r in results
        ]
        sources = [r["doc_id"] for r in results]

        return RetrievalResult(
            context="\n\n".join(context_parts) or "No relevant documents found.",
            sources=sources,
        )


class NoneRetriever(BaseRetriever):
    """
    No-retrieval pass-through — sends the query directly to the LLM.

    Useful as a baseline to compare against RAG and GraphRAG.
    """

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        return RetrievalResult(context="", sources=[])
