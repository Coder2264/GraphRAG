"""
QueryService — business logic for all three query modes.

SRP:  Owns the query use case; never touches infrastructure directly.
DIP:  Depends on BaseRetriever and BaseLLM, not on concrete classes.
OCP:  Adding a new query mode = new retriever subclass + registry entry.
"""

from app.core.llm import BaseLLM
from app.core.retriever import BaseRetriever
from app.models.query import QueryMode, QueryRequest, QueryResponse


class QueryService:
    """
    Executes a query using the injected retriever and LLM.

    The same service class handles all query modes (GraphRAG, RAG, None) —
    the behaviour difference lives entirely in the injected BaseRetriever.
    """

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM) -> None:
        self._retriever = retriever
        self._llm = llm

    async def answer(self, request: QueryRequest, mode: QueryMode) -> QueryResponse:
        """
        Retrieve context and generate an answer.

        Args:
            request: Parsed QueryRequest.
            mode:    The active query mode (for response metadata).

        Returns:
            QueryResponse with the LLM answer and retrieved context.
        """
        # 1. Retrieve context via whichever strategy is injected
        retrieval = await self._retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        # 2. Generate answer with the LLM
        answer = await self._llm.generate(
            prompt=request.question,
            context=retrieval.context,
        )

        return QueryResponse(
            question=request.question,
            mode=mode,
            answer=answer,
            context=retrieval.context,
            sources=retrieval.sources,
        )
