"""
QueryService — business logic for all three query modes.

SRP:  Owns the query use case; never touches infrastructure directly.
DIP:  Depends on BaseRetriever and BaseLLM, not on concrete classes.
OCP:  Adding a new query mode = new retriever subclass + registry entry.
"""

import time

from app.core.llm import BaseLLM
from app.core.retriever import BaseRetriever
from app.models.query import QueryMode, QueryRequest, QueryResponse
from app.prompts import (
    GRAPH_RAG_SYSTEM_PROMPT,
    NO_RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    graph_rag_user_prompt,
    no_rag_user_prompt,
    rag_user_prompt,
)


class QueryService:
    """
    Executes a query using the injected retriever and LLM.

    The same service class handles all query modes (GraphRAG, RAG, None) —
    the behaviour difference lives entirely in the injected BaseRetriever
    and the prompt template selected per mode.
    """

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM) -> None:
        self._retriever = retriever
        self._llm = llm

    async def answer(self, request: QueryRequest, mode: QueryMode) -> QueryResponse:
        """
        Retrieve context and generate an answer.

        Tracks wall-clock time from start of retrieval to end of generation
        and includes it in the response as `elapsed_seconds`.

        Args:
            request: Parsed QueryRequest.
            mode:    The active query mode (for prompt selection + response metadata).

        Returns:
            QueryResponse with the LLM answer, retrieved context, sources,
            and elapsed wall-clock seconds.
        """
        t_start = time.perf_counter()

        # 1. Retrieve context via whichever strategy is injected
        retrieval = await self._retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        # 2. Build mode-specific prompt + system prompt from prompts.py
        if mode == QueryMode.RAG:
            system_prompt = RAG_SYSTEM_PROMPT
            prompt = rag_user_prompt(request.question, retrieval.context)
        elif mode == QueryMode.GRAPHRAG:
            system_prompt = GRAPH_RAG_SYSTEM_PROMPT
            prompt = graph_rag_user_prompt(request.question, retrieval.context)
        else:  # QueryMode.NONE
            system_prompt = NO_RAG_SYSTEM_PROMPT
            prompt = no_rag_user_prompt(request.question)

        # 3. Generate answer — system_prompt is sent as a dedicated system
        #    message so the model treats it with higher authority than user text
        answer = await self._llm.generate(
            prompt=prompt,
            context=retrieval.context,
            system_prompt=system_prompt,
        )

        elapsed = round(time.perf_counter() - t_start, 3)

        return QueryResponse(
            question=request.question,
            mode=mode,
            answer=answer,
            context=retrieval.context,
            sources=retrieval.sources,
            elapsed_seconds=elapsed,
        )
