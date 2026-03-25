"""
IterativeGraphRAGRetriever — beam-search graph traversal guided by the LLM.

Implements the Think-on-Graph (ToG) algorithm:
  1. LLM extracts seed keywords from the question.
  2. Seed keywords are resolved to graph node IDs via search_nodes().
  3. Iterative loop (up to max_iterations):
       a. Expand depth-1 neighbours of every frontier node.
       b. LLM evaluates candidates → decides early stop OR selects top-K next frontier.
       c. LLM compresses the accumulated context, discarding dead-ends.
  4. The compressed context is returned to QueryService for final answer generation.

SRP: Only owns the iterative retrieval concern.
DIP: Depends on BaseGraphStore and BaseLLM abstractions — no concrete imports.
LSP: Fully substitutable for BaseRetriever.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.core.graph_store import BaseGraphStore
from app.core.llm import BaseLLM
from app.core.retriever import BaseRetriever, RetrievalResult
from app.prompts import (
    BEAM_SEARCH_COMPRESS_SYSTEM_PROMPT,
    BEAM_SEARCH_SEED_SYSTEM_PROMPT,
    beam_search_compress_user_prompt,
    beam_search_eval_system_prompt,
    beam_search_eval_user_prompt,
    beam_search_seed_user_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal Pydantic response models (not part of the public API surface)
# ---------------------------------------------------------------------------

class _SeedResult(BaseModel):
    keywords: list[str] = Field(default_factory=list)


class _EvalResult(BaseModel):
    has_sufficient_context: bool = False
    selected_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class IterativeGraphRAGRetriever(BaseRetriever):
    """
    Iterative beam-search retriever over a knowledge graph.

    At each hop the LLM prunes the frontier to the most promising nodes,
    so the search stays focused even over many iterations.

    Args:
        graph_store:    Any BaseGraphStore backend (e.g. Neo4jGraphStore).
        llm:            Any BaseLLM backend used for seed extraction, evaluation,
                        and context compression.
        max_iterations: Hard cap on the number of expansion steps (default 20).
        beam_width:     Maximum nodes to keep in the frontier per iteration (default 8).
    """

    def __init__(
        self,
        graph_store: BaseGraphStore,
        llm: BaseLLM,
        max_iterations: int = 20,
        beam_width: int = 8,
    ) -> None:
        self._graph_store = graph_store
        self._llm = llm
        self._max_iterations = max_iterations
        self._beam_width = beam_width

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Run the iterative beam-search and return a compressed context string.

        Args:
            query:  The user's natural-language question.
            top_k:  Seeds per keyword resolved via search_nodes (default 5).

        Returns:
            RetrievalResult with compressed context and visited node IDs as sources.
        """
        logger.info("IterativeGraphRAGRetriever: starting for query=%r", query)

        # ---- Step 0: seed extraction ----------------------------------------
        keywords = await self._extract_seeds(query)
        logger.info("Beam search seeds: keywords=%s", keywords)

        # Resolve keywords → node IDs
        frontier: set[str] = set()
        id_to_name: dict[str, str] = {}
        for kw in keywords:
            nodes = await self._graph_store.search_nodes(kw, top_k=top_k)
            for n in nodes:
                nid = n.get("id")
                if nid:
                    frontier.add(nid)
                    id_to_name[nid] = n.get("name", nid)
        frontier_names = [id_to_name.get(nid, nid) for nid in frontier]
        logger.info("Beam search: initial frontier size=%d  nodes=%s", len(frontier), frontier_names)

        if not frontier:
            logger.warning("Beam search: no seed nodes found — returning empty context")
            return RetrievalResult(
                context="No relevant graph context found.",
                sources=[],
            )

        # ---- Iterative loop -------------------------------------------------
        visited: set[str] = set()
        all_nodes: list[dict[str, Any]] = []
        all_edges: list[dict[str, Any]] = []
        seen_node_ids: set[str] = set()
        compressed_summary: str = ""

        for iteration in range(self._max_iterations):
            if not frontier:
                logger.info("Beam search: frontier empty at iteration %d — stopping", iteration)
                break

            frontier_names = [id_to_name.get(nid, nid) for nid in frontier]
            logger.info("Beam search iteration %d: frontier=%s", iteration, frontier_names)

            # Expand all frontier nodes (skip already visited)
            raw_nodes: list[dict[str, Any]] = []
            raw_edges: list[dict[str, Any]] = []
            for node_id in frontier:
                if node_id in visited:
                    continue
                subgraph = await self._graph_store.get_subgraph(node_id, depth=1)
                raw_nodes.extend(subgraph.get("nodes", []))
                raw_edges.extend(subgraph.get("edges", []))
                visited.add(node_id)

            logger.info(
                "Beam search iteration %d: expanded %d frontier nodes → %d candidate nodes",
                iteration, len(frontier), len(raw_nodes),
            )

            # Accumulate unique nodes and update name map
            for n in raw_nodes:
                nid = n.get("id")
                if nid:
                    id_to_name[nid] = n.get("name", nid)
                    if nid not in seen_node_ids:
                        all_nodes.append(n)
                        seen_node_ids.add(nid)
            all_edges.extend(raw_edges)

            # Build slim candidate list for the LLM (id, name, type, description only)
            candidates = [
                {
                    "id": n.get("id", ""),
                    "name": n.get("name", n.get("id", "")),
                    "type": n.get("type", ""),
                    "description": n.get("description", ""),
                }
                for n in raw_nodes
            ]

            # LLM evaluation: enough context? which top-K to visit next?
            eval_result = await self._evaluate(query, candidates, compressed_summary)
            selected_names = [id_to_name.get(nid, nid) for nid in eval_result.selected_ids]
            logger.info(
                "Beam search iteration %d: has_sufficient_context=%s  selected=%s",
                iteration, eval_result.has_sufficient_context, selected_names,
            )

            if eval_result.has_sufficient_context:
                logger.info("Beam search: LLM signalled sufficient context — stopping")
                break

            # Compress accumulated context after the LLM has decided to continue
            compressed_summary = await self._compress(query, all_nodes, all_edges)
            logger.info(
                "Beam search iteration %d: compressed summary length=%d chars",
                iteration, len(compressed_summary),
            )

            # Update frontier to the LLM-selected nodes not yet visited
            frontier = set(eval_result.selected_ids) - visited

        # ---- Build final context --------------------------------------------
        context = compressed_summary or self._format_raw(all_nodes, all_edges)
        sources = list(seen_node_ids)
        logger.info(
            "Beam search complete: %d unique nodes visited, context length=%d chars",
            len(sources), len(context),
        )
        return RetrievalResult(
            context=context or "No relevant graph context found.",
            sources=sources,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _extract_seeds(self, question: str) -> list[str]:
        """Ask the LLM to extract entity keywords from the question.

        Falls back to a single keyword equal to the full question if the LLM
        returns an empty list (e.g. structured output parsing failed).
        """
        prompt = beam_search_seed_user_prompt(question)
        result: _SeedResult = await self._llm.generate_structured(
            prompt=prompt,
            response_model=_SeedResult,
            system_prompt=BEAM_SEARCH_SEED_SYSTEM_PROMPT,
        )
        keywords = result.keywords if result and result.keywords else [question]
        return keywords

    async def _evaluate(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        compressed_summary: str,
    ) -> _EvalResult:
        """Ask the LLM whether we have enough context and which nodes to expand next."""
        system_prompt = beam_search_eval_system_prompt(self._beam_width)
        prompt = beam_search_eval_user_prompt(
            question=question,
            beam_width=self._beam_width,
            compressed_summary=compressed_summary,
            candidates=candidates,
        )
        result: _EvalResult = await self._llm.generate_structured(
            prompt=prompt,
            response_model=_EvalResult,
            system_prompt=system_prompt,
        )
        return result if result else _EvalResult()

    async def _compress(
        self,
        question: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> str:
        """Ask the LLM to distil accumulated graph data into a relevant summary."""
        raw_context = self._format_raw(nodes, edges)
        prompt = beam_search_compress_user_prompt(question, raw_context)
        summary = await self._llm.generate(
            prompt=prompt,
            system_prompt=BEAM_SEARCH_COMPRESS_SYSTEM_PROMPT,
        )
        return summary.strip()

    @staticmethod
    def _format_raw(
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> str:
        """Serialise nodes and edges into a plain-text context string."""
        node_lines = [
            f"Node({n.get('id', '?')}): name={n.get('name', '?')}  "
            f"type={n.get('type', '?')}  desc={n.get('description', '')}"
            for n in nodes
        ]
        edge_lines = []
        for e in edges:
            src = e.get("src", "?")
            dst = e.get("dst", "?")
            rel = e.get("relation", "?")
            if rel == "RAW_RELATION":
                raw_text = e.get("raw_text") or "?"
                edge_lines.append(f'{src} --["{raw_text}"]--> {dst}')
            else:
                edge_lines.append(f"{src} --[{rel}]--> {dst}")
        parts = []
        if node_lines:
            parts.append("Nodes:\n" + "\n".join(node_lines))
        if edge_lines:
            parts.append("Edges:\n" + "\n".join(edge_lines))
        return "\n\n".join(parts)
