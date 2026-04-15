"""Think-on-Graph (ToG) retriever — paper-faithful implementation.

Implements the Think-on-Graph algorithm from:
  "Think-on-Graph: Deep and Responsible Reasoning of LLM on Knowledge Graphs"
  Sun et al., ICLR 2024.  https://arxiv.org/abs/2307.07697

Section 2.1 defines the four-phase algorithm executed per query:
  Phase 1 — Topic entity extraction (seeds the initial beam).
  Phase 2 — Iterative graph exploration: relation pruning → entity search
             → entity pruning, repeated up to ``depth_max`` times.
  Phase 3 — Reasoning sufficiency check after each exploration iteration.
  Phase 4 — Final answer generation from the accumulated reasoning paths.

SRP: Owns only the ToG retrieval concern; delegates I/O to injected collaborators.
DIP: Depends on BaseGraphStore and BaseLLM abstractions — no concrete imports.
LSP: Fully substitutable for BaseRetriever.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from pydantic import BaseModel, Field

from app.core.graph_store import BaseGraphStore
from app.core.llm import BaseLLM
from app.core.retriever import BaseRetriever, RetrievalResult
from app.core.vector_store import BaseVectorStore
from app.prompts import (
    TOG_BATCH_ENTITY_PRUNE_SYSTEM_PROMPT,
    TOG_BATCH_RELATION_PRUNE_SYSTEM_PROMPT,
    TOG_DEFAULT_EXAMPLES,
    TOG_ENTITY_PRUNE_SYSTEM_PROMPT,
    TOG_GENERATE_SYSTEM_PROMPT,
    TOG_REASONING_SYSTEM_PROMPT,
    TOG_RELATION_PRUNE_SYSTEM_PROMPT,
    tog_batch_entity_prune_user_prompt,
    tog_batch_relation_prune_user_prompt,
    tog_entity_prune_user_prompt,
    tog_generate_user_prompt,
    tog_reasoning_user_prompt,
    tog_relation_prune_user_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal Pydantic response models (not part of the public API surface)
# ---------------------------------------------------------------------------


class _TopicEntities(BaseModel):
    """Structured output for topic entity extraction (Phase 1)."""

    entities: list[str] = Field(default_factory=list)


class _ScoredRelation(BaseModel):
    """A relation label paired with its relevance score."""

    relation: str
    score: float


class _RelationPruneResult(BaseModel):
    """Structured output for relation pruning (Phase 2, Step B)."""

    relations: list[_ScoredRelation] = Field(default_factory=list)


class _ScoredEntity(BaseModel):
    """An entity name paired with its relevance score."""

    entity: str
    score: float


class _EntityPruneResult(BaseModel):
    """Structured output for entity pruning (Phase 2, Step D)."""

    entities: list[_ScoredEntity] = Field(default_factory=list)


class _BatchRelationEntry(BaseModel):
    """One entity's pruned relations in a batched relation-pruning response."""

    idx: int
    relations: list[_ScoredRelation] = Field(default_factory=list)


class _BatchRelationPruneResult(BaseModel):
    """Structured output for batched relation pruning (Step B, all entities)."""

    results: list[_BatchRelationEntry] = Field(default_factory=list)


class _BatchEntityEntry(BaseModel):
    """One path's scored entities in a batched entity-pruning response."""

    idx: int
    entities: list[_ScoredEntity] = Field(default_factory=list)


class _BatchEntityPruneResult(BaseModel):
    """Structured output for batched entity pruning (Step D, all paths)."""

    results: list[_BatchEntityEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Internal path representation:
#   {"nodes": list[dict], "relations": list[str], "edge_chunk_ids": list[str]}
#   node dict: {"id": str, "name": str, "source_chunk_id": str}
#   edge_chunk_ids[i] = source_chunk_id of the edge between nodes[i] and nodes[i+1]
#   Invariant for a *complete* path: len(nodes) == len(relations) + 1
#   A path of depth D has D+1 nodes and D relations.
#
# During Step B→D of Phase 2, paths temporarily exist in a *pending* state where
#   len(relations) == len(nodes)  (one extra relation, tail entity not yet chosen).
# These are never returned to callers.


def _format_path(path: dict[str, Any]) -> str:
    """Format a complete path dict as a triple-chain string.

    A single-node path (no traversal yet) is formatted as just the entity name.
    Deeper paths produce a comma-separated chain of ``(head, relation, tail)``
    triples, matching the paper's best-performing context format.

    Args:
        path: Dict with keys ``"nodes"`` (``list[dict]``) and
              ``"relations"`` (``list[str]``), satisfying
              ``len(nodes) == len(relations) + 1``.
              Each node dict must have a ``"name"`` key.

    Returns:
        Formatted string, e.g.
        ``"(node0, rel0, node1), (node1, rel1, node2)"``
        or just ``"node0"`` when ``len(nodes) == 1``.
    """
    raw_nodes = path.get("nodes", [])
    nodes: list[str] = [
        n.get("name", str(n)) if isinstance(n, dict) else str(n)
        for n in raw_nodes
    ]
    relations: list[str] = path.get("relations", [])

    if not nodes:
        return ""
    if len(nodes) == 1:
        return nodes[0]

    triples = [
        f"({nodes[i]}, {rel}, {nodes[i + 1]})"
        for i, rel in enumerate(relations)
    ]
    return ", ".join(triples)


async def _collect_source_chunks(
    paths: list[dict[str, Any]],
    vector_store: BaseVectorStore | None,
) -> list[str]:
    """Fetch up to 3 unique non-empty source chunk texts for the given paths.

    Collects chunk IDs from node source_chunk_id and edge_chunk_ids, then
    fetches the content from the vector store (PostgreSQL).  Caps at 3 chunks
    × 1500 chars each to bound prompt size.

    Args:
        paths:        List of complete path dicts.
        vector_store: The vector store to fetch chunk text from.
                      Returns empty list when None.

    Returns:
        List of chunk text strings (up to 3, each truncated to 1500 chars).
    """
    if vector_store is None:
        return []
    seen_ids: set[str] = set()
    chunk_ids: list[str] = []
    for path in paths:
        for node in path.get("nodes", []):
            cid = node.get("source_chunk_id", "") if isinstance(node, dict) else ""
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                chunk_ids.append(cid)
        for cid in path.get("edge_chunk_ids", []):
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                chunk_ids.append(cid)
        if len(chunk_ids) >= 3:
            break
    ids_to_fetch = chunk_ids[:3]
    records = await asyncio.gather(*[vector_store.get(cid) for cid in ids_to_fetch])
    return [r["content"][:1500] for r in records if r and r.get("content")]


# ---------------------------------------------------------------------------
# ToGRetriever
# ---------------------------------------------------------------------------


class ToGRetriever(BaseRetriever):
    """
    Think-on-Graph retriever: paper-faithful implementation of the ToG algorithm.

    The algorithm seeds a beam from topic entities extracted by the LLM, then
    iteratively prunes relations and entities on the knowledge graph to build
    reasoning paths.  After each exploration step the LLM judges whether the
    accumulated paths are sufficient to answer the question.  Once sufficient
    (or once the depth limit is reached), a final answer is generated.

    The ``context`` field of the returned :class:`~app.core.retriever.RetrievalResult`
    contains the final **answer string** rather than raw graph context, because
    ToG performs its own answer generation internally.  ``sources`` lists every
    entity ID visited during traversal.

    Args:
        graph_store: Any :class:`~app.core.graph_store.BaseGraphStore` backend
                     (e.g. ``Neo4jGraphStore``).  Must implement
                     ``get_relations``, ``get_tail_entities``, and
                     ``get_head_entities``.
        llm:         Any :class:`~app.core.llm.BaseLLM` backend used for entity
                     extraction, relation/entity pruning, reasoning checks, and
                     answer generation.
        beam_width:  Maximum relations kept per entity per hop (paper default N=3).
        depth_max:   Hard cap on exploration iterations (paper default D_max=3).
        max_paths:   Maximum surviving paths after Step D entity pruning at each
                     depth.  Paths are ranked by the LLM entity score and the
                     top ``max_paths`` are kept, bounding prompt growth across
                     depths (default 8).
        examples:    Few-shot demonstrations forwarded to all four ToG prompts.
                     Defaults to :data:`~app.prompts.TOG_DEFAULT_EXAMPLES`.

    Note:
        Temperature hint for relation/entity pruning LLM calls: **0.4** per the
        paper.  ``BaseLLM`` does not expose a temperature parameter; callers
        should configure temperature directly on their LLM backend if supported.
    """

    def __init__(
        self,
        graph_store: BaseGraphStore,
        llm: BaseLLM,
        beam_width: int = 3,
        depth_max: int = 3,
        max_paths: int = 8,
        examples: list[dict] | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self._graph_store = graph_store
        self._llm = llm
        self._beam_width = beam_width
        self._depth_max = depth_max
        self._max_paths = max_paths
        self._examples: list[dict] = (
            examples if examples is not None else TOG_DEFAULT_EXAMPLES
        )
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Execute the Think-on-Graph algorithm and return the generated answer.

        ``top_k`` is accepted for interface compatibility but is not used by
        the ToG algorithm; breadth is controlled by ``beam_width`` instead.

        Args:
            query: The user's natural-language question.
            top_k: Unused (kept for :class:`~app.core.retriever.BaseRetriever`
                   interface compatibility).

        Returns:
            :class:`~app.core.retriever.RetrievalResult` where ``context`` is
            the generated answer string and ``sources`` are the entity IDs of
            every node visited during graph traversal.
        """
        t_total = time.monotonic()
        logger.info("ToGRetriever: starting for query=%r", query)

        # ── Phase 1: Topic entity extraction ───────────────────────────────
        # P = [] (empty paths); E^0 = topic entities extracted by the LLM.
        topic_entity_names = await self._extract_topic_entities(query)
        logger.info("ToGRetriever: topic entities=%s", topic_entity_names)

        if not topic_entity_names:
            logger.warning(
                "ToGRetriever: no topic entities found — falling back to LLM-only"
            )
            answer = await self._llm.generate(prompt=query)
            return RetrievalResult(context=answer, sources=[])

        # Resolve entity name strings → graph node dicts via search_nodes in parallel.
        # Stored node IDs are "{doc_id}__slug"; querying by raw name would miss them.
        t1 = time.monotonic()
        logger.info(
            "ToGRetriever [Phase 1]: resolving %d topic entities from graph (parallel)",
            len(topic_entity_names),
        )

        async def _resolve_entity(name: str) -> dict[str, Any]:
            matches = await self._graph_store.search_nodes(name, top_k=2)
            if matches:
                best = matches[0]
                entity = {
                    "id": best.get("id", name),
                    "name": best.get("name", name),
                    "source_chunk_id": best.get("source_chunk_id", ""),
                }
                logger.info(
                    "ToGRetriever [Phase 1]: resolved %r → id=%r", name, entity["id"]
                )
                return entity
            logger.warning(
                "ToGRetriever [Phase 1]: no graph node found for %r — using name as ID", name
            )
            return {"id": name, "name": name, "source_chunk_id": ""}

        topic_entities: list[dict[str, Any]] = list(
            await asyncio.gather(*[_resolve_entity(n) for n in topic_entity_names])
        )
        logger.info(
            "ToGRetriever [Phase 1]: entity resolution done in %.1fs", time.monotonic() - t1
        )

        # Initialise beam: one single-node path per topic entity.
        # Invariant: len(path["nodes"]) == len(path["relations"]) + 1
        paths: list[dict[str, Any]] = [
            {"nodes": [e], "relations": [], "edge_chunk_ids": []}
            for e in topic_entities
        ]
        visited_entities: set[str] = {e["id"] for e in topic_entities}

        # ── Phase 2: Iterative exploration ──────────────────────────────────
        answer: str | None = None

        for depth in range(1, self._depth_max + 1):
            t_depth = time.monotonic()
            logger.info("ToGRetriever: exploration depth=%d / %d", depth, self._depth_max)

            # E^{D-1}: tail entity dict of each current path (last node).
            # Deduplicate by entity ID so we don't fetch the same entity twice.
            unique_tails: dict[str, dict[str, Any]] = {}
            for p in paths:
                td = p["nodes"][-1]
                unique_tails.setdefault(td["id"], td)

            # ── Step A: Relation search (parallel) ───────────────────────
            t_a = time.monotonic()
            logger.info(
                "ToGRetriever [A]: fetching relations for %d unique entities (parallel)",
                len(unique_tails),
            )
            relation_lists = await asyncio.gather(
                *[self._graph_store.get_relations(eid) for eid in unique_tails]
            )
            entity_relations: dict[str, list[str]] = dict(
                zip(unique_tails.keys(), relation_lists)
            )
            total_rels = sum(len(r) for r in entity_relations.values())
            logger.info(
                "ToGRetriever [A]: done in %.1fs — %d total candidate relations",
                time.monotonic() - t_a, total_rels,
            )

            # ── Step B: Relation pruning (batched) ───────────────────────
            # All entities are scored in a single LLM call instead of one
            # call per entity, eliminating N-1 network round-trips.
            entities_with_rels = [(eid, rels) for eid, rels in entity_relations.items() if rels]
            t_b = time.monotonic()
            logger.info(
                "ToGRetriever [B]: pruning relations for %d entities via LLM (batched)",
                len(entities_with_rels),
            )
            name_rel_pairs = [
                (unique_tails[eid].get("name", eid), rels)
                for eid, rels in entities_with_rels
            ]
            pruned_rels_list = await self._prune_relations_batch(query, name_rel_pairs)
            entity_pruned_relations: dict[str, list[str]] = {
                eid: pruned_rels
                for (eid, _), pruned_rels in zip(entities_with_rels, pruned_rels_list)
            }
            for eid in entity_relations:
                entity_pruned_relations.setdefault(eid, [])
            logger.info(
                "ToGRetriever [B]: done in %.1fs", time.monotonic() - t_b
            )

            # Extend each path with each of its tail entity's pruned relations,
            # producing *pending* paths (len(relations) == len(nodes), no new entity yet).
            pending_paths: list[dict[str, Any]] = []
            for path in paths:
                tail_id = path["nodes"][-1]["id"]
                for rel in entity_pruned_relations.get(tail_id, []):
                    pending_paths.append({
                        "nodes": list(path["nodes"]),
                        "relations": list(path["relations"]) + [rel],
                        "edge_chunk_ids": list(path["edge_chunk_ids"]),
                    })

            if not pending_paths:
                logger.info(
                    "ToGRetriever: no expandable paths at depth=%d — stopping", depth
                )
                break

            # ── Step C: Entity search (parallel) ─────────────────────────
            # For each pending path retrieve candidate entity dicts in both directions.
            t_c = time.monotonic()
            logger.info(
                "ToGRetriever [C]: entity search for %d pending paths (parallel)",
                len(pending_paths),
            )

            async def _get_candidates(
                pending: dict[str, Any],
            ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
                tail_id = pending["nodes"][-1]["id"]
                relation = pending["relations"][-1]
                tail_results, head_results = await asyncio.gather(
                    self._graph_store.get_tail_entities(tail_id, relation),
                    self._graph_store.get_head_entities(tail_id, relation),
                )
                seen_ids: set[str] = set()
                candidates: list[dict[str, Any]] = []
                for c in tail_results + head_results:
                    if c["id"] not in seen_ids:
                        seen_ids.add(c["id"])
                        candidates.append(c)
                logger.debug(
                    "ToGRetriever [C]: entity_id=%r relation=%r → %d candidate(s)",
                    tail_id, relation, len(candidates),
                )
                return pending, candidates

            path_candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = list(
                await asyncio.gather(*[_get_candidates(p) for p in pending_paths])
            )
            logger.info(
                "ToGRetriever [C]: done in %.1fs", time.monotonic() - t_c
            )

            # ── Step D: Entity pruning (batched) ─────────────────────────
            # All paths are scored in a single LLM call instead of one call
            # per path, eliminating M-1 network round-trips.
            non_empty_candidates = [(p, c) for p, c in path_candidates if c]
            t_d = time.monotonic()
            logger.info(
                "ToGRetriever [D]: entity pruning for %d candidate sets via LLM (batched)",
                len(non_empty_candidates),
            )
            prune_results = await self._entity_prune_batch(query, non_empty_candidates)
            scored_paths: list[tuple[dict[str, Any], float]] = []
            for new_path, score in prune_results:
                if new_path is not None:
                    scored_paths.append((new_path, score))
                    logger.debug(
                        "ToGRetriever [D]: finalised path (score=%.2f) → %s",
                        score, _format_path(new_path),
                    )
            scored_paths.sort(key=lambda x: x[1], reverse=True)
            scored_paths = scored_paths[: self._max_paths]
            new_paths: list[dict[str, Any]] = []
            for new_path, _ in scored_paths:
                new_paths.append(new_path)
                visited_entities.add(new_path["nodes"][-1]["id"])
            logger.info(
                "ToGRetriever [D]: done in %.1fs — %d paths finalised (capped at max_paths=%d)",
                time.monotonic() - t_d, len(new_paths), self._max_paths,
            )

            if not new_paths:
                logger.info(
                    "ToGRetriever: no paths finalised at depth=%d — stopping", depth
                )
                break

            paths = new_paths  # E^D is now the tail of each path in new_paths

            # ── Phase 3: Reasoning sufficiency check ─────────────────────
            # Skipped at max depth: the loop ends regardless, so the check
            # would be wasted — the fallback below handles generation instead.
            path_strings = [_format_path(p) for p in paths]
            source_chunks = await _collect_source_chunks(paths, self._vector_store)
            if depth < self._depth_max:
                sufficient = await self._check_reasoning(query, path_strings, source_chunks)
                logger.info(
                    "ToGRetriever [Phase 3]: depth=%d sufficient=%s paths=%d chunks=%d "
                    "(depth wall time=%.1fs)",
                    depth, sufficient, len(path_strings), len(source_chunks),
                    time.monotonic() - t_depth,
                )
                if sufficient:
                    # Phase 4: Generate the final answer from accumulated paths.
                    answer = await self._generate_answer(query, path_strings, source_chunks)
                    break
            else:
                logger.info(
                    "ToGRetriever [Phase 3]: depth=%d is max — skipping reasoning check "
                    "(depth wall time=%.1fs)",
                    depth, time.monotonic() - t_depth,
                )
            # "No" and depth < depth_max → implicit continue to next iteration.
            # depth == depth_max → loop ends; fallback applied below.

        # If depth_max reached without a "Yes", generate with available paths + chunks.
        if answer is None:
            if paths and any(len(p["nodes"]) > 1 for p in paths):
                logger.info(
                    "ToGRetriever: depth limit reached — generating from accumulated paths"
                )
                path_strings = [_format_path(p) for p in paths]
                source_chunks = await _collect_source_chunks(paths, self._vector_store)
                answer = await self._generate_answer(query, path_strings, source_chunks)
            else:
                logger.info(
                    "ToGRetriever: no traversal completed — falling back to LLM-only"
                )
                answer = await self._llm.generate(prompt=query)

        logger.info(
            "ToGRetriever: done — total=%.1fs visited=%d entities answer=%d chars",
            time.monotonic() - t_total, len(visited_entities), len(answer),
        )
        return RetrievalResult(
            context=answer,
            sources=list(visited_entities),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _extract_topic_entities(self, question: str) -> list[str]:
        """Extract topic entities from the question via structured LLM generation.

        Uses a minimal inline system prompt so that callers who have not
        yet wired up a dedicated ToG-extraction prompt still get reasonable
        behaviour (Phase 1 of the paper).

        Args:
            question: The user's natural-language question.

        Returns:
            List of entity name strings extracted by the LLM, or an empty
            list if the structured output parse failed.
        """
        t0 = time.monotonic()
        logger.info("ToGRetriever [Phase 1]: calling LLM for topic entity extraction")
        result: _TopicEntities = await self._llm.generate_structured(
            prompt=question,
            response_model=_TopicEntities,
            system_prompt=(
                "Extract the main topic entities (proper nouns, named entities, "
                "key concepts) from the question. "
                'Return ONLY a JSON object: {"entities": ["entity1", "entity2", ...]}'
            ),
        )
        logger.info(
            "ToGRetriever [Phase 1]: LLM responded in %.1fs → entities=%s",
            time.monotonic() - t0,
            result.entities if result else [],
        )
        return result.entities if result and result.entities else []

    async def _prune_relations(
        self,
        question: str,
        entity: str,
        relations: list[str],
    ) -> list[str]:
        """Score and prune candidate relations for one entity (Phase 2, Step B).

        Calls the LLM with the ToG relation-pruning prompt and returns the
        top-``beam_width`` relation labels sorted by descending score.

        Temperature hint: **0.4** (paper recommendation).  ``BaseLLM`` does not
        expose a temperature parameter; configure the LLM backend directly if
        temperature control is needed.

        Args:
            question:  The user's natural-language question.
            entity:    The entity whose outgoing/incoming relations are pruned.
            relations: Candidate relation label strings from the graph.

        Returns:
            Up to ``beam_width`` relation labels, highest-scored first.
            Returns an empty list if the LLM returns no valid output.
        """
        system_prompt = TOG_RELATION_PRUNE_SYSTEM_PROMPT.format(
            beam_width=self._beam_width
        )
        user_prompt = tog_relation_prune_user_prompt(
            question=question,
            entity=entity,
            relations=relations,
            beam_width=self._beam_width,
            examples=self._examples,
        )
        t0 = time.monotonic()
        logger.debug(
            "ToGRetriever [B]: calling LLM for relation pruning (entity=%r, %d relations)",
            entity, len(relations),
        )
        result: _RelationPruneResult = await self._llm.generate_structured(
            prompt=user_prompt,
            response_model=_RelationPruneResult,
            system_prompt=system_prompt,
        )
        logger.debug(
            "ToGRetriever [B]: LLM responded in %.2fs (entity=%r)", time.monotonic() - t0, entity
        )
        if not result or not result.relations:
            return []
        sorted_rels = sorted(result.relations, key=lambda r: r.score, reverse=True)
        return [sr.relation for sr in sorted_rels[: self._beam_width]]

    async def _entity_prune(
        self,
        question: str,
        relation: str,
        candidates: list[str],
        path: dict[str, Any],
    ) -> str | None:
        """Score candidate entities and return the top-scored one (Phase 2, Step D).

        This method is intended to be overridden by subclasses that want to
        replace the LLM-based scoring strategy (e.g. random sampling).

        Temperature hint: **0.4** (paper recommendation; see ``_prune_relations``).

        Args:
            question:   The user's natural-language question.
            relation:   The relation label being traversed to reach the candidates.
            candidates: Candidate entity names reachable via ``relation``.
            path:       The pending path dict being extended (``{"nodes": ...,
                        "relations": ...}``).  Provided so subclasses can use
                        path context if needed; unused by the default
                        LLM-based implementation.

        Returns:
            The name of the highest-scoring entity, or ``None`` if the LLM
            returns no valid output.
        """
        user_prompt = tog_entity_prune_user_prompt(
            question=question,
            relation=relation,
            entities=candidates,
            examples=self._examples,
        )
        t0 = time.monotonic()
        logger.debug(
            "ToGRetriever [D]: calling LLM for entity pruning (relation=%r, %d candidates)",
            relation, len(candidates),
        )
        result: _EntityPruneResult = await self._llm.generate_structured(
            prompt=user_prompt,
            response_model=_EntityPruneResult,
            system_prompt=TOG_ENTITY_PRUNE_SYSTEM_PROMPT,
        )
        logger.debug(
            "ToGRetriever [D]: LLM responded in %.2fs (relation=%r)", time.monotonic() - t0, relation
        )
        if not result or not result.entities:
            return None
        best = max(result.entities, key=lambda e: e.score)
        return best.entity

    async def _prune_relations_batch(
        self,
        question: str,
        name_rel_pairs: list[tuple[str, list[str]]],
    ) -> list[list[str]]:
        """Prune relations for all entities in a single LLM call (Step B).

        Replaces the ``asyncio.gather`` fan-out over ``_prune_relations`` with
        one batched request, eliminating per-entity network round-trips.

        Args:
            question:      The user's natural-language question.
            name_rel_pairs: List of (entity_name, candidate_relation_list) tuples,
                            one entry per entity.

        Returns:
            List of pruned relation lists, same length and order as
            ``name_rel_pairs``.  An empty list at position *i* means the LLM
            selected nothing for that entity.
        """
        system_prompt = TOG_BATCH_RELATION_PRUNE_SYSTEM_PROMPT.format(
            beam_width=self._beam_width
        )
        user_prompt = tog_batch_relation_prune_user_prompt(
            question, name_rel_pairs, self._beam_width
        )
        result: _BatchRelationPruneResult = await self._llm.generate_structured(
            prompt=user_prompt,
            response_model=_BatchRelationPruneResult,
            system_prompt=system_prompt,
        )
        idx_to_rels: dict[int, list[str]] = {}
        if result and result.results:
            for entry in result.results:
                sorted_rels = sorted(entry.relations, key=lambda r: r.score, reverse=True)
                idx_to_rels[entry.idx] = [sr.relation for sr in sorted_rels[: self._beam_width]]
        return [idx_to_rels.get(i, []) for i in range(len(name_rel_pairs))]

    async def _entity_prune_batch(
        self,
        question: str,
        path_candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    ) -> list[tuple[dict[str, Any] | None, float]]:
        """Prune and select entities for all paths in a single LLM call (Step D).

        Replaces the ``asyncio.gather`` fan-out over ``_entity_prune`` with
        one batched request, eliminating per-path network round-trips.

        Args:
            question:        The user's natural-language question.
            path_candidates: List of (pending_path_dict, candidate_entity_dicts)
                             tuples, one entry per pending path.

        Returns:
            List of (finalised_path_dict | None, score) tuples, same length and
            order as ``path_candidates``.  Score is the LLM relevance score of
            the selected entity (0.0 when nothing was selected), used by callers
            to rank and cap the surviving path set.
        """
        rel_name_pairs = [
            (pending["relations"][-1], [c["name"] for c in candidates])
            for pending, candidates in path_candidates
        ]
        user_prompt = tog_batch_entity_prune_user_prompt(question, rel_name_pairs)
        result: _BatchEntityPruneResult = await self._llm.generate_structured(
            prompt=user_prompt,
            response_model=_BatchEntityPruneResult,
            system_prompt=TOG_BATCH_ENTITY_PRUNE_SYSTEM_PROMPT,
        )
        idx_to_selection: dict[int, tuple[str, float]] = {}
        if result and result.results:
            for entry in result.results:
                if entry.entities:
                    best = max(entry.entities, key=lambda e: e.score)
                    idx_to_selection[entry.idx] = (best.entity, best.score)

        new_paths: list[tuple[dict[str, Any] | None, float]] = []
        for i, (pending, candidates) in enumerate(path_candidates):
            selection = idx_to_selection.get(i)
            if selection is None:
                new_paths.append((None, 0.0))
                continue
            selected_name, score = selection
            selected_dict = next(
                (c for c in candidates if c["name"].lower() == selected_name.lower()),
                {"id": selected_name, "name": selected_name, "source_chunk_id": "", "edge_chunk_id": ""},
            )
            new_paths.append((
                {
                    "nodes": pending["nodes"] + [selected_dict],
                    "relations": list(pending["relations"]),
                    "edge_chunk_ids": list(pending["edge_chunk_ids"]) + [selected_dict.get("edge_chunk_id", "")],
                },
                score,
            ))
        return new_paths

    async def _check_reasoning(
        self,
        question: str,
        path_strings: list[str],
        source_chunks: list[str] | None = None,
    ) -> bool:
        """Ask the LLM whether the current paths are sufficient to answer (Phase 3).

        Uses :func:`~app.core.llm.BaseLLM.generate` (not ``generate_structured``)
        and parses "Yes" / "No" from the free-form response (case-insensitive,
        leading/trailing whitespace ignored).

        Args:
            question:      The user's natural-language question.
            path_strings:  Formatted triple-chain strings from all current paths.
            source_chunks: Optional source text excerpts from the document chunks
                           that produced the entities/edges on the path.

        Returns:
            ``True`` if the LLM response starts with "yes" (case-insensitive),
            ``False`` otherwise.
        """
        user_prompt = tog_reasoning_user_prompt(
            question=question,
            paths=path_strings,
            examples=self._examples,
            source_chunks=source_chunks or [],
        )
        t0 = time.monotonic()
        logger.info(
            "ToGRetriever [Phase 3]: calling LLM for reasoning check (%d paths)", len(path_strings)
        )
        response = await self._llm.generate(
            prompt=user_prompt,
            system_prompt=TOG_REASONING_SYSTEM_PROMPT,
        )
        logger.info(
            "ToGRetriever [Phase 3]: LLM responded in %.1fs → %r",
            time.monotonic() - t0, response.strip()[:30],
        )
        return response.strip().lower().startswith("yes")

    async def _generate_answer(
        self,
        question: str,
        path_strings: list[str],
        source_chunks: list[str] | None = None,
    ) -> str:
        """Generate the final answer from accumulated reasoning paths (Phase 4).

        Paths are provided as triple chains — ``"(e1, r1, e2), (e2, r2, e3)"`` —
        which is the best-performing format reported in the paper (Appendix E.3.4).

        Args:
            question:      The user's natural-language question.
            path_strings:  Formatted triple-chain strings from all current paths.
            source_chunks: Optional source text excerpts from the document chunks
                           that produced the entities/edges on the path.

        Returns:
            The LLM-generated answer string.
        """
        user_prompt = tog_generate_user_prompt(
            question=question,
            paths=path_strings,
            examples=self._examples,
            source_chunks=source_chunks or [],
        )
        t0 = time.monotonic()
        logger.info(
            "ToGRetriever [Phase 4]: calling LLM for answer generation (%d paths)", len(path_strings)
        )
        answer = await self._llm.generate(
            prompt=user_prompt,
            system_prompt=TOG_GENERATE_SYSTEM_PROMPT,
        )
        logger.info(
            "ToGRetriever [Phase 4]: LLM responded in %.1fs", time.monotonic() - t0
        )
        return answer


# ---------------------------------------------------------------------------
# ToGRRetriever
# ---------------------------------------------------------------------------


class ToGRRetriever(ToGRetriever):
    """Think-on-Graph with Random entity pruning (ToG-R).

    Identical to :class:`ToGRetriever` except that Step D (entity pruning)
    selects an entity by **random sampling** instead of asking the LLM to
    score candidates.  This halves the number of LLM calls per query from
    ``2*N*D + D + 1`` to ``N*D + D + 1``, where N is ``beam_width`` and D
    is the exploration depth actually taken.

    All constructor arguments, the ``retrieve()`` loop, and every other
    private helper are inherited unchanged from :class:`ToGRetriever`.
    Only ``_entity_prune()`` is overridden.

    Args:
        graph_store: Any :class:`~app.core.graph_store.BaseGraphStore` backend.
        llm:         Any :class:`~app.core.llm.BaseLLM` backend used for
                     relation pruning, reasoning checks, and answer generation.
                     **Not** called during entity pruning.
        beam_width:  Maximum relations kept per entity per hop (paper default N=3).
        depth_max:   Hard cap on exploration iterations (paper default D_max=3).
        examples:    Few-shot demonstrations forwarded to relation-pruning,
                     reasoning, and generation prompts.
                     Defaults to :data:`~app.prompts.TOG_DEFAULT_EXAMPLES`.
    """

    async def _entity_prune_batch(
        self,
        question: str,
        path_candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    ) -> list[tuple[dict[str, Any] | None, float]]:
        """Randomly select one entity per path (Phase 2, Step D) — no LLM calls.

        Overrides :meth:`ToGRetriever._entity_prune_batch` to use uniform
        random sampling instead of LLM scoring, keeping Step D cost at O(1)
        regardless of the number of pending paths.

        If the candidate pool is smaller than ``beam_width``, the full pool
        is used; otherwise ``beam_width`` entities are sampled and one is
        chosen from that sample uniformly at random.

        Args:
            question:        Unused — present only to satisfy the overridden
                             method signature.
            path_candidates: List of (pending_path_dict, candidate_entity_dicts)
                             tuples.

        Returns:
            List of (finalised_path_dict | None, score) tuples, same length and
            order as ``path_candidates``.  Score is always 1.0 (uniform random
            selection assigns equal weight to all paths).
        """
        new_paths: list[tuple[dict[str, Any] | None, float]] = []
        for pending, candidates in path_candidates:
            if not candidates:
                new_paths.append((None, 0.0))
                continue
            pool = (
                candidates
                if len(candidates) <= self._beam_width
                else random.sample(candidates, self._beam_width)
            )
            selected = random.choice(pool)
            new_paths.append((
                {
                    "nodes": pending["nodes"] + [selected],
                    "relations": list(pending["relations"]),
                    "edge_chunk_ids": list(pending["edge_chunk_ids"]) + [selected.get("edge_chunk_id", "")],
                },
                1.0,
            ))
        return new_paths
