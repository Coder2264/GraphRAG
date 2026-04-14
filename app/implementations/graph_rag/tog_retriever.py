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

import logging
import random
from typing import Any

from pydantic import BaseModel, Field

from app.core.graph_store import BaseGraphStore
from app.core.llm import BaseLLM
from app.core.retriever import BaseRetriever, RetrievalResult
from app.core.vector_store import BaseVectorStore
from app.prompts import (
    TOG_DEFAULT_EXAMPLES,
    TOG_ENTITY_PRUNE_SYSTEM_PROMPT,
    TOG_GENERATE_SYSTEM_PROMPT,
    TOG_REASONING_SYSTEM_PROMPT,
    TOG_RELATION_PRUNE_SYSTEM_PROMPT,
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
    chunks: list[str] = []
    for cid in chunk_ids[:3]:
        record = await vector_store.get(cid)
        if record and record.get("content"):
            chunks.append(record["content"][:1500])
    return chunks


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
        examples: list[dict] | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self._graph_store = graph_store
        self._llm = llm
        self._beam_width = beam_width
        self._depth_max = depth_max
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

        # Resolve entity name strings → graph node dicts via search_nodes.
        # Stored node IDs are "{doc_id}__slug"; querying by raw name would miss them.
        topic_entities: list[dict[str, Any]] = []
        for name in topic_entity_names:
            matches = await self._graph_store.search_nodes(name, top_k=2)
            if matches:
                best = matches[0]
                topic_entities.append({
                    "id": best.get("id", name),
                    "name": best.get("name", name),
                    "source_chunk_id": best.get("source_chunk_id", ""),
                })
                logger.info(
                    "ToGRetriever [Phase 1]: resolved %r → id=%r",
                    name, best.get("id", name),
                )
            else:
                logger.warning(
                    "ToGRetriever [Phase 1]: no graph node found for %r — using name as ID",
                    name,
                )
                topic_entities.append({"id": name, "name": name, "source_chunk_id": ""})

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
            logger.info("ToGRetriever: exploration depth=%d / %d", depth, self._depth_max)

            # E^{D-1}: tail entity dict of each current path (last node).
            tail_dicts: list[dict[str, Any]] = [p["nodes"][-1] for p in paths]

            # ── Step A: Relation search ──────────────────────────────────
            # Retrieve all candidate relations for each unique tail entity ID.
            entity_relations: dict[str, list[str]] = {}
            seen_tail_ids: dict[str, dict] = {}
            for td in tail_dicts:
                eid = td["id"]
                if eid in seen_tail_ids:
                    continue
                seen_tail_ids[eid] = td
                relations = await self._graph_store.get_relations(eid)
                entity_relations[eid] = relations
                logger.debug(
                    "ToGRetriever [A]: entity_id=%r → %d relation(s)", eid, len(relations)
                )

            # ── Step B: Relation pruning ──────────────────────────────────
            # For each entity ask the LLM to score and keep ≤ beam_width relations.
            # Pass the human-readable name to the LLM, not the internal ID.
            entity_pruned_relations: dict[str, list[str]] = {}
            for eid, relations in entity_relations.items():
                if not relations:
                    entity_pruned_relations[eid] = []
                    continue
                entity_name = seen_tail_ids[eid].get("name", eid)
                pruned = await self._prune_relations(query, entity_name, relations)
                entity_pruned_relations[eid] = pruned
                logger.debug(
                    "ToGRetriever [B]: pruned relations for entity_id=%r → %s",
                    eid,
                    pruned,
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

            # ── Step C: Entity search ─────────────────────────────────────
            # For each pending path retrieve candidate entity dicts in both directions.
            path_candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
            for pending in pending_paths:
                tail_dict = pending["nodes"][-1]
                tail_id = tail_dict["id"]
                relation = pending["relations"][-1]

                tail_results = await self._graph_store.get_tail_entities(tail_id, relation)
                head_results = await self._graph_store.get_head_entities(tail_id, relation)
                # Deduplicate by id, tail-direction first.
                seen_ids: set[str] = set()
                candidates: list[dict[str, Any]] = []
                for c in tail_results + head_results:
                    if c["id"] not in seen_ids:
                        seen_ids.add(c["id"])
                        candidates.append(c)
                logger.debug(
                    "ToGRetriever [C]: entity_id=%r relation=%r → %d candidate(s)",
                    tail_id,
                    relation,
                    len(candidates),
                )
                path_candidates.append((pending, candidates))

            # ── Step D: Entity pruning ────────────────────────────────────
            # Pass candidate names to LLM; look up the selected name in candidate dicts.
            new_paths: list[dict[str, Any]] = []
            for pending, candidates in path_candidates:
                if not candidates:
                    logger.debug(
                        "ToGRetriever [D]: no candidates for pending path — skipping"
                    )
                    continue
                relation = pending["relations"][-1]
                candidate_names = [c["name"] for c in candidates]
                selected_name = await self._entity_prune(
                    query, relation, candidate_names, pending
                )
                if selected_name is None:
                    continue
                # Look up the full dict by name (case-insensitive).
                selected_dict = next(
                    (c for c in candidates if c["name"].lower() == selected_name.lower()),
                    {"id": selected_name, "name": selected_name, "source_chunk_id": "", "edge_chunk_id": ""},
                )
                edge_chunk_id = selected_dict.get("edge_chunk_id", "")
                # Finalise: append selected entity → complete path at depth D.
                new_path: dict[str, Any] = {
                    "nodes": pending["nodes"] + [selected_dict],
                    "relations": list(pending["relations"]),
                    "edge_chunk_ids": list(pending["edge_chunk_ids"]) + [edge_chunk_id],
                }
                new_paths.append(new_path)
                visited_entities.add(selected_dict["id"])
                logger.debug("ToGRetriever [D]: finalised path → %s", _format_path(new_path))

            if not new_paths:
                logger.info(
                    "ToGRetriever: no paths finalised at depth=%d — stopping", depth
                )
                break

            paths = new_paths  # E^D is now the tail of each path in new_paths

            # ── Phase 3: Reasoning sufficiency check ─────────────────────
            path_strings = [_format_path(p) for p in paths]
            source_chunks = await _collect_source_chunks(paths, self._vector_store)
            sufficient = await self._check_reasoning(query, path_strings, source_chunks)
            logger.info(
                "ToGRetriever [Phase 3]: depth=%d sufficient=%s paths=%d chunks=%d",
                depth,
                sufficient,
                len(path_strings),
                len(source_chunks),
            )

            if sufficient:
                # Phase 4: Generate the final answer from accumulated paths.
                answer = await self._generate_answer(query, path_strings, source_chunks)
                break
            # "No" and depth < depth_max → implicit continue to next iteration.
            # "No" and depth == depth_max → loop ends; fallback applied below.

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
            "ToGRetriever: done; visited=%d entity/entities, answer=%d char(s)",
            len(visited_entities),
            len(answer),
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
        result: _TopicEntities = await self._llm.generate_structured(
            prompt=question,
            response_model=_TopicEntities,
            system_prompt=(
                "Extract the main topic entities (proper nouns, named entities, "
                "key concepts) from the question. "
                'Return ONLY a JSON object: {"entities": ["entity1", "entity2", ...]}'
            ),
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
        result: _RelationPruneResult = await self._llm.generate_structured(
            prompt=user_prompt,
            response_model=_RelationPruneResult,
            system_prompt=system_prompt,
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
        result: _EntityPruneResult = await self._llm.generate_structured(
            prompt=user_prompt,
            response_model=_EntityPruneResult,
            system_prompt=TOG_ENTITY_PRUNE_SYSTEM_PROMPT,
        )
        if not result or not result.entities:
            return None
        best = max(result.entities, key=lambda e: e.score)
        return best.entity

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
        response = await self._llm.generate(
            prompt=user_prompt,
            system_prompt=TOG_REASONING_SYSTEM_PROMPT,
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
        return await self._llm.generate(
            prompt=user_prompt,
            system_prompt=TOG_GENERATE_SYSTEM_PROMPT,
        )


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

    async def _entity_prune(
        self,
        question: str,
        relation: str,
        candidates: list[str],
        path: dict[str, Any],
    ) -> str | None:
        """Randomly sample one entity from the candidates (Phase 2, Step D).

        Replaces the LLM-based scoring of :class:`ToGRetriever` with a
        uniform random draw, eliminating one LLM call per candidate set.

        If the candidate pool is smaller than ``beam_width``, the full pool
        is used; otherwise ``beam_width`` entities are sampled and one is
        chosen from that sample uniformly at random.

        Args:
            question:   The user's natural-language question (unused — present
                        only to satisfy the overridden method signature).
            relation:   The relation label being traversed (unused — present
                        only to satisfy the overridden method signature).
            candidates: Candidate entity names reachable via ``relation``.
            path:       The pending path dict being extended (unused — present
                        only to satisfy the overridden method signature).

        Returns:
            A randomly chosen entity name, or ``None`` if ``candidates`` is
            empty.
        """
        if not candidates:
            return None
        pool = (
            candidates
            if len(candidates) <= self._beam_width
            else random.sample(candidates, self._beam_width)
        )
        return random.choice(pool)
