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
#   {"nodes": list[str], "relations": list[str]}
#   Invariant for a *complete* path: len(nodes) == len(relations) + 1
#   A path of depth D has D+1 nodes and D relations.
#   Example (depth 2): {"nodes": ["e0", "e1", "e2"], "relations": ["r1", "r2"]}
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
        path: Dict with keys ``"nodes"`` (``list[str]``) and
              ``"relations"`` (``list[str]``), satisfying
              ``len(nodes) == len(relations) + 1``.

    Returns:
        Formatted string, e.g.
        ``"(node0, rel0, node1), (node1, rel1, node2)"``
        or just ``"node0"`` when ``len(nodes) == 1``.
    """
    nodes: list[str] = path.get("nodes", [])
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
    ) -> None:
        self._graph_store = graph_store
        self._llm = llm
        self._beam_width = beam_width
        self._depth_max = depth_max
        self._examples: list[dict] = (
            examples if examples is not None else TOG_DEFAULT_EXAMPLES
        )

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
        topic_entities = await self._extract_topic_entities(query)
        logger.info("ToGRetriever: topic entities=%s", topic_entities)

        if not topic_entities:
            logger.warning(
                "ToGRetriever: no topic entities found — falling back to LLM-only"
            )
            answer = await self._llm.generate(prompt=query)
            return RetrievalResult(context=answer, sources=[])

        # Initialise beam: one single-node path per topic entity.
        # Invariant: len(path["nodes"]) == len(path["relations"]) + 1
        paths: list[dict[str, Any]] = [
            {"nodes": [entity], "relations": []} for entity in topic_entities
        ]
        visited_entities: set[str] = set(topic_entities)

        # ── Phase 2: Iterative exploration ──────────────────────────────────
        answer: str | None = None

        for depth in range(1, self._depth_max + 1):
            logger.info("ToGRetriever: exploration depth=%d / %d", depth, self._depth_max)

            # E^{D-1}: tail entity of each current path (last node).
            tail_entities: list[str] = [p["nodes"][-1] for p in paths]

            # ── Step A: Relation search ──────────────────────────────────
            # Retrieve all candidate relations for each unique tail entity.
            entity_relations: dict[str, list[str]] = {}
            for entity in dict.fromkeys(tail_entities):  # deduplicate, preserve order
                relations = await self._graph_store.get_relations(entity)
                entity_relations[entity] = relations
                logger.debug(
                    "ToGRetriever [A]: entity=%r → %d relation(s)", entity, len(relations)
                )

            # ── Step B: Relation pruning ──────────────────────────────────
            # For each entity ask the LLM to score and keep ≤ beam_width relations.
            entity_pruned_relations: dict[str, list[str]] = {}
            for entity, relations in entity_relations.items():
                if not relations:
                    entity_pruned_relations[entity] = []
                    continue
                pruned = await self._prune_relations(query, entity, relations)
                entity_pruned_relations[entity] = pruned
                logger.debug(
                    "ToGRetriever [B]: pruned relations for entity=%r → %s",
                    entity,
                    pruned,
                )

            # Extend each path with each of its tail entity's pruned relations,
            # producing *pending* paths (len(relations) == len(nodes), no new entity yet).
            pending_paths: list[dict[str, Any]] = []
            for path in paths:
                tail = path["nodes"][-1]
                for rel in entity_pruned_relations.get(tail, []):
                    pending_paths.append(
                        {
                            "nodes": list(path["nodes"]),
                            "relations": list(path["relations"]) + [rel],
                        }
                    )

            if not pending_paths:
                logger.info(
                    "ToGRetriever: no expandable paths at depth=%d — stopping", depth
                )
                break

            # ── Step C: Entity search ─────────────────────────────────────
            # For each pending path retrieve candidate entities in both directions.
            path_candidates: list[tuple[dict[str, Any], list[str]]] = []
            for pending in pending_paths:
                tail_entity = pending["nodes"][-1]  # last confirmed node
                relation = pending["relations"][-1]  # the newly appended relation

                tail_results: list[str] = await self._graph_store.get_tail_entities(
                    tail_entity, relation
                )
                head_results: list[str] = await self._graph_store.get_head_entities(
                    tail_entity, relation
                )
                # Deduplicate while preserving order (tail-direction first).
                candidates: list[str] = list(
                    dict.fromkeys(tail_results + head_results)
                )
                logger.debug(
                    "ToGRetriever [C]: entity=%r relation=%r → %d candidate(s)",
                    tail_entity,
                    relation,
                    len(candidates),
                )
                path_candidates.append((pending, candidates))

            # ── Step D: Entity pruning ────────────────────────────────────
            # Ask the LLM to score candidates; keep the top-scored entity per path.
            new_paths: list[dict[str, Any]] = []
            for pending, candidates in path_candidates:
                if not candidates:
                    logger.debug(
                        "ToGRetriever [D]: no candidates for pending path %s — skipping",
                        pending,
                    )
                    continue
                relation = pending["relations"][-1]
                selected = await self._entity_prune(query, relation, candidates, pending)
                if selected is None:
                    continue
                # Finalise: append selected entity → complete path at depth D.
                new_path: dict[str, Any] = {
                    "nodes": pending["nodes"] + [selected],
                    "relations": list(pending["relations"]),
                }
                new_paths.append(new_path)
                visited_entities.add(selected)
                logger.debug("ToGRetriever [D]: finalised path → %s", _format_path(new_path))

            if not new_paths:
                logger.info(
                    "ToGRetriever: no paths finalised at depth=%d — stopping", depth
                )
                break

            paths = new_paths  # E^D is now the tail of each path in new_paths

            # ── Phase 3: Reasoning sufficiency check ─────────────────────
            path_strings = [_format_path(p) for p in paths]
            sufficient = await self._check_reasoning(query, path_strings)
            logger.info(
                "ToGRetriever [Phase 3]: depth=%d sufficient=%s paths=%d",
                depth,
                sufficient,
                len(path_strings),
            )

            if sufficient:
                # Phase 4: Generate the final answer from accumulated paths.
                answer = await self._generate_answer(query, path_strings)
                break
            # "No" and depth < depth_max → implicit continue to next iteration.
            # "No" and depth == depth_max → loop ends; fallback applied below.

        # If depth_max reached without a "Yes", generate from LLM-only knowledge
        # (no paths — per Section 2.1 of the paper).
        if answer is None:
            logger.info(
                "ToGRetriever: depth limit reached without sufficient context "
                "— using LLM-only generation"
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
    ) -> bool:
        """Ask the LLM whether the current paths are sufficient to answer (Phase 3).

        Uses :func:`~app.core.llm.BaseLLM.generate` (not ``generate_structured``)
        and parses "Yes" / "No" from the free-form response (case-insensitive,
        leading/trailing whitespace ignored).

        Args:
            question:     The user's natural-language question.
            path_strings: Formatted triple-chain strings from all current paths.

        Returns:
            ``True`` if the LLM response starts with "yes" (case-insensitive),
            ``False`` otherwise.
        """
        user_prompt = tog_reasoning_user_prompt(
            question=question,
            paths=path_strings,
            examples=self._examples,
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
    ) -> str:
        """Generate the final answer from accumulated reasoning paths (Phase 4).

        Paths are provided as triple chains — ``"(e1, r1, e2), (e2, r2, e3)"`` —
        which is the best-performing format reported in the paper (Appendix E.3.4).

        Args:
            question:     The user's natural-language question.
            path_strings: Formatted triple-chain strings from all current paths.

        Returns:
            The LLM-generated answer string.
        """
        user_prompt = tog_generate_user_prompt(
            question=question,
            paths=path_strings,
            examples=self._examples,
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
