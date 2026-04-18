"""
ManualSessionService — human-in-the-loop query state machine.

Replaces all LLM calls in RAG, GraphRAG, and ToG query modes with
pauses that return the exact prompt for the user to paste into ChatGPT
or Gemini.  All Neo4j and vector-search steps run automatically.

SRP: Owns only session state and step dispatch; delegates I/O to injected
     graph_store / vector_store / embedder passed per-call.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.config import settings
from app.core.embedder import BaseEmbedder
from app.core.graph_store import BaseGraphStore
from app.core.vector_store import BaseVectorStore
from app.models.manual import (
    LLMPromptPayload,
    SessionInfoResponse,
    SessionResponse,
    SessionStatus,
)
from app.prompts import (
    BEAM_SEARCH_COMPRESS_SYSTEM_PROMPT,
    BEAM_SEARCH_SEED_SYSTEM_PROMPT,
    GRAPH_RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    TOG_BATCH_ENTITY_PRUNE_SYSTEM_PROMPT,
    TOG_BATCH_RELATION_PRUNE_SYSTEM_PROMPT,
    TOG_DEFAULT_EXAMPLES,
    TOG_GENERATE_SYSTEM_PROMPT,
    TOG_REASONING_SYSTEM_PROMPT,
    beam_search_compress_user_prompt,
    beam_search_eval_system_prompt,
    beam_search_eval_user_prompt,
    beam_search_seed_user_prompt,
    graph_rag_user_prompt,
    rag_user_prompt,
    tog_batch_entity_prune_user_prompt,
    tog_batch_relation_prune_user_prompt,
    tog_generate_user_prompt,
    tog_reasoning_user_prompt,
)


# ---------------------------------------------------------------------------
# Robust JSON parser for LLM responses
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_llm_json(raw: str) -> Any:
    """Parse a JSON-ish string that came from an LLM or was pasted by a user.

    Tries in order:
    1. Strip markdown code fences, then json.loads  (double-quoted JSON)
    2. ast.literal_eval  (single-quoted Python dict — common when pasting into
       the Swagger UI without escaping inner quotes)
    3. Raise ValueError with a clear message.
    """
    text = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Strategy 1: standard JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Python literal (handles single-quoted dicts/lists)
    try:
        result = ast.literal_eval(text)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError):
        pass

    raise ValueError(
        f"Could not parse as JSON. Received: {text[:120]!r}. "
        "Tip: use single quotes inside llm_response to avoid escaping "
        "(e.g. \"llm_response\": \"{'entities': ['Foo', 'Bar']}\")."
    )


# ---------------------------------------------------------------------------
# Path helpers (mirrors tog_retriever.py but standalone to avoid coupling)
# ---------------------------------------------------------------------------


def _format_path(path: dict[str, Any]) -> str:
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
    triples = [f"({nodes[i]}, {rel}, {nodes[i + 1]})" for i, rel in enumerate(relations)]
    return ", ".join(triples)


async def _collect_source_chunks(
    paths: list[dict[str, Any]],
    vector_store: BaseVectorStore | None,
) -> list[str]:
    if vector_store is None:
        return []
    seen: set[str] = set()
    chunk_ids: list[str] = []
    for path in paths:
        for node in path.get("nodes", []):
            cid = node.get("source_chunk_id", "") if isinstance(node, dict) else ""
            if cid and cid not in seen:
                seen.add(cid)
                chunk_ids.append(cid)
        for cid in path.get("edge_chunk_ids", []):
            if cid and cid not in seen:
                seen.add(cid)
                chunk_ids.append(cid)
        if len(chunk_ids) >= 3:
            break
    records = await asyncio.gather(*[vector_store.get(cid) for cid in chunk_ids[:3]])
    return [r["content"][:1500] for r in records if r and r.get("content")]


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


@dataclass
class ManualSession:
    session_id: str
    mode: str
    query: str
    top_k: int
    step: str
    step_number: int
    data: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ManualSessionService:
    """
    In-memory session store and step dispatcher for human-in-the-loop querying.

    Sessions expire after TTL_SECONDS (1 hour).  Expiry is checked lazily on
    each get_session call.
    """

    TTL_SECONDS = 3600

    def __init__(self) -> None:
        self._sessions: dict[str, ManualSession] = {}

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(self, query: str, mode: str, top_k: int) -> ManualSession:
        self._purge_expired()
        session_id = uuid.uuid4().hex[:12]
        session = ManualSession(
            session_id=session_id,
            mode=mode,
            query=query,
            top_k=top_k,
            step="start",
            step_number=0,
            data=self._init_data(mode),
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ManualSession:
        self._purge_expired()
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def session_info(self, session_id: str) -> SessionInfoResponse:
        session = self.get_session(session_id)
        status = "complete" if session.step == "done" else "needs_llm"
        return SessionInfoResponse(
            session_id=session.session_id,
            mode=session.mode,
            query=session.query,
            status=status,
            step=session.step,
            step_number=session.step_number,
            accumulated_context=self._context_snapshot(session),
            created_at=session.created_at.isoformat(),
        )

    def _purge_expired(self) -> None:
        now = datetime.utcnow()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if (now - s.created_at).total_seconds() > self.TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]

    # ------------------------------------------------------------------
    # Advance (main entry point)
    # ------------------------------------------------------------------

    async def advance(
        self,
        session_id: str,
        llm_response: str | None,
        graph_store: BaseGraphStore,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
    ) -> SessionResponse:
        """Run the next step(s) for the session.

        If llm_response is None this is the initial advance (start step).
        Otherwise llm_response is the text the user got from their LLM webapp.

        Raises:
            KeyError: if session_id is unknown or expired.
            RuntimeError: if the session is already complete.
            ValueError: if llm_response cannot be parsed for the expected format.
        """
        session = self.get_session(session_id)

        if session.step == "done":
            raise RuntimeError("Session is already complete.")

        if session.mode == "rag":
            return await self._advance_rag(session, llm_response, vector_store, embedder)
        if session.mode == "graphrag":
            return await self._advance_graphrag(session, llm_response, graph_store)
        if session.mode == "tog":
            return await self._advance_tog(session, llm_response, graph_store, vector_store)

        raise ValueError(f"Unknown mode: {session.mode!r}")

    # ------------------------------------------------------------------
    # RAG flow
    # ------------------------------------------------------------------

    async def _advance_rag(
        self,
        session: ManualSession,
        llm_response: str | None,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
    ) -> SessionResponse:
        data = session.data

        if session.step == "start":
            vector = await embedder.embed(session.query)
            results = await vector_store.search(vector, top_k=session.top_k)
            data["chunks"] = [
                {
                    "content": r.get("content", ""),
                    "score": float(r.get("score", 0)),
                    "doc_id": r.get("doc_id", ""),
                }
                for r in results
            ]
            session.step = "answer_generation"
            session.step_number = 1

            context = "\n\n".join(
                f"[score={c['score']:.2f}] {c['content']}" for c in data["chunks"]
            )
            return self._make_response(
                session=session,
                system_prompt=RAG_SYSTEM_PROMPT,
                user_prompt=rag_user_prompt(session.query, context),
                response_format="Free text — paste the LLM's answer directly.",
            )

        if session.step == "answer_generation":
            data["answer"] = llm_response
            session.step = "done"
            session.step_number = 2
            return self._make_done(session, llm_response or "")

        raise ValueError(f"Unexpected RAG step: {session.step!r}")

    # ------------------------------------------------------------------
    # GraphRAG flow
    # ------------------------------------------------------------------

    async def _advance_graphrag(
        self,
        session: ManualSession,
        llm_response: str | None,
        graph_store: BaseGraphStore,
    ) -> SessionResponse:
        data = session.data

        if session.step == "start":
            session.step = "seed_extraction"
            session.step_number = 1
            return self._make_response(
                session=session,
                system_prompt=BEAM_SEARCH_SEED_SYSTEM_PROMPT,
                user_prompt=beam_search_seed_user_prompt(session.query),
                response_format='JSON: {"keywords": ["entity1", "entity2"]}',
            )

        if session.step == "seed_extraction":
            try:
                parsed = _parse_llm_json(llm_response or "")
                keywords: list[str] = list(parsed["keywords"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f'Missing "keywords" key in response. Expected: {{"keywords": ["entity1", "entity2"]}}. Got: {exc}'
                )

            data["keywords"] = keywords

            # Resolve keywords → frontier nodes via Neo4j
            all_nodes: list[dict[str, Any]] = []
            for kw in keywords:
                nodes = await graph_store.search_nodes(kw, top_k=5)
                all_nodes.extend(nodes)

            seen: set[str] = set()
            frontier: list[dict[str, Any]] = []
            for n in all_nodes:
                nid = n.get("id", "")
                if nid and nid not in seen:
                    seen.add(nid)
                    frontier.append(n)

            data["frontier_nodes"] = frontier
            data["visited_node_ids"] = [n.get("id") for n in frontier]

            # Expand subgraph for this frontier
            await self._graphrag_expand(data, graph_store)

            session.step = "node_evaluation"
            session.step_number = 2
            return self._make_graphrag_eval(session)

        if session.step == "node_evaluation":
            try:
                parsed = _parse_llm_json(llm_response or "")
                sufficient = bool(parsed["has_sufficient_context"])
                selected_ids: list[str] = list(parsed.get("selected_ids", []))
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f'Missing key in response: {exc}. Expected: {{"has_sufficient_context": true/false, "selected_ids": ["id1"]}}'
                )

            data["sufficient_context"] = sufficient
            data["selected_ids_pending"] = selected_ids
            session.step = "context_compression"
            session.step_number += 1

            return self._make_response(
                session=session,
                system_prompt=BEAM_SEARCH_COMPRESS_SYSTEM_PROMPT,
                user_prompt=beam_search_compress_user_prompt(
                    session.query, data.get("raw_graph_data", "")
                ),
                response_format="Free text — paste the LLM's distilled summary directly.",
            )

        if session.step == "context_compression":
            # Append to accumulated summary
            prev = data.get("compressed_summary", "")
            data["compressed_summary"] = (prev + "\n" + (llm_response or "")).strip()
            data["iteration"] = data.get("iteration", 0) + 1

            # Update frontier to selected IDs
            selected_ids = data.get("selected_ids_pending", [])
            selected_set = set(selected_ids)
            candidates = data.get("candidate_nodes", [])
            new_frontier = [n for n in candidates if n.get("id") in selected_set]

            visited: list[str] = data.get("visited_node_ids", [])
            visited_set = set(visited)
            for n in new_frontier:
                nid = n.get("id", "")
                if nid and nid not in visited_set:
                    visited.append(nid)
                    visited_set.add(nid)
            data["visited_node_ids"] = visited

            sufficient = data.get("sufficient_context", False)
            at_limit = data["iteration"] >= data["max_iterations"]

            if sufficient or at_limit or not new_frontier:
                data["frontier_nodes"] = new_frontier
                session.step = "answer_generation"
                session.step_number += 1
                return self._make_response(
                    session=session,
                    system_prompt=GRAPH_RAG_SYSTEM_PROMPT,
                    user_prompt=graph_rag_user_prompt(
                        session.query, data["compressed_summary"]
                    ),
                    response_format="Free text — paste the LLM's final answer directly.",
                )

            # Continue loop: expand new frontier
            data["frontier_nodes"] = new_frontier
            await self._graphrag_expand(data, graph_store)
            session.step = "node_evaluation"
            session.step_number += 1
            return self._make_graphrag_eval(session)

        if session.step == "answer_generation":
            data["answer"] = llm_response
            session.step = "done"
            session.step_number += 1
            return self._make_done(session, llm_response or "")

        raise ValueError(f"Unexpected GraphRAG step: {session.step!r}")

    async def _graphrag_expand(
        self, data: dict[str, Any], graph_store: BaseGraphStore
    ) -> None:
        """Expand all frontier nodes by one hop and populate candidate_nodes + raw_graph_data."""
        frontier = data.get("frontier_nodes", [])
        visited_set = set(data.get("visited_node_ids", []))

        all_candidate_nodes: list[dict[str, Any]] = []
        all_edges: list[dict[str, Any]] = []

        for node in frontier:
            nid = node.get("id", "")
            if not nid:
                continue
            subgraph = await graph_store.get_subgraph(nid, depth=1)
            for n in subgraph.get("nodes", []):
                if n.get("id") not in visited_set:
                    all_candidate_nodes.append(n)
            all_edges.extend(subgraph.get("edges", []))

        # Deduplicate
        seen: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for n in all_candidate_nodes:
            nid = n.get("id", "")
            if nid and nid not in seen:
                seen.add(nid)
                candidates.append(n)

        data["candidate_nodes"] = candidates

        nodes_text = "\n".join(
            f"  Node: id={n.get('id','?')}  name={n.get('name','?')}  "
            f"type={n.get('type','?')}  desc={n.get('description','')}"
            for n in candidates
        )
        edges_text = "\n".join(
            f"  Edge: ({e.get('src', e.get('source', '?'))})"
            f"-[{e.get('relation', e.get('type', '?'))}]->"
            f"({e.get('dst', e.get('target', '?'))})"
            for e in all_edges
        )
        data["raw_graph_data"] = f"Nodes:\n{nodes_text or '(none)'}\n\nEdges:\n{edges_text or '(none)'}"

    # ------------------------------------------------------------------
    # ToG flow
    # ------------------------------------------------------------------

    async def _advance_tog(
        self,
        session: ManualSession,
        llm_response: str | None,
        graph_store: BaseGraphStore,
        vector_store: BaseVectorStore,
    ) -> SessionResponse:
        data = session.data

        if session.step == "start":
            session.step = "entity_extraction"
            session.step_number = 1
            return self._make_response(
                session=session,
                system_prompt=(
                    "Extract the main topic entities (proper nouns, named entities, "
                    "key concepts) from the question. "
                    'Return ONLY a JSON object: {"entities": ["entity1", "entity2", ...]}'
                ),
                user_prompt=f"Question: {session.query}\n\nJSON:",
                response_format='JSON: {"entities": ["entity1", "entity2"]}',
            )

        if session.step == "entity_extraction":
            try:
                parsed = _parse_llm_json(llm_response or "")
                entity_names: list[str] = list(parsed["entities"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f'Missing "entities" key in response: {exc}. Expected: {{"entities": ["entity1", "entity2"]}}'
                )

            data["topic_entities"] = entity_names

            # Resolve entity names → graph node dicts
            async def _resolve(name: str) -> dict[str, Any]:
                matches = await graph_store.search_nodes(name, top_k=2)
                if matches:
                    best = matches[0]
                    return {
                        "id": best.get("id", name),
                        "name": best.get("name", name),
                        "source_chunk_id": best.get("source_chunk_id", ""),
                    }
                return {"id": name, "name": name, "source_chunk_id": ""}

            resolved = list(await asyncio.gather(*[_resolve(n) for n in entity_names]))

            paths: list[dict[str, Any]] = [
                {"nodes": [e], "relations": [], "edge_chunk_ids": []}
                for e in resolved
            ]
            data["paths"] = paths
            data["visited_entity_ids"] = [e["id"] for e in resolved]

            if not paths:
                session.step = "answer_generation"
                session.step_number = 2
                return self._make_response(
                    session=session,
                    system_prompt=TOG_GENERATE_SYSTEM_PROMPT,
                    user_prompt=tog_generate_user_prompt(
                        session.query, [], TOG_DEFAULT_EXAMPLES
                    ),
                    response_format="Free text — paste the LLM's answer directly.",
                )

            # Auto: fetch relations for all frontier entities
            await self._tog_fetch_relations(data, graph_store)
            session.step = "relation_pruning"
            session.step_number = 2
            return self._make_tog_relation_pruning(session)

        if session.step == "relation_pruning":
            try:
                parsed = _parse_llm_json(llm_response or "")
                results: list[dict[str, Any]] = list(parsed["results"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f'Missing "results" key in response: {exc}. Expected: {{"results": [{{"idx": 0, "relations": [{{"relation": "...", "score": 0.9}}]}}]}}'
                )

            entities_with_rels = [
                (eid, rels)
                for eid, rels in data["pending_entity_relations"].items()
                if rels
            ]
            beam_width: int = data["beam_width"]
            entity_pruned: dict[str, list[str]] = {}
            for entry in results:
                idx = entry.get("idx", -1)
                if 0 <= idx < len(entities_with_rels):
                    eid = entities_with_rels[idx][0]
                    sorted_rels = sorted(
                        entry.get("relations", []),
                        key=lambda r: r.get("score", 0),
                        reverse=True,
                    )
                    entity_pruned[eid] = [r["relation"] for r in sorted_rels[:beam_width]]

            # Build pending paths (one per path × selected relation)
            pending_paths: list[dict[str, Any]] = []
            for path in data["paths"]:
                tail_id = path["nodes"][-1]["id"]
                for rel in entity_pruned.get(tail_id, []):
                    pending_paths.append({
                        "nodes": list(path["nodes"]),
                        "relations": list(path["relations"]) + [rel],
                        "edge_chunk_ids": list(path["edge_chunk_ids"]),
                    })

            if not pending_paths:
                session.step = "answer_generation"
                session.step_number += 1
                paths_str = [_format_path(p) for p in data["paths"]]
                chunks = await _collect_source_chunks(data["paths"], vector_store)
                return self._make_response(
                    session=session,
                    system_prompt=TOG_GENERATE_SYSTEM_PROMPT,
                    user_prompt=tog_generate_user_prompt(
                        session.query, paths_str, TOG_DEFAULT_EXAMPLES, chunks
                    ),
                    response_format="Free text — paste the LLM's answer directly.",
                )

            # Auto: fetch candidate entities for each pending path
            async def _get_candidates(
                pending: dict[str, Any],
            ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
                tail_id = pending["nodes"][-1]["id"]
                relation = pending["relations"][-1]
                tail_res, head_res = await asyncio.gather(
                    graph_store.get_tail_entities(tail_id, relation),
                    graph_store.get_head_entities(tail_id, relation),
                )
                seen: set[str] = set()
                candidates: list[dict[str, Any]] = []
                for c in tail_res + head_res:
                    if c["id"] not in seen:
                        seen.add(c["id"])
                        candidates.append(c)
                return pending, candidates

            path_cands = list(
                await asyncio.gather(*[_get_candidates(p) for p in pending_paths])
            )
            non_empty = [(p, c) for p, c in path_cands if c]

            data["pending_path_candidates"] = [
                {
                    "path": p,
                    "candidates": [
                        {
                            "id": c["id"],
                            "name": c["name"],
                            "source_chunk_id": c.get("source_chunk_id", ""),
                            "edge_chunk_id": c.get("edge_chunk_id", ""),
                        }
                        for c in cands
                    ],
                }
                for p, cands in non_empty
            ]

            session.step = "entity_pruning"
            session.step_number += 1
            return self._make_tog_entity_pruning(session)

        if session.step == "entity_pruning":
            try:
                parsed = _parse_llm_json(llm_response or "")
                results = list(parsed["results"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f'Missing "results" key in response: {exc}. Expected: {{"results": [{{"idx": 0, "entities": [{{"entity": "...", "score": 0.9}}]}}]}}'
                )

            path_candidates = data["pending_path_candidates"]
            max_paths: int = data["max_paths"]

            idx_to_selection: dict[int, tuple[str, float]] = {}
            for entry in results:
                idx = entry.get("idx", -1)
                entities = entry.get("entities", [])
                if entities and 0 <= idx < len(path_candidates):
                    best = max(entities, key=lambda e: e.get("score", 0))
                    idx_to_selection[idx] = (best["entity"], float(best.get("score", 0)))

            scored_paths: list[tuple[dict[str, Any], float]] = []
            for i, pc in enumerate(path_candidates):
                pending = pc["path"]
                candidates = pc["candidates"]
                selection = idx_to_selection.get(i)
                if selection is None:
                    continue
                sel_name, score = selection
                sel_dict = next(
                    (c for c in candidates if c["name"].lower() == sel_name.lower()),
                    {"id": sel_name, "name": sel_name, "source_chunk_id": "", "edge_chunk_id": ""},
                )
                scored_paths.append((
                    {
                        "nodes": pending["nodes"] + [sel_dict],
                        "relations": list(pending["relations"]),
                        "edge_chunk_ids": list(pending["edge_chunk_ids"])
                            + [sel_dict.get("edge_chunk_id", "")],
                    },
                    score,
                ))

            scored_paths.sort(key=lambda x: x[1], reverse=True)
            new_paths = [p for p, _ in scored_paths[:max_paths]]

            visited: list[str] = data["visited_entity_ids"]
            visited_set = set(visited)
            for p in new_paths:
                eid = p["nodes"][-1].get("id", "")
                if eid and eid not in visited_set:
                    visited.append(eid)
                    visited_set.add(eid)

            if new_paths:
                data["paths"] = new_paths
            data["depth"] = data.get("depth", 0) + 1

            paths_str = [_format_path(p) for p in data["paths"]]
            chunks = await _collect_source_chunks(data["paths"], vector_store)

            if not new_paths or data["depth"] >= data["max_depth"]:
                session.step = "answer_generation"
                session.step_number += 1
                return self._make_response(
                    session=session,
                    system_prompt=TOG_GENERATE_SYSTEM_PROMPT,
                    user_prompt=tog_generate_user_prompt(
                        session.query, paths_str, TOG_DEFAULT_EXAMPLES, chunks
                    ),
                    response_format="Free text — paste the LLM's answer directly.",
                )

            session.step = "reasoning_check"
            session.step_number += 1
            return self._make_response(
                session=session,
                system_prompt=TOG_REASONING_SYSTEM_PROMPT,
                user_prompt=tog_reasoning_user_prompt(
                    session.query, paths_str, TOG_DEFAULT_EXAMPLES, chunks
                ),
                response_format='Plain text: answer with just "Yes" or "No".',
            )

        if session.step == "reasoning_check":
            sufficient = (llm_response or "").strip().lower().startswith("yes")
            paths_str = [_format_path(p) for p in data["paths"]]
            chunks = await _collect_source_chunks(data["paths"], vector_store)

            if sufficient or data["depth"] >= data["max_depth"]:
                session.step = "answer_generation"
                session.step_number += 1
                return self._make_response(
                    session=session,
                    system_prompt=TOG_GENERATE_SYSTEM_PROMPT,
                    user_prompt=tog_generate_user_prompt(
                        session.query, paths_str, TOG_DEFAULT_EXAMPLES, chunks
                    ),
                    response_format="Free text — paste the LLM's answer directly.",
                )

            # Continue: fetch relations for new frontier
            await self._tog_fetch_relations(data, graph_store)
            session.step = "relation_pruning"
            session.step_number += 1
            return self._make_tog_relation_pruning(session)

        if session.step == "answer_generation":
            data["answer"] = llm_response
            session.step = "done"
            session.step_number += 1
            return self._make_done(session, llm_response or "")

        raise ValueError(f"Unexpected ToG step: {session.step!r}")

    async def _tog_fetch_relations(
        self, data: dict[str, Any], graph_store: BaseGraphStore
    ) -> None:
        """Fetch relations for all unique tail entities and store in data."""
        paths: list[dict[str, Any]] = data["paths"]
        unique_tails: dict[str, dict[str, Any]] = {}
        for p in paths:
            tail = p["nodes"][-1]
            unique_tails.setdefault(tail["id"], tail)

        rel_lists = await asyncio.gather(
            *[graph_store.get_relations(eid) for eid in unique_tails]
        )
        data["pending_entity_relations"] = dict(zip(unique_tails.keys(), rel_lists))
        data["pending_tail_names"] = {
            eid: unique_tails[eid].get("name", eid) for eid in unique_tails
        }

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------

    def _make_response(
        self,
        session: ManualSession,
        system_prompt: str,
        user_prompt: str,
        response_format: str,
    ) -> SessionResponse:
        step_label = session.step.replace("_", " ").title()
        return SessionResponse(
            session_id=session.session_id,
            status=SessionStatus.NEEDS_LLM,
            step_name=f"{session.mode.upper()}: {step_label}",
            step_number=session.step_number,
            llm_prompt=LLMPromptPayload(
                step_name=session.step,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_format,
            ),
            resume_endpoint=f"POST /api/v1/manual/sessions/{session.session_id}/resume",
            accumulated_context=self._context_snapshot(session),
            final_answer=None,
        )

    def _make_done(self, session: ManualSession, answer: str) -> SessionResponse:
        return SessionResponse(
            session_id=session.session_id,
            status=SessionStatus.COMPLETE,
            step_name="Complete",
            step_number=session.step_number,
            llm_prompt=None,
            resume_endpoint=None,
            accumulated_context=self._context_snapshot(session),
            final_answer=answer,
        )

    def _make_graphrag_eval(self, session: ManualSession) -> SessionResponse:
        data = session.data
        bw: int = data["beam_width"]
        return self._make_response(
            session=session,
            system_prompt=beam_search_eval_system_prompt(bw),
            user_prompt=beam_search_eval_user_prompt(
                session.query,
                bw,
                data.get("compressed_summary", ""),
                data.get("candidate_nodes", []),
            ),
            response_format=(
                'JSON: {"has_sufficient_context": true/false, "selected_ids": ["id1", "id2"]}'
            ),
        )

    def _make_tog_relation_pruning(self, session: ManualSession) -> SessionResponse:
        data = session.data
        bw: int = data["beam_width"]
        entity_relation_pairs = [
            (data["pending_tail_names"].get(eid, eid), rels)
            for eid, rels in data["pending_entity_relations"].items()
            if rels
        ]
        return self._make_response(
            session=session,
            system_prompt=TOG_BATCH_RELATION_PRUNE_SYSTEM_PROMPT.format(beam_width=bw),
            user_prompt=tog_batch_relation_prune_user_prompt(
                session.query, entity_relation_pairs, bw
            ),
            response_format=(
                '{"results": [{"idx": 0, "relations": [{"relation": "...", "score": 0.9}]}]}'
            ),
        )

    def _make_tog_entity_pruning(self, session: ManualSession) -> SessionResponse:
        data = session.data
        pcs = data["pending_path_candidates"]
        rel_name_pairs = [
            (pc["path"]["relations"][-1], [c["name"] for c in pc["candidates"]])
            for pc in pcs
        ]
        return self._make_response(
            session=session,
            system_prompt=TOG_BATCH_ENTITY_PRUNE_SYSTEM_PROMPT,
            user_prompt=tog_batch_entity_prune_user_prompt(session.query, rel_name_pairs),
            response_format=(
                '{"results": [{"idx": 0, "entities": [{"entity": "...", "score": 0.9}]}]}'
            ),
        )

    # ------------------------------------------------------------------
    # Context snapshot (for accumulated_context field)
    # ------------------------------------------------------------------

    def _context_snapshot(self, session: ManualSession) -> dict[str, Any]:
        data = session.data
        if session.mode == "rag":
            return {
                "chunk_count": len(data.get("chunks", [])),
                "chunks": data.get("chunks", []),
            }
        if session.mode == "graphrag":
            return {
                "iteration": data.get("iteration", 0),
                "keywords": data.get("keywords", []),
                "frontier_node_count": len(data.get("frontier_nodes", [])),
                "candidate_node_count": len(data.get("candidate_nodes", [])),
                "visited_node_count": len(data.get("visited_node_ids", [])),
                "compressed_summary": data.get("compressed_summary", ""),
                "sufficient_context": data.get("sufficient_context", False),
            }
        if session.mode == "tog":
            return {
                "depth": data.get("depth", 0),
                "max_depth": data.get("max_depth", 3),
                "topic_entities": data.get("topic_entities", []),
                "path_count": len(data.get("paths", [])),
                "paths": [_format_path(p) for p in data.get("paths", [])],
                "visited_entity_count": len(data.get("visited_entity_ids", [])),
            }
        return {}

    # ------------------------------------------------------------------
    # Data initialiser
    # ------------------------------------------------------------------

    @staticmethod
    def _init_data(mode: str) -> dict[str, Any]:
        if mode == "rag":
            return {"chunks": [], "answer": None}
        if mode == "graphrag":
            return {
                "iteration": 0,
                "max_iterations": 20,
                "beam_width": settings.beam_search_beam_width,
                "keywords": [],
                "frontier_nodes": [],
                "visited_node_ids": [],
                "candidate_nodes": [],
                "compressed_summary": "",
                "raw_graph_data": "",
                "sufficient_context": False,
                "selected_ids_pending": [],
                "answer": None,
            }
        if mode == "tog":
            return {
                "depth": 0,
                "max_depth": settings.tog_depth_max,
                "beam_width": settings.beam_search_beam_width,
                "max_paths": settings.tog_max_paths,
                "topic_entities": [],
                "paths": [],
                "visited_entity_ids": [],
                "pending_entity_relations": {},
                "pending_tail_names": {},
                "pending_path_candidates": [],
                "answer": None,
            }
        return {}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service: ManualSessionService | None = None


def get_manual_session_service() -> ManualSessionService:
    """Return the module-level ManualSessionService singleton."""
    global _service
    if _service is None:
        _service = ManualSessionService()
    return _service
