"""
ManualExtractService — human-in-the-loop document ingestion state machine.

Replaces LLM API calls during ingestion with pauses that return the exact
prompt for the user to paste into ChatGPT or Gemini (with the PDF uploaded).
Automatic steps (embedding, Neo4j writes, PostgreSQL writes) still run on
the server side — only the LLM extraction steps are externalised.

Two-step flow:
  1. chunk_extraction  — LLM reads the uploaded PDF and returns JSON chunks.
                         Server embeds each chunk and writes to PostgreSQL.
  2. entity_extraction — LLM extracts entities and relations from the combined
                         chunk text.  Server deduplicates and writes to Neo4j.

SRP: Owns only extraction-session state and step dispatch.
DIP: Depends on BaseEmbedder, BaseVectorStore, BaseGraphStore abstractions.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.embedder import BaseEmbedder
from app.core.graph_store import BaseGraphStore
from app.core.vector_store import BaseVectorStore
from app.models.manual import LLMPromptPayload, SessionStatus
from app.models.manual_extract import (
    ExtractionSessionInfoResponse,
    ExtractionSessionResponse,
)
from app.prompts import (
    CHUNK_EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    chunk_extraction_user_prompt,
    extraction_user_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Robust JSON parser (mirrors manual_session_service._parse_llm_json)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_llm_json(raw: str) -> Any:
    """Parse a JSON-ish string that may include markdown fences or single quotes.

    Args:
        raw: Raw text from the user (pasted LLM response).

    Returns:
        Parsed Python dict or list.

    Raises:
        ValueError: If the text cannot be parsed as JSON or a Python literal.
    """
    text = raw.strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        result = ast.literal_eval(text)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError):
        pass

    raise ValueError(
        f"Could not parse as JSON. Received: {text[:120]!r}. "
        "Tip: use single quotes inside llm_response to avoid escaping."
    )


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExtractionSession:
    """In-memory state for one manual extraction session."""

    session_id: str
    source: str
    processing_instruction: str
    doc_id: str
    step: str
    step_number: int
    data: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ManualExtractService:
    """
    In-memory session store and step dispatcher for human-in-the-loop ingestion.

    Sessions expire after TTL_SECONDS (1 hour).  Expiry is checked lazily on
    each get_session call.
    """

    TTL_SECONDS = 3600

    def __init__(self) -> None:
        self._sessions: dict[str, ExtractionSession] = {}

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        source: str,
        processing_instruction: str,
        doc_id: str | None,
    ) -> ExtractionSession:
        """Create a new extraction session and return it.

        Args:
            source:                  Document filename or label.
            processing_instruction:  Optional extraction hint for the LLM.
            doc_id:                  Optional pre-supplied document ID.
                                     A UUID is generated when omitted.

        Returns:
            The newly created ExtractionSession.
        """
        self._purge_expired()
        session_id = uuid.uuid4().hex[:12]
        session = ExtractionSession(
            session_id=session_id,
            source=source,
            processing_instruction=processing_instruction,
            doc_id=doc_id or str(uuid.uuid4()),
            step="start",
            step_number=0,
            data={"chunks": [], "entity_count": 0, "relation_count": 0},
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ExtractionSession:
        """Retrieve a live session by ID.

        Args:
            session_id: The session identifier.

        Returns:
            The ExtractionSession.

        Raises:
            KeyError: If the session does not exist or has expired.
        """
        self._purge_expired()
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def session_info(self, session_id: str) -> ExtractionSessionInfoResponse:
        """Return a metadata snapshot for the session without advancing it.

        Args:
            session_id: The session identifier.

        Returns:
            ExtractionSessionInfoResponse with current state.

        Raises:
            KeyError: If the session does not exist or has expired.
        """
        session = self.get_session(session_id)
        status = (
            SessionStatus.COMPLETE if session.step == "done" else SessionStatus.NEEDS_LLM
        )
        return ExtractionSessionInfoResponse(
            session_id=session.session_id,
            doc_id=session.doc_id,
            source=session.source,
            processing_instruction=session.processing_instruction,
            status=status.value,
            step=session.step,
            step_number=session.step_number,
            accumulated_context=self._context_snapshot(session),
            created_at=session.created_at.isoformat(),
        )

    # ------------------------------------------------------------------
    # Main dispatcher
    # ------------------------------------------------------------------

    async def advance(
        self,
        session_id: str,
        llm_response: str | None,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore,
        embedder: BaseEmbedder,
    ) -> ExtractionSessionResponse:
        """Advance the session by one step.

        When llm_response is None the session is at "start" and advances
        automatically to the first LLM pause.  On subsequent calls the
        provided LLM response is processed and the next prompt returned,
        until the session completes.

        Args:
            session_id:   The session to advance.
            llm_response: The text pasted back from the external LLM, or None
                          on the first call.
            vector_store: Injected vector store for chunk persistence.
            graph_store:  Injected graph store for entity/relation persistence.
            embedder:     Injected embedder for chunk embedding.

        Returns:
            ExtractionSessionResponse with the next prompt or final summary.

        Raises:
            KeyError:     If the session does not exist or has expired.
            RuntimeError: If the session is already complete.
            ValueError:   If the LLM response cannot be parsed.
        """
        session = self.get_session(session_id)

        if session.step == "done":
            raise RuntimeError(
                f"Session {session_id!r} is already complete. Start a new session."
            )

        if session.step == "start":
            return self._build_chunk_extraction_prompt(session)

        if session.step == "chunk_extraction":
            return await self._advance_chunk_extraction(
                session, llm_response, vector_store, embedder
            )

        if session.step == "entity_extraction":
            return await self._advance_entity_extraction(
                session, llm_response, graph_store
            )

        raise RuntimeError(f"Unknown step {session.step!r} in session {session_id!r}.")

    # ------------------------------------------------------------------
    # Step handlers
    # ------------------------------------------------------------------

    def _build_chunk_extraction_prompt(
        self, session: ExtractionSession
    ) -> ExtractionSessionResponse:
        """Transition from 'start' to 'chunk_extraction' and return the prompt."""
        session.step = "chunk_extraction"
        session.step_number = 1
        prompt = LLMPromptPayload(
            step_name="chunk_extraction",
            system_prompt=CHUNK_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=chunk_extraction_user_prompt(session.source),
            response_format='{"chunks": [{"index": 0, "content": "..."}, ...]}',
        )
        return ExtractionSessionResponse(
            session_id=session.session_id,
            doc_id=session.doc_id,
            status=SessionStatus.NEEDS_LLM,
            step_name=session.step,
            step_number=session.step_number,
            llm_prompt=prompt,
            resume_endpoint=f"/api/v1/manual/extract/sessions/{session.session_id}/resume",
            accumulated_context=self._context_snapshot(session),
            final_summary=None,
        )

    async def _advance_chunk_extraction(
        self,
        session: ExtractionSession,
        llm_response: str | None,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
    ) -> ExtractionSessionResponse:
        """Parse chunk JSON, embed each chunk, upsert to PostgreSQL, then build entity prompt."""
        if not llm_response:
            raise ValueError("llm_response is required for the chunk_extraction step.")

        parsed = _parse_llm_json(llm_response)
        if not isinstance(parsed, dict) or "chunks" not in parsed:
            raise ValueError(
                'Expected JSON with a "chunks" key, e.g. {"chunks": [{"index": 0, "content": "..."}]}.'
            )

        raw_chunks: list[Any] = parsed["chunks"]
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise ValueError("chunks must be a non-empty list.")

        # Sort by index if present, fall back to list order
        try:
            raw_chunks = sorted(raw_chunks, key=lambda c: int(c.get("index", 0)))
        except (TypeError, ValueError):
            pass

        doc_id = session.doc_id
        total = len(raw_chunks)
        stored_chunks: list[dict[str, Any]] = []

        for i, chunk in enumerate(raw_chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Each chunk must be a JSON object; got {type(chunk).__name__}.")
            content = chunk.get("content", "")
            if not isinstance(content, str) or not content.strip():
                logger.warning("ManualExtractService: skipping empty chunk at index %d", i)
                continue

            chunk_id = f"{doc_id}__chunk_{i}"
            vector = await embedder.embed(content)
            await vector_store.upsert(
                doc_id=chunk_id,
                vector=vector,
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": total,
                    "source": session.source,
                },
                content=content,
            )
            stored_chunks.append({"index": i, "content": content, "chunk_id": chunk_id})
            logger.info(
                "ManualExtractService [%s]: stored chunk %d/%d (id=%s)",
                doc_id, i + 1, total, chunk_id,
            )

        session.data["chunks"] = stored_chunks
        logger.info(
            "ManualExtractService [%s]: %d chunks embedded and stored in PostgreSQL",
            doc_id, len(stored_chunks),
        )

        # Build combined text for entity extraction
        combined_text = "\n\n".join(c["content"] for c in stored_chunks)

        session.step = "entity_extraction"
        session.step_number = 2
        prompt = LLMPromptPayload(
            step_name="entity_extraction",
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=extraction_user_prompt(combined_text, session.processing_instruction),
            response_format='{"entities": [...], "relations": [...]}',
        )
        return ExtractionSessionResponse(
            session_id=session.session_id,
            doc_id=session.doc_id,
            status=SessionStatus.NEEDS_LLM,
            step_name=session.step,
            step_number=session.step_number,
            llm_prompt=prompt,
            resume_endpoint=f"/api/v1/manual/extract/sessions/{session.session_id}/resume",
            accumulated_context=self._context_snapshot(session),
            final_summary=None,
        )

    async def _advance_entity_extraction(
        self,
        session: ExtractionSession,
        llm_response: str | None,
        graph_store: BaseGraphStore,
    ) -> ExtractionSessionResponse:
        """Parse entity/relation JSON, deduplicate, and persist to Neo4j."""
        if not llm_response:
            raise ValueError("llm_response is required for the entity_extraction step.")

        parsed = _parse_llm_json(llm_response)
        if not isinstance(parsed, dict):
            raise ValueError(
                'Expected JSON with "entities" and "relations" keys.'
            )

        entities: list[dict[str, Any]] = parsed.get("entities", [])
        relations: list[dict[str, Any]] = parsed.get("relations", [])

        if not isinstance(entities, list):
            raise ValueError('"entities" must be a list.')
        if not isinstance(relations, list):
            raise ValueError('"relations" must be a list.')

        doc_id = session.doc_id

        # --- Deduplicate entities by normalised name (first-seen wins) --------
        all_entities: dict[str, dict[str, Any]] = {}
        name_to_canonical_id: dict[str, str] = {}

        for entity in entities:
            local_id = entity.get("id", "")
            name = entity.get("name", "")
            if not local_id or not name:
                continue
            key = name.strip().lower()
            if key not in all_entities:
                all_entities[key] = entity
                name_to_canonical_id[key] = local_id

        # Re-assign canonical IDs on each entity record
        merged_entities = list(all_entities.values())
        for entity in merged_entities:
            key = entity.get("name", "").strip().lower()
            entity["id"] = name_to_canonical_id.get(key, entity["id"])

        # Deduplicate by final canonical id
        seen_ids: set[str] = set()
        deduped_entities: list[dict[str, Any]] = []
        for entity in merged_entities:
            eid = entity.get("id", "")
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                deduped_entities.append(entity)

        # Build canonical id → global id map
        entity_id_map: dict[str, str] = {}
        for entity in deduped_entities:
            local_id = entity.get("id", "")
            if not local_id:
                continue
            global_id = f"{doc_id}__{local_id}"
            entity_id_map[local_id] = global_id

        # --- Validate relations -----------------------------------------------
        valid_entity_ids = set(entity_id_map.keys())
        valid_relations = [
            r for r in relations
            if isinstance(r.get("src_id"), str)
            and isinstance(r.get("dst_id"), str)
            and r.get("src_id") in valid_entity_ids
            and r.get("dst_id") in valid_entity_ids
            and r.get("src_id") != r.get("dst_id")
        ]

        # Deduplicate relations by (src, dst, relation) triple
        seen_rels: set[tuple[str, str, str]] = set()
        deduped_relations: list[dict[str, Any]] = []
        for rel in valid_relations:
            key = (rel.get("src_id", ""), rel.get("dst_id", ""), rel.get("relation", ""))
            if key not in seen_rels:
                seen_rels.add(key)
                deduped_relations.append(rel)

        # --- Persist Entity nodes to Neo4j -----------------------------------
        for entity in deduped_entities:
            local_id = entity.get("id", "")
            if not local_id:
                continue
            global_id = entity_id_map[local_id]
            entity_type = entity.get("type", "Entity")
            await graph_store.add_node(
                node_id=global_id,
                labels=["Entity", entity_type],
                data={
                    "id": global_id,
                    "name": entity.get("name", local_id),
                    "type": entity_type,
                    "description": entity.get("description", ""),
                    "doc_id": doc_id,
                    "source_chunk_id": "",
                },
            )

        # --- Persist relationships to Neo4j ----------------------------------
        for rel in deduped_relations:
            src_local = rel.get("src_id", "")
            dst_local = rel.get("dst_id", "")
            if src_local not in entity_id_map or dst_local not in entity_id_map:
                continue
            relation_type = rel.get("relation", "RELATED_TO")
            edge_data = {**rel.get("properties", {})}
            await graph_store.add_edge(
                src_id=entity_id_map[src_local],
                dst_id=entity_id_map[dst_local],
                relation=relation_type,
                data=edge_data,
            )

        entity_count = len(deduped_entities)
        relation_count = len(deduped_relations)
        session.data["entity_count"] = entity_count
        session.data["relation_count"] = relation_count

        logger.info(
            "ManualExtractService [%s]: wrote %d entities and %d relations to Neo4j",
            doc_id, entity_count, relation_count,
        )

        session.step = "done"
        session.step_number = 3

        final_summary = {
            "doc_id": doc_id,
            "source": session.source,
            "chunks_stored": len(session.data.get("chunks", [])),
            "entities_stored": entity_count,
            "relations_stored": relation_count,
        }

        return ExtractionSessionResponse(
            session_id=session.session_id,
            doc_id=session.doc_id,
            status=SessionStatus.COMPLETE,
            step_name=session.step,
            step_number=session.step_number,
            llm_prompt=None,
            resume_endpoint=None,
            accumulated_context=self._context_snapshot(session),
            final_summary=final_summary,
        )

    # ------------------------------------------------------------------
    # Context snapshot
    # ------------------------------------------------------------------

    def _context_snapshot(self, session: ExtractionSession) -> dict[str, Any]:
        """Return a summary of accumulated state for the response payload.

        Args:
            session: The current extraction session.

        Returns:
            Dict suitable for the accumulated_context response field.
        """
        chunks = session.data.get("chunks", [])
        return {
            "doc_id": session.doc_id,
            "source": session.source,
            "chunks_count": len(chunks),
            "entity_count": session.data.get("entity_count", 0),
            "relation_count": session.data.get("relation_count", 0),
        }

    # ------------------------------------------------------------------
    # TTL helpers
    # ------------------------------------------------------------------

    def _purge_expired(self) -> None:
        """Remove sessions that have exceeded TTL_SECONDS since creation."""
        now = datetime.utcnow()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if (now - s.created_at).total_seconds() > self.TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service: ManualExtractService | None = None


def get_manual_extract_service() -> ManualExtractService:
    """Return the module-level ManualExtractService singleton.

    Returns:
        The shared ManualExtractService instance.
    """
    global _service
    if _service is None:
        _service = ManualExtractService()
    return _service
