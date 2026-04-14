"""
PostgresVectorStore — BaseVectorStore backed by PostgreSQL + pgvector.

Requires:
  - PostgreSQL >= 15 with the pgvector extension
  - `asyncpg` and `pgvector` Python packages

Schema is defined in db/schema.sql and mirrored here for auto-creation.
The table uses explicit columns (no JSONB blob) for all known metadata fields.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from app.config import settings
from app.core.vector_store import BaseVectorStore

# Columns that are stored as real table columns (not freeform metadata)
_KNOWN_METADATA_COLS = {"doc_id", "chunk_index", "total_chunks", "source"}

# Filterable columns and their SQL types for WHERE-clause generation
_FILTERABLE_COLS: dict[str, str] = {
    "doc_id": "TEXT",
    "chunk_index": "INTEGER",
    "total_chunks": "INTEGER",
    "source": "TEXT",
}


class PostgresVectorStore(BaseVectorStore):
    """
    Vector store using PostgreSQL + pgvector for ANN search.

    Table layout (mirrors db/schema.sql):
        chunk_id      TEXT PRIMARY KEY   — "{doc_id}__chunk_{i}"
        doc_id        TEXT               — parent document UUID
        chunk_index   INTEGER
        total_chunks  INTEGER
        source        TEXT               — filename or URL
        content       TEXT
        embedding     vector(N)
        created_at    TIMESTAMPTZ

    Swap in by setting DEFAULT_VECTOR_STORE=postgres in .env.
    """

    def __init__(
        self,
        dsn: str,
        table: str = "rag_chunks",
        embedding_dim: int | None = None,
    ) -> None:
        self._dsn = dsn
        self._table = table
        self._embedding_dim = embedding_dim or settings.embedding_dim
        self._pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create connection pool, register pgvector codec, ensure schema."""
        self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=10)
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await self._register_vector_codec(conn)
        await self._ensure_table()

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def _register_vector_codec(self, conn: asyncpg.Connection) -> None:
        from pgvector.asyncpg import register_vector
        await register_vector(conn)

    async def _ensure_table(self) -> None:
        """
        Create rag_chunks table and indexes if they do not exist.

        Mirrors db/schema.sql — keep both in sync when modifying structure.
        """
        assert self._pool, "Call connect() first."
        async with self._pool.acquire() as conn:
            await self._register_vector_codec(conn)
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    chunk_id      TEXT        PRIMARY KEY,
                    doc_id        TEXT        NOT NULL,
                    chunk_index   INTEGER     NOT NULL,
                    total_chunks  INTEGER     NOT NULL,
                    source        TEXT        NOT NULL DEFAULT '',
                    content       TEXT        NOT NULL DEFAULT '',
                    embedding     vector({self._embedding_dim}),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT {self._table}_doc_chunk_unique
                        UNIQUE (doc_id, chunk_index)
                );
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table}_doc_id_idx
                    ON {self._table} (doc_id);
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx
                    ON {self._table}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert(
        self,
        doc_id: str,
        vector: list[float] | None,
        metadata: dict[str, Any],
        content: str = "",
    ) -> None:
        """
        Insert or update a chunk row.

        `doc_id` here is the chunk_id (e.g. "{parent_doc_id}__chunk_{i}").
        Known metadata fields are extracted into their respective columns.
        When `vector` is None the embedding column is stored as NULL — the row
        is accessible via get() but skipped by ivfflat ANN index searches.
        """
        assert self._pool, "Call connect() first."
        import numpy as np

        chunk_doc_id  = str(metadata.get("doc_id", ""))
        chunk_index   = int(metadata.get("chunk_index", 0))
        total_chunks  = int(metadata.get("total_chunks", 1))
        source        = str(metadata.get("source", ""))
        embedding_val = np.array(vector, dtype=np.float32) if vector is not None else None

        async with self._pool.acquire() as conn:
            await self._register_vector_codec(conn)
            await conn.execute(
                f"""
                INSERT INTO {self._table}
                    (chunk_id, doc_id, chunk_index, total_chunks, source, content, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (chunk_id) DO UPDATE
                    SET doc_id       = EXCLUDED.doc_id,
                        chunk_index  = EXCLUDED.chunk_index,
                        total_chunks = EXCLUDED.total_chunks,
                        source       = EXCLUDED.source,
                        content      = EXCLUDED.content,
                        embedding    = EXCLUDED.embedding;
                """,
                doc_id,           # chunk_id  (the PK)
                chunk_doc_id,     # doc_id    (parent)
                chunk_index,
                total_chunks,
                source,
                content,
                embedding_val,
            )

    async def delete(self, doc_id: str) -> None:
        """Delete a single chunk by its chunk_id."""
        assert self._pool, "Call connect() first."
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE chunk_id = $1;", doc_id
            )

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete ALL chunks belonging to a parent doc_id."""
        assert self._pool, "Call connect() first."
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE doc_id = $1;", doc_id
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search(
        self,
        vector: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        ANN cosine search using pgvector's <=> operator.

        `metadata_filter` supports filtering on any of the known columns:
            doc_id, chunk_index, total_chunks, source
        Unknown keys in metadata_filter are silently ignored.
        """
        assert self._pool, "Call connect() first."
        import numpy as np

        params: list[Any] = [np.array(vector, dtype=np.float32), top_k]
        where_parts: list[str] = []
        param_idx = 3  # $1 = vector, $2 = top_k

        if metadata_filter:
            for col, val in metadata_filter.items():
                if col in _FILTERABLE_COLS:
                    where_parts.append(f"{col} = ${param_idx}")
                    params.append(val)
                    param_idx += 1

        where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        query = f"""
            SELECT chunk_id, doc_id, chunk_index, total_chunks, source, content,
                   1 - (embedding <=> $1::vector) AS score
            FROM {self._table}
            {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2;
        """

        async with self._pool.acquire() as conn:
            await self._register_vector_codec(conn)
            rows = await conn.fetch(query, *params)

        return [
            {
                "doc_id": row["chunk_id"],   # keeps BaseVectorStore contract
                "content": row["content"],
                "score": float(row["score"]),
                "metadata": {
                    "doc_id":       row["doc_id"],
                    "chunk_index":  row["chunk_index"],
                    "total_chunks": row["total_chunks"],
                    "source":       row["source"],
                },
            }
            for row in rows
        ]

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch a single chunk by chunk_id."""
        assert self._pool, "Call connect() first."
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT chunk_id, doc_id, chunk_index, total_chunks, source, content
                FROM {self._table}
                WHERE chunk_id = $1;
                """,
                doc_id,
            )
        if row is None:
            return None
        return {
            "doc_id": row["chunk_id"],
            "content": row["content"],
            "metadata": {
                "doc_id":       row["doc_id"],
                "chunk_index":  row["chunk_index"],
                "total_chunks": row["total_chunks"],
                "source":       row["source"],
            },
        }
