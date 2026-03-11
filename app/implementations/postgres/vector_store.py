"""
PostgresVectorStore — BaseVectorStore backed by PostgreSQL + pgvector.

Requires:
  - PostgreSQL >= 15 with the pgvector extension (`CREATE EXTENSION vector;`)
  - `asyncpg` and `pgvector` Python packages (in requirements.txt)

Table schema (run once):
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS documents (
        doc_id   TEXT PRIMARY KEY,
        content  TEXT,
        metadata JSONB,
        embedding vector(1536)
    );
    CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
"""

from __future__ import annotations

from typing import Any

import asyncpg

from app.core.vector_store import BaseVectorStore


class PostgresVectorStore(BaseVectorStore):
    """
    Vector store using PostgreSQL + pgvector for ANN search.

    This is the production implementation for the RAG pipeline.
    Swap in by registering "postgres" in registry.py.
    """

    def __init__(self, dsn: str, table: str = "documents") -> None:
        self._dsn = dsn
        self._table = table
        self._pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create an asyncpg connection pool."""
        self._pool = await asyncpg.create_pool(dsn=self._dsn)
        await self._ensure_table()

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def _ensure_table(self) -> None:
        """Create the vector table if it does not exist."""
        assert self._pool, "Call connect() first."
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    doc_id    TEXT PRIMARY KEY,
                    content   TEXT DEFAULT '',
                    metadata  JSONB DEFAULT '{{}}',
                    embedding vector(1536)
                );
                """
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert(
        self,
        doc_id: str,
        vector: list[float],
        metadata: dict[str, Any],
        content: str = "",
    ) -> None:
        assert self._pool, "Call connect() first."
        import json

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._table} (doc_id, content, metadata, embedding)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (doc_id) DO UPDATE
                    SET content   = EXCLUDED.content,
                        metadata  = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding;
                """,
                doc_id,
                content,
                json.dumps(metadata),
                str(vector),
            )

    async def delete(self, doc_id: str) -> None:
        assert self._pool, "Call connect() first."
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._table} WHERE doc_id = $1;", doc_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search(
        self,
        vector: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """ANN cosine search using pgvector's <=> operator."""
        assert self._pool, "Call connect() first."

        # Basic metadata filtering via JSONB containment
        where_clause = ""
        params: list[Any] = [str(vector), top_k]
        if metadata_filter:
            import json
            where_clause = "WHERE metadata @> $3"
            params.append(json.dumps(metadata_filter))

        query = f"""
            SELECT doc_id, content, metadata,
                   1 - (embedding <=> $1::vector) AS score
            FROM {self._table}
            {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2;
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            {
                "doc_id": row["doc_id"],
                "content": row["content"],
                "metadata": row["metadata"],
                "score": float(row["score"]),
            }
            for row in rows
        ]

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        assert self._pool, "Call connect() first."
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT doc_id, content, metadata FROM {self._table} WHERE doc_id = $1;",
                doc_id,
            )
        if row is None:
            return None
        return {"doc_id": row["doc_id"], "content": row["content"], "metadata": row["metadata"]}
