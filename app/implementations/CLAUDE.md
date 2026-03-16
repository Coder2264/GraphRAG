# CLAUDE.md — app/implementations/

## Purpose

Concrete provider implementations, grouped by vendor/technology. Each subdirectory is a self-contained plugin — it subclasses the relevant `app/core/` abstract class and is registered in `app/registry.py`.

## Directory Structure

```
implementations/
├── document_processor.py    ← DefaultDocumentProcessor (shared, provider-agnostic)
├── in_memory/               ← Stubs for local dev / fallback
│   ├── llm.py               ← InMemoryLLM
│   ├── embedder.py          ← InMemoryEmbedder
│   ├── vector_store.py      ← InMemoryVectorStore
│   ├── graph_store.py       ← InMemoryGraphStore
│   ├── entity_extractor.py  ← InMemoryEntityExtractor
│   ├── retrievers.py        ← NoneRetriever, RAGRetriever, GraphRAGRetriever
│   └── ingestion.py         ← InMemoryIngestionPipeline
├── ollama/                  ← Ollama LLM + embedder + entity extractor
├── postgres/                ← pgvector vector store
├── neo4j/                   ← Neo4j graph store
├── rag/                     ← RAGIngestionPipeline (postgres-backed)
└── graph_rag/               ← GraphRAGIngestionPipeline (Neo4j-backed)
```

## Rules for All Implementation Files

1. **Subclass exactly one `app/core/` abstract class** per file.
2. **Implement every `@abstractmethod`** — do not leave any unimplemented; raise `NotImplementedError` only for methods the base class explicitly marks as optional.
3. **LSP**: The implementation must be a drop-in substitute. Never add public methods that calling code might rely on, unless they are on the base class.
4. **No config hardcoding** — all credentials, URLs, model names, and tunable values come from `app.config.settings`, passed in through the constructor by `ServiceFactory`.
5. **Module docstring required**: Every file must open with a docstring stating what it provides and noting `LSP: Fully substitutable for <BaseClass>`.
6. **All I/O is async** (`async def`, `await`) — no synchronous DB or HTTP calls.

## Adding a New Provider

Only two existing files ever need to change:
- `app/registry.py` — add `"<key>": MyNewClass` to the appropriate registry dict.
- `app/factory.py` — add constructor wiring in the matching `_build_*` method if the constructor takes non-trivial arguments.

Do **not** modify `QueryService`, `IngestionService`, or any `app/core/` file.

## Provider-Specific Notes

### `in_memory/`
Used for local development and as a fallback when no real backend is configured. All stubs should return empty/zero values and log a warning rather than raising errors, so the app starts cleanly without any external services.

### `ollama/`
Uses the Ollama HTTP API (`ollama_base_url` from settings). The LLM and embedder share the same base URL. The entity extractor reuses the same Ollama instance as the LLM.

### `postgres/`
Uses `asyncpg` + `pgvector`. The `connect()` method creates the connection pool; `close()` closes it. The `embedding_dim` must match `settings.embedding_dim` exactly — the pgvector column is fixed-width.

### `neo4j/`
Uses the async `neo4j` Python driver. Credentials (`neo4j_uri`, `neo4j_user`, `neo4j_password`) come from settings. `connect()` opens the driver; `close()` closes it.

### `rag/` and `graph_rag/`
Ingestion pipelines. `rag/` writes chunks + embeddings to the vector store. `graph_rag/` extracts entities/relations via `BaseEntityExtractor` and writes to the graph store (and optionally the vector store for hybrid search).
