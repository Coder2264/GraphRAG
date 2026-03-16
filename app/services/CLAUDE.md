# CLAUDE.md — app/services/

## Purpose

Business logic layer. Services orchestrate abstract interfaces to fulfil use cases. They never import from `app/implementations/`, never touch infrastructure directly, and never instantiate their own dependencies.

## Files

| File | Class | Responsibility |
|---|---|---|
| `query_service.py` | `QueryService` | Execute a query: retrieve context → select prompt → generate answer |
| `ingestion_service.py` | `IngestionService` | Orchestrate document ingestion through the active pipeline(s) |

## Rules

### DIP — Depend on abstractions
Services receive `BaseLLM`, `BaseRetriever`, `BaseIngestionPipeline`, etc. via constructor injection. They must never import concrete classes or call `ServiceFactory` directly.

### SRP — One responsibility per service
`QueryService` only answers questions. `IngestionService` only ingests documents. If you find a service doing two independent things, split it.

### OCP — Open without modification
- Adding a new query mode must **not** require changing `QueryService` — add a new `BaseRetriever` subclass and the corresponding prompt template in `app/prompts.py`, then wire it in `factory.py`.
- `QueryService` selects system/user prompts by `QueryMode` value. New modes require a new branch here, but the retrieval and generation logic stays the same.

### Prompts
Services import named constants and builder functions from `app/prompts.py`. They must never define raw prompt strings inline.

## QueryService Flow

```
QueryRequest
    → BaseRetriever.retrieve(query, top_k)   → RetrievalResult
    → select system_prompt + user_prompt      (from app/prompts.py, by QueryMode)
    → BaseLLM.generate(prompt, system_prompt)
    → QueryResponse (answer, context, sources, elapsed_seconds)
```

## IngestionService Flow

```
UploadFile + doc_id
    → BaseIngestionPipeline.ingest(file_path, doc_id)   → chunk_count (RAG path)
    → BaseIngestionPipeline.ingest(file_path, doc_id)   → chunk_count (GraphRAG path, optional)
    → IngestResponse (doc_id, chunk_count, elapsed_seconds)
```

The GraphRAG ingestion pipeline may be `None` (when using the in-memory graph store) — `IngestionService` skips it in that case.
