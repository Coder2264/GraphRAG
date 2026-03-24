# CLAUDE.md — Thinking on Graph

## Project Overview

**Thinking on Graph** is a FastAPI backend that answers natural-language questions using three interchangeable retrieval strategies backed by a common LLM interface.

| Query Mode | Retriever | Context Source |
|---|---|---|
| `none` | `NoneRetriever` | No context — LLM only |
| `rag` | `RAGRetriever` | PostgreSQL + pgvector |
| `graphrag` | `GraphRAGRetriever` | Neo4j knowledge graph |

## Run / Dev

```bash
uvicorn main:app --reload          # dev server
# Interactive API docs at http://localhost:8000/docs
```

All config comes from `.env`. Copy `.env.example` and fill in values before starting.

## Architecture (4-Layer)

```
HTTP Request
     ↓
app/api/v1/          ← FastAPI routers, Pydantic request/response schemas
     ↓  Depends(...)
app/services/        ← Business logic only; no direct infrastructure access
     ↓  injected via factory
app/core/            ← Abstract base classes ONLY (BaseLLM, BaseEmbedder, …)
     ↓  concrete classes
app/implementations/ ← Provider-specific code (ollama/, postgres/, neo4j/, …)
```

**Wiring files** (the only files that know about concrete classes):
- `app/registry.py` — maps string keys → implementation classes
- `app/factory.py` — builds singletons, manages `connect()`/`close()` lifecycle
- `app/dependencies.py` — exposes the factory to FastAPI `Depends`

## Key Rules

### Adding a New Provider (checklist)
1. Create `app/implementations/<provider>/<component>.py`
2. Subclass the correct `app/core/*.py` abstract class
3. Implement every `@abstractmethod`
4. Add the key → class entry in `app/registry.py`
5. Add constructor wiring in the matching `_build_*` method in `app/factory.py`
6. Add config fields to both `app/config.py` AND `.env.example`
7. Activate with `DEFAULT_<COMPONENT>=<key>` in `.env`

**Never** touch `QueryService`, `IngestionService`, or any `app/core/` file when adding a provider.

### Prompts
- All prompt strings live in `app/prompts.py` — never define them inline in services or handlers.
- System prompts are passed via `BaseLLM.generate(system_prompt=...)`, not concatenated into the user turn.
- Context always appears **before** the question in user prompt builders.
- New query mode → new system prompt constant + user prompt builder in `prompts.py`.

### SOLID Principles
- **SRP**: Each class owns one responsibility. Services orchestrate; implementations do I/O.
- **OCP**: New providers/modes extend via registry entries; existing code is untouched.
- **LSP**: All implementations are drop-in substitutes for their base class — no `isinstance` guards in calling code.
- **ISP**: Base classes stay small (≤5 abstract methods). Split if growing.
- **DIP**: Services and routes depend on `app.core.*` abstractions, never on concrete classes.

### Python Style
- Python 3.11+; `from __future__ import annotations` on modules using forward references.
- All public methods fully type-annotated; use `X | None` not `Optional[X]`.
- All I/O is `async`; never use blocking calls inside async functions.
- Google-style docstrings on all public classes and methods.
- Every file starts with a module-level docstring stating what it provides and which SOLID principle(s) apply.
- Import order: stdlib → third-party → `app.*` (absolute imports only, no relative imports).
- All config through `app.config.settings`; never `os.environ` or hardcoded values.

### Testing Policy
**No automated tests.** Do not write unit tests, integration tests, or test scripts. Verification is done by running the dev server and exercising endpoints manually (curl, FastAPI `/docs` UI, Neo4j Browser).

## Important File Map

| File | Purpose |
|---|---|
| `main.py` | App entrypoint; lifespan startup/shutdown |
| `app/config.py` | All settings via `pydantic_settings.BaseSettings` |
| `app/registry.py` | `LLM_REGISTRY`, `EMBEDDER_REGISTRY`, `GRAPH_STORE_REGISTRY`, `VECTOR_STORE_REGISTRY` |
| `app/factory.py` | `ServiceFactory` — builds and wires all singletons |
| `app/dependencies.py` | FastAPI `Depends` providers |
| `app/prompts.py` | All prompt strings and builder functions |
| `app/core/` | Abstract base classes — no concrete imports allowed |
| `app/implementations/` | Concrete providers grouped by vendor |
| `app/services/` | `QueryService`, `IngestionService` |
| `app/models/` | Pydantic request/response models |
| `db/schema.sql` | PostgreSQL schema (pgvector) |

## RAG Pipeline Summary

**Ingestion**: `Document → DefaultDocumentProcessor (chunk) → BaseEmbedder.embed → BaseVectorStore.upsert`

**Query**: `Question → BaseEmbedder.embed → BaseVectorStore.search → rag_user_prompt → BaseLLM.generate`

Critical: The embedding model used at ingest **must match** the model used at query time. `EMBEDDING_DIM` in `.env` must match the model's actual output dimension. Re-ingest after changing `CHUNK_SIZE` or `CHUNK_OVERLAP`.

## Current Providers

| Component | Key | Class |
|---|---|---|
| LLM | `ollama` | `OllamaLLM` |
| LLM | `in_memory` | `InMemoryLLM` (stub) |
| Embedder | `ollama` | `OllamaEmbedder` |
| Embedder | `in_memory` | `InMemoryEmbedder` (stub) |
| Vector Store | `postgres` | `PostgresVectorStore` |
| Vector Store | `in_memory` | `InMemoryVectorStore` (stub) |
| Graph Store | `neo4j` | `Neo4jGraphStore` |
| Graph Store | `in_memory` | `InMemoryGraphStore` (stub) |
| Entity Extractor | (auto) | `OllamaEntityExtractor` (when LLM=ollama) |

# To Ignore
- All files under Tests/ - this contains files for uploading and manually testing. Don't read it