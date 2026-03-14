# Architecture Overview

## System Summary

**Thinking on Graph** is a FastAPI backend that supports three query modes:

| Mode | Retriever used | Context source |
|---|---|---|
| `none` | `NoneRetriever` | No context |
| `rag` | `RAGRetriever` | PostgreSQL + pgvector |
| `graphrag` | `GraphRAGRetriever` | Neo4j knowledge graph |

All three modes share the same `QueryService` and `BaseLLM` contract. The only thing that changes per mode is the injected retriever and the prompt template.

---

## Layer Diagram

```
HTTP Request
     │
     ▼
┌─────────────────────────────┐
│   API Layer  (app/api/v1/)  │  FastAPI routers, request/response schemas
└────────────┬────────────────┘
             │ Depends(...)
             ▼
┌─────────────────────────────┐
│  Service Layer (app/services/) │  Business logic, no infrastructure
│  QueryService, IngestionService│
└────────────┬────────────────┘
             │ injected via factory
             ▼
┌─────────────────────────────┐
│  Core Interfaces (app/core/) │  Abstract base classes only
│  BaseLLM, BaseEmbedder,      │
│  BaseVectorStore, ...        │
└────────────┬────────────────┘
             │ concrete classes
             ▼
┌─────────────────────────────────────────────────┐
│  Implementations (app/implementations/)          │
│  ollama/, postgres/, neo4j/, in_memory/, rag/    │
└─────────────────────────────────────────────────┘
             ▲
             │ registered in
┌────────────┴───────────┐
│   registry.py           │  str key → class mapping
└───────────────────────-┘
             ▲
             │ instantiated by
┌────────────┴───────────┐
│   factory.py            │  Builds singletons, manages lifecycle
└─────────────────────────┘
```

---

## Key Components

### `app/core/` — Abstractions
Pure abstract base classes. **No concrete imports allowed here.** Each file documents exactly what the contract guarantees so that any implementation can be swapped without touching calling code.

- `llm.py` → `BaseLLM`
- `embedder.py` → `BaseEmbedder`
- `vector_store.py` → `BaseVectorStore`
- `graph_store.py` → `BaseGraphStore`
- `retriever.py` → `BaseRetriever`
- `ingestion.py` → `BaseIngestionPipeline`
- `document_processor.py` → `BaseDocumentProcessor`

### `app/implementations/` — Concretions
Concrete implementations grouped by provider:

- `in_memory/` — Stubs for local/test use
- `ollama/` — Ollama LLM + embedder
- `postgres/` — pgvector vector store
- `neo4j/` — Neo4j graph store
- `rag/` — RAG-specific ingestion pipeline

### `app/registry.py` — Extension Point
Maps string keys to implementation classes. This is the **only file to edit** when adding a new provider.

### `app/factory.py` — Dependency Wiring
Constructs singletons from registry entries, passes settings from `config.py`, manages `connect()`/`close()` lifecycle. The only place that knows about concrete classes.

### `app/services/` — Business Logic
Services orchestrate abstract interfaces. They never touch infrastructure directly.

### `app/config.py` — Settings
All configuration comes from `pydantic_settings.BaseSettings`, loaded exclusively from `.env`. No hardcoded values anywhere else.

### `app/prompts.py` — Prompt Templates
All LLM prompt strings and templates live here. Services import named constants / factory functions — never build raw strings inline.

---

## Adding a New Provider (Checklist)

1. Create `app/implementations/<provider>/<component>.py`
2. Subclass the appropriate `app/core/*.py` abstract class
3. Implement every `@abstractmethod`
4. Add the key → class entry in `app/registry.py`
5. If the provider needs config values, add fields to `app/config.py` and `.env.example`
6. Update `ServiceFactory._build_*` in `factory.py` if the constructor signature is non-trivial
7. Smoke-test by setting `DEFAULT_<COMPONENT>=<key>` in `.env`
