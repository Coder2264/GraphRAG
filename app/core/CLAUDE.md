# CLAUDE.md — app/core/

## Purpose

This directory contains **abstract base classes only**. It defines the contracts that all implementations must honour.

**Rule: No concrete imports.** A file in `app/core/` must never import from `app/implementations/`, third-party infrastructure libraries (asyncpg, neo4j, httpx, …), or `app/factory.py`.

## Files

| File | Abstract Class | Contract |
|---|---|---|
| `llm.py` | `BaseLLM` | `generate`, `generate_structured` |
| `embedder.py` | `BaseEmbedder` | `embed` |
| `vector_store.py` | `BaseVectorStore` | `connect`, `close`, `upsert`, `delete`, `search`, `get` |
| `graph_store.py` | `BaseGraphStore` | `connect`, `close`, + graph CRUD |
| `retriever.py` | `BaseRetriever` + `RetrievalResult` | `retrieve(query, top_k) → RetrievalResult` |
| `ingestion.py` | `BaseIngestionPipeline` | `ingest(file_path, doc_id) → int` |
| `document_processor.py` | `BaseDocumentProcessor` | `process(file_path) → list[str]` |
| `entity_extractor.py` | `BaseEntityExtractor` | `extract(text) → …` |

## Rules When Editing This Directory

1. **Every abstract method must have a full Google-style docstring** — these serve as the living contract for implementors.
2. Keep each base class **minimal** (≤5 abstract methods). Split before growing beyond that.
3. Do not add `@property` attributes or default implementations unless they are truly universal across all conceivable providers.
4. Adding a new abstract method to an existing base class is a **breaking change** — all existing implementations must be updated at the same time.
5. `RetrievalResult` (in `retriever.py`) is a `@dataclass`; keep it a plain data container with no behaviour.

## ISP Reminder

`BaseLLM` only exposes `generate` and `generate_structured`. Streaming, batching, or fine-tuning go in a separate mixin or subinterface — do not add them here.
