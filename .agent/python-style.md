# Python Style Guide

## Language Version & Runtime

- **Python 3.11+** is required.
- Use `from __future__ import annotations` at the top of every module that uses forward references in type hints.
- The project runs via `uvicorn` in ASGI mode — all I/O-bound operations must be `async`.

---

## Type Hints

All public functions and methods **must** be fully annotated.

```python
# ✅ Good
async def generate(self, prompt: str, context: str = "") -> str: ...

# ❌ Bad
async def generate(self, prompt, context=""):  ...
```

- Use `X | None` (union) syntax, not `Optional[X]`, unless the project's minimum Python version makes this necessary.
- Use `list[T]`, `dict[K, V]`, `tuple[T, ...]` (lowercase), not `List`, `Dict`, `Tuple` from `typing`.
- Avoid `Any` unless explicitly dealing with truly dynamic data (e.g. metadata dicts in store interfaces). Annotate it with a comment explaining why.

---

## Async Conventions

- All I/O operations (database, HTTP, file I/O) must be `async def` and use `await`.
- Never call blocking code (`time.sleep`, `requests.get`, synchronous DB drivers) inside async functions. Use `asyncio.to_thread` if a sync library is unavoidable.
- Lifecycle methods on stores (`connect` / `close`) must be called only by `ServiceFactory.startup()` / `ServiceFactory.shutdown()` — never inside a request handler.

---

## Docstrings

Use Google-style docstrings for all public classes and methods.

```python
async def search(
    self,
    vector: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Nearest-neighbour search.

    Args:
        vector: Query embedding produced by BaseEmbedder.
        top_k:  Maximum number of results to return.

    Returns:
        List of dicts with keys: doc_id, score, content, metadata.

    Raises:
        RuntimeError: If the connection has not been established.
    """
```

All abstract methods in `app/core/` must have full docstrings — these serve as the living contract for implementors.

---

## Module-Level Docstrings

Every file must start with a module-level docstring that briefly states:
1. What the module provides.
2. Which SOLID principle(s) are intentionally applied.

```python
"""
Abstract interface for vector storage implementations.

Targets PostgreSQL + pgvector as the primary real backend while keeping
the contract compatible with other ANN stores (Pinecone, Qdrant, Chroma, ...).
"""
```

---

## Naming Conventions

| Kind | Convention | Example |
|---|---|---|
| Class | `PascalCase` | `OllamaLLM`, `RAGRetriever` |
| Function / method | `snake_case` | `generate_structured` |
| Variable | `snake_case` | `top_k`, `doc_id` |
| Constant | `UPPER_SNAKE_CASE` | `RAG_SYSTEM_PROMPT`, `LLM_REGISTRY` |
| Private attr | leading `_` | `self._llm`, `self._vector_store` |
| Abstract base | `Base` prefix | `BaseLLM`, `BaseEmbedder` |

---

## Error Handling

- **Raise specific exceptions**, not bare `Exception`.
- Services should catch infrastructure exceptions and re-raise as domain-level errors where appropriate, so route handlers never see raw `asyncpg` or `neo4j` errors.
- Use `assert` only for internal invariants that must hold (`assert self._llm is not None`) — never for user-input validation.

---

## Imports

Order (enforced by `isort`):
1. Standard library
2. Third-party packages
3. Internal `app.*` packages

Within groups, alphabetical order. Absolute imports always; no relative imports.

```python
# ✅ Good
import time
from typing import Any

from fastapi import Depends
from pydantic import BaseModel

from app.core.llm import BaseLLM
from app.models.query import QueryMode
```

---

## Configuration

All runtime config must come from `app.config.settings` (a `pydantic_settings.BaseSettings` instance loaded from `.env`).

- **Never** hardcode credentials, URLs, or model names in implementation files.
- **Never** import `os.environ` directly — always go through `settings`.
- Add every new config field to both `app/config.py` **and** `.env.example`.
