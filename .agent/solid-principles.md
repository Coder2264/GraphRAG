# SOLID Principles

This project is built around SOLID object-oriented principles. Every new component should adhere to these rules. Violations must be justified in a code comment.

---

## S — Single Responsibility Principle (SRP)

Each class owns **one** well-scoped responsibility.

| Class | Its single responsibility |
|---|---|
| `QueryService` | Execute a query (retrieve + generate) |
| `IngestionService` | Orchestrate document ingestion |
| `ServiceFactory` | Wire and lifecycle-manage dependencies |
| `DefaultDocumentProcessor` | Parse and chunk raw documents |
| `registry.py` | Map string keys → concrete implementation classes |

**Rule:** If you find a class doing two independent things (e.g. fetching data AND formatting output), split it into two.

---

## O — Open/Closed Principle (OCP)

Classes are **open for extension, closed for modification**.

- Adding a new LLM backend? Add an entry to `LLM_REGISTRY` in `registry.py`. Do **not** modify `ServiceFactory`.
- Adding a new query mode? Add a new `BaseRetriever` subclass and the corresponding prompt template. Do **not** modify `QueryService`.
- The `_build_*` methods in `ServiceFactory` are the only place allowed to branch on provider keys; do not replicate that branching elsewhere.

---

## L — Liskov Substitution Principle (LSP)

Any concrete implementation must be a **drop-in replacement** for its base class, with no surprises.

- `OllamaLLM`, `InMemoryLLM`, and any future `OpenAILLM` must all satisfy the full `BaseLLM` contract.
- Never add provider-specific behaviour that calling code has to guard against (e.g. `if isinstance(llm, OllamaLLM): ...`).
- Implementations may raise `NotImplementedError` only for methods clearly marked optional in the base class docstring.

---

## I — Interface Segregation Principle (ISP)

Interfaces are **focused and minimal**; callers are not burdened with methods they don't use.

- `BaseLLM` exposes only `generate` and `generate_structured`. Streaming, batching, or fine-tuning concerns belong in a separate mixin or subinterface.
- `BaseVectorStore` covers only vector I/O. Metadata-only queries or admin operations belong elsewhere.
- If a new base class grows beyond ~5 abstract methods, consider splitting it.

---

## D — Dependency Inversion Principle (DIP)

High-level modules depend on **abstractions**, not concrete classes.

- `QueryService` depends on `BaseRetriever` and `BaseLLM` — never on `OllamaLLM` or `PostgresVectorStore` directly.
- `ServiceFactory` is the **only** place that knows about concrete classes; every other module imports from `app.core.*`.
- FastAPI route handlers receive dependencies via `Depends(...)` — they must never instantiate services themselves.

---

## Quick Reference: Where Each Principle Lives

```
app/
├── core/            ← Abstractions (ISP, LSP contract definitions)
├── implementations/ ← Concretions (LSP implementations)
├── registry.py      ← OCP extension point
├── factory.py       ← DIP wiring
└── services/        ← SRP business logic
```
