# Prompts & RAG Pipeline

## Prompt Management

All prompt strings live in `app/prompts.py`. **Never** define raw prompt strings inside services, route handlers, or LLM implementations.

### Structure

```python
# System prompts — constant strings
RAG_SYSTEM_PROMPT: str = "..."
GRAPH_RAG_SYSTEM_PROMPT: str = "..."
NO_RAG_SYSTEM_PROMPT: str = "..."

# User prompt builders — functions that accept dynamic values
def rag_user_prompt(question: str, context: str) -> str: ...
def graph_rag_user_prompt(question: str, context: str) -> str: ...
def no_rag_user_prompt(question: str) -> str: ...
```

### Rules

- System prompts are passed as a **separate system-role message** via `BaseLLM.generate(system_prompt=...)`. Do not concatenate them into the user turn.
- User prompt builder functions must **always** place retrieved `context` before the `question` in the message so the model sees evidence first.
- If you add a new query mode, add a corresponding system prompt constant and a user prompt builder to `prompts.py` — then import them into `QueryService`.

---

## RAG Pipeline: How It Works

```
Document (PDF / TXT)
        │
        ▼
DefaultDocumentProcessor     ← chunks text into (chunk_size, chunk_overlap) pieces
        │
        ▼
RAGIngestionPipeline
  ├─ BaseEmbedder.embed(chunk)  ← produces a float vector
  └─ BaseVectorStore.upsert(doc_id, vector, metadata, content)
```

On query:

```
User question
        │
        ▼
BaseEmbedder.embed(question)    ← same model used during ingestion!
        │
        ▼
BaseVectorStore.search(vector, top_k)
        │
        ▼
RetrievalResult(context, sources)
        │
        ▼
rag_user_prompt(question, context)  ← injected into LLM call
        │
        ▼
BaseLLM.generate(prompt, system_prompt=RAG_SYSTEM_PROMPT)
```

### Critical Constraints

| Constraint | Why |
|---|---|
| The embedding model used at ingest **must match** the model used at query time | Vectors from different models are not comparable |
| `EMBEDDING_DIM` in `.env` **must match** the actual output dimension of the embed model | The `pgvector` column is fixed-width |
| `CHUNK_SIZE` and `CHUNK_OVERLAP` affect retrieval quality | Tune per domain; re-ingest after changing |

---

## Retriever Contract

Each `BaseRetriever` subclass returns a `RetrievalResult`:

```python
@dataclass
class RetrievalResult:
    context: str           # Formatted context string passed to the LLM
    sources: list[str]     # Source identifiers (doc IDs, node IDs, etc.)
```

- `NoneRetriever` returns an empty `RetrievalResult` (no-RAG mode).
- `RAGRetriever` encodes the question, searches `BaseVectorStore`, and formats the returned chunks into `context`.
- `GraphRAGRetriever` searches `BaseGraphStore`, traverses the subgraph, and formats entities/relations into `context`.

### Adding a New Retriever

1. Subclass `BaseRetriever` in `app/implementations/`.
2. Implement `retrieve(query, top_k)` to return a `RetrievalResult`.
3. Wire it in `ServiceFactory.get_retriever(mode)`.
4. Add the corresponding prompt template to `app/prompts.py`.

---

## Ingestion Response & Timing

- `IngestionService` returns the number of chunks ingested (`chunk_count`).
- `QueryService` records wall-clock time from the start of retrieval to the end of generation and returns it as `elapsed_seconds`.
- Both values are included in the API response so callers can diagnose latency.
