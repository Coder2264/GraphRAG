# End-to-End Verification Checklist — Think-on-Graph (ToG)

## 1. Prerequisite Setup

### 1a. Minimal .env for stub mode (no Neo4j, no Ollama)

```dotenv
# Provider selection — all stubs
DEFAULT_LLM=in_memory
DEFAULT_EMBEDDER=in_memory
DEFAULT_VECTOR_STORE=in_memory
DEFAULT_GRAPH_STORE=in_memory
DEFAULT_ENTITY_EXTRACTOR=in_memory

# Chunking (required to avoid zero-value errors)
CHUNK_SIZE=512
CHUNK_OVERLAP=64
EMBEDDING_DIM=4

# Enable DEBUG so all ToG phase logs are visible
LOG_LEVEL=DEBUG
LOG_FILE=logs/app.log
```

Leave all other keys (Ollama, Postgres, Neo4j, Gemini) blank or omitted entirely.

### 1b. Known limitations of stub mode

| Stub | Behaviour |
|---|---|
| `InMemoryLLM.generate()` | Returns a canned `[InMemoryLLM] Stub response...` string |
| `InMemoryLLM.generate_structured()` | Calls `response_model.model_construct()` — always returns the model's default, e.g. `_TopicEntities(entities=[])` |
| `InMemoryGraphStore` | Holds nodes/edges in-process; data is lost on restart |

**Critical consequence for ToG in stub mode:** Because `generate_structured` always returns
`entities=[]`, Phase 1 (topic entity extraction) will always find zero entities and trigger
the "falling back to LLM-only" path. Phases 2–4 never execute with `DEFAULT_LLM=in_memory`.

To exercise Phases 2–4 you must use a real LLM (`DEFAULT_LLM=ollama`) that can perform
structured generation. See section 5 for guidance on diagnosing each failure mode.

**Additional gap:** `get_tail_entities` and `get_head_entities` are called by the ToG
retriever at Step C but are not yet declared on `BaseGraphStore` nor implemented in
`InMemoryGraphStore` or `Neo4jGraphStore`. If Phase 1 ever succeeds, Step C will raise
`AttributeError`. These methods must be added before Phases 2–4 can run end-to-end.

---

## 2. Step-by-Step curl Commands

Start the dev server first:

```bash
uvicorn main:app --reload
```

### 2a. Ingest the Canberra/Australia document

```bash
curl -s -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Canberra is the capital city of Australia. It is located in the Australian Capital Territory (ACT). Australia is governed by the Australian Labor Party, which holds a majority in the House of Representatives. The Prime Minister of Australia is the head of government. Australia is a federal parliamentary constitutional monarchy.",
    "metadata": {"topic": "australia_politics"},
    "source": "tog-verification",
    "processing_instruction": "Extract cities, countries, political parties, and government roles and their relationships."
  }' | python3 -m json.tool
```

**Expected response shape:**
```json
{
  "doc_id": "<uuid>",
  "message": "Ingestion complete",
  "chunks_count": 1,
  "graph_entities_count": 0,
  "metadata": {"topic": "australia_politics"}
}
```

`graph_entities_count` will be 0 in stub mode because `InMemoryEntityExtractor` returns no
entities. With a real extractor (Gemini/Ollama), this will be > 0.

### 2b. Query /api/v1/query/tog

```bash
curl -s -X POST http://localhost:8000/api/v1/query/tog \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the majority party in the country where Canberra is located?",
    "top_k": 5
  }' | python3 -m json.tool
```

### 2c. Query /api/v1/query/tog_r

```bash
curl -s -X POST http://localhost:8000/api/v1/query/tog_r \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the majority party in the country where Canberra is located?",
    "top_k": 5
  }' | python3 -m json.tool
```

### 2d. Query /api/v1/query/graphrag (comparison baseline)

```bash
curl -s -X POST http://localhost:8000/api/v1/query/graphrag \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the majority party in the country where Canberra is located?",
    "top_k": 5
  }' | python3 -m json.tool
```

---

## 3. What to Look for in logs/app.log

Set `LOG_LEVEL=DEBUG` to see all phase-level messages. Steps A/B/C/D are at `DEBUG`; phase
boundaries and final summaries are at `INFO`.

### Phase 1 — Topic entity extraction

```
INFO  ToGRetriever: starting for query='What is the majority party...'
INFO  ToGRetriever: topic entities=['Canberra', 'Australia']
```

If you see this instead, Phase 1 failed (stub LLM or parse error):

```
WARNING  ToGRetriever: no topic entities found — falling back to LLM-only
```

### Phase 2, Step A — Relation search

```
DEBUG  ToGRetriever [A]: entity='Canberra' → 2 relation(s)
DEBUG  ToGRetriever [A]: entity='Australia' → 3 relation(s)
```

Zero relations (`→ 0 relation(s)`) means those entities are not in the graph — see section 5.

### Phase 2, Step B — Relation pruning

```
DEBUG  ToGRetriever [B]: pruned relations for entity='Canberra' → ['located_in']
```

### Phase 2, Step C — Entity search

```
DEBUG  ToGRetriever [C]: entity='Canberra' relation='located_in' → 1 candidate(s)
```

### Phase 2, Step D — Entity pruning

```
DEBUG  ToGRetriever [D]: finalised path → (Canberra, located_in, Australia)
```

### Phase 3 — Reasoning sufficiency check

```
INFO  ToGRetriever [Phase 3]: depth=1 sufficient=False paths=2
INFO  ToGRetriever [Phase 3]: depth=2 sufficient=True paths=3
```

### Phase 4 — Answer generation (triggered when sufficient=True)

Phase 4 produces no dedicated log line; look for the final summary:

```
INFO  ToGRetriever: done; visited=4 entity/entities, answer=187 char(s)
```

If the depth limit is hit without `sufficient=True`:

```
INFO  ToGRetriever: depth limit reached without sufficient context — using LLM-only generation
INFO  ToGRetriever: done; visited=3 entity/entities, answer=132 char(s)
```

---

## 4. Expected Response Shape (Successful ToG Response)

```json
{
  "question": "What is the majority party in the country where Canberra is located?",
  "mode": "tog",
  "answer": "<string — the LLM-generated answer>",
  "context": "<string — same as answer for ToG; ToG generates its own answer internally>",
  "sources": ["Canberra", "Australia", "Australian Labor Party"],
  "elapsed_seconds": 2.34
}
```

Key points:
- `mode` will be `"tog"` or `"tog_r"` depending on the endpoint called.
- `context` and `answer` contain the same string for ToG — unlike RAG where `context` holds
  the raw retrieved chunks. ToG performs answer generation internally and returns the answer
  as the context.
- `sources` lists every entity node visited during graph traversal, not document chunk IDs.
- `sources` is empty `[]` when the LLM-only fallback fires (no entities were found/traversed).

---

## 5. Common Failure Modes and Diagnosis

### 5a. InMemoryLLM returns stub responses (expected in stub mode)

**Symptom:** `answer` field contains `[InMemoryLLM] Stub response for prompt: '...'`.

**Cause:** `DEFAULT_LLM=in_memory` is set. The stub LLM is working correctly.

**What to check in logs:**
```
WARNING  ToGRetriever: no topic entities found — falling back to LLM-only
INFO     ToGRetriever: done; visited=0 entity/entities, answer=N char(s)
```

**Resolution:** Switch to a real LLM. For Ollama:
```dotenv
DEFAULT_LLM=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2
```

### 5b. get_relations returns an empty list

**Symptom:** Step A logs show `→ 0 relation(s)` for every entity.

**Cause (a) — Graph is empty:** The document was ingested but no entities were written to
the graph store. This happens when `DEFAULT_ENTITY_EXTRACTOR=in_memory` (the stub extractor
produces no entities).

Check in logs after ingest:
```
INFO  Ingest complete — doc_id=... chunks=1 graph_entities=0
```

Resolution: Use a real entity extractor (Gemini or Ollama) and re-ingest:
```dotenv
DEFAULT_ENTITY_EXTRACTOR=gemini
GEMINI_API_KEY=<your-key>
```

**Cause (b) — Entity name mismatch:** The entity names extracted by the LLM in Phase 1
do not match the node IDs stored in the graph. For example, Phase 1 extracts `"Australia"`
but the graph has `"australia"` or `"Australia (country)"`.

Check the in-memory graph directly: restart the server, re-ingest, then call:
```bash
curl -s http://localhost:8000/docs   # use /docs UI to call GET /api/v1/ingest or debug endpoint
```
Or add a temporary log in `InMemoryGraphStore.get_relations` to print `self._nodes.keys()`.

**Cause (c) — Step C will crash before relations matter:** If `get_tail_entities` and
`get_head_entities` are not yet implemented on the graph store, the retriever will raise
`AttributeError` as soon as Step C is reached. Check:
```
AttributeError: 'InMemoryGraphStore' object has no attribute 'get_tail_entities'
```
These two methods must be added to `BaseGraphStore` and both `InMemoryGraphStore` and
`Neo4jGraphStore` before the full ToG pipeline can run.

### 5c. Reasoning check never returns "Yes" (hits depth_max)

**Symptom:** Logs show `sufficient=False` at every depth, then:
```
INFO  ToGRetriever: depth limit reached without sufficient context — using LLM-only generation
```

The `answer` field will contain a response generated from the LLM's parametric knowledge
alone (no graph context), and `sources` will contain only the entities visited (not empty).

**Cause (a) — Stub LLM:** `InMemoryLLM.generate()` always returns the canned stub string,
which does not start with "yes". The reasoning check therefore always returns `False`.
This is expected when `DEFAULT_LLM=in_memory`.

**Cause (b) — LLM format non-compliance:** The real LLM is not responding with a bare
"Yes" or "No". The check parses `response.strip().lower().startswith("yes")`, so any
preamble ("Yes, the paths...") will pass, but responses like "Based on the above..." will
fail. Inspect the raw LLM output by temporarily adding a debug log in `_check_reasoning`.

**Cause (c) — Insufficient graph depth:** The paths do not contain enough information to
answer the question. Increase `TOG_DEPTH_MAX` in `.env` (e.g. `TOG_DEPTH_MAX=5`) or ingest
richer source material so the graph has more multi-hop connections.

**Cause (d) — Beam width too narrow:** With `beam_width=3` (default), only 3 relations are
kept per entity per hop. Relevant relations may be pruned. Try:
```dotenv
TOG_DEPTH_MAX=5
```
Note: `beam_width` is a constructor argument to `ToGRetriever`, not currently a `.env`
setting — it defaults to 3 in the retriever code.
