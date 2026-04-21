# Manual Extraction Flow — Human-in-the-Loop Guide

Use this when you want to ingest a PDF without the server calling any LLM API.
You upload the PDF directly into Gemini or ChatGPT, paste the prompts provided by the system, and return the responses. The server handles all embedding and database writes automatically.

**Server**: `uvicorn main:app --reload`
**Docs UI**: `http://localhost:8000/docs` (easiest way to interact)

---

## Core Concept: Extraction Sessions

An **extraction session** tracks where you are in the ingestion pipeline. You create one session per document.

```
POST /api/v1/manual/extract/sessions              ← Start (returns first LLM prompt)
POST /api/v1/manual/extract/sessions/{id}/resume  ← Submit LLM response, get next prompt
GET  /api/v1/manual/extract/sessions/{id}         ← Check current state at any time
```

Sessions expire after **1 hour** of inactivity.

---

## Step-by-Step Flow

### Step 0 — Start the session

```json
POST /api/v1/manual/extract/sessions
{
  "source": "my_document.pdf",
  "processing_instruction": "Focus on people, organisations, and their relationships"
}
```

| Field | Required | Description |
|---|---|---|
| `source` | Yes | Filename or label for the document (used in prompts and chunk metadata) |
| `processing_instruction` | No | Optional hint for the entity extraction step (domain, entity types to focus on) |
| `doc_id` | No | Pre-assign a document ID; one is generated automatically if omitted |

The response includes a `session_id`, the current `step_name` (`chunk_extraction`), and an `llm_prompt` object containing a `system_prompt` and `user_prompt`.

---

### Step 1 — Chunk Extraction (LLM pause)

**What to do:**

1. Open a new conversation in [Gemini](https://gemini.google.com) or [ChatGPT](https://chat.openai.com).
2. **Upload your PDF** to the conversation.
3. Paste the **`system_prompt`** into the system instructions (or as the first message if there is no system slot).
4. Paste the **`user_prompt`** as your next message.
5. Copy the entire JSON response from the LLM.
6. POST it back:

```json
POST /api/v1/manual/extract/sessions/{id}/resume
{
  "llm_response": "<paste the entire LLM response here>"
}
```

**Expected LLM response format:**
```json
{
  "chunks": [
    {"index": 0, "content": "Section 1 text..."},
    {"index": 1, "content": "Section 2 text..."},
    {"index": 2, "content": "Section 3 text..."}
  ]
}
```

**What the server does automatically after this step:**
- Embeds each chunk using the configured embedder
- Stores every chunk in PostgreSQL with its vector embedding
- Builds the combined text for the next step
- Returns the entity extraction prompt

> **Chunk size matters.** Each chunk is embedded individually, so it must fit within your embedder's token limit (typically 2 000–8 000 tokens depending on the model). If a chunk is too large the embedding call will fail. Ask Gemini/ChatGPT to produce **4–6 chunks** for a typical 5-section document rather than one giant chunk.

---

### Step 2 — Entity & Relation Extraction (LLM pause)

**What to do:**

You are still in the **same Gemini/ChatGPT conversation** (the PDF context is still active). Paste the new `system_prompt` and `user_prompt` from the response.

Copy the JSON output and POST it back:

```json
POST /api/v1/manual/extract/sessions/{id}/resume
{
  "llm_response": "<paste the entity/relation JSON here>"
}
```

**Expected LLM response format:**
```json
{
  "entities": [
    {
      "id": "helix_db",
      "name": "Helix DB",
      "type": "Organization",
      "description": "Enterprise database company founded in the late 1990s"
    },
    {
      "id": "dr_aris_thorne",
      "name": "Dr. Aris Thorne",
      "type": "Person",
      "description": "Computer scientist and chief architect at Helix DB"
    }
  ],
  "relations": [
    {
      "src_id": "helix_db",
      "dst_id": "dr_aris_thorne",
      "relation": "EMPLOYED",
      "properties": {}
    }
  ]
}
```

**What the server does automatically after this step:**
- Deduplicates entities by normalised name
- Assigns globally unique IDs (`{doc_id}__{local_id}`)
- Writes `:Entity` nodes to Neo4j
- Writes relationships to Neo4j
- Returns `status: "complete"` with a `final_summary`

---

### Completion

When `status` equals `"complete"`, the `final_summary` field shows the ingestion counts:

```json
{
  "final_summary": {
    "doc_id": "3f9a1b2c-...",
    "source": "my_document.pdf",
    "chunks_stored": 5,
    "entities_stored": 12,
    "relations_stored": 18
  }
}
```

The document is now fully ingested and available for querying via `/api/v1/query` or the manual query flow (`/api/v1/manual/sessions`).

---

## Common Mistakes

### Submitting one giant chunk
**Problem:** Putting the entire document content into a single chunk will exceed your embedding model's token limit and cause a server error.

**Fix:** Ask Gemini/ChatGPT to produce multiple chunks (4–8 for a typical document). Each chunk should be roughly 1 500–2 500 words.

### Pasting plain text instead of JSON
**Problem:** The resume endpoint expects the LLM's raw output, which must be valid JSON. If the LLM wraps its response in a markdown code block (` ```json ... ``` `), that is fine — the server strips it automatically. If the LLM returns prose instead of JSON, the server will return HTTP 422.

**Fix:** If the LLM does not return JSON, re-prompt it: *"Return only the JSON object with no explanation."*

### Wrong step format
The two steps expect different JSON shapes:

| Step | Expected top-level keys |
|---|---|
| `chunk_extraction` | `chunks` (list of `{index, content}`) |
| `entity_extraction` | `entities` and `relations` |

Submitting entity JSON at the chunk step (or vice versa) returns HTTP 422.

### Session not found (HTTP 404)
Sessions live in memory and expire after 1 hour. If the server restarts or the TTL passes, you must start a new session. You cannot resume an expired session.

---

## Full curl Example

```bash
# 1. Start session
SESSION=$(curl -s -X POST http://localhost:8000/api/v1/manual/extract/sessions \
  -H "Content-Type: application/json" \
  -d '{"source": "report.pdf", "processing_instruction": "Extract companies, people, and products"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

echo "Session: $SESSION"

# 2. Resume after chunk extraction (paste your LLM response as the value)
curl -s -X POST http://localhost:8000/api/v1/manual/extract/sessions/$SESSION/resume \
  -H "Content-Type: application/json" \
  -d '{"llm_response": "{\"chunks\": [{\"index\": 0, \"content\": \"...section 1...\"}, {\"index\": 1, \"content\": \"...section 2...\"}]}"}'

# 3. Resume after entity extraction (paste your LLM response as the value)
curl -s -X POST http://localhost:8000/api/v1/manual/extract/sessions/$SESSION/resume \
  -H "Content-Type: application/json" \
  -d '{"llm_response": "{\"entities\": [{\"id\": \"acme\", \"name\": \"Acme Corp\", \"type\": \"Organization\", \"description\": \"\"}], \"relations\": []}"}'
```

---

## Response Reference

Every response from start/resume follows this shape:

```json
{
  "session_id": "a3f9c1b2e4d0",
  "doc_id": "3f9a1b2c-d5e6-...",
  "status": "needs_llm",
  "step_name": "chunk_extraction",
  "step_number": 1,
  "llm_prompt": {
    "step_name": "chunk_extraction",
    "system_prompt": "You are a document extraction assistant...",
    "user_prompt": "Extract and chunk all text from the uploaded PDF...",
    "response_format": "{\"chunks\": [{\"index\": 0, \"content\": \"...\"}, ...]}"
  },
  "resume_endpoint": "/api/v1/manual/extract/sessions/a3f9c1b2e4d0/resume",
  "accumulated_context": {
    "doc_id": "3f9a1b2c-...",
    "source": "report.pdf",
    "chunks_count": 0,
    "entity_count": 0,
    "relation_count": 0
  },
  "final_summary": null
}
```

When `status` is `"complete"`, `llm_prompt` and `resume_endpoint` are `null`, and `final_summary` is populated.
