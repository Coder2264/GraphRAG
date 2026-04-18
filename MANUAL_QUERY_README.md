# Manual Query Flow — Human-in-the-Loop Guide

Use this when you're low on Ollama/API credits. Instead of the LLM running automatically, the system pauses at every LLM step and hands you the exact prompt. You paste it into ChatGPT or Gemini, copy the response back, and the system continues with the Neo4j/vector searches.

All database work (Neo4j graph traversal, pgvector search, chunk lookup) still runs **automatically**. Only the LLM reasoning steps require your input.

**Server**: `uvicorn main:app --reload`  
**Docs UI**: `http://localhost:8000/docs` (easiest way to interact)

---

## Core Concept: Sessions

A **session** is a stateful object that tracks where you are in the query pipeline. You create one session per question.

```
POST /api/v1/manual/sessions          ← Start (auto-runs first searches, returns first prompt)
POST /api/v1/manual/sessions/{id}/resume  ← Submit LLM response, get next prompt
GET  /api/v1/manual/sessions/{id}     ← Check current state at any time
```

Every response from start/resume looks like this:

```json
{
  "session_id": "a3f9c1b2e4d0",
  "status": "needs_llm",
  "step_name": "TOG: Relation Pruning",
  "step_number": 4,
  "llm_prompt": {
    "step_name": "relation_pruning",
    "system_prompt": "You are a graph reasoning assistant...",
    "user_prompt": "Question: ...\n\n[0] Entity: ...",
    "response_format": "{\"results\": [{\"idx\": 0, \"relations\": [...]}]}"
  },
  "resume_endpoint": "POST /api/v1/manual/sessions/a3f9c1b2e4d0/resume",
  "accumulated_context": { ... },
  "final_answer": null
}
```

When `status` is `"complete"`, `final_answer` contains the answer and `llm_prompt` is null.

Sessions expire after **1 hour** of inactivity.

---

## RAG Mode

**2 steps, 1 LLM pause.**

```
POST /api/v1/manual/sessions
{"query": "What is the mechanism of action of Velorazine?", "mode": "rag", "top_k": 5}
```

**Step 1 (automatic):** Embeds your query → runs pgvector cosine similarity search → returns the top-k chunks.

**Step 1 LLM pause — Answer Generation:**
The `accumulated_context.chunks` field shows you what was retrieved. The prompt instructs the LLM to answer strictly from those chunks.

```
POST /api/v1/manual/sessions/{id}/resume
{"llm_response": "Velorazine works by..."}  ← paste LLM's free-text answer
```

Done. `final_answer` is set.

---

## GraphRAG Mode

**3 LLM pauses per iteration + 1 for the final answer.** Typically 1–3 iterations.

```
POST /api/v1/manual/sessions
{"query": "Which organization funds the institute where the collaborator of Velorazine's lead investigator works?", "mode": "graphrag"}
```

### Pause 1 — Seed Extraction

The system asks the LLM to identify which named entities from the question should be looked up in Neo4j.

**You paste into ChatGPT/Gemini:**
- System: `BEAM_SEARCH_SEED_SYSTEM_PROMPT` (keyword extractor)
- User: the question

**You paste back:**
```json
{"keywords": ["Velorazine", "lead investigator", "collaborator"]}
```

The system then searches Neo4j for each keyword and loads matching nodes as the starting frontier.

---

### Pause 2 — Node Evaluation (repeat per iteration)

The system expanded one hop of the graph from the frontier and now shows you all neighbouring candidate nodes. You decide:
1. Whether the current context is already enough to answer the question.
2. Which candidate node IDs to explore next.

The `accumulated_context.candidate_node_count` shows how many candidates were found. The `compressed_summary` shows everything collected so far.

**You paste back:**
```json
{
  "has_sufficient_context": false,
  "selected_ids": ["doc_031__james_whitfield", "doc_031__meridian_institute"]
}
```

> **Tip:** The node IDs are in the `user_prompt` under "Candidate neighbouring nodes". Copy them exactly — they look like `doc_031__entity_name`.

> **Tip:** If you already have enough context to answer, set `has_sufficient_context: true` and leave `selected_ids` empty. The loop will stop after compression.

---

### Pause 3 — Context Compression (repeat per iteration)

The system serialized all nodes and edges found so far into `raw_graph_data` and asks the LLM to distill only what's relevant.

**You paste back:** Free text — a concise summary of the relevant facts. Example:
```
- James Whitfield is the lead clinical investigator for Velorazine.
- His collaborator is Dr. Sarah Chen at the Meridian Institute.
- The Meridian Institute is funded by the Global Health Foundation.
```

The system appends this to the accumulated summary, updates the frontier to the nodes you selected, and either loops back to Pause 2 or moves to the final answer.

---

### Final Pause — Answer Generation

Uses the full accumulated compressed summary as context. The prompt is the standard GraphRAG answer prompt.

**You paste back:** Free-text final answer.

---

## ToG (Think-on-Graph) Mode

**Most complex. Typically 8–12 LLM pauses for depth=2, but you can cut it short.**

```
POST /api/v1/manual/sessions
{"query": "Which organization funds the institute where the collaborator of Velorazine's lead investigator works?", "mode": "tog"}
```

The ToG algorithm builds **reasoning paths** — chains of entity→relation→entity triples from Neo4j — then generates an answer from those paths. Each depth level is one hop further into the graph.

---

### Phase 1 — Entity Extraction (Pause 1)

Extract the topic entities that seed the graph traversal.

**You paste back:**
```json
{"entities": ["Velorazine", "James Whitfield"]}
```

The system searches Neo4j for each entity name (top 2 matches per name) and initializes one reasoning **path** per resolved entity.

After this, `accumulated_context.paths` shows the initial one-node paths:
```
["Velorazine", "James Whitfield"]
```

---

### Phase 2 — Iterative Exploration (repeats up to depth_max=3 times)

Each iteration goes one hop deeper and has 3 steps: relation pruning → entity pruning → reasoning check.

#### Step A (automatic): Relation Fetch

The system calls `GET_RELATIONS` on Neo4j for every entity at the tail of each current path. You don't need to do anything. The candidate relations appear in the next pause's prompt.

#### Pause 2N — Relation Pruning (batched)

Given all current frontier entities and their candidate relations, pick which relations are most likely to lead toward the answer.

The prompt uses **few-shot examples** (5 examples from the paper) before the actual question. The format is:

```
Question: Which organization funds the institute where...

Select up to 3 relations per entity.

[0] Entity: Velorazine
    Relations: DEVELOPED_BY, TESTED_IN, HAS_INVESTIGATOR, APPROVED_BY

[1] Entity: James Whitfield
    Relations: WORKS_AT, COLLABORATES_WITH, LED_TRIAL_FOR

JSON:
```

**You paste back:**
```json
{
  "results": [
    {
      "idx": 0,
      "relations": [
        {"relation": "HAS_INVESTIGATOR", "score": 0.8},
        {"relation": "TESTED_IN", "score": 0.2}
      ]
    },
    {
      "idx": 1,
      "relations": [
        {"relation": "COLLABORATES_WITH", "score": 0.7},
        {"relation": "WORKS_AT", "score": 0.3}
      ]
    }
  ]
}
```

> **Rules:**
> - `idx` matches the `[0]`, `[1]`, ... numbers in the prompt — don't change the order.
> - Scores per entity must sum to 1.0.
> - Select 1–3 relations per entity (beam_width = 3 by default).
> - Relations must match **exactly** what's in the prompt (case-sensitive).

#### Step B (automatic): Entity Search

For each selected relation, the system calls `GET_TAIL_ENTITIES` and `GET_HEAD_ENTITIES` on Neo4j (both directions) to find all reachable entities. You don't need to do anything.

#### Pause 2N+1 — Entity Pruning (batched)

For each pending path (path + relation), score the candidate entities that were retrieved.

```
Question: Which organization funds the institute where...

For each path, score the candidate entities reachable via the given relation.

[0] Relation: HAS_INVESTIGATOR | Candidates: James Whitfield, Maria Torres, Kevin Park
[1] Relation: TESTED_IN | Candidates: Meridian Clinical Center, St. Luke's Hospital
[2] Relation: COLLABORATES_WITH | Candidates: Dr. Sarah Chen, Dr. Raj Patel
[3] Relation: WORKS_AT | Candidates: Meridian Institute, Harvard Medical School

JSON:
```

**You paste back:**
```json
{
  "results": [
    {"idx": 0, "entities": [{"entity": "James Whitfield", "score": 1.0}]},
    {"idx": 1, "entities": [{"entity": "Meridian Clinical Center", "score": 1.0}]},
    {"idx": 2, "entities": [{"entity": "Dr. Sarah Chen", "score": 1.0}]},
    {"idx": 3, "entities": [{"entity": "Meridian Institute", "score": 1.0}]}
  ]
}
```

> **Rules:**
> - Scores per path must sum to 1.0.
> - Pick exactly one entity per path (highest score wins).
> - Entity name must match a candidate from that path's list (case-insensitive match).
> - If you have no idea, just pick the most plausible one and give it `"score": 1.0`.

After this, `accumulated_context.paths` updates to show the extended chains:
```
[
  "(Velorazine, HAS_INVESTIGATOR, James Whitfield)",
  "(Velorazine, TESTED_IN, Meridian Clinical Center)",
  "(James Whitfield, COLLABORATES_WITH, Dr. Sarah Chen)",
  "(James Whitfield, WORKS_AT, Meridian Institute)"
]
```

#### Pause 2N+2 — Reasoning Check

The system shows you all current reasoning paths and asks: is there enough information here to answer the original question?

```
Q: Which organization funds the institute where...
Knowledge triples:
(Velorazine, HAS_INVESTIGATOR, James Whitfield)
(James Whitfield, COLLABORATES_WITH, Dr. Sarah Chen)
(James Whitfield, WORKS_AT, Meridian Institute)
A:
```

**You paste back:** `Yes` or `No`

- **Yes** → jumps to the final answer generation.
- **No** → another depth level begins (back to Relation Pruning for the new frontier entities).

> **Tip:** If the paths clearly contain enough facts to answer the question, say `Yes` even if you're at depth 1 — it saves several more pauses.

> **Tip:** If at `depth == max_depth` (default 3), the reasoning check is skipped and it goes straight to answer generation regardless.

---

### Phase 4 — Answer Generation (Final Pause)

The full set of reasoning paths (plus optional source text excerpts from the original documents) is provided. The few-shot examples show the format.

**You paste back:** Free-text final answer, e.g.:
```
James Whitfield is the lead clinical investigator for Velorazine. His collaborator, Dr. Sarah Chen, works at the Meridian Institute, which is funded by the Global Health Foundation.
```

Done. `status: "complete"`, `final_answer` is set.

---

## Full ToG Example (Condensed)

```
# 1. Start
POST /sessions  {"query": "...", "mode": "tog"}
→ step: entity_extraction

# 2. Provide entities
POST /sessions/{id}/resume  {"llm_response": "{\"entities\": [\"Velorazine\"]}"}
→ system resolves entity in Neo4j, fetches relations
→ step: relation_pruning  (depth 1)

# 3. Prune relations
POST /sessions/{id}/resume  {"llm_response": "{\"results\": [{\"idx\": 0, \"relations\": [...]}]}"}
→ system fetches candidate entities
→ step: entity_pruning  (depth 1)

# 4. Prune entities
POST /sessions/{id}/resume  {"llm_response": "{\"results\": [{\"idx\": 0, \"entities\": [...]}]}"}
→ system finalises paths
→ step: reasoning_check  (depth 1)

# 5. Reasoning check
POST /sessions/{id}/resume  {"llm_response": "No"}
→ system fetches relations for new frontier
→ step: relation_pruning  (depth 2)

# 6–7. Relation + entity pruning (depth 2) ...

# 8. Reasoning check (depth 2)
POST /sessions/{id}/resume  {"llm_response": "Yes"}
→ step: answer_generation

# 9. Final answer
POST /sessions/{id}/resume  {"llm_response": "The Global Health Foundation funds..."}
→ status: complete, final_answer: "The Global Health Foundation funds..."
```

---

## Tool Endpoints (Ad-hoc Database Access)

These don't require a session. Use them to explore the graph manually or to understand what data exists before starting a session.

| Endpoint | Body | Use for |
|---|---|---|
| `POST /tools/vector-search` | `{"query": "Velorazine mechanism", "top_k": 5}` | Find relevant text chunks |
| `POST /tools/node-search` | `{"keyword": "Velorazine", "top_k": 5}` | Find Neo4j entities by name |
| `POST /tools/subgraph` | `{"node_id": "doc_031__velorazine", "depth": 1}` | See all neighbours of a node |
| `POST /tools/relations` | `{"entity_id": "doc_031__james_whitfield"}` | List all relation types for an entity |
| `POST /tools/tail-entities` | `{"entity_id": "doc_031__james_whitfield", "relation": "WORKS_AT"}` | Outgoing edges |
| `POST /tools/head-entities` | `{"entity_id": "doc_031__meridian_institute", "relation": "WORKS_AT"}` | Incoming edges |
| `POST /tools/chunk-lookup` | `{"chunk_id": "doc_031__chunk_003"}` | Fetch raw source text |

**Finding node IDs**: Use `/tools/node-search` first — it returns node dicts with the `id` field. Node IDs are typically `{doc_id}__{entity_slug}`.

---

## Tips

- **Use the `/docs` UI.** The Swagger UI at `http://localhost:8000/docs` makes it easy to try endpoints without curl.

- **Pasting JSON into `llm_response` — use single quotes.** When the LLM returns JSON and you paste it into the `llm_response` field, double quotes inside the string would break the outer JSON. Use single quotes instead and the parser accepts it:
  ```json
  {"llm_response": "{'entities': ['Velorazine', 'lead clinical investigator']}"}
  ```
  The parser also accepts markdown code fences (` ```json ... ``` `) and standard escaped JSON.

- **Check `accumulated_context` on every response.** It shows paths found, nodes visited, and iteration count — useful for deciding "Yes"/"No" on reasoning checks.

- **For ToG, entity names in entity pruning must match candidates exactly** (case-insensitive). If the LLM makes up a name not in the list, the path is dropped.

- **Sessions are in-memory** — they don't survive server restarts.

- **The `response_format` field** in every `llm_prompt` tells you exactly what to paste back. JSON steps need valid JSON; reasoning check just needs `Yes` or `No`; answer generation is free text.

- **Stuck in a loop?** For GraphRAG, set `"has_sufficient_context": true` in node evaluation to force it to the answer step. For ToG, respond `Yes` to any reasoning check.
