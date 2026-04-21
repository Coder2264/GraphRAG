"""
Prompt templates for all query modes.

Centralising prompts here makes iteration easy — tweak wording without
touching service or LLM code.  Each mode exposes:
  - A SYSTEM prompt string
  - A helper that formats the USER turn
"""

from __future__ import annotations

# ===========================================================================
# RAG — answer grounded in retrieved context chunks
# ===========================================================================

RAG_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based strictly on the \
provided context.

Rules:
- Use ONLY information present in the context below.
- If the context does not contain enough information to answer, say:
  "I don't have enough context to answer that."
- Be concise and factual.
- Do NOT fabricate or extrapolate beyond what is given.\
"""


def rag_user_prompt(question: str, context: str) -> str:
    """Format the user turn for a RAG query."""
    return f"""\
Context:
{context}

Question: {question}

Answer:\
"""


# ===========================================================================
# No-RAG — pure LLM, no retrieved context
# ===========================================================================

NO_RAG_SYSTEM_PROMPT = """\
You are a helpful, knowledgeable assistant.
Answer the user's question as accurately and concisely as possible.\
"""


def no_rag_user_prompt(question: str) -> str:
    """Format the user turn for a no-RAG (LLM-only) query."""
    return f"Question: {question}\n\nAnswer:"


# ===========================================================================
# GraphRAG — answer grounded in knowledge-graph context (future)
# ===========================================================================

GRAPH_RAG_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based strictly on the \
provided knowledge graph context.

Rules:
- Use ONLY information present in the knowledge graph context below.
- Do NOT use general knowledge or information from outside the provided context.
- If the context does not contain enough information to answer, say:
  "I don't have enough context to answer that."
- Be concise and factual.
- Do NOT fabricate or extrapolate beyond what is given.\
"""


def graph_rag_user_prompt(question: str, context: str) -> str:
    """Format the user turn for a GraphRAG query."""
    return f"""\
Knowledge Graph Context:
{context}

Question: {question}

Answer:\
"""


# ===========================================================================
# GraphRAG Beam-Search — used during iterative graph traversal at query time
# ===========================================================================

BEAM_SEARCH_SEED_SYSTEM_PROMPT = """\
You are a keyword extractor for a knowledge graph search engine.
Extract the specific entity names or keywords from the question that are most \
likely to match nodes in the graph.

Rules:
- Focus on proper nouns, named entities, and domain-specific terms.
- Return ONLY a JSON object with a single key "keywords" containing a list of strings.
- Do NOT wrap the JSON in a code block or add any text outside the JSON.\
"""


def beam_search_seed_user_prompt(question: str) -> str:
    """Format the user turn for seed keyword extraction.

    Args:
        question: The user's natural-language question.
    """
    return f"""\
Question: {question}

JSON:\
"""


def beam_search_eval_system_prompt(beam_width: int) -> str:
    """Build the system prompt for neighbour evaluation.

    Args:
        beam_width: Maximum number of node IDs to select.
    """
    return (
        "You are navigating a knowledge graph to answer a question.\n"
        "You are given the current context summary (what has been gathered so far) and a "
        "list of candidate neighbouring nodes to explore next.\n\n"
        "Your task:\n"
        "1. Decide if the current context is already sufficient to answer the question.\n"
        f"2. If not, select up to {beam_width} candidate node IDs that are most likely to "
        "lead to the answer.\n\n"
        "Rules:\n"
        '- Return ONLY a JSON object: {"has_sufficient_context": true/false, "selected_ids": [...]}\n'
        "- If has_sufficient_context is true, selected_ids may be empty.\n"
        "- Do NOT wrap the JSON in a code block or add any text outside the JSON."
    )


def beam_search_eval_user_prompt(
    question: str,
    beam_width: int,
    compressed_summary: str,
    candidates: list[dict],
) -> str:
    """Format the user turn for neighbour evaluation and selection.

    Args:
        question:           The user's natural-language question.
        beam_width:         Maximum number of node IDs to select.
        compressed_summary: Distilled context accumulated so far (empty on first iteration).
        candidates:         List of candidate nodes as {id, name, type, description}.
    """
    summary_section = (
        f"Current context summary:\n{compressed_summary}"
        if compressed_summary
        else "Current context summary: (none yet — first iteration)"
    )
    candidates_text = "\n".join(
        f"  id={c.get('id', '?')}  name={c.get('name', '?')}  "
        f"type={c.get('type', '?')}  desc={c.get('description', '')}"
        for c in candidates
    )
    return f"""\
Question: {question}

{summary_section}

Candidate neighbouring nodes (select up to {beam_width}):
{candidates_text or '(none)'}

JSON:\
"""


BEAM_SEARCH_COMPRESS_SYSTEM_PROMPT = """\
You are a context distiller for a knowledge graph traversal.
Given a question and raw graph data (nodes and edges gathered so far), produce a \
concise natural-language summary that retains ONLY information relevant to \
answering the question.

Rules:
- Discard dead-ends and nodes clearly unrelated to the question.
- Preserve entity names, relationships, and facts that could contribute to the answer.
- Be concise — prefer bullet points or short sentences over paragraphs.
- Do not answer the question; only distil the relevant context.\
"""


def beam_search_compress_user_prompt(question: str, raw_context: str) -> str:
    """Format the user turn for context compression.

    Args:
        question:    The user's natural-language question.
        raw_context: Serialised nodes and edges accumulated so far.
    """
    return f"""\
Question: {question}

Raw graph context:
{raw_context}

Relevant summary:\
"""


# ===========================================================================
# GraphRAG Extraction — used during ingestion to extract entities + relations
# ===========================================================================

EXTRACTION_SYSTEM_PROMPT = """\
You are a precise information-extraction engine.
Your sole job is to read the provided text and return a single, valid JSON object.

Output format (strict — no markdown, no explanation, no extra keys):
{
  "entities": [
    {
      "id": "<unique_slug_no_spaces>",
      "name": "<entity name as it appears in text>",
      "type": "<entity type>",
      "description": "<one sentence description, or empty string>"
    }
  ],
  "relations": [
    {
      "src_id": "<id of source entity>",
      "dst_id": "<id of target entity>",
      "relation": "<UPPER_SNAKE_CASE relation type, or RAW_RELATION>",
      "properties": {}
    }
  ]
}

Rules:
- IDs must be lowercase, underscore-separated slugs unique within this response.
- Relations must reference IDs that appear in the entities list.
- Use descriptive UPPER_SNAKE_CASE relation names (e.g. WORKS_AT, LOCATED_IN, PART_OF).
- If no canonical name fits, set "relation" to "RAW_RELATION" and add two keys to "properties":
    "raw_text": "<the original relationship phrasing from the text>",
    "canonical": null
- Extract ALL meaningful entities — do not omit any entity even if it appears only once.
- Include isolated entities (those with no extracted relation to others) — they are valid.
- Capture every stated or clearly implied relation, including fine-grained ones (dates,
  locations, product specs, supply agreements, subsidiaries, roles, events, etc.).
- Prefer more relations over fewer; it is better to over-extract than to miss information.
- DEDUPLICATION: If the same real-world entity is mentioned multiple times or referred to
  in different ways (e.g. "Oxford University" and "Oxford" both meaning the same institution),
  emit it ONCE with a single canonical id and attach ALL its relations to that one entry.
  Do NOT create separate entity entries for the same underlying entity.
- Do NOT wrap the JSON in a code block or add any text outside the JSON.\
"""


def extraction_user_prompt(text: str, processing_instruction: str = "") -> str:
    """
    Format the user turn for entity/relation extraction.

    Args:
        text:                   Document text to extract from.
        processing_instruction: Optional free-text hint guiding extraction
                                (e.g. domain, entity types to focus on).
    """
    if processing_instruction:
        instruction_section = (
            f"Processing instruction: {processing_instruction}\n"
            "Use this instruction to focus your extraction on the most relevant "
            "entities and relationships for the described domain or purpose."
        )
    else:
        instruction_section = (
            "No specific instruction provided — extract all meaningful named entities "
            "(people, organisations, locations, concepts, products, events, etc.) "
            "and their relationships."
        )

    return f"""\
{instruction_section}

Text to extract from:
\"\"\"
{text}
\"\"\"

JSON output:\
"""


# ===========================================================================
# ToG — Think-on-Graph (paper-faithful implementation)
# Based on: "Think-on-Graph: Deep and Responsible Reasoning of LLM on KG"
# Paper appendix E.3.1 - E.3.4
# ===========================================================================

# Five few-shot examples (KBQA-style) shared across all four ToG prompts.
# Each dict must contain: question, entity, relation, entities (list),
# relations (list), paths (list of triple-chain strings), answer,
# relation_answer_json, entity_answer_json, reasoning_answer ("Yes"/"No").
TOG_DEFAULT_EXAMPLES: list[dict] = [
    # Example 1 — based on the paper's running Canberra/Australia example.
    {
        "question": "What is the capital of Australia and which party does its prime minister lead?",
        "entity": "Australia",
        "relation": "HAS_PRIME_MINISTER",
        "entities": ["Anthony Albanese", "Scott Morrison", "Malcolm Turnbull"],
        "relations": ["CAPITAL_OF", "HAS_PRIME_MINISTER", "LOCATED_IN", "MEMBER_OF"],
        "paths": [
            "(Australia, capital of, Canberra)",
            "(Australia, prime minister, Anthony Albanese)",
            "(Anthony Albanese, leader of, Australian Labor Party)",
        ],
        "answer": (
            "Canberra is the capital of Australia. "
            "Anthony Albanese leads the Australian Labor Party."
        ),
        "relation_answer_json": (
            '{"relations": [{"relation": "CAPITAL_OF", "score": 0.5},'
            ' {"relation": "HAS_PRIME_MINISTER", "score": 0.5}]}'
        ),
        "entity_answer_json": '{"entities": [{"entity": "Anthony Albanese", "score": 1.0}]}',
        "reasoning_answer": "Yes",
    },
    # Example 2 — birthplace lookup.
    {
        "question": "In which city was Marie Curie born?",
        "entity": "Marie Curie",
        "relation": "BORN_IN",
        "entities": ["Warsaw", "Paris", "London"],
        "relations": ["BORN_IN", "NATIONALITY", "FIELD_OF_STUDY", "AWARD_WINNER"],
        "paths": [
            "(Marie Curie, born in, Warsaw)",
            "(Warsaw, located in, Poland)",
        ],
        "answer": "Marie Curie was born in Warsaw, Poland.",
        "relation_answer_json": (
            '{"relations": [{"relation": "BORN_IN", "score": 0.8},'
            ' {"relation": "NATIONALITY", "score": 0.2}]}'
        ),
        "entity_answer_json": '{"entities": [{"entity": "Warsaw", "score": 1.0}]}',
        "reasoning_answer": "Yes",
    },
    # Example 3 — film director and studio.
    {
        "question": "Who directed the film Inception and which studio produced it?",
        "entity": "Inception",
        "relation": "DIRECTED_BY",
        "entities": ["Christopher Nolan", "James Cameron", "Steven Spielberg"],
        "relations": ["DIRECTED_BY", "PRODUCED_BY", "STARRING", "RELEASED_IN"],
        "paths": [
            "(Inception, directed by, Christopher Nolan)",
            "(Inception, produced by, Warner Bros.)",
        ],
        "answer": "Inception was directed by Christopher Nolan and produced by Warner Bros.",
        "relation_answer_json": (
            '{"relations": [{"relation": "DIRECTED_BY", "score": 0.5},'
            ' {"relation": "PRODUCED_BY", "score": 0.5}]}'
        ),
        "entity_answer_json": '{"entities": [{"entity": "Christopher Nolan", "score": 1.0}]}',
        "reasoning_answer": "Yes",
    },
    # Example 4 — company CEO and headquarters.
    {
        "question": "Who is the CEO of Tesla and in which city is the company headquartered?",
        "entity": "Tesla",
        "relation": "CEO_OF",
        "entities": ["Elon Musk", "Tim Cook", "Satya Nadella"],
        "relations": ["CEO_OF", "HEADQUARTERED_IN", "PRODUCES", "FOUNDED_BY"],
        "paths": [
            "(Tesla, CEO, Elon Musk)",
            "(Tesla, headquartered in, Austin)",
            "(Austin, located in, Texas)",
        ],
        "answer": (
            "Elon Musk is the CEO of Tesla. "
            "The company is headquartered in Austin, Texas."
        ),
        "relation_answer_json": (
            '{"relations": [{"relation": "CEO_OF", "score": 0.5},'
            ' {"relation": "HEADQUARTERED_IN", "score": 0.5}]}'
        ),
        "entity_answer_json": '{"entities": [{"entity": "Elon Musk", "score": 1.0}]}',
        "reasoning_answer": "Yes",
    },
    # Example 5 — sports team home city.
    {
        "question": "Which city is home to the NBA team the Golden State Warriors?",
        "entity": "Golden State Warriors",
        "relation": "HOME_CITY",
        "entities": ["San Francisco", "Oakland", "Los Angeles"],
        "relations": ["HOME_CITY", "PLAYS_IN", "FOUNDED_IN", "MEMBER_OF"],
        "paths": [
            "(Golden State Warriors, home city, San Francisco)",
            "(Golden State Warriors, plays in, NBA)",
            "(San Francisco, located in, California)",
        ],
        "answer": "The Golden State Warriors play in San Francisco, California.",
        "relation_answer_json": (
            '{"relations": [{"relation": "HOME_CITY", "score": 0.7},'
            ' {"relation": "PLAYS_IN", "score": 0.3}]}'
        ),
        "entity_answer_json": '{"entities": [{"entity": "San Francisco", "score": 1.0}]}',
        "reasoning_answer": "Yes",
    },
]

# ---------------------------------------------------------------------------
# ToG E.3.1 — Relation pruning
# ---------------------------------------------------------------------------

# Call .format(beam_width=N) before use; all other braces are literal JSON.
TOG_RELATION_PRUNE_SYSTEM_PROMPT = """\
You are a graph reasoning assistant helping answer a question by \
traversing a knowledge graph.

Your task: Given a question, a topic entity, and a list of candidate \
relations connected to that entity, select the relations most likely \
to lead toward the answer. Score each selected relation's contribution \
on a scale from 0 to 1 (all scores must sum to 1).

Rules:
- Return ONLY a JSON object: {{"relations": [{{"relation": "...", \
"score": 0.X}}, ...]}}
- Select between 1 and {beam_width} relations.
- Scores must sum to 1.0 (rounded to 2 decimal places).
- Do NOT wrap in markdown or add text outside the JSON.\
"""


def tog_relation_prune_user_prompt(
    question: str,
    entity: str,
    relations: list[str],
    beam_width: int,
    examples: list[dict],
) -> str:
    """Build the few-shot user prompt for the relation-pruning step.

    Args:
        question:   The user's natural-language question.
        entity:     The topic entity currently being explored.
        relations:  Candidate relation labels connected to the entity.
        beam_width: Maximum number of relations to select (informational;
                    the constraint is enforced by the system prompt).
        examples:   Few-shot demonstrations; each dict must contain
                    ``question``, ``entity``, ``relations`` (list), and
                    ``relation_answer_json``.
    """
    shots = "\n\n".join(
        f"Q: {ex['question']}\n"
        f"Topic Entity: {ex['entity']}\n"
        f"Relations: {', '.join(ex['relations'])}\n"
        f"A: {ex['relation_answer_json']}"
        for ex in examples
    )
    query = (
        f"Q: {question}\n"
        f"Topic Entity: {entity}\n"
        f"Relations: {', '.join(relations)}\n"
        "A:"
    )
    return f"{shots}\n\n{query}" if shots else query


# ---------------------------------------------------------------------------
# ToG E.3.2 — Entity pruning
# ---------------------------------------------------------------------------

TOG_ENTITY_PRUNE_SYSTEM_PROMPT = """\
You are a graph reasoning assistant helping answer a question by \
traversing a knowledge graph.

Your task: Given a question, the current relation being explored, and \
a list of candidate entities reachable via that relation, score each \
entity's contribution to answering the question on a scale from 0 to 1 \
(all scores must sum to 1).

Rules:
- Return ONLY a JSON object: {"entities": [{"entity": "...", \
"score": 0.X}, ...]}
- Scores must sum to 1.0 (rounded to 2 decimal places).
- Do NOT wrap in markdown or add text outside the JSON.\
"""


def tog_entity_prune_user_prompt(
    question: str,
    relation: str,
    entities: list[str],
    examples: list[dict],
) -> str:
    """Build the few-shot user prompt for the entity-pruning step.

    Args:
        question:  The user's natural-language question.
        relation:  The relation currently being traversed.
        entities:  Candidate entity names reachable via the relation.
        examples:  Few-shot demonstrations; each dict must contain
                   ``question``, ``relation``, ``entities`` (list), and
                   ``entity_answer_json``.
    """
    shots = "\n\n".join(
        f"Q: {ex['question']}\n"
        f"Relation: {ex['relation']}\n"
        f"Entities: {', '.join(ex['entities'])}\n"
        f"Score:\n{ex['entity_answer_json']}"
        for ex in examples
    )
    query = (
        f"Q: {question}\n"
        f"Relation: {relation}\n"
        f"Entities: {', '.join(entities)}\n"
        "Score:"
    )
    return f"{shots}\n\n{query}" if shots else query


# ---------------------------------------------------------------------------
# ToG E.3.3 — Reasoning sufficiency check
# ---------------------------------------------------------------------------

TOG_REASONING_SYSTEM_PROMPT = """\
You are a graph reasoning assistant. Given a question and a set of \
reasoning paths retrieved from a knowledge graph (each path is a \
sequence of entity-relation-entity triples), decide whether the \
information in these paths alone is sufficient to answer the question.

Answer ONLY "Yes" or "No".
- "Yes" if the paths contain enough information to answer the question \
WITHOUT relying on any external or general knowledge.
- "No" if more graph traversal is needed.\
"""


def tog_reasoning_user_prompt(
    question: str,
    paths: list[str],
    examples: list[dict],
    source_chunks: list[str] | None = None,
) -> str:
    """Build the few-shot user prompt for the reasoning-sufficiency check.

    Args:
        question:      The user's natural-language question.
        paths:         Reasoning paths accumulated so far; each is a
                       triple-chain string such as
                       ``"(Canberra, capital of, Australia), (Australia, ...)"``
        examples:      Few-shot demonstrations; each dict must contain
                       ``question``, ``paths`` (list of strings), and
                       ``reasoning_answer`` ("Yes" or "No").
        source_chunks: Optional source text excerpts from the document chunks
                       that produced the graph entities/edges on the path.
                       Appended after the triples so the model can ground
                       its Yes/No decision in the original source text.
    """
    shots = "\n\n".join(
        f"Q: {ex['question']}\n"
        f"Knowledge triples: {chr(10).join(ex['paths'])}\n"
        f"A: {ex['reasoning_answer']}"
        for ex in examples
    )
    excerpts = ""
    if source_chunks:
        numbered = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(source_chunks))
        excerpts = f"\nSource excerpts:\n{numbered}\n"
    query = (
        f"Q: {question}\n"
        f"Knowledge triples: {chr(10).join(paths)}\n"
        f"{excerpts}"
        "A:"
    )
    return f"{shots}\n\n{query}" if shots else query


# ---------------------------------------------------------------------------
# ToG E.3.4 — Final answer generation
# ---------------------------------------------------------------------------

TOG_GENERATE_SYSTEM_PROMPT = """\
You are a precise assistant. Given a question and a set of reasoning \
paths retrieved from a knowledge graph (each path is a sequence of \
entity-relation-entity triples), answer the question using ONLY the \
information in the provided knowledge paths.

Rules:
- Use ONLY facts present in the provided knowledge paths.
- Do NOT use general knowledge or information from outside the provided paths.
- If the paths do not contain enough information to answer, say:
  "I don't have enough context to answer that."
- Be concise and direct. State the answer clearly.\
"""


def tog_generate_user_prompt(
    question: str,
    paths: list[str],
    examples: list[dict],
    source_chunks: list[str] | None = None,
) -> str:
    """Build the few-shot user prompt for final answer generation.

    Args:
        question:      The user's natural-language question.
        paths:         Reasoning paths accumulated so far; each is a
                       triple-chain string such as
                       ``"(Canberra, capital of, Australia), (Australia, ...)"``
        examples:      Few-shot demonstrations; each dict must contain
                       ``question``, ``paths`` (list of strings), and
                       ``answer``.
        source_chunks: Optional source text excerpts from the document chunks
                       that produced the graph entities/edges on the path.
                       Appended after the triples to ground the answer in
                       the original source text.
    """
    shots = "\n\n".join(
        f"Q: {ex['question']}\n"
        f"Knowledge triples: {chr(10).join(ex['paths'])}\n"
        f"A: {ex['answer']}"
        for ex in examples
    )
    excerpts = ""
    if source_chunks:
        numbered = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(source_chunks))
        excerpts = f"\nSource excerpts:\n{numbered}\n"
    query = (
        f"Q: {question}\n"
        f"Knowledge triples: {chr(10).join(paths)}\n"
        f"{excerpts}"
        "A:"
    )
    return f"{shots}\n\n{query}" if shots else query


# ---------------------------------------------------------------------------
# ToG E.3.1b — Batched relation pruning (all entities in one LLM call)
# ---------------------------------------------------------------------------

# Call .format(beam_width=N) before use.
TOG_BATCH_RELATION_PRUNE_SYSTEM_PROMPT = """\
You are a graph reasoning assistant helping answer a question by \
traversing a knowledge graph.

Your task: Given a question and several topic entities (each with candidate \
relations), select the relations most likely to lead toward the answer for \
EACH entity. Score each selected relation from 0 to 1 (scores per entity \
must sum to 1).

Rules:
- Return ONLY a JSON object:
  {{"results": [{{"idx": 0, "relations": [{{"relation": "...", "score": 0.X}}, ...]}}, ...]}}
- For each entity, select between 1 and {beam_width} relations.
- Scores per entity must sum to 1.0 (rounded to 2 decimal places).
- Include one entry per entity idx (0-based), in the same order as the input.
- Do NOT wrap in markdown or add text outside the JSON.\
"""


def tog_batch_relation_prune_user_prompt(
    question: str,
    entity_relation_pairs: list[tuple[str, list[str]]],
    beam_width: int,
) -> str:
    """Build the batched user prompt for relation pruning (all entities, one call).

    Args:
        question:             The user's natural-language question.
        entity_relation_pairs: List of (entity_name, relation_list) tuples.
        beam_width:           Maximum relations to keep per entity.

    Returns:
        Prompt string with all entities numbered [0], [1], … ready for one
        LLM call.
    """
    lines = [
        f"Question: {question}",
        "",
        f"Select up to {beam_width} relations per entity.",
        "",
    ]
    for idx, (name, rels) in enumerate(entity_relation_pairs):
        lines.append(f"[{idx}] Entity: {name}")
        lines.append(f"    Relations: {', '.join(rels)}")
    lines += ["", "JSON:"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ToG E.3.2b — Batched entity pruning (all paths in one LLM call)
# ---------------------------------------------------------------------------

TOG_BATCH_ENTITY_PRUNE_SYSTEM_PROMPT = """\
You are a graph reasoning assistant helping answer a question by \
traversing a knowledge graph.

Your task: Given a question and several reasoning paths (each extended via \
a relation to candidate entities), score the candidate entities for EACH \
path. Scores per path must sum to 1.

Rules:
- Return ONLY a JSON object:
  {{"results": [{{"idx": 0, "entities": [{{"entity": "...", "score": 0.X}}, ...]}}, ...]}}
- Scores per path must sum to 1.0 (rounded to 2 decimal places).
- Include one entry per path idx (0-based), in the same order as the input.
- Do NOT wrap in markdown or add text outside the JSON.\
"""


def tog_batch_entity_prune_user_prompt(
    question: str,
    path_candidates: list[tuple[str, list[str]]],
) -> str:
    """Build the batched user prompt for entity pruning (all paths, one call).

    Args:
        question:        The user's natural-language question.
        path_candidates: List of (relation_label, candidate_name_list) tuples,
                         one per pending path.

    Returns:
        Prompt string with all paths numbered [0], [1], … ready for one
        LLM call.
    """
    lines = [
        f"Question: {question}",
        "",
        "For each path, score the candidate entities reachable via the given relation.",
        "",
    ]
    for idx, (relation, candidates) in enumerate(path_candidates):
        lines.append(f"[{idx}] Relation: {relation} | Candidates: {', '.join(candidates)}")
    lines += ["", "JSON:"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ToG / ToG-R — QueryService integration prompts
# ---------------------------------------------------------------------------

TOG_SYSTEM_PROMPT = (
    "You are answering a question based strictly on reasoning paths retrieved "
    "from a knowledge graph. Use ONLY the information in the provided context. "
    "Do NOT use general knowledge or information from outside the provided context. "
    "If the context does not contain enough information to answer, say: "
    "\"I don't have enough context to answer that.\""
)


def tog_user_prompt(question: str, context: str) -> str:
    """Build the user prompt for ToG and ToG-R query modes.

    Args:
        question: The user's natural-language question.
        context:  The answer string produced by ToGRetriever / ToGRRetriever.

    Returns:
        Formatted prompt string with context before the question.
    """
    return f"Knowledge Graph Reasoning:\n{context}\n\nQuestion: {question}\n\nAnswer:"


# ===========================================================================
# Manual Extraction — human-in-the-loop ingestion via Gemini / ChatGPT
# ===========================================================================

CHUNK_EXTRACTION_SYSTEM_PROMPT = """\
You are a document extraction assistant.
Your job is to read the uploaded PDF and extract its full text, then split it into
logical, self-contained chunks of roughly 1500–2500 words each.

Output format (strict — no markdown, no explanation, no extra keys):
{
  "chunks": [
    {"index": 0, "content": "<chunk text>"},
    {"index": 1, "content": "<chunk text>"}
  ]
}

Rules:
- Preserve the original wording; do not paraphrase or summarise.
- Each chunk should be self-contained and cover a coherent topic or section.
- Respect natural paragraph and section boundaries — do not split mid-sentence.
- Do NOT wrap the JSON in a code block or add any text outside the JSON.\
"""


def chunk_extraction_user_prompt(source: str) -> str:
    """Build the user turn for the PDF chunk-extraction step.

    Args:
        source: The filename or label for the uploaded document.

    Returns:
        Formatted prompt string to paste into Gemini / ChatGPT.
    """
    return (
        f'Extract and chunk all text from the uploaded PDF "{source}" '
        "following the instructions above.\n\nJSON output:"
    )
