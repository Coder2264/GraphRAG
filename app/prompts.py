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
You are a helpful assistant with access to a knowledge graph.
Use the provided graph context (nodes and relationships) to answer \
the question accurately and concisely.

Rules:
- Prefer information from the graph context over general knowledge.
- If the graph context is insufficient, say so clearly.\
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
