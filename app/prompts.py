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
      "type": "<one of the allowed entity types>",
      "description": "<one sentence description, or empty string>"
    }
  ],
  "relations": [
    {
      "src_id": "<id of source entity>",
      "dst_id": "<id of target entity>",
      "relation": "<one of the allowed relation types>",
      "properties": {}
    }
  ]
}

Rules:
- Use ONLY the entity types and relation types listed below.
- If no entities are found, return {"entities": [], "relations": []}.
- IDs must be lowercase, underscore-separated slugs unique within this response.
- Relations must reference IDs that appear in the entities list.
- Do NOT wrap the JSON in a code block or add any text outside the JSON.\
"""


def extraction_user_prompt(
    text: str,
    entity_types: list[dict],
    relation_types: list[dict],
) -> str:
    """
    Format the user turn for entity/relation extraction.

    Args:
        text:           Document text to extract from.
        entity_types:   Rows from graph_entity_types [{name, description}].
        relation_types: Rows from graph_relation_types [{name, description}].
    """
    entity_lines = "\n".join(
        f"  - {et['name']}: {et['description']}" for et in entity_types
    )
    relation_lines = "\n".join(
        f"  - {rt['name']}: {rt['description']}" for rt in relation_types
    )
    return f"""\
Allowed entity types:
{entity_lines}

Allowed relation types:
{relation_lines}

Text to extract from:
\"\"\"
{text}
\"\"\"

JSON output:\
"""
