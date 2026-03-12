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
