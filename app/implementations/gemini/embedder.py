"""
GeminiEmbedder — BaseEmbedder implementation backed by the Gemini Embedding API.

Swap instantly by setting DEFAULT_EMBEDDER=gemini in .env.
Requires the google-generativeai package and a valid GEMINI_API_KEY.

LSP: Fully substitutable for BaseEmbedder.
"""

from __future__ import annotations

import asyncio

import google.generativeai as genai

from app.core.embedder import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):
    """
    Embedder backed by Google's Gemini Embedding API.

    Uses `genai.embed_content` with `semantic_similarity` task type so the
    same model instance works correctly for both document ingestion and query
    embedding (the BaseEmbedder interface makes no distinction between the two).

    Constructor args map directly from settings:
        api_key:    Google Gemini API key.
        model_name: Gemini embedding model identifier.
                    e.g. "models/gemini-embedding-2-preview"
    """

    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-2-preview") -> None:
        genai.configure(api_key=api_key)
        self._model_name = model_name

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single string using the Gemini Embedding API.

        The underlying SDK call is synchronous; it is wrapped in
        `asyncio.to_thread` to avoid blocking the event loop.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        result = await asyncio.to_thread(
            genai.embed_content,
            model=self._model_name,
            content=text,
            task_type="semantic_similarity",
        )
        return result["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple strings in a single Gemini API call.

        The Gemini `embed_content` endpoint accepts a list of strings
        and returns a list of embeddings in the same order.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []
        result = await asyncio.to_thread(
            genai.embed_content,
            model=self._model_name,
            content=texts,
            task_type="semantic_similarity",
        )
        return result["embedding"]
