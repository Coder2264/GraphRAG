"""
GeminiEmbedder — BaseEmbedder implementation backed by the Gemini Embedding API.

Swap instantly by setting DEFAULT_EMBEDDER=gemini in .env.
Requires the google-genai package and a valid GEMINI_API_KEY.

LSP: Fully substitutable for BaseEmbedder.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from app.core.embedder import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):
    """
    Embedder backed by Google's Gemini Embedding API.

    Uses the google-genai SDK's async client (httpx-based) so embed calls
    never block the event loop.

    Constructor args map directly from settings:
        api_key:    Google Gemini API key.
        model_name: Gemini embedding model identifier.
                    e.g. "gemini-embedding-2-preview"
    """

    def __init__(self, api_key: str, model_name: str = "gemini-embedding-2-preview") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single string using the Gemini Embedding API.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        result = await self._client.aio.models.embed_content(
            model=self._model_name,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        return list(result.embeddings[0].values)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple strings in a single Gemini API call.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []
        result = await self._client.aio.models.embed_content(
            model=self._model_name,
            contents=texts,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        return [list(e.values) for e in result.embeddings]
