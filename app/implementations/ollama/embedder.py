"""
OllamaEmbedder — BaseEmbedder implementation using a local Ollama server.

Swap instantly by setting DEFAULT_EMBEDDER=in_memory in .env.
Requires: `ollama` Python package and a running `ollama serve` process.
"""

from __future__ import annotations

from app.core.embedder import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """
    Embedder backed by a local Ollama embedding model.

    Tested with:
      - nomic-embed-text   (768-dim)  — default, good quality/speed
      - mxbai-embed-large  (1024-dim) — higher quality
      - all-minilm         (384-dim)  — fastest

    IMPORTANT: embedding_dim in config.py MUST match the model you use.

    Constructor args map directly from settings:
        model_name  — e.g. "nomic-embed-text"
        base_url    — e.g. "http://localhost:11434"
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url

    def _client(self):
        import ollama  # local import keeps startup fast if not installed

        return ollama.AsyncClient(host=self.base_url)

    async def embed(self, text: str) -> list[float]:
        """Embed a single string and return a float vector."""
        client = self._client()
        response = await client.embeddings(model=self.model_name, prompt=text)
        return response.embedding  # type: ignore[attr-defined]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts sequentially.

        Ollama's /api/embeddings endpoint is single-text only, so we loop.
        For large batches this can be slow — consider parallel tasks if needed.
        """
        return [await self.embed(text) for text in texts]
