"""
InMemoryEmbedder — in-process stub embedder for development and testing.
"""

from app.core.embedder import BaseEmbedder

_STUB_DIMENSION = 1536  # Match OpenAI text-embedding-3-small dimension


class InMemoryEmbedder(BaseEmbedder):
    """
    Stub embedder that returns zero vectors.

    Replace with OpenAIEmbedder, CohereEmbedder, etc. by subclassing BaseEmbedder.
    """

    def __init__(self, dimension: int = _STUB_DIMENSION) -> None:
        self.dimension = dimension

    async def embed(self, text: str) -> list[float]:
        """Return a zero vector of the configured dimension."""
        return [0.0] * self.dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return a list of zero vectors."""
        return [[0.0] * self.dimension for _ in texts]
