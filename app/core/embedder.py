"""
Abstract interface for text embedding implementations.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Contract for any embedding backend.

    Examples: InMemoryEmbedder, OpenAIEmbedder, CohereEmbedder, ...
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Embed a string into a dense vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple strings in a single call (more efficient for bulk ops).

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, same order as input.
        """
        ...
