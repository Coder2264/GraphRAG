"""
Abstract interface for Language Model implementations.

ISP: BaseLLM exposes only text generation. Structured output helpers
     (e.g. instructor-powered) are optional mixins, not part of this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Contract for any LLM backend.

    Implementations must be swappable without changing calling code (LSP).
    Examples: InMemoryLLM, OpenAILLM, AnthropicLLM, OllamaLLM, ...
    """

    @abstractmethod
    async def generate(self, prompt: str, context: str = "") -> str:
        """
        Generate a text response.

        Args:
            prompt:  The user's question or instruction.
            context: Retrieved context to ground the response (empty = no context).

        Returns:
            The model's text response.
        """
        ...

    @abstractmethod
    async def generate_structured(self, prompt: str, response_model: type, context: str = "") -> object:
        """
        Generate a response parsed into a Pydantic model via instructor.

        Args:
            prompt:         The user's question or instruction.
            response_model: Pydantic model class to parse into.
            context:        Retrieved context.

        Returns:
            An instance of `response_model`.
        """
        ...
