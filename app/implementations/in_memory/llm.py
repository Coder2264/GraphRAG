"""
InMemoryLLM — in-process stub LLM for development and testing.

Returns canned responses without any external API calls.
Demonstrates how instructor's generate_structured would be wired.

NOTE: instructor is imported lazily inside generate_structured to avoid
breaking Python 3.9 at import time (instructor >= 1.x uses 3.10+ syntax
in its own source). On Python 3.10+ the top-level import can be restored.
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from app.core.llm import BaseLLM


class InMemoryLLM(BaseLLM):
    """
    Stub LLM that returns hard-coded responses.

    Replace with a real implementation (OpenAILLM, AnthropicLLM, OllamaLLM)
    by creating a new class that extends BaseLLM — no other code changes needed.
    """

    def __init__(self, model_name: str = "in-memory-v1") -> None:
        self.model_name = model_name

    async def generate(self, prompt: str, context: str = "") -> str:
        """Return a stub response incorporating the query and context summary."""
        ctx_note = f" (context length: {len(context)} chars)" if context else " (no context)"
        return (
            f"[InMemoryLLM] Stub response for prompt: '{prompt[:80]}'{ctx_note}. "
            "Replace InMemoryLLM with a real LLM implementation."
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        context: str = "",
    ) -> Any:
        """
        Stub structured generation.

        In a real implementation this would call:
            import instructor, openai
            client = instructor.from_openai(openai.AsyncOpenAI())
            return await client.chat.completions.create(
                model=self.model_name,
                response_model=response_model,
                messages=[{"role": "user", "content": prompt}],
            )

        Here we just construct a default instance of the response_model.
        instructor is imported lazily to avoid Python 3.9 compatibility issues
        with the latest releases; swap to a top-level import on Python 3.10+.
        """
        try:
            import instructor  # noqa: F401 — validates dep is installed
        except ImportError:
            pass  # graceful degradation in stub

        try:
            return response_model.model_construct()
        except Exception:
            return None
