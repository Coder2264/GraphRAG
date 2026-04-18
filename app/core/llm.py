"""
Abstract interface for Language Model implementations.

ISP: BaseLLM exposes only text generation. Structured output helpers
     (e.g. instructor-powered) are optional mixins, not part of this contract.

Template-method pattern: concrete generate()/generate_structured() handle
logging to app.llm / logs/LLM.log; subclasses implement _generate() and
_generate_structured() with the actual provider API calls.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

_logger = logging.getLogger("app.llm")

_SEP = "═" * 72
_HDR = "─" * 72


class BaseLLM(ABC):
    """
    Contract for any LLM backend.

    Implementations must be swappable without changing calling code (LSP).
    Examples: InMemoryLLM, OpenAILLM, AnthropicLLM, OllamaLLM, GeminiLLM, ...
    """

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def _generate(
        self, prompt: str, context: str = "", system_prompt: str = ""
    ) -> str:
        """
        Generate a text response.

        Args:
            prompt:        The user's question or instruction (user turn).
            context:       Retrieved context already embedded in `prompt` by the
                           service layer.  Implementations may ignore this arg.
            system_prompt: Optional system-role instruction sent as a separate
                           message so the model treats it with higher priority
                           than plain user text.

        Returns:
            The model's text response.
        """
        ...

    @abstractmethod
    async def _generate_structured(
        self,
        prompt: str,
        response_model: type,
        context: str = "",
        system_prompt: str = "",
    ) -> object:
        """
        Generate a response parsed into a Pydantic model.

        Args:
            prompt:         The user's question or instruction.
            response_model: Pydantic model class to parse into.
            context:        Retrieved context.
            system_prompt:  Optional system-role instruction.

        Returns:
            An instance of `response_model`.
        """
        ...

    # ------------------------------------------------------------------
    # Public API — logs every call to logs/LLM.log via the app.llm logger
    # ------------------------------------------------------------------

    @property
    def _model_label(self) -> str:
        """Best-effort model name for logging; falls back to class name."""
        return getattr(self, "_model_name", getattr(self, "model_name", self.__class__.__name__))

    async def generate(
        self, prompt: str, context: str = "", system_prompt: str = ""
    ) -> str:
        """
        Generate a text response and log the full call to LLM.log.

        Args:
            prompt:        The user's question or instruction (user turn).
            context:       Retrieved context; passed through to the implementation.
            system_prompt: Optional system-role instruction.

        Returns:
            The model's text response.
        """
        start = time.monotonic()
        response = await self._generate(prompt, context, system_prompt)
        elapsed = time.monotonic() - start
        _logger.info(
            "\n".join([
                _SEP,
                f"  METHOD  : generate",
                f"  CLASS   : {self.__class__.__name__}",
                f"  MODEL   : {self._model_label}",
                f"  DURATION: {elapsed:.3f}s",
                "─── system_prompt " + "─" * 54,
                system_prompt or "(none)",
                "─── context " + "─" * 60,
                context or "(none)",
                "─── prompt " + "─" * 61,
                prompt,
                "─── response " + "─" * 59,
                response,
                _SEP,
            ])
        )
        return response

    async def generate_structured(
        self,
        prompt: str,
        response_model: type,
        context: str = "",
        system_prompt: str = "",
    ) -> object:
        """
        Generate a structured response and log the full call to LLM.log.

        Args:
            prompt:         The user's question or instruction.
            response_model: Pydantic model class to parse into.
            context:        Retrieved context.
            system_prompt:  Optional system-role instruction.

        Returns:
            An instance of `response_model`.
        """
        start = time.monotonic()
        response = await self._generate_structured(prompt, response_model, context, system_prompt)
        elapsed = time.monotonic() - start
        _logger.info(
            "\n".join([
                _SEP,
                f"  METHOD  : generate_structured",
                f"  CLASS   : {self.__class__.__name__}",
                f"  MODEL   : {self._model_label}",
                f"  SCHEMA  : {response_model.__name__}",
                f"  DURATION: {elapsed:.3f}s",
                "─── system_prompt " + "─" * 54,
                system_prompt or "(none)",
                "─── context " + "─" * 60,
                context or "(none)",
                "─── prompt " + "─" * 61,
                prompt,
                "─── response " + "─" * 59,
                str(response),
                _SEP,
            ])
        )
        return response
