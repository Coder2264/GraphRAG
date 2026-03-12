"""
OllamaLLM — BaseLLM implementation using a local Ollama server.

Swap instantly by setting DEFAULT_LLM=in_memory in your .env.
Requires: `ollama` Python package and a running `ollama serve` process.
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from app.core.llm import BaseLLM


class OllamaLLM(BaseLLM):
    """
    LLM backed by a local Ollama server.

    Uses the chat endpoint with a single user message built from
    (system_prompt + user_prompt) forwarded by the QueryService.

    Constructor args map directly from settings:
        model_name  — e.g. "llama3.2", "mistral", "gemma2"
        base_url    — e.g. "http://localhost:11434"
    """

    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self):
        """Create a fresh AsyncClient (lightweight; no persistent state)."""
        import ollama  # local import keeps startup fast if not installed

        return ollama.AsyncClient(host=self.base_url)

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    async def generate(self, prompt: str, context: str = "") -> str:
        """
        Send `prompt` (already-formatted by QueryService) to Ollama chat.

        The `context` arg is embedded in the prompt by QueryService via
        prompts.py before this method is called — no further handling needed.
        """
        client = self._client()
        response = await client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content  # type: ignore[attr-defined]

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        context: str = "",
    ) -> Any:
        """
        Structured generation via Ollama.

        Ollama does not natively support JSON-schema-constrained output in the
        same way instructor/OpenAI does, so we fall back to a best-effort
        attempt: ask the model to return JSON, then parse it.

        For production structured output, prefer an OpenAI-compatible endpoint
        with the instructor library.
        """
        import json

        json_prompt = (
            f"{prompt}\n\n"
            "Respond with a valid JSON object matching this schema:\n"
            f"{response_model.model_json_schema()}\n"
            "Return ONLY the JSON object, no explanation."
        )
        raw = await self.generate(json_prompt, context)

        # Strip markdown fences if present
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(raw)
            return response_model.model_validate(data)
        except Exception:
            return response_model.model_construct()
