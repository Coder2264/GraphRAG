"""
GeminiLLM — BaseLLM implementation backed by Google Gemini.

Swap instantly by setting DEFAULT_LLM=gemini in .env.
Requires the google-genai package and a valid GEMINI_API_KEY.

LSP: Fully substitutable for BaseLLM.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Type

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.core.llm import BaseLLM

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """
    LLM backed by Google Gemini.

    Uses the google-genai SDK which provides a truly async HTTP client
    (httpx-based), avoiding thread-pool saturation under concurrent calls.

    Constructor args map directly from settings:
        api_key:    Google Gemini API key.
        model_name: Gemini model identifier (e.g. "gemini-2.0-flash").
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    async def generate(
        self, prompt: str, context: str = "", system_prompt: str = ""
    ) -> str:
        """
        Send `prompt` to Gemini and return the text response.

        If `system_prompt` is provided it is passed as Gemini's
        system_instruction, ensuring the model treats it with higher
        priority than the user turn.
        """
        config = types.GenerateContentConfig(
            temperature=0.7,
            system_instruction=system_prompt or None,
        )
        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )
        return response.text

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        context: str = "",
        system_prompt: str = "",
    ) -> Any:
        """
        Structured generation via Gemini JSON mode.

        Uses response_mime_type="application/json" for constrained output.
        Falls back to model_construct() on any parse failure.
        """
        json_prompt = (
            f"{prompt}\n\n"
            "Respond with a valid JSON object matching this schema:\n"
            f"{response_model.model_json_schema()}\n"
            "Return ONLY the JSON object, no explanation."
        )
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            system_instruction=system_prompt or None,
        )
        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=json_prompt,
            config=config,
        )
        raw = response.text.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(raw)
            return response_model.model_validate(data)
        except Exception:
            logger.warning("GeminiLLM.generate_structured: failed to parse JSON response.")
            return response_model.model_construct()
