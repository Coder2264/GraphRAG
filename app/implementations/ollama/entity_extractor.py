"""
OllamaEntityExtractor — BaseEntityExtractor backed by the local Ollama LLM.

Sends an extraction prompt to Ollama and parses the returned JSON.
Degrades gracefully: on any parse failure returns empty lists so the
ingestion pipeline is not blocked.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from app.config import settings
from app.core.entity_extractor import BaseEntityExtractor
from app.prompts import EXTRACTION_SYSTEM_PROMPT, extraction_user_prompt

logger = logging.getLogger(__name__)
_llm_logger = logging.getLogger("app.llm")

_SEP = "═" * 72


class OllamaEntityExtractor(BaseEntityExtractor):
    """
    Entity/relationship extractor that uses an Ollama-served LLM.

    Uses the Ollama /api/chat endpoint with a structured extraction prompt.
    The *same* Ollama base URL and LLM model as OllamaLLM are reused.
    """

    def __init__(self, model_name: str, base_url: str) -> None:
        self._model = model_name
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # BaseEntityExtractor interface
    # ------------------------------------------------------------------

    async def extract(
        self,
        text: str,
        processing_instruction: str = "",
    ) -> dict:
        """
        Call Ollama to extract entities and relations from *text*.

        Args:
            text:                   The document text to analyse.
            processing_instruction: Optional free-text hint guiding extraction.

        Returns:
            {"entities": [...], "relations": [...]}
            Empty lists on any error (graceful degradation).
        """
        user_msg = extraction_user_prompt(text, processing_instruction)

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "stream": False,
            # Some models respect format:"json" for guaranteed JSON output
            "format": "json",
            "options": {
                "temperature": 0.0,   # deterministic extraction
            },
        }

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=settings.ollama_extraction_timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            elapsed = time.monotonic() - start
            raw = data["message"]["content"]
            _llm_logger.info(
                "\n".join([
                    _SEP,
                    "  METHOD  : extract (entity extraction)",
                    f"  CLASS   : {self.__class__.__name__}",
                    f"  MODEL   : {self._model}",
                    f"  DURATION: {elapsed:.3f}s",
                    "─── system_prompt " + "─" * 54,
                    EXTRACTION_SYSTEM_PROMPT,
                    "─── prompt " + "─" * 61,
                    user_msg,
                    "─── response " + "─" * 59,
                    raw,
                    _SEP,
                ])
            )
            return self._parse_response(raw)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Entity extraction failed: %s: %s", type(exc).__name__, exc)
            return {"entities": [], "relations": []}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """
        Parse the LLM output as JSON with fallback strategies.

        1. Try direct json.loads on the raw string.
        2. Try extracting the first {...} block with regex (LLM sometimes
           wraps JSON in markdown fences or adds preamble text).
        3. Return empty result if everything fails.
        """
        # Strategy 1: direct parse
        try:
            parsed = json.loads(raw)
            return {
                "entities":  parsed.get("entities", []),
                "relations": parsed.get("relations", []),
            }
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return {
                    "entities":  parsed.get("entities", []),
                    "relations": parsed.get("relations", []),
                }
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse entity extraction JSON from LLM response.")
        return {"entities": [], "relations": []}
