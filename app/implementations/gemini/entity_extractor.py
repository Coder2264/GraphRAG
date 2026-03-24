"""
GeminiEntityExtractor — BaseEntityExtractor backed by Google Gemini.

Used exclusively for document ingestion (entity/relation extraction).
Calls the Gemini API asynchronously and returns structured JSON.

LSP: Fully substitutable for BaseEntityExtractor.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import google.generativeai as genai

from app.core.entity_extractor import BaseEntityExtractor
from app.prompts import EXTRACTION_SYSTEM_PROMPT, extraction_user_prompt

logger = logging.getLogger(__name__)


class GeminiEntityExtractor(BaseEntityExtractor):
    """
    Entity/relationship extractor backed by Google Gemini.

    Sends the extraction prompt to a Gemini model and parses the returned JSON.
    Degrades gracefully on any API or parse failure by returning empty lists.

    Args:
        api_key:    Google Gemini API key.
        model_name: Gemini model identifier (e.g. "gemini-2.5-pro").
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro") -> None:
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # BaseEntityExtractor interface
    # ------------------------------------------------------------------

    async def extract(self, text: str, processing_instruction: str = "") -> dict:
        """
        Call Gemini to extract entities and relations from *text*.

        Args:
            text:                   The document text to analyse.
            processing_instruction: Optional free-text hint guiding extraction.

        Returns:
            {"entities": [...], "relations": [...]}
            Empty lists on any error (graceful degradation).
        """
        user_msg = extraction_user_prompt(text, processing_instruction)
        try:
            response = await self._model.generate_content_async(
                user_msg,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            return self._parse_response(response.text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GeminiEntityExtractor: extraction failed: %s", exc)
            return {"entities": [], "relations": []}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """
        Parse the Gemini response as JSON with fallback strategies.

        1. Direct json.loads.
        2. Extract first {...} block via regex (handles any markdown fences).
        3. Return empty result if both fail.
        """
        try:
            parsed = json.loads(raw)
            return {
                "entities":  parsed.get("entities", []),
                "relations": parsed.get("relations", []),
            }
        except json.JSONDecodeError:
            pass

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

        logger.warning("GeminiEntityExtractor: could not parse JSON from response.")
        return {"entities": [], "relations": []}
