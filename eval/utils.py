"""
Shared utilities for the eval pipeline.

Provides:
  - text_to_pdf:     convert plain text (with ## Section headers) to a PDF
  - gemini_generate: async Gemini API call with retry/backoff
  - parse_json_response: robustly extract JSON from an LLM response string
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import textwrap
from pathlib import Path

import google.generativeai as genai

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def text_to_pdf(text: str, output_path: str | Path, title: str = "") -> None:
    """
    Convert plain text with "## Section N: Title" headers to a PDF.

    Uses reportlab with A4 page, 1-inch margins, and readable typography.
    Section headers render as bold Heading 2; body text wraps at page width.

    Args:
        text:        Document text (may contain "## Section N: ..." headers).
        output_path: Destination .pdf file path (parent directory must exist).
        title:       Optional document title rendered at the top of page 1.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=16,
        spaceAfter=14,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=6,
        alignment=TA_LEFT,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        spaceAfter=4,
        alignment=TA_LEFT,
    )

    story = []

    if title:
        story.append(Paragraph(_esc(title), title_style))
        story.append(Spacer(1, 0.15 * inch))

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 0.08 * inch))
        elif stripped.startswith("## "):
            heading_text = stripped[3:].strip()
            story.append(Paragraph(_esc(heading_text), heading_style))
        elif stripped.startswith("# "):
            story.append(Paragraph(_esc(stripped[2:].strip()), title_style))
        else:
            # Wrap long lines and escape XML special chars
            story.append(Paragraph(_esc(stripped), body_style))

    doc.build(story)


def _esc(text: str) -> str:
    """Escape characters that break reportlab's XML parser."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Gemini API wrapper
# ---------------------------------------------------------------------------

async def gemini_generate(
    prompt: str,
    model: str,
    api_key: str,
    system: str = "",
    json_mode: bool = False,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    """
    Async Gemini generation with exponential-backoff retry.

    Args:
        prompt:      User-turn prompt text.
        model:       Gemini model name (e.g. "gemini-2.0-flash").
        api_key:     Google Gemini API key.
        system:      Optional system instruction.
        json_mode:   If True, sets response_mime_type to application/json.
        temperature: Sampling temperature (lower = more deterministic).
        max_retries: Number of retry attempts on transient errors.

    Returns:
        Raw text response from Gemini.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    genai.configure(api_key=api_key)

    model_kwargs: dict = {}
    if system:
        model_kwargs["system_instruction"] = system

    gen_model = genai.GenerativeModel(model_name=model, **model_kwargs)

    gen_config = genai.types.GenerationConfig(
        temperature=temperature,
        response_mime_type="application/json" if json_mode else "text/plain",
    )

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await gen_model.generate_content_async(
                prompt, generation_config=gen_config
            )
            return response.text
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Gemini call failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)

    raise RuntimeError(
        f"Gemini API failed after {max_retries} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def parse_json_response(raw: str) -> dict | list:
    """
    Robustly parse a JSON object or array from an LLM response.

    Strips markdown code fences if present, then parses.

    Args:
        raw: Raw LLM response text.

    Returns:
        Parsed Python dict or list.

    Raises:
        ValueError: If the response cannot be parsed as JSON.
    """
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # If the response has leading prose before the JSON object, extract the JSON
    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if json_match:
        text = json_match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse JSON from LLM response: {exc}\n\nRaw:\n{raw[:500]}") from exc
