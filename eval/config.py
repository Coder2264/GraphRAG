"""
Evaluation configuration — all tunable constants for the eval pipeline.

Reads GEMINI_API_KEY from the project .env (or environment).
All other values are hardcoded defaults that can be overridden at the top
of each script before calling the main function.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the project .env so GEMINI_API_KEY is available
load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# LLM models
# ---------------------------------------------------------------------------

# Used by validate_qa.py — stronger model for nuanced multi-hop checks
VALIDATION_MODEL: str = "gemini-2.5-flash-preview-05-20"

# Used by eval_rag.py LLM judge — Flash is cost-efficient for bulk scoring
JUDGE_MODEL: str = "gemini-2.0-flash"

# Gemini API key — loaded from .env
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------

# Running GraphRAG server base URL
API_BASE_URL: str = "http://localhost:8000"

# top_k passed to all query modes
TOP_K: int = 5

# The three modes to benchmark
EVAL_MODES: list[str] = ["tog", "rag", "none"]

# How many mode queries to fire concurrently per QA pair
PARALLEL_QUERIES: int = 3

# Seconds to wait between document uploads to avoid overwhelming the server
UPLOAD_DELAY: float = 1.0

# Seconds to wait between Gemini API calls (rate limiting)
GEMINI_CALL_DELAY: float = 0.5

# ---------------------------------------------------------------------------
# QA validation thresholds
# ---------------------------------------------------------------------------

# Mean score (0–1) across 4 dimensions required to pass
VALIDATION_PASS_THRESHOLD: float = 0.7

# Any single dimension below this → fail regardless of mean
VALIDATION_MIN_DIMENSION: float = 0.4

# Any single dimension below this (but mean OK) → warn
VALIDATION_WARN_DIMENSION: float = 0.5

# ---------------------------------------------------------------------------
# Paths  (relative to project root, created at runtime)
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).parent.parent

MANUAL_INPUT_DIR: Path = PROJECT_ROOT / "eval" / "manual_input"
GENERATED_DIR: Path = PROJECT_ROOT / "eval" / "generated"
DOCS_DIR: Path = GENERATED_DIR / "docs"
QA_PAIRS_DIR: Path = GENERATED_DIR / "qa_pairs"
MANIFEST_PATH: Path = GENERATED_DIR / "manifest.json"
VALIDATED_QA_PATH: Path = GENERATED_DIR / "qa_pairs_validated.json"
RESULTS_DIR: Path = PROJECT_ROOT / "eval" / "results"
