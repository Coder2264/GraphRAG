"""
validate_qa.py — Quality validation pipeline for manually generated QA pairs.

Usage:
    python eval/validate_qa.py [--manifest PATH] [--output PATH]

For each QA pair in the manifest, scores it on 4 dimensions using Gemini:
  1. consistency         — Is the answer correct per the document?
  2. multi_hop_necessity — Would a single chunk be insufficient?
  3. specificity         — Is the question clear with one correct answer?
  4. difficulty          — Would top-5 RAG retrieval fail?

Scoring thresholds (from config.py):
  PASS:  mean >= 0.7 AND all dimensions >= 0.4
  WARN:  mean >= 0.5 AND any dimension < 0.5
  FAIL:  mean < 0.5 OR any dimension < 0.4

Requires GEMINI_API_KEY in .env (loaded automatically via config.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.config import (
    GEMINI_API_KEY,
    GEMINI_CALL_DELAY,
    MANIFEST_PATH,
    VALIDATED_QA_PATH,
    VALIDATION_MIN_DIMENSION,
    VALIDATION_MODEL,
    VALIDATION_PASS_THRESHOLD,
    VALIDATION_WARN_DIMENSION,
)
from eval.prompts import (
    VALIDATION_SYSTEM,
    consistency_prompt,
    difficulty_prompt,
    multi_hop_necessity_prompt,
    specificity_prompt,
)
from eval.utils import gemini_generate, parse_json_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-dimension validation
# ---------------------------------------------------------------------------

async def _score_dimension(prompt_text: str, dimension_name: str) -> tuple[float, str]:
    """Call Gemini for a single validation dimension. Returns (score, reason)."""
    try:
        raw = await gemini_generate(
            prompt=prompt_text,
            model=VALIDATION_MODEL,
            api_key=GEMINI_API_KEY,
            system=VALIDATION_SYSTEM,
            json_mode=True,
            temperature=0.1,
        )
        data = parse_json_response(raw)
        score = float(data.get("score", 0.0))
        reason = str(data.get("reason", ""))
        score = max(0.0, min(1.0, score))  # clamp to [0, 1]
        return score, reason
    except Exception as exc:
        logger.warning("  Dimension '%s' failed: %s", dimension_name, exc)
        return 0.0, f"Error: {exc}"


async def validate_qa_pair(
    doc_text: str,
    qa: dict,
    chunk_size: int = 512,
) -> dict:
    """
    Run all 4 validation dimensions for a single QA pair.

    Returns a validation result dict with per-dimension scores, mean, and status.
    """
    question = qa["question"]
    answer = qa["answer"]

    # Run the 4 dimensions concurrently
    consistency_task = _score_dimension(
        consistency_prompt(doc_text, question, answer), "consistency"
    )
    multi_hop_task = _score_dimension(
        multi_hop_necessity_prompt(doc_text, question, chunk_size), "multi_hop_necessity"
    )
    specificity_task = _score_dimension(
        specificity_prompt(question, answer), "specificity"
    )
    difficulty_task = _score_dimension(
        difficulty_prompt(doc_text, question), "difficulty"
    )

    (cons_score, cons_reason), (mh_score, mh_reason), (spec_score, spec_reason), (diff_score, diff_reason) = (
        await asyncio.gather(consistency_task, multi_hop_task, specificity_task, difficulty_task)
    )

    mean_score = (cons_score + mh_score + spec_score + diff_score) / 4.0
    min_score = min(cons_score, mh_score, spec_score, diff_score)

    # Determine status
    if mean_score < VALIDATION_PASS_THRESHOLD or min_score < VALIDATION_MIN_DIMENSION:
        status = "fail"
    elif min_score < VALIDATION_WARN_DIMENSION:
        status = "warn"
    else:
        status = "pass"

    return {
        "consistency": cons_score,
        "consistency_reason": cons_reason,
        "multi_hop_necessity": mh_score,
        "multi_hop_necessity_reason": mh_reason,
        "specificity": spec_score,
        "specificity_reason": spec_reason,
        "difficulty": diff_score,
        "difficulty_reason": diff_reason,
        "mean_score": round(mean_score, 3),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Document-level validation
# ---------------------------------------------------------------------------

async def validate_document(doc_entry: dict, qa_pairs_dir: Path) -> list[dict]:
    """
    Validate all QA pairs for a single document.

    Returns list of validated QA pair dicts (original fields + validation sub-dict).
    """
    qa_path = Path(doc_entry["qa_path"])
    if not qa_path.exists():
        logger.error("QA file not found: %s", qa_path)
        return []

    with qa_path.open("r", encoding="utf-8") as f:
        qa_data = json.load(f)

    doc_text: str = qa_data.get("document_text", "")
    qa_pairs: list[dict] = qa_data.get("qa_pairs", [])

    if not doc_text:
        logger.error("No document_text in %s", qa_path.name)
        return []

    validated_pairs: list[dict] = []
    for i, qa in enumerate(qa_pairs):
        logger.info(
            "  [%s] Validating QA %d/%d: %s",
            doc_entry["doc_index"],
            i + 1,
            len(qa_pairs),
            qa.get("id", "?"),
        )

        validation_result = await validate_qa_pair(doc_text, qa)

        validated_qa = dict(qa)
        validated_qa["doc_index"] = doc_entry["doc_index"]
        validated_qa["domain"] = doc_entry["domain"]
        validated_qa["topic"] = doc_entry["topic"]
        validated_qa["doc_path"] = doc_entry["doc_path"]
        validated_qa["validation"] = validation_result

        log_level = {
            "pass": logging.INFO,
            "warn": logging.WARNING,
            "fail": logging.ERROR,
        }[validation_result["status"]]
        logger.log(
            log_level,
            "    → %s (mean=%.2f, consistency=%.2f, multi_hop=%.2f, spec=%.2f, diff=%.2f)",
            validation_result["status"].upper(),
            validation_result["mean_score"],
            validation_result["consistency"],
            validation_result["multi_hop_necessity"],
            validation_result["specificity"],
            validation_result["difficulty"],
        )

        validated_pairs.append(validated_qa)

        # Brief pause between API calls
        await asyncio.sleep(GEMINI_CALL_DELAY)

    return validated_pairs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_validation(manifest_path: Path, output_path: Path) -> None:
    """Load manifest, validate all QA pairs, write validated output."""
    if not GEMINI_API_KEY:
        logger.error(
            "GEMINI_API_KEY not set. Add it to .env or set the environment variable."
        )
        sys.exit(1)

    if not manifest_path.exists():
        logger.error(
            "Manifest not found at %s\nRun python eval/import_manual.py first.",
            manifest_path,
        )
        sys.exit(1)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    docs = manifest.get("docs", [])
    if not docs:
        logger.error("No documents found in manifest.")
        sys.exit(1)

    logger.info("Validating QA pairs for %d document(s)...", len(docs))

    all_validated: list[dict] = []
    for doc_entry in docs:
        logger.info(
            "Document %s — %s: %s",
            doc_entry["doc_index"],
            doc_entry["domain"],
            doc_entry["topic"][:60],
        )
        validated = await validate_document(doc_entry, Path(manifest_path).parent / "qa_pairs")
        all_validated.extend(validated)

    # Compute summary
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for qa in all_validated:
        counts[qa["validation"]["status"]] += 1

    mean_scores = {
        "consistency": 0.0,
        "multi_hop_necessity": 0.0,
        "specificity": 0.0,
        "difficulty": 0.0,
        "overall": 0.0,
    }
    if all_validated:
        for qa in all_validated:
            v = qa["validation"]
            mean_scores["consistency"] += v["consistency"]
            mean_scores["multi_hop_necessity"] += v["multi_hop_necessity"]
            mean_scores["specificity"] += v["specificity"]
            mean_scores["difficulty"] += v["difficulty"]
            mean_scores["overall"] += v["mean_score"]
        n = len(all_validated)
        mean_scores = {k: round(v / n, 3) for k, v in mean_scores.items()}

    summary = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(all_validated),
        "passed": counts["pass"],
        "warned": counts["warn"],
        "failed": counts["fail"],
        "pass_rate": round(counts["pass"] / len(all_validated), 3) if all_validated else 0.0,
        "mean_scores": mean_scores,
    }

    output_data = {
        "summary": summary,
        "qa_pairs": all_validated,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=== Validation Summary ===")
    logger.info("  Total QA pairs: %d", summary["total"])
    logger.info("  PASS:  %d (%.0f%%)", summary["passed"], summary["pass_rate"] * 100)
    logger.info("  WARN:  %d", summary["warned"])
    logger.info("  FAIL:  %d", summary["failed"])
    logger.info("  Mean scores:")
    for dim, score in mean_scores.items():
        logger.info("    %-22s %.3f", dim + ":", score)
    logger.info("  Output: %s", output_path)
    logger.info("")
    logger.info("Next step: uvicorn main:app --reload  →  python eval/eval_rag.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate QA pairs using Gemini as a 4-dimension judge."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to eval/generated/manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VALIDATED_QA_PATH,
        help="Output path for validated QA pairs JSON",
    )
    args = parser.parse_args()

    asyncio.run(run_validation(args.manifest, args.output))


if __name__ == "__main__":
    main()
