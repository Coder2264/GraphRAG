"""
import_manual.py — Import manually generated JSON files into the eval pipeline.

Usage:
    python eval/import_manual.py [--input DIR] [--output DIR]

Workflow:
  1. Scan eval/manual_input/*.json for document JSON files.
  2. Validate each file against the required schema.
  3. Create a PDF from the document_text using reportlab.
  4. Write per-document QA JSON (preserving document_text for validate_qa.py).
  5. Write eval/generated/manifest.json with an index of all documents.

After running this script, run eval/validate_qa.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root: python eval/import_manual.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.config import (
    DOCS_DIR,
    GENERATED_DIR,
    MANUAL_INPUT_DIR,
    MANIFEST_PATH,
    QA_PAIRS_DIR,
)
from eval.utils import text_to_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_REQUIRED_DOC_FIELDS = {
    "doc_index", "domain", "topic", "document_text", "entities", "qa_pairs"
}

_REQUIRED_QA_FIELDS = {
    "id", "question", "answer", "hop_count",
    "entity_chain", "sections_involved", "answer_justification",
}

_VALID_DOMAINS = {
    "corporate_history", "scientific_research", "academic_collaboration",
    "medical_research", "tech_genealogy", "legal_proceedings",
    "historical_politics", "supply_chain", "nonprofit_foundations",
    "cultural_institutions", "fantasy",
}


def validate_doc(data: dict, source_file: str) -> list[str]:
    """Return list of validation error strings (empty = valid)."""
    errors: list[str] = []

    # Top-level required fields
    missing = _REQUIRED_DOC_FIELDS - set(data.keys())
    if missing:
        errors.append(f"Missing top-level fields: {sorted(missing)}")
        return errors  # Can't continue without these

    # Domain check
    if data["domain"] not in _VALID_DOMAINS:
        errors.append(
            f"Unknown domain '{data['domain']}'. "
            f"Valid: {sorted(_VALID_DOMAINS)}"
        )

    # document_text must be non-empty and contain sections
    doc_text: str = data["document_text"]
    if not doc_text or len(doc_text.strip()) < 500:
        errors.append("document_text is too short (< 500 chars)")
    section_count = doc_text.count("## Section")
    if section_count < 3:
        errors.append(
            f"document_text has only {section_count} '## Section' headers "
            f"(expected at least 3)"
        )

    # entities list
    if not isinstance(data["entities"], list) or len(data["entities"]) < 5:
        errors.append("entities must be a list with at least 5 items")

    # qa_pairs list
    if not isinstance(data["qa_pairs"], list) or len(data["qa_pairs"]) < 3:
        errors.append("qa_pairs must be a list with at least 3 items")
    else:
        for i, qa in enumerate(data["qa_pairs"]):
            missing_qa = _REQUIRED_QA_FIELDS - set(qa.keys())
            if missing_qa:
                errors.append(f"qa_pairs[{i}] missing fields: {sorted(missing_qa)}")
            else:
                if not isinstance(qa["hop_count"], int) or qa["hop_count"] < 2:
                    errors.append(f"qa_pairs[{i}].hop_count must be int >= 2")
                if not isinstance(qa["entity_chain"], list) or len(qa["entity_chain"]) < 2:
                    errors.append(f"qa_pairs[{i}].entity_chain must have >= 2 items")
                if not isinstance(qa["sections_involved"], list) or len(qa["sections_involved"]) < 2:
                    errors.append(f"qa_pairs[{i}].sections_involved must have >= 2 items")

    return errors


# ---------------------------------------------------------------------------
# Import logic
# ---------------------------------------------------------------------------

def import_document(json_path: Path, docs_dir: Path, qa_pairs_dir: Path) -> dict | None:
    """
    Load, validate, and process a single document JSON file.

    Returns a manifest entry dict on success, None on validation failure.
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error in %s: %s", json_path.name, exc)
        return None

    errors = validate_doc(data, json_path.name)
    if errors:
        logger.error("Validation failed for %s:", json_path.name)
        for err in errors:
            logger.error("  • %s", err)
        return None

    raw_index = str(data["doc_index"]).strip().lstrip("0") or "0"
    padded = str(data["doc_index"]).strip().zfill(3)
    doc_stem = f"doc_{padded}"

    # Create PDF
    pdf_path = docs_dir / f"{doc_stem}.pdf"
    try:
        text_to_pdf(
            text=data["document_text"],
            output_path=pdf_path,
            title=data["topic"],
        )
        logger.info("  PDF created: %s", pdf_path.name)
    except Exception as exc:
        logger.error("  Failed to create PDF for %s: %s", json_path.name, exc)
        return None

    # Write per-document QA JSON (includes document_text for validate_qa.py)
    qa_out_path = qa_pairs_dir / f"{doc_stem}_qa.json"
    qa_data = {
        "doc_index": padded,
        "doc_path": str(pdf_path.relative_to(pdf_path.parent.parent)),
        "domain": data["domain"],
        "topic": data["topic"],
        "document_text": data["document_text"],
        "entities": data["entities"],
        "qa_pairs": data["qa_pairs"],
    }
    with qa_out_path.open("w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)
    logger.info("  QA JSON written: %s (%d pairs)", qa_out_path.name, len(data["qa_pairs"]))

    return {
        "doc_index": padded,
        "domain": data["domain"],
        "topic": data["topic"],
        "doc_path": str(pdf_path),
        "qa_path": str(qa_out_path),
        "qa_count": len(data["qa_pairs"]),
        "source_file": json_path.name,
    }


def run_import(input_dir: Path, generated_dir: Path) -> None:
    """
    Scan input_dir for .json files, import each, and write manifest.json.
    """
    docs_dir = generated_dir / "docs"
    qa_pairs_dir = generated_dir / "qa_pairs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    qa_pairs_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        logger.error(
            "No .json files found in %s\n"
            "Generate documents using Claude/ChatGPT/Gemini and save them there "
            "(e.g. doc_001.json).",
            input_dir,
        )
        sys.exit(1)

    logger.info("Found %d JSON file(s) in %s", len(json_files), input_dir)

    manifest_entries: list[dict] = []
    failed: list[str] = []

    for json_path in json_files:
        logger.info("Processing %s ...", json_path.name)
        entry = import_document(json_path, docs_dir, qa_pairs_dir)
        if entry:
            manifest_entries.append(entry)
        else:
            failed.append(json_path.name)

    # Sort manifest by doc_index
    manifest_entries.sort(key=lambda e: e["doc_index"])

    total_qa = sum(e["qa_count"] for e in manifest_entries)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_docs": len(manifest_entries),
        "total_qa_pairs": total_qa,
        "failed_imports": failed,
        "docs": manifest_entries,
    }

    manifest_path = generated_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=== Import Summary ===")
    logger.info("  Imported:    %d documents", len(manifest_entries))
    logger.info("  QA pairs:    %d total", total_qa)
    logger.info("  Failed:      %d", len(failed))
    if failed:
        logger.warning("  Failed files: %s", ", ".join(failed))
    logger.info("  Manifest:    %s", manifest_path)
    logger.info("")
    logger.info("Next step: python eval/validate_qa.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import manually generated JSON documents into the eval pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=MANUAL_INPUT_DIR,
        help="Directory containing manually generated doc_NNN.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GENERATED_DIR,
        help="Output directory for PDFs, QA JSONs, and manifest",
    )
    args = parser.parse_args()

    run_import(args.input, args.output)


if __name__ == "__main__":
    main()
