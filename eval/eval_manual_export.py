import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.config import (
    API_BASE_URL,
    EVAL_MODES,
    PARALLEL_QUERIES,
    RESULTS_DIR,
    TOP_K,
    UPLOAD_DELAY,
    VALIDATED_QA_PATH,
)
from eval.eval_rag import check_server, upload_document, query_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

async def collect_qa_responses(
    qa: dict,
    doc_id: str,
    modes: list[str],
    client: httpx.AsyncClient,
    base_url: str,
    top_k: int,
) -> list[dict]:
    """Query modes for a single QA pair and return the results for export."""
    question = qa["question"]
    gold_answer = qa["answer"]
    justification = qa.get("answer_justification", "")
    
    logger.info("    Q: %s", question[:80])

    mode_results: dict[str, dict] = {}
    for i in range(0, len(modes), PARALLEL_QUERIES):
        batch = modes[i : i + PARALLEL_QUERIES]
        tasks = [query_mode(question, m, top_k, client, base_url) for m in batch]
        responses = await asyncio.gather(*tasks)
        for mode, response in zip(batch, responses):
            mode_results[mode] = response
            if response["error"]:
                logger.warning("      [%s] error: %s", mode, response["error"])
            else:
                logger.info(
                    "      [%s] %.1fs — answer: %s",
                    mode,
                    response["elapsed_seconds"],
                    response["answer"][:60],
                )

    exported_items = []
    for mode in modes:
        resp = mode_results.get(mode, {})
        # Only export successful ones. Errors handle as 0 manually or skip.
        if resp.get("error"):
            continue
            
        exported_items.append({
            "qa_id": qa.get("id", ""),
            "mode": mode,
            "question": question,
            "gold_answer": gold_answer,
            "answer_justification": justification,
            "system_answer": resp.get("answer", ""),
            "retrieved_context": resp.get("context", ""),
            "hop_count": qa.get("hop_count", 0),
            "domain": qa.get("domain", ""),
            "doc_index": qa.get("doc_index", ""),
            "sections_involved": qa.get("sections_involved", []),
            "elapsed_seconds": resp.get("elapsed_seconds", 0.0),
        })

    return exported_items

async def run_export(
    validated_qa_path: Path,
    base_url: str,
    modes: list[str],
    top_k: int,
) -> None:
    """Generate answers using the local GraphRAG server and export them."""
    if not validated_qa_path.exists():
        logger.error(
            "Validated QA file not found: %s\nRun python eval/validate_qa.py first.",
            validated_qa_path,
        )
        sys.exit(1)

    with validated_qa_path.open("r", encoding="utf-8") as f:
        validated_data = json.load(f)

    # Filter to pass + warn only
    all_pairs = validated_data.get("qa_pairs", [])
    qa_pairs = [qa for qa in all_pairs if qa.get("validation", {}).get("status") in ("pass", "warn")]
    
    if not qa_pairs:
        logger.error("No valid QA pairs to evaluate.")
        sys.exit(1)

    all_exported_items = []

    async with httpx.AsyncClient() as client:
        # Health check
        if not await check_server(client, base_url):
            logger.error("Server not reachable at %s", base_url)
            sys.exit(1)
        logger.info("Server healthy at %s", base_url)

        # Upload documents
        doc_id_map: dict[str, str] = {}
        seen_indices = set()
        docs_to_upload = []
        for qa in qa_pairs:
            idx = qa.get("doc_index", "")
            if idx and idx not in seen_indices:
                seen_indices.add(idx)
                docs_to_upload.append(qa)

        for qa in docs_to_upload:
            idx = qa["doc_index"]
            pdf_path = Path(qa["doc_path"])
            doc_id = await upload_document(pdf_path, client, base_url)
            if doc_id:
                doc_id_map[idx] = doc_id
            await asyncio.sleep(UPLOAD_DELAY)

        # Evaluate QA pairs
        total = len(qa_pairs)
        for i, qa in enumerate(qa_pairs):
            doc_id = doc_id_map.get(qa.get("doc_index", ""), "unknown")
            logger.info("Processing [%d/%d]", i + 1, total)
            items = await collect_qa_responses(qa, doc_id, modes, client, base_url, top_k)
            all_exported_items.extend(items)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    export_path = RESULTS_DIR / f"manual_eval_export_{timestamp}.json"
    
    export_payload = {
        "timestamp": timestamp,
        "config": {"modes": modes, "top_k": top_k},
        "instructions_for_web_llm": (
            "You are an evaluator. Look at each item in the 'items_to_judge' array. "
            "For each item, provide a retrieval_score (0-10) by comparing "
            "'retrieved_context' against 'answer_justification' and 'gold_answer'. "
            "Provide an answer_score (0-10) by comparing 'system_answer' against 'gold_answer'. "
            "Output ONLY a valid JSON list of objects, each containing: "
            "'qa_id', 'mode', 'retrieval_score', 'retrieval_reason', 'answer_score', 'answer_reason'."
        ),
        "items_to_judge": all_exported_items
    }

    with export_path.open("w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)

    logger.info("Export complete! File: %s", export_path)
    logger.info("Upload this file to ChatGPT or Gemini web UI to get the judgments.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Export QA responses for manual web LLM evaluation.")
    parser.add_argument("--validated", type=Path, default=VALIDATED_QA_PATH)
    parser.add_argument("--api", default=API_BASE_URL)
    parser.add_argument("--modes", nargs="+", default=EVAL_MODES)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    asyncio.run(run_export(args.validated, args.api, args.modes, args.top_k))

if __name__ == "__main__":
    main()
