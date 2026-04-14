import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.config import RESULTS_DIR, JUDGE_MODEL
from eval.eval_rag import compute_stats, generate_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_import(export_path: Path, judgments_path: Path) -> None:
    if not export_path.exists():
        logger.error(f"Export file not found: {export_path}")
        sys.exit(1)
    if not judgments_path.exists():
        logger.error(f"Judgments file not found: {judgments_path}")
        sys.exit(1)

    with export_path.open("r", encoding="utf-8") as f:
        export_data = json.load(f)

    with judgments_path.open("r", encoding="utf-8") as f:
        try:
            judgments_array = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse judgments file. Must be valid JSON list: {exc}")
            sys.exit(1)

    if not isinstance(judgments_array, list):
        logger.error("Judgments JSON must be a list of objects.")
        sys.exit(1)

    # Index judgments by (qa_id, mode)
    judgment_map = {}
    for j in judgments_array:
        qa_id = j.get("qa_id")
        mode = j.get("mode")
        if qa_id and mode:
            judgment_map[(qa_id, mode)] = j

    items = export_data.get("items_to_judge", [])
    config_modes = export_data.get("config", {}).get("modes", [])
    
    # Reconstruct the results structure expected by compute_stats
    qa_groups = {}
    for item in items:
        qa_id = item["qa_id"]
        if qa_id not in qa_groups:
            qa_groups[qa_id] = {
                "qa_id": qa_id,
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "hop_count": item["hop_count"],
                "domain": item["domain"],
                "doc_index": item["doc_index"],
                "sections_involved": item["sections_involved"],
                "answer_justification": item["answer_justification"],
                "responses": {}
            }
        
        mode = item["mode"]
        judg = judgment_map.get((qa_id, mode), {})
        
        # Parse scores strictly ensuring they are ints
        try: r_score = int(judg.get("retrieval_score", 0))
        except: r_score = 0
            
        try: a_score = int(judg.get("answer_score", 0))
        except: a_score = 0

        qa_groups[qa_id]["responses"][mode] = {
            "answer": item["system_answer"],
            "context": item["retrieved_context"],
            "elapsed_seconds": item["elapsed_seconds"],
            "error": None,
            "retrieval_score": r_score,
            "retrieval_reason": str(judg.get("retrieval_reason", "Missing judgment")),
            "answer_score": a_score,
            "answer_reason": str(judg.get("answer_reason", "Missing judgment")),
        }

    all_results = list(qa_groups.values())
    
    if not all_results:
        logger.error("No valid QA results parsed. Exiting.")
        sys.exit(1)

    logger.info(f"Reconstructed {len(all_results)} QA evaluations.")

    # Compute stats
    agg = compute_stats(all_results, config_modes)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    full_results = {
        "timestamp": timestamp,
        "config": {
            "modes": config_modes,
            "top_k": export_data.get("config", {}).get("top_k", 5),
            "judge_model": "manual_web_llm",
            "api_base_url": "manual",
        },
        "total_qa_pairs_evaluated": len(all_results),
        "aggregate": agg,
        "results": all_results,
    }

    # Write outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"manual_eval_results_{timestamp}.json"
    report_path = RESULTS_DIR / f"manual_report_{timestamp}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    report_md = generate_report(full_results)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("\n=== Manual Evaluation Complete ===")
    logger.info(f"  QA pairs evaluated: {len(all_results)}")
    logger.info(f"  Modes: {', '.join(config_modes)}")
    logger.info("\n  %-8s  Retrieval  Answer" % "Mode")
    for mode in config_modes:
        s = agg["per_mode"].get(mode, {})
        m_r_score = s.get("mean_retrieval_score", 0)
        m_a_score = s.get("mean_answer_score", 0)
        logger.info(f"  {mode:<8}  {m_r_score:5.2f}/10   {m_a_score:5.2f}/10")
    logger.info(f"\n  Results: {json_path}")
    logger.info(f"  Report:  {report_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Import web LLM judgments and generate a report.")
    parser.add_argument("--export", type=Path, required=True, help="Path to the JSON file generated by eval_manual_export.py")
    parser.add_argument("--judgments", type=Path, required=True, help="Path to the JSON file containing the LLM judgments array")
    args = parser.parse_args()

    run_import(args.export, args.judgments)

if __name__ == "__main__":
    main()
