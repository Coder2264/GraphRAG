"""
eval_rag.py — Automated benchmark comparing tog vs rag vs none.

Usage:
    python eval/eval_rag.py [--validated PATH] [--api URL] [--modes MODE ...]

Workflow:
  1. Load validated QA pairs from eval/generated/qa_pairs_validated.json.
  2. For each unique document: upload the PDF to the running GraphRAG server.
  3. For each QA pair (pass or warn status): query all 3 modes concurrently.
  4. LLM judge (Gemini) scores each response on:
       - retrieval_score (0-10): does the context cover the required answer chain?
       - answer_score   (0-10): how correct/complete is the answer vs. gold?
  5. Aggregate results and write:
       - eval/results/eval_results_{timestamp}.json  (full data)
       - eval/results/report_{timestamp}.md           (human-readable summary)

Requires:
  - GraphRAG server running at API_BASE_URL (default http://localhost:8000)
  - GEMINI_API_KEY in .env
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.config import (
    API_BASE_URL,
    EVAL_MODES,
    GEMINI_API_KEY,
    GEMINI_CALL_DELAY,
    JUDGE_MODEL,
    PARALLEL_QUERIES,
    RESULTS_DIR,
    TOP_K,
    UPLOAD_DELAY,
    VALIDATED_QA_PATH,
)
from eval.prompts import (
    ANSWER_JUDGE_SYSTEM,
    RETRIEVAL_JUDGE_SYSTEM,
    answer_judge_prompt,
    retrieval_judge_prompt,
)
from eval.utils import gemini_generate, parse_json_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server interaction
# ---------------------------------------------------------------------------

async def check_server(client: httpx.AsyncClient, base_url: str) -> bool:
    """Return True if the server is reachable and healthy."""
    try:
        resp = await client.get(f"{base_url}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


async def upload_document(
    pdf_path: Path,
    client: httpx.AsyncClient,
    base_url: str,
) -> str | None:
    """
    Upload a PDF file to the GraphRAG server.

    Returns doc_id string on success, None on failure.
    """
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return None

    try:
        with pdf_path.open("rb") as f:
            files = {"file": (pdf_path.name, f, "application/pdf")}
            data = {"source": pdf_path.stem}
            resp = await client.post(
                f"{base_url}/api/v1/ingest/upload",
                files=files,
                data=data,
                timeout=300.0,  # entity extraction can be slow
            )

        if resp.status_code != 200:
            logger.error(
                "Upload failed for %s: HTTP %d — %s",
                pdf_path.name,
                resp.status_code,
                resp.text[:200],
            )
            return None

        result = resp.json()
        doc_id = result.get("doc_id", "")
        chunks = result.get("chunks_count", 0)
        entities = result.get("graph_entities_count", 0)
        logger.info(
            "  Uploaded %s → doc_id=%s  chunks=%d  entities=%d",
            pdf_path.name,
            doc_id,
            chunks,
            entities,
        )
        return doc_id

    except Exception as exc:
        logger.error("Upload exception for %s: %s", pdf_path.name, exc)
        return None


async def query_mode(
    question: str,
    mode: str,
    top_k: int,
    client: httpx.AsyncClient,
    base_url: str,
) -> dict[str, Any]:
    """
    Query a single mode and return the response dict.

    Returns a dict with keys: answer, context, elapsed_seconds, error (if any).
    """
    payload = {"question": question, "top_k": top_k}
    try:
        resp = await client.post(
            f"{base_url}/api/v1/query/{mode}",
            json=payload,
            timeout=120.0,
        )
        if resp.status_code != 200:
            return {
                "answer": "",
                "context": "",
                "elapsed_seconds": 0.0,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
        data = resp.json()
        return {
            "answer": data.get("answer", ""),
            "context": data.get("context", ""),
            "elapsed_seconds": data.get("elapsed_seconds", 0.0),
            "error": None,
        }
    except Exception as exc:
        return {
            "answer": "",
            "context": "",
            "elapsed_seconds": 0.0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

async def judge_response(
    question: str,
    gold_answer: str,
    answer: str,
    context: str,
    answer_justification: str,
) -> dict[str, Any]:
    """
    Score a system response on retrieval quality and answer quality.

    Returns dict with retrieval_score, retrieval_reason, answer_score, answer_reason.
    """
    retrieval_task = gemini_generate(
        prompt=retrieval_judge_prompt(question, gold_answer, context, answer_justification),
        model=JUDGE_MODEL,
        api_key=GEMINI_API_KEY,
        system=RETRIEVAL_JUDGE_SYSTEM,
        json_mode=True,
        temperature=0.0,
    )
    answer_task = gemini_generate(
        prompt=answer_judge_prompt(question, gold_answer, answer),
        model=JUDGE_MODEL,
        api_key=GEMINI_API_KEY,
        system=ANSWER_JUDGE_SYSTEM,
        json_mode=True,
        temperature=0.0,
    )

    raw_retrieval, raw_answer = await asyncio.gather(retrieval_task, answer_task)

    def _parse_score(raw: str, name: str) -> tuple[int, str]:
        try:
            data = parse_json_response(raw)
            score = int(data.get("score", 0))
            score = max(0, min(10, score))
            return score, str(data.get("reason", ""))
        except Exception as exc:
            logger.warning("Judge parse error (%s): %s", name, exc)
            return 0, f"Parse error: {exc}"

    r_score, r_reason = _parse_score(raw_retrieval, "retrieval")
    a_score, a_reason = _parse_score(raw_answer, "answer")

    return {
        "retrieval_score": r_score,
        "retrieval_reason": r_reason,
        "answer_score": a_score,
        "answer_reason": a_reason,
    }


# ---------------------------------------------------------------------------
# Per-QA-pair evaluation
# ---------------------------------------------------------------------------

async def evaluate_qa_pair(
    qa: dict,
    doc_id: str,
    modes: list[str],
    client: httpx.AsyncClient,
    base_url: str,
    top_k: int,
) -> dict[str, Any]:
    """
    Query all modes for a single QA pair and judge each response.

    Returns a result dict for this QA pair with per-mode scores.
    """
    question = qa["question"]
    gold_answer = qa["answer"]
    justification = qa.get("answer_justification", "")

    logger.info("    Q: %s", question[:80])

    # Query all modes concurrently (batched by PARALLEL_QUERIES)
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

    # Judge each mode's response
    judge_tasks = {}
    for mode, resp in mode_results.items():
        if resp["error"]:
            continue
        judge_tasks[mode] = judge_response(
            question,
            gold_answer,
            resp["answer"],
            resp["context"],
            justification,
        )

    if judge_tasks:
        mode_names = list(judge_tasks.keys())
        judgements_list = await asyncio.gather(*judge_tasks.values())
        judgements = dict(zip(mode_names, judgements_list))
        await asyncio.sleep(GEMINI_CALL_DELAY)
    else:
        judgements = {}

    # Combine into per-mode response dicts
    final_responses: dict[str, dict] = {}
    for mode in modes:
        resp = mode_results.get(mode, {})
        judge = judgements.get(mode, {"retrieval_score": 0, "retrieval_reason": "no response", "answer_score": 0, "answer_reason": "no response"})
        final_responses[mode] = {
            "answer": resp.get("answer", ""),
            "context": resp.get("context", ""),
            "elapsed_seconds": resp.get("elapsed_seconds", 0.0),
            "error": resp.get("error"),
            "retrieval_score": judge.get("retrieval_score", 0),
            "retrieval_reason": judge.get("retrieval_reason", ""),
            "answer_score": judge.get("answer_score", 0),
            "answer_reason": judge.get("answer_reason", ""),
        }
        logger.info(
            "      [%s] retrieval=%d/10  answer=%d/10",
            mode,
            final_responses[mode]["retrieval_score"],
            final_responses[mode]["answer_score"],
        )

    return {
        "qa_id": qa.get("id", ""),
        "question": question,
        "gold_answer": gold_answer,
        "hop_count": qa.get("hop_count", 0),
        "domain": qa.get("domain", ""),
        "doc_index": qa.get("doc_index", ""),
        "sections_involved": qa.get("sections_involved", []),
        "answer_justification": justification,
        "responses": final_responses,
    }


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def compute_stats(results: list[dict], modes: list[str]) -> dict[str, Any]:
    """Compute aggregate statistics from all result dicts."""
    stats: dict[str, Any] = {}

    for mode in modes:
        retrieval_scores = [
            r["responses"][mode]["retrieval_score"]
            for r in results
            if mode in r["responses"] and r["responses"][mode]["error"] is None
        ]
        answer_scores = [
            r["responses"][mode]["answer_score"]
            for r in results
            if mode in r["responses"] and r["responses"][mode]["error"] is None
        ]
        elapsed = [
            r["responses"][mode]["elapsed_seconds"]
            for r in results
            if mode in r["responses"] and r["responses"][mode]["error"] is None
        ]

        stats[mode] = {
            "n": len(answer_scores),
            "mean_retrieval_score": round(mean(retrieval_scores), 2) if retrieval_scores else 0.0,
            "mean_answer_score": round(mean(answer_scores), 2) if answer_scores else 0.0,
            "mean_elapsed_seconds": round(mean(elapsed), 2) if elapsed else 0.0,
        }

    # Head-to-head win rates (tog vs rag, tog vs none, rag vs none)
    comparisons = [
        ("tog", "rag"),
        ("tog", "none"),
        ("rag", "none"),
    ]
    head_to_head = {}
    for a, b in comparisons:
        if a not in modes or b not in modes:
            continue
        wins_a = wins_b = ties = 0
        for r in results:
            sa = r["responses"].get(a, {}).get("answer_score", 0)
            sb = r["responses"].get(b, {}).get("answer_score", 0)
            if sa > sb:
                wins_a += 1
            elif sb > sa:
                wins_b += 1
            else:
                ties += 1
        total = wins_a + wins_b + ties
        head_to_head[f"{a}_vs_{b}"] = {
            f"{a}_wins": wins_a,
            f"{b}_wins": wins_b,
            "ties": ties,
            f"{a}_win_rate": round(wins_a / total, 3) if total else 0.0,
        }

    # By hop count
    by_hop: dict[int, dict] = {}
    for hop in [2, 3, 4]:
        hop_results = [r for r in results if r.get("hop_count") == hop]
        if not hop_results:
            continue
        by_hop[hop] = {}
        for mode in modes:
            scores = [
                r["responses"][mode]["answer_score"]
                for r in hop_results
                if mode in r["responses"] and r["responses"][mode]["error"] is None
            ]
            by_hop[hop][mode] = {
                "n": len(scores),
                "mean_answer_score": round(mean(scores), 2) if scores else 0.0,
            }

    # By domain
    domains = sorted({r.get("domain", "unknown") for r in results})
    by_domain: dict[str, dict] = {}
    for domain in domains:
        domain_results = [r for r in results if r.get("domain") == domain]
        by_domain[domain] = {}
        for mode in modes:
            scores = [
                r["responses"][mode]["answer_score"]
                for r in domain_results
                if mode in r["responses"] and r["responses"][mode]["error"] is None
            ]
            by_domain[domain][mode] = {
                "n": len(scores),
                "mean_answer_score": round(mean(scores), 2) if scores else 0.0,
            }

    return {
        "per_mode": stats,
        "head_to_head": head_to_head,
        "by_hop_count": by_hop,
        "by_domain": by_domain,
    }


def generate_report(full_results: dict) -> str:
    """Render a Markdown summary report."""
    agg = full_results["aggregate"]
    modes = full_results["config"]["modes"]
    total = full_results["total_qa_pairs_evaluated"]
    ts = full_results["timestamp"]

    lines: list[str] = []
    lines.append(f"# GraphRAG Evaluation Report")
    lines.append(f"\n**Generated:** {ts}  |  **QA pairs evaluated:** {total}")
    lines.append(f"**Modes:** {', '.join(modes)}  |  **Top-K:** {full_results['config']['top_k']}")

    # Overall summary table
    lines.append("\n## Overall Summary\n")
    lines.append("| Mode | N | Mean Retrieval Score | Mean Answer Score | Mean Time (s) |")
    lines.append("|---|---|---|---|---|")
    for mode in modes:
        s = agg["per_mode"].get(mode, {})
        lines.append(
            f"| **{mode}** | {s.get('n',0)} | {s.get('mean_retrieval_score',0):.2f}/10 "
            f"| {s.get('mean_answer_score',0):.2f}/10 | {s.get('mean_elapsed_seconds',0):.2f}s |"
        )

    # Head-to-head
    lines.append("\n## Head-to-Head (Answer Quality)\n")
    h2h = agg.get("head_to_head", {})
    lines.append("| Matchup | Winner Win Rate | Loser Wins | Ties |")
    lines.append("|---|---|---|---|")
    for key, data in h2h.items():
        a, _, b = key.partition("_vs_")
        win_rate = data.get(f"{a}_win_rate", 0)
        a_wins = data.get(f"{a}_wins", 0)
        b_wins = data.get(f"{b}_wins", 0)
        ties = data.get("ties", 0)
        winner = a if win_rate >= 0.5 else b
        winner_wins = a_wins if win_rate >= 0.5 else b_wins
        loser_wins = b_wins if win_rate >= 0.5 else a_wins
        lines.append(
            f"| **{a}** vs **{b}** | **{winner}** {win_rate:.0%} | {loser_wins} | {ties} |"
        )

    # By hop count
    by_hop = agg.get("by_hop_count", {})
    if by_hop:
        lines.append("\n## Performance by Hop Count (Mean Answer Score)\n")
        lines.append("| Hop Count | " + " | ".join(f"**{m}**" for m in modes) + " |")
        lines.append("|---|" + "---|" * len(modes))
        for hop in sorted(by_hop.keys()):
            row = f"| {hop}-hop"
            for mode in modes:
                s = by_hop[hop].get(mode, {})
                row += f" | {s.get('mean_answer_score', 0):.2f}/10 (n={s.get('n',0)})"
            row += " |"
            lines.append(row)

    # By domain
    by_domain = agg.get("by_domain", {})
    if by_domain:
        lines.append("\n## Performance by Domain (Mean Answer Score)\n")
        lines.append("| Domain | " + " | ".join(f"**{m}**" for m in modes) + " |")
        lines.append("|---|" + "---|" * len(modes))
        for domain in sorted(by_domain.keys()):
            row = f"| {domain}"
            for mode in modes:
                s = by_domain[domain].get(mode, {})
                row += f" | {s.get('mean_answer_score', 0):.2f}/10"
            row += " |"
            lines.append(row)

    # Failure analysis — bottom 5 by tog answer score
    results_list = full_results.get("results", [])
    if results_list and "tog" in modes:
        worst = sorted(
            results_list,
            key=lambda r: r["responses"].get("tog", {}).get("answer_score", 10),
        )[:5]
        lines.append("\n## Worst ToG Results (for investigation)\n")
        for i, r in enumerate(worst):
            tog_resp = r["responses"].get("tog", {})
            lines.append(f"**{i+1}. [{r.get('domain','')}] {r['question']}**")
            lines.append(f"- Gold: {r['gold_answer']}")
            lines.append(f"- ToG answer: {tog_resp.get('answer','')[:120]}")
            lines.append(
                f"- Retrieval: {tog_resp.get('retrieval_score',0)}/10  |  "
                f"Answer: {tog_resp.get('answer_score',0)}/10"
            )
            lines.append(f"- Reason: {tog_resp.get('answer_reason','')[:100]}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def run_evaluation(
    validated_qa_path: Path,
    base_url: str,
    modes: list[str],
    top_k: int,
) -> None:
    """Load validated QA pairs, upload docs, query all modes, judge, report."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set. Add it to .env.")
        sys.exit(1)

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
    logger.info(
        "Loaded %d QA pairs (%d total, %d failed excluded)",
        len(qa_pairs),
        len(all_pairs),
        len(all_pairs) - len(qa_pairs),
    )

    if not qa_pairs:
        logger.error("No valid QA pairs to evaluate.")
        sys.exit(1)

    async with httpx.AsyncClient() as client:
        # Health check
        if not await check_server(client, base_url):
            logger.error(
                "Server not reachable at %s\nStart it with: uvicorn main:app --reload",
                base_url,
            )
            sys.exit(1)
        logger.info("Server healthy at %s", base_url)

        # Upload documents (deduplicated by doc_index)
        doc_id_map: dict[str, str] = {}  # doc_index → doc_id
        seen_indices = set()
        docs_to_upload = []
        for qa in qa_pairs:
            idx = qa.get("doc_index", "")
            if idx and idx not in seen_indices:
                seen_indices.add(idx)
                docs_to_upload.append(qa)

        logger.info("Uploading %d unique document(s)...", len(docs_to_upload))
        for qa in docs_to_upload:
            idx = qa["doc_index"]
            pdf_path = Path(qa["doc_path"])
            doc_id = await upload_document(pdf_path, client, base_url)
            if doc_id:
                doc_id_map[idx] = doc_id
            else:
                logger.error("Skipping doc %s — upload failed", idx)
            await asyncio.sleep(UPLOAD_DELAY)

        # Evaluate QA pairs
        all_results: list[dict] = []
        total = len(qa_pairs)
        for i, qa in enumerate(qa_pairs):
            doc_id = doc_id_map.get(qa.get("doc_index", ""), "unknown")
            logger.info(
                "[%d/%d] doc=%s  hop=%d  id=%s",
                i + 1,
                total,
                qa.get("doc_index", "?"),
                qa.get("hop_count", 0),
                qa.get("id", "?"),
            )
            result = await evaluate_qa_pair(qa, doc_id, modes, client, base_url, top_k)
            all_results.append(result)

    # Compute aggregate stats
    agg = compute_stats(all_results, modes)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    full_results = {
        "timestamp": timestamp,
        "config": {
            "modes": modes,
            "top_k": top_k,
            "judge_model": JUDGE_MODEL,
            "api_base_url": base_url,
        },
        "total_qa_pairs_evaluated": len(all_results),
        "aggregate": agg,
        "results": all_results,
    }

    # Write outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"eval_results_{timestamp}.json"
    report_path = RESULTS_DIR / f"report_{timestamp}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    report_md = generate_report(full_results)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_md)

    # Print summary to console
    logger.info("")
    logger.info("=== Evaluation Complete ===")
    logger.info("  QA pairs evaluated: %d", len(all_results))
    logger.info("  Modes: %s", ", ".join(modes))
    logger.info("")
    logger.info("  %-8s  Retrieval  Answer", "Mode")
    for mode in modes:
        s = agg["per_mode"].get(mode, {})
        logger.info(
            "  %-8s  %5.2f/10   %5.2f/10",
            mode,
            s.get("mean_retrieval_score", 0),
            s.get("mean_answer_score", 0),
        )
    logger.info("")
    logger.info("  Results: %s", json_path)
    logger.info("  Report:  %s", report_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark tog vs rag vs none on multi-hop QA pairs."
    )
    parser.add_argument(
        "--validated",
        type=Path,
        default=VALIDATED_QA_PATH,
        help="Path to eval/generated/qa_pairs_validated.json",
    )
    parser.add_argument(
        "--api",
        default=API_BASE_URL,
        help="GraphRAG server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=EVAL_MODES,
        choices=["tog", "tog_r", "graphrag", "rag", "none"],
        help="Query modes to evaluate (default: tog rag none)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="top_k passed to all query modes (default: 5)",
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.validated, args.api, args.modes, args.top_k))


if __name__ == "__main__":
    main()
