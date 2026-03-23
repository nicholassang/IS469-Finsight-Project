"""
run_eval.py
Runs the complete evaluation benchmark across all three pipeline variants.
Saves structured JSON results to evaluation/results/<variant>.json

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --variants v1_baseline v2_advanced_a
    python evaluation/run_eval.py --benchmark evaluation/benchmark.csv --limit 10
    python evaluation/run_eval.py --skip-guardrail-tests
"""

import sys
import csv
import json
import time
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb_compat  # noqa: F401  — must precede any chromadb/pipeline import

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

VARIANTS = {
    "v0_llm_only": ("src.pipeline.llm_only", "LLMOnlyPipeline"),
    "v1_baseline": ("src.pipeline.baseline", "BaselinePipeline"),
    "v2_advanced_a": ("src.pipeline.advanced_a", "AdvancedAPipeline"),
    "v3_advanced_b": ("src.pipeline.advanced_b", "AdvancedBPipeline"),
    "v4_advanced_c": ("src.pipeline.advanced_c", "AdvancedCPipeline"),
    "v5_advanced_d": ("src.pipeline.advanced_d", "AdvancedDPipeline"),
    "v6_advanced_e": ("src.pipeline.advanced_e", "AdvancedEPipeline"),
}

# Question types to exclude from automated metric scoring (handled separately)
GUARDRAIL_TYPES = {"investment_advice", "out_of_scope"}


def load_benchmark(path: str, limit: int = None, skip_guardrail: bool = False) -> list:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if skip_guardrail and row.get("question_type", "") in GUARDRAIL_TYPES:
                continue
            questions.append(row)
            if limit and len(questions) >= limit:
                break
    return questions


def load_pipeline(variant_key: str):
    module_path, class_name = VARIANTS[variant_key]
    mod = __import__(module_path, fromlist=[class_name])
    cls = getattr(mod, class_name)
    return cls()


def run_single_question(pipeline, question_row: dict) -> dict:
    """Run one question through one pipeline and return structured result."""
    q = question_row["question"]
    t0 = time.time()

    try:
        result = pipeline.ask(q)
        elapsed_ms = (time.time() - t0) * 1000

        return {
            "id": question_row["id"],
            "question": q,
            "expected_answer_summary": question_row.get("expected_answer_summary", ""),
            "target_doc_type": question_row.get("target_doc_type", ""),
            "target_fiscal_period": question_row.get("target_fiscal_period", ""),
            "question_type": question_row.get("question_type", ""),
            "difficulty": question_row.get("difficulty", ""),
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "retrieved_chunk_ids": [
                c.get("metadata", {}).get("chunk_id", "")
                for c in result.get("retrieved_chunks", [])
            ],
            "n_citations": len(result.get("citations", [])),
            "n_chunks_retrieved": len(result.get("retrieved_chunks", [])),
            "latency_ms": result.get("latency_ms", elapsed_ms),
            "generation_latency_ms": result.get("generation_latency_ms", 0),
            "variant": result.get("variant", pipeline.VARIANT_NAME),
            "model": result.get("model", ""),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "insufficient_evidence": result.get("insufficient_evidence", False),
            "top_rerank_score": result.get("top_rerank_score", None),
            "error": result.get("error"),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        logger.error(f"Question {question_row['id']} failed: {e}")
        return {
            "id": question_row["id"],
            "question": q,
            "expected_answer_summary": question_row.get("expected_answer_summary", ""),
            "question_type": question_row.get("question_type", ""),
            "difficulty": question_row.get("difficulty", ""),
            "answer": f"ERROR: {str(e)}",
            "citations": [],
            "latency_ms": elapsed_ms,
            "variant": variant_key,
            "model": "",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat(),
        }


def run_variant_eval(
    variant_key: str,
    questions: list,
    output_dir: Path,
    max_workers: int = 1,
) -> dict:
    """Run all questions for one variant, save results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating variant: {variant_key}")
    logger.info(f"Questions: {len(questions)}")
    logger.info(f"{'='*60}")

    pipeline = load_pipeline(variant_key)
    results = []
    t0 = time.time()

    if max_workers > 1:
        # Parallel evaluation (use carefully — OpenAI rate limits)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_question, pipeline, q): q
                for q in questions
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                logger.info(
                    f"  [{result['id']}] {result['latency_ms']:.0f}ms | "
                    f"tokens={result.get('total_tokens', 0)} | "
                    f"{'INSUFFICIENT' if result.get('insufficient_evidence') else 'OK'}"
                )
    else:
        # Sequential (safer for rate limits and debugging)
        for i, q in enumerate(questions, 1):
            logger.info(f"  [{i}/{len(questions)}] {q['id']}: {q['question'][:60]}...")
            result = run_single_question(pipeline, q)
            results.append(result)
            logger.info(
                f"    → {result['latency_ms']:.0f}ms | "
                f"tokens={result.get('total_tokens', 0)} | "
                f"{'INSUFFICIENT' if result.get('insufficient_evidence') else 'answered'}"
            )

            # Small delay to avoid rate limiting
            time.sleep(0.5)

    elapsed = time.time() - t0

    # Summary stats
    answered = [r for r in results if not r.get("insufficient_evidence") and not r.get("error")]
    insufficient = [r for r in results if r.get("insufficient_evidence")]
    errors = [r for r in results if r.get("error")]
    avg_latency = sum(r["latency_ms"] for r in results) / max(len(results), 1)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)

    summary = {
        "variant": variant_key,
        "n_questions": len(questions),
        "n_answered": len(answered),
        "n_insufficient_evidence": len(insufficient),
        "n_errors": len(errors),
        "avg_latency_ms": round(avg_latency, 2),
        "total_tokens": total_tokens,
        "elapsed_seconds": round(elapsed, 1),
        "evaluated_at": datetime.utcnow().isoformat(),
    }

    output = {"summary": summary, "results": results}

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{variant_key}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"\nVariant {variant_key} complete:")
    logger.info(f"  Answered: {len(answered)}/{len(questions)}")
    logger.info(f"  Insufficient evidence: {len(insufficient)}")
    logger.info(f"  Errors: {len(errors)}")
    logger.info(f"  Avg latency: {avg_latency:.0f}ms")
    logger.info(f"  Total tokens used: {total_tokens:,}")
    logger.info(f"  Results saved: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="FinSight evaluation runner")
    parser.add_argument("--benchmark", default="evaluation/benchmark.csv")
    parser.add_argument("--output", default="evaluation/results")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()),
                        help="Which variants to evaluate (default: all seven)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N questions (for testing)")
    parser.add_argument("--skip-guardrail-tests", action="store_true",
                        help="Skip investment_advice and out_of_scope questions")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1, sequential)")
    args = parser.parse_args()

    benchmark_path = PROJECT_ROOT / args.benchmark
    if not benchmark_path.exists():
        logger.error(f"Benchmark file not found: {benchmark_path}")
        sys.exit(1)

    questions = load_benchmark(
        str(benchmark_path),
        limit=args.limit,
        skip_guardrail=args.skip_guardrail_tests,
    )
    logger.info(f"Loaded {len(questions)} questions from benchmark")

    output_dir = PROJECT_ROOT / args.output
    all_summaries = []

    for variant_key in args.variants:
        summary = run_variant_eval(
            variant_key,
            questions,
            output_dir,
            max_workers=args.workers,
        )
        all_summaries.append(summary)

    # Save combined summary
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation complete. Summary saved to {summary_path}")
    logger.info(f"Next step: python evaluation/metrics.py --results {args.output}")


if __name__ == "__main__":
    main()
