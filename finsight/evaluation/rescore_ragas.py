"""
rescore_ragas.py
Re-compute RAGAS metrics on previously saved Q&A results.

Run this SEPARATELY from run_evaluation.py so the vLLM server has
time to recover before being used as the RAGAS judge.

If OPENAI_API_KEY is set (in .env or environment), uses gpt-4o-mini as the
judge LLM — much more stable than the local vLLM for 80 sequential calls.
Falls back to local vLLM if no API key is found.

Also back-fills numerical_accuracy and category_retrieval metrics from
per-question results so the comparison table is complete.

Usage:
    python evaluation/rescore_ragas.py
    python evaluation/rescore_ragas.py --input evaluation/results/eval_results.json
    python evaluation/rescore_ragas.py --input evaluation/results/eval_results.json --output evaluation/results/eval_results_rescored.json
    python evaluation/rescore_ragas.py --skip-ragas   # back-fill only numerical/category metrics
"""
import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from evaluation.metrics import compute_numeric_match

logger = get_logger(__name__)


def _load_env():
    """Load .env file if present."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def compute_ragas_metrics_stable(results: list, cfg: dict) -> dict:
    """
    Compute RAGAS metrics with max_workers=1 (sequential) to avoid crashing
    the local vLLM. Uses OpenAI API if OPENAI_API_KEY is available.
    """
    try:
        from ragas import evaluate, RunConfig
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_recall": 0.0, "context_precision": 0.0}

    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        logger.info("Using OpenAI API (gpt-4o-mini) as RAGAS judge.")
        judge_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-4o-mini",
                api_key=openai_key,
                temperature=0.0,
                max_tokens=2048,  # Increased from 512 to avoid LLMDidNotFinishException
                timeout=60,
            )
        )
        ragas_run_cfg = RunConfig(timeout=60, max_workers=4, max_retries=3, max_wait=30)
    else:
        gen_cfg = cfg["generation"]
        logger.info(f"No OPENAI_API_KEY found — using local vLLM ({gen_cfg['model']}) sequentially.")
        logger.warning("Add OPENAI_API_KEY to .env for faster, more reliable RAGAS scoring.")
        judge_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=gen_cfg["model"],
                base_url=gen_cfg.get("base_url", "http://localhost:8000/v1"),
                api_key=gen_cfg.get("api_key", "dummy"),
                temperature=0.0,
                max_tokens=2048,  # Increased from 512 to avoid LLMDidNotFinishException
                timeout=gen_cfg.get("timeout_seconds", 120),
            )
        )
        # max_workers=1: send one judge request at a time to avoid crashing vLLM
        ragas_run_cfg = RunConfig(
            timeout=gen_cfg.get("timeout_seconds", 120),
            max_workers=1,
            max_retries=2,
            max_wait=30,
        )

    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=cfg["embeddings"]["model"])
    )

    valid_results = [r for r in results if not r.get("error")]
    if not valid_results:
        logger.warning("No valid results to score.")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_recall": 0.0, "context_precision": 0.0}

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in valid_results],
        "answer": [r["answer"] for r in valid_results],
        "contexts": [r["contexts"] for r in valid_results],
        "ground_truth": [r["ground_truth"] for r in valid_results],
    })

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    try:
        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=ragas_run_cfg,
        )
        import numpy as np
        def _safe_mean(val):
            if isinstance(val, (list, np.ndarray)):
                clean = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
                return float(np.mean(clean)) if clean else 0.0
            return float(val)

        scores = {
            "faithfulness": round(_safe_mean(eval_result["faithfulness"]), 4),
            "answer_relevancy": round(_safe_mean(eval_result["answer_relevancy"]), 4),
            "context_recall": round(_safe_mean(eval_result["context_recall"]), 4),
            "context_precision": round(_safe_mean(eval_result["context_precision"]), 4),
        }
        logger.info(f"RAGAS scores: {scores}")
        return scores
    except Exception as e:
        logger.error(f"RAGAS scoring failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_recall": 0.0, "context_precision": 0.0}


def backfill_numerical_accuracy(per_question: list) -> tuple:
    """
    Back-fill numerical_match into per-question records (if absent) and
    compute aggregate numerical_accuracy + category_retrieval breakdown.

    Returns (updated_per_question, numerical_accuracy, category_retrieval).
    """
    CATEGORIES = ["factual_retrieval", "temporal_reasoning",
                  "multi_hop_reasoning", "comparative_analysis"]

    cat_data = defaultdict(lambda: {"num_match": [], "has_context": []})

    for r in per_question:
        if r.get("error"):
            continue
        # Back-fill numerical_match if missing
        if "numerical_match" not in r:
            r["numerical_match"] = compute_numeric_match(
                r.get("answer", ""), r.get("ground_truth", "")
            )
        cat = r.get("category", "unknown")
        cat_data[cat]["num_match"].append(int(r["numerical_match"]))
        cat_data[cat]["has_context"].append(int(len(r.get("contexts", [])) > 0))

    valid = [r for r in per_question if not r.get("error")]
    n_valid = max(len(valid), 1)
    numerical_accuracy = round(
        sum(1 for r in valid if r.get("numerical_match")) / n_valid, 4
    )

    category_retrieval = {}
    for cat in CATEGORIES:
        data = cat_data.get(cat, {"num_match": [], "has_context": []})
        n = len(data["num_match"])
        category_retrieval[cat] = {
            "n": n,
            "numerical_accuracy": round(sum(data["num_match"]) / n, 4) if n else 0.0,
            "context_coverage": round(sum(data["has_context"]) / n, 4) if n else 0.0,
        }

    return per_question, numerical_accuracy, category_retrieval


def main():
    _load_env()

    parser = argparse.ArgumentParser(description="Re-score RAGAS metrics on saved results")
    parser.add_argument(
        "--input",
        default="evaluation/results/eval_results.json",
        help="Path to the saved Q&A results JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save scored results (default: overwrites input file)",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS re-scoring; only back-fill numerical_accuracy and category_retrieval.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset of variants to re-score (default: all in file)",
    )
    args = parser.parse_args()

    cfg = load_config()
    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / (args.output or args.input)

    if not input_path.exists():
        logger.error(f"Results file not found: {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        all_results = json.load(f)

    variants = args.variants or list(all_results.keys())

    for mode in variants:
        if mode not in all_results:
            logger.warning(f"Variant {mode} not found in results file; skipping.")
            continue

        logger.info(f"\nProcessing variant: {mode} ...")
        per_question = all_results[mode]["per_question"]

        # ── Back-fill numerical accuracy (always) ────────────────────────
        per_question, numerical_accuracy, category_retrieval = backfill_numerical_accuracy(
            per_question
        )
        all_results[mode]["per_question"] = per_question

        valid = [r for r in per_question if not r.get("error")]
        latencies = [r["latency_seconds"] for r in valid]
        avg_latency = round(sum(latencies) / max(len(latencies), 1), 4)
        avg_retrieval = round(
            sum(r.get("retrieval_latency_ms", 0) for r in valid) / max(len(valid), 1), 2
        )
        avg_reranking = round(
            sum(r.get("reranking_latency_ms", 0) for r in valid) / max(len(valid), 1), 2
        )
        avg_generation = round(
            sum(r.get("generation_latency_ms", 0) for r in valid) / max(len(valid), 1), 2
        )

        # Preserve existing RAGAS scores
        existing_agg = all_results[mode].get("aggregate", {})
        all_results[mode]["aggregate"] = {
            **existing_agg,
            "numerical_accuracy": numerical_accuracy,
            "avg_latency_seconds": avg_latency,
            "avg_retrieval_ms": avg_retrieval,
            "avg_reranking_ms": avg_reranking,
            "avg_generation_ms": avg_generation,
            "category_retrieval": category_retrieval,
        }

        logger.info(
            f"  numerical_accuracy={numerical_accuracy:.4f}, "
            f"n_valid={len(valid)}/{len(per_question)}"
        )
        for cat, stats in category_retrieval.items():
            if stats["n"] > 0:
                logger.info(
                    f"    {cat}: n={stats['n']}, "
                    f"num_acc={stats['numerical_accuracy']:.2%}, "
                    f"ctx_cov={stats['context_coverage']:.2%}"
                )

        if args.skip_ragas:
            continue

        # ── Re-compute RAGAS scores ──────────────────────────────────────
        logger.info(f"  Re-scoring RAGAS for {mode} ({len(valid)} valid results)...")
        ragas_scores = compute_ragas_metrics_stable(per_question, cfg)
        all_results[mode]["aggregate"].update(ragas_scores)
        logger.info(f"  RAGAS: {json.dumps(ragas_scores)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_path}")

    # ── Print summary table ──────────────────────────────────────────────
    VARIANT_ORDER = [
        "v0_llm_only", "v1_baseline", "v2_advanced_a", "v3_advanced_b",
        "v4_advanced_c", "v5_advanced_d", "v6_advanced_e",
    ]
    modes = [v for v in VARIANT_ORDER if v in all_results] + \
            [v for v in all_results if v not in VARIANT_ORDER]

    metrics_keys = [
        "faithfulness", "answer_relevancy", "context_recall", "context_precision",
        "numerical_accuracy", "avg_latency_seconds",
        "avg_retrieval_ms", "avg_reranking_ms", "avg_generation_ms",
    ]
    col_w = 14
    header = f"{'Metric':<28}" + "".join(f" | {m:>{col_w}}" for m in modes)
    sep = "=" * (28 + (col_w + 3) * len(modes))
    print(f"\n{sep}")
    print("  RAGAS EVALUATION RESULTS — FinSight (Outline §6.2)")
    print(sep)
    print(header)
    print("-" * len(sep))
    for k in metrics_keys:
        row = f"{k:<28}"
        for m in modes:
            val = all_results[m]["aggregate"].get(k)
            if val is None:
                row += f" | {'N/A':>{col_w}}"
            elif k.endswith("_ms"):
                row += f" | {val:>{col_w}.0f}"
            else:
                row += f" | {val:>{col_w}.4f}"
        print(row)
    print("-" * len(sep))
    row = f"{'n_questions':<28}"
    for m in modes:
        row += f" | {len(all_results[m]['per_question']):>{col_w}}"
    print(row)
    print(sep)

    # ── Per-category numerical accuracy ──────────────────────────────────
    CATEGORIES = ["factual_retrieval", "temporal_reasoning",
                  "multi_hop_reasoning", "comparative_analysis"]
    print(f"\n{sep}")
    print("  NUMERICAL ACCURACY BY QUERY CATEGORY (Outline §10.3)")
    print(sep)
    cat_hdr = f"  {'Category':<24}" + "".join(f" | {m:>{col_w}}" for m in modes)
    print(cat_hdr)
    print("-" * len(sep))
    for cat in CATEGORIES:
        row = f"  {cat:<24}"
        for m in modes:
            acc = all_results[m]["aggregate"].get(
                "category_retrieval", {}
            ).get(cat, {}).get("numerical_accuracy", 0.0)
            row += f" | {acc:>{col_w}.1%}"
        print(row)
    print(sep)


if __name__ == "__main__":
    main()
