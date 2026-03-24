"""
run_evaluation.py
RAGAS-based evaluation pipeline for FinSight.

Runs every question from eval_dataset.json through all 7 pipeline variants
(V0–V6), computes RAGAS metrics (faithfulness, answer_relevancy,
context_recall, context_precision), and saves a side-by-side comparison.

Aligned with outline §6.2 (quantitative evaluation) and §6.3 (category-based).

Uses the same qwen2.5-14b via vLLM (http://localhost:8000/v1) as the RAGAS
judge LLM.

Usage:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --limit 5
    python evaluation/run_evaluation.py --variants v1_baseline v3_advanced_b
    python evaluation/run_evaluation.py --skip-ragas
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb_compat  # noqa: F401  — must precede any chromadb/pipeline import

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Variant registry (matches src/pipeline/__init__.py) ──────────────────────

VARIANTS = {
    "v0_llm_only":    ("src.pipeline.llm_only",    "LLMOnlyPipeline"),
    "v1_baseline":    ("src.pipeline.baseline",     "BaselinePipeline"),
    "v2_advanced_a":  ("src.pipeline.advanced_a",   "AdvancedAPipeline"),
    "v3_advanced_b":  ("src.pipeline.advanced_b",   "AdvancedBPipeline"),
    "v4_advanced_c":  ("src.pipeline.advanced_c",   "AdvancedCPipeline"),
    "v5_advanced_d":  ("src.pipeline.advanced_d",   "AdvancedDPipeline"),
    "v6_advanced_e":  ("src.pipeline.advanced_e",   "AdvancedEPipeline"),
}

VARIANT_ORDER = list(VARIANTS.keys())


def load_eval_dataset(path: str) -> list:
    """Load the evaluation dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} questions from {path}")
    return dataset


def build_pipeline(variant_key: str, cfg: dict):
    """Dynamically import and instantiate a pipeline variant."""
    module_path, class_name = VARIANTS[variant_key]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(cfg)


def run_questions(pipeline, dataset: list, variant_key: str) -> list:
    """Run all questions through a pipeline, collecting results with category."""
    results = []
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        category = item.get("category", "")
        source_doc = item.get("source_doc", "")
        logger.info(f"  [{i}/{len(dataset)}] {item['id']}: {question[:60]}...")

        t0 = time.time()
        try:
            result = pipeline.ask(question)
            latency_s = time.time() - t0

            contexts = [
                c.get("text", "") for c in result.get("retrieved_chunks", [])
            ]

            results.append({
                "id": item["id"],
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "ground_truth": ground_truth,
                "category": category,
                "source_doc": source_doc,
                "variant": variant_key,
                "latency_seconds": round(latency_s, 3),
                "retrieval_latency_ms": result.get("retrieval_latency_ms", 0),
                "reranking_latency_ms": result.get("reranking_latency_ms", 0),
                "generation_latency_ms": result.get("generation_latency_ms", 0),
                "insufficient_evidence": result.get("insufficient_evidence", False),
                "error": result.get("error"),
            })
            logger.info(f"    -> {latency_s:.1f}s | answered")
        except Exception as e:
            latency_s = time.time() - t0
            logger.error(f"    -> ERROR: {e}")
            results.append({
                "id": item["id"],
                "question": question,
                "answer": f"ERROR: {e}",
                "contexts": [],
                "ground_truth": ground_truth,
                "category": category,
                "source_doc": source_doc,
                "variant": variant_key,
                "latency_seconds": round(latency_s, 3),
                "retrieval_latency_ms": 0,
                "reranking_latency_ms": 0,
                "generation_latency_ms": 0,
                "insufficient_evidence": False,
                "error": str(e),
            })

        time.sleep(0.3)

    return results


def compute_ragas_metrics(results: list, cfg: dict) -> dict:
    """Compute RAGAS metrics using the local vLLM endpoint as the judge LLM."""
    try:
        from ragas import evaluate
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
        logger.error(
            f"Missing dependency for RAGAS evaluation: {e}. "
            f"Install with: pip install ragas langchain langchain-openai langchain-huggingface datasets"
        )
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
        }

    from ragas import RunConfig

    gen_cfg = cfg["generation"]
    base_url = gen_cfg.get("base_url", "http://localhost:8000/v1")
    api_key = gen_cfg.get("api_key", "dummy")
    model = gen_cfg.get("model", "qwen2.5-14b")
    timeout_s = gen_cfg.get("timeout_seconds", 120)

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            max_tokens=1024,  # 512 causes LLMDidNotFinishException on RAGAS judge prompts
            timeout=timeout_s,
        )
    )

    # Use local sentence-transformers for embeddings (vLLM doesn't serve /embeddings)
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=cfg["embeddings"]["model"],
        )
    )

    # Limit concurrency to 2 so the local vLLM isn't flooded with parallel judge requests.
    ragas_run_cfg = RunConfig(
        timeout=timeout_s,
        max_workers=2,
        max_retries=3,
        max_wait=60,
    )

    valid_results = [r for r in results if not r.get("error")]
    if not valid_results:
        logger.warning("No valid results to evaluate with RAGAS")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
        }

    eval_data = {
        "question": [r["question"] for r in valid_results],
        "answer": [r["answer"] for r in valid_results],
        "contexts": [r["contexts"] for r in valid_results],
        "ground_truth": [r["ground_truth"] for r in valid_results],
    }
    dataset = Dataset.from_dict(eval_data)

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

        # ── Per-question RAGAS scores for category breakdown (§6.3) ───────
        per_question_ragas = []
        for idx, r in enumerate(valid_results):
            pq = {"id": r["id"], "category": r.get("category", "")}
            for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
                val = eval_result[metric_name]
                if isinstance(val, (list, np.ndarray)) and idx < len(val):
                    pq[metric_name] = float(val[idx]) if val[idx] is not None else None
                else:
                    pq[metric_name] = None
            per_question_ragas.append(pq)

        scores["per_question_ragas"] = per_question_ragas

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        logger.info("Falling back to manual metric computation...")
        scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
        }

    return scores


def compute_category_ragas(per_question_ragas: list) -> dict:
    """
    Break down RAGAS scores by query category (outline §6.3).

    Returns dict: category -> { metric -> mean_score }
    """
    import statistics as stats

    cat_scores = defaultdict(lambda: defaultdict(list))
    for pq in per_question_ragas:
        cat = pq.get("category", "unknown")
        for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            val = pq.get(metric)
            if val is not None:
                cat_scores[cat][metric].append(val)

    result = {}
    for cat, metrics in cat_scores.items():
        result[cat] = {
            metric: round(stats.mean(vals), 4) if vals else 0.0
            for metric, vals in metrics.items()
        }
    return result


def print_comparison_table(all_results: dict):
    """Print a clean side-by-side comparison table (outline §10.3)."""
    variants = [v for v in VARIANT_ORDER if v in all_results]
    metrics_keys = [
        "faithfulness", "answer_relevancy", "context_recall",
        "context_precision", "avg_latency_seconds",
        "avg_retrieval_ms", "avg_reranking_ms", "avg_generation_ms",
    ]

    col_w = 14
    header = f"{'Metric':<28}"
    for v in variants:
        header += f" | {v:>{col_w}}"
    sep = "=" * (28 + (col_w + 3) * len(variants))

    print(f"\n{sep}")
    print("  RAGAS EVALUATION RESULTS — FinSight (Outline §6.2)")
    print(sep)
    print(header)
    print("-" * len(sep))

    for metric in metrics_keys:
        row = f"{metric:<28}"
        for v in variants:
            val = all_results[v]["aggregate"].get(metric, 0.0)
            if val is None:
                row += f" | {'N/A':>{col_w}}"
            elif metric.endswith("_ms"):
                row += f" | {val:>{col_w}.0f}"
            else:
                row += f" | {val:>{col_w}.4f}"
        print(row)

    print("-" * len(sep))
    row = f"{'n_questions':<28}"
    for v in variants:
        n = len(all_results[v]["per_question"])
        row += f" | {n:>{col_w}}"
    print(row)
    print(sep)

    # ── Category breakdown (§6.3) ─────────────────────────────────────────
    for v in variants:
        cat_ragas = all_results[v]["aggregate"].get("category_ragas")
        if cat_ragas:
            print(f"\n  Category RAGAS — {v}")
            cat_header = f"  {'Category':<24}" + "".join(
                f"{'Faith':>8}{'Relev':>8}{'C.Rec':>8}{'C.Prec':>8}"
            )
            print(cat_header)
            print("  " + "-" * 56)
            for cat in ["factual_retrieval", "temporal_reasoning",
                        "multi_hop_reasoning", "comparative_analysis"]:
                scores = cat_ragas.get(cat, {})
                print(
                    f"  {cat:<24}"
                    f"{scores.get('faithfulness', 0):>8.3f}"
                    f"{scores.get('answer_relevancy', 0):>8.3f}"
                    f"{scores.get('context_recall', 0):>8.3f}"
                    f"{scores.get('context_precision', 0):>8.3f}"
                )


def main():
    parser = argparse.ArgumentParser(description="FinSight RAGAS evaluation pipeline (V0–V6)")
    parser.add_argument(
        "--dataset",
        default="evaluation/eval_dataset.json",
        help="Path to the evaluation dataset JSON file",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/eval_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N questions (for quick testing)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
        help="Which variants to evaluate (default: all seven)",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS scoring. Saves Q&A results only — run rescore_ragas.py separately.",
    )
    args = parser.parse_args()

    cfg = load_config()

    dataset_path = PROJECT_ROOT / args.dataset
    if not dataset_path.exists():
        logger.error(f"Evaluation dataset not found: {dataset_path}")
        sys.exit(1)

    dataset = load_eval_dataset(str(dataset_path))
    if args.limit:
        dataset = dataset[: args.limit]
        logger.info(f"Limited to {len(dataset)} questions")

    # Load existing results to merge with (instead of overwriting)
    output_path = PROJECT_ROOT / args.output
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        logger.info(f"Loaded existing results with {len(all_results)} variants")
    else:
        all_results = {}

    for variant_key in args.variants:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating variant: {variant_key}")
        logger.info(f"{'=' * 60}")

        pipeline = build_pipeline(variant_key, cfg)
        per_question = run_questions(pipeline, dataset, variant_key)

        valid = [r for r in per_question if not r.get("error")]
        n_valid = max(len(valid), 1)
        avg_latency = sum(r["latency_seconds"] for r in valid) / n_valid
        avg_retrieval = sum(r.get("retrieval_latency_ms", 0) for r in valid) / n_valid
        avg_reranking = sum(r.get("reranking_latency_ms", 0) for r in valid) / n_valid
        avg_generation = sum(r.get("generation_latency_ms", 0) for r in valid) / n_valid

        # Save Q&A results immediately so they are not lost if RAGAS crashes.
        all_results[variant_key] = {
            "per_question": per_question,
            "aggregate": {
                "avg_latency_seconds": round(avg_latency, 4),
                "avg_retrieval_ms": round(avg_retrieval, 2),
                "avg_reranking_ms": round(avg_reranking, 2),
                "avg_generation_ms": round(avg_generation, 2),
            },
        }
        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Q&A results saved to {output_path}")

        if args.skip_ragas:
            logger.info(f"Skipping RAGAS scoring for {variant_key} (--skip-ragas).")
            continue

        logger.info(f"Computing RAGAS metrics for {variant_key}...")
        ragas_scores = compute_ragas_metrics(per_question, cfg)

        # Category breakdown from per-question RAGAS
        per_q_ragas = ragas_scores.pop("per_question_ragas", [])
        category_ragas = compute_category_ragas(per_q_ragas) if per_q_ragas else {}

        all_results[variant_key]["aggregate"].update(ragas_scores)
        all_results[variant_key]["aggregate"]["category_ragas"] = category_ragas
        all_results[variant_key]["per_question_ragas"] = per_q_ragas

        logger.info(
            f"Variant {variant_key} complete: "
            f"{json.dumps({k: v for k, v in all_results[variant_key]['aggregate'].items() if k != 'category_ragas'}, indent=2)}"
        )

    # Final save
    output_path = PROJECT_ROOT / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_path}")

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
