"""
ablation_study.py
Run ablation experiments to isolate the impact of each pipeline component.

Experiments:
1. Dense only (baseline)
2. BM25 only
3. Hybrid (Dense + BM25 with RRF) - no rerank
4. Hybrid + Rerank (full advanced pipeline)

Usage:
    python evaluation/ablation_study.py
    python evaluation/ablation_study.py --limit 5
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb_compat  # noqa: F401  — must precede any chromadb/pipeline import

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_eval_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def run_dense_only(question: str, cfg: dict, top_k: int = 5) -> dict:
    """Dense retrieval only (V1 baseline)."""
    from src.retrieval.dense_retriever import DenseRetriever
    from src.generation.generator import Generator

    retriever = DenseRetriever(cfg)
    generator = Generator(cfg)

    t0 = time.time()
    chunks = retriever.retrieve(question, top_k=top_k)
    retrieval_time = time.time() - t0

    t1 = time.time()
    result = generator.generate(question, chunks)
    generation_time = time.time() - t1

    return {
        "answer": result.get("answer", ""),
        "contexts": [c.get("text", "") for c in chunks],
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": retrieval_time + generation_time,
        "method": "dense_only"
    }


def run_sparse_only(question: str, cfg: dict, top_k: int = 5) -> dict:
    """BM25 retrieval only."""
    from src.retrieval.sparse_retriever import SparseRetriever
    from src.generation.generator import Generator

    retriever = SparseRetriever(cfg)
    generator = Generator(cfg)

    t0 = time.time()
    chunks = retriever.retrieve(question, top_k=top_k)
    retrieval_time = time.time() - t0

    t1 = time.time()
    result = generator.generate(question, chunks)
    generation_time = time.time() - t1

    return {
        "answer": result.get("answer", ""),
        "contexts": [c.get("text", "") for c in chunks],
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": retrieval_time + generation_time,
        "method": "sparse_only"
    }


def run_hybrid_no_rerank(question: str, cfg: dict, top_k: int = 10) -> dict:
    """Hybrid retrieval (Dense + BM25 with RRF) without reranking."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.generation.generator import Generator

    retriever = HybridRetriever(cfg)
    generator = Generator(cfg)

    t0 = time.time()
    # Get fused results without reranking
    chunks = retriever.retrieve(question, skip_rerank=True)[:top_k]
    retrieval_time = time.time() - t0

    t1 = time.time()
    result = generator.generate(question, chunks)
    generation_time = time.time() - t1

    return {
        "answer": result.get("answer", ""),
        "contexts": [c.get("text", "") for c in chunks],
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": retrieval_time + generation_time,
        "method": "hybrid_no_rerank"
    }


def run_hybrid_with_rerank(question: str, cfg: dict) -> dict:
    """Full advanced pipeline: Hybrid + Rerank."""
    from src.pipeline.advanced_b import AdvancedBPipeline

    pipeline = AdvancedBPipeline(cfg)

    t0 = time.time()
    result = pipeline.ask(question)
    total_time = time.time() - t0

    return {
        "answer": result.get("answer", ""),
        "contexts": [c.get("text", "") for c in result.get("retrieved_chunks", [])],
        "retrieval_time": 0,  # Combined in pipeline
        "generation_time": 0,
        "total_time": total_time,
        "method": "hybrid_with_rerank"
    }


def main():
    parser = argparse.ArgumentParser(description="FinSight Ablation Study")
    parser.add_argument("--dataset", default="evaluation/eval_dataset.json")
    parser.add_argument("--output", default="evaluation/results/ablation_results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--methods", nargs="+",
                       default=["dense_only", "sparse_only", "hybrid_no_rerank", "hybrid_with_rerank"],
                       choices=["dense_only", "sparse_only", "hybrid_no_rerank", "hybrid_with_rerank"])
    args = parser.parse_args()

    cfg = load_config()
    dataset_path = PROJECT_ROOT / args.dataset
    dataset = load_eval_dataset(str(dataset_path))

    if args.limit:
        dataset = dataset[:args.limit]

    logger.info(f"Running ablation study on {len(dataset)} questions")
    logger.info(f"Methods: {args.methods}")

    method_runners = {
        "dense_only": run_dense_only,
        "sparse_only": run_sparse_only,
        "hybrid_no_rerank": run_hybrid_no_rerank,
        "hybrid_with_rerank": run_hybrid_with_rerank,
    }

    all_results = {}

    for method in args.methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {method}")
        logger.info(f"{'='*60}")

        runner = method_runners[method]
        method_results = []

        for i, item in enumerate(dataset, 1):
            logger.info(f"  [{i}/{len(dataset)}] {item['id']}: {item['question'][:50]}...")

            try:
                result = runner(item["question"], cfg)
                method_results.append({
                    "id": item["id"],
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "category": item.get("category", ""),
                    **result,
                    "error": None
                })
                logger.info(f"    -> {result['total_time']:.1f}s")
            except Exception as e:
                logger.error(f"    -> ERROR: {e}")
                method_results.append({
                    "id": item["id"],
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "category": item.get("category", ""),
                    "answer": f"ERROR: {e}",
                    "contexts": [],
                    "total_time": 0,
                    "method": method,
                    "error": str(e)
                })

            time.sleep(0.3)

        # Compute aggregate stats
        valid_results = [r for r in method_results if not r.get("error")]
        avg_latency = sum(r["total_time"] for r in valid_results) / max(len(valid_results), 1)
        error_rate = (len(method_results) - len(valid_results)) / len(method_results)

        all_results[method] = {
            "per_question": method_results,
            "aggregate": {
                "avg_latency_seconds": round(avg_latency, 4),
                "error_rate": round(error_rate, 4),
                "n_questions": len(method_results),
                "n_errors": len(method_results) - len(valid_results)
            }
        }

        logger.info(f"  {method} complete: avg_latency={avg_latency:.2f}s, errors={len(method_results) - len(valid_results)}")

    # Save results
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {output_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"{'Method':<25} | {'Avg Latency':>12} | {'Error Rate':>12}")
    print("-" * 70)
    for method, data in all_results.items():
        agg = data["aggregate"]
        print(f"{method:<25} | {agg['avg_latency_seconds']:>10.2f}s | {agg['error_rate']:>10.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
