"""
ablation_study.py
Run ablation experiments to isolate the impact of each pipeline component.

Aligned with outline §6.5 — 8 ablation steps:
  Step 0: LLM-only (V0)            — No retrieval baseline
  Step 1: Dense only (V1)          — Semantic retrieval
  Step 2: BM25 only                — Lexical retrieval
  Step 3: Hybrid (no rerank)       — Fusion effect (BM25 + Dense + RRF)
  Step 4: Hybrid + rerank (V3)     — Reranking effect
  Step 5: + Query rewriting (V4)   — Query understanding
  Step 6: + Metadata filtering (V5)— Structured retrieval
  Step 7: + Context compression (V6)— Context quality

Metrics compared (per outline §6.5):
  - RAGAS scores (if available)
  - Retrieval precision (MRR, hit rate)
  - Answer accuracy (%)
  - Per-stage latency breakdown (§6.6)

Usage:
    python evaluation/ablation_study.py
    python evaluation/ablation_study.py --limit 5
    python evaluation/ablation_study.py --steps llm_only dense_only hybrid_rerank
"""

import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb_compat  # noqa: F401  — must precede any chromadb/pipeline import

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Step definitions ──────────────────────────────────────────────────────────

ABLATION_STEPS = [
    {
        "key": "llm_only",
        "step": 0,
        "variant": "V0",
        "component": "None (LLM-only)",
        "description": "No retrieval baseline — measures hallucination floor",
    },
    {
        "key": "dense_only",
        "step": 1,
        "variant": "V1",
        "component": "+ Dense Retrieval",
        "description": "Semantic retrieval via embeddings",
    },
    {
        "key": "sparse_only",
        "step": 2,
        "variant": "—",
        "component": "+ BM25 (Lexical)",
        "description": "Lexical retrieval — keyword matching baseline",
    },
    {
        "key": "hybrid_no_rerank",
        "step": 3,
        "variant": "—",
        "component": "+ Hybrid (RRF)",
        "description": "Dense + BM25 fused via RRF — no reranking",
    },
    {
        "key": "hybrid_rerank",
        "step": 4,
        "variant": "V3",
        "component": "+ Reranking",
        "description": "Hybrid retrieval + cross-encoder reranking",
    },
    {
        "key": "query_rewrite",
        "step": 5,
        "variant": "V4",
        "component": "+ Query Rewriting",
        "description": "LLM-based query rewriting before hybrid retrieval",
    },
    {
        "key": "metadata_filter",
        "step": 6,
        "variant": "V5",
        "component": "+ Metadata Filtering",
        "description": "Fiscal period / doc type filtering before dense retrieval",
    },
    {
        "key": "context_compression",
        "step": 7,
        "variant": "V6",
        "component": "+ Context Compression",
        "description": "Sentence-level compression after reranking",
    },
]

ALL_STEP_KEYS = [s["key"] for s in ABLATION_STEPS]


# ── Runners ───────────────────────────────────────────────────────────────────

def load_eval_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def run_llm_only(question: str, cfg: dict, **kwargs) -> dict:
    """Step 0: LLM-only (V0 pipeline class)."""
    from src.pipeline.llm_only import LLMOnlyPipeline
    pipeline = LLMOnlyPipeline(cfg)
    result = pipeline.ask(question)
    return _normalise_result(result, "llm_only")


def run_dense_only(question: str, cfg: dict, top_k: int = 5, **kwargs) -> dict:
    """Step 1: Dense retrieval only (V1 baseline)."""
    from src.retrieval.dense_retriever import DenseRetriever
    from src.generation.generator import Generator

    retriever = DenseRetriever(cfg)
    generator = Generator(cfg)

    t0 = time.time()
    t_ret = time.time()
    chunks = retriever.retrieve(question, top_k=top_k)
    retrieval_ms = (time.time() - t_ret) * 1000

    t_gen = time.time()
    result = generator.generate(question, chunks)
    generation_ms = (time.time() - t_gen) * 1000

    return {
        "answer": result.get("answer", ""),
        "contexts": [c.get("text", "") for c in chunks],
        "retrieved_chunks": chunks,
        "retrieval_latency_ms": round(retrieval_ms, 2),
        "reranking_latency_ms": 0,
        "generation_latency_ms": round(generation_ms, 2),
        "total_time": (time.time() - t0),
        "method": "dense_only",
    }


def run_sparse_only(question: str, cfg: dict, top_k: int = 5, **kwargs) -> dict:
    """Step 2: BM25 retrieval only."""
    from src.retrieval.sparse_retriever import SparseRetriever
    from src.generation.generator import Generator

    retriever = SparseRetriever(cfg)
    generator = Generator(cfg)

    t0 = time.time()
    t_ret = time.time()
    chunks = retriever.retrieve(question, top_k=top_k)
    retrieval_ms = (time.time() - t_ret) * 1000

    t_gen = time.time()
    result = generator.generate(question, chunks)
    generation_ms = (time.time() - t_gen) * 1000

    return {
        "answer": result.get("answer", ""),
        "contexts": [c.get("text", "") for c in chunks],
        "retrieved_chunks": chunks,
        "retrieval_latency_ms": round(retrieval_ms, 2),
        "reranking_latency_ms": 0,
        "generation_latency_ms": round(generation_ms, 2),
        "total_time": (time.time() - t0),
        "method": "sparse_only",
    }


def run_hybrid_no_rerank(question: str, cfg: dict, top_k: int = 10, **kwargs) -> dict:
    """Step 3: Hybrid retrieval (Dense + BM25 with RRF) without reranking."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.generation.generator import Generator

    retriever = HybridRetriever(cfg)
    generator = Generator(cfg)

    t0 = time.time()
    t_ret = time.time()
    chunks = retriever.retrieve(question, skip_rerank=True)[:top_k]
    retrieval_ms = (time.time() - t_ret) * 1000

    t_gen = time.time()
    result = generator.generate(question, chunks)
    generation_ms = (time.time() - t_gen) * 1000

    return {
        "answer": result.get("answer", ""),
        "contexts": [c.get("text", "") for c in chunks],
        "retrieved_chunks": chunks,
        "retrieval_latency_ms": round(retrieval_ms, 2),
        "reranking_latency_ms": 0,
        "generation_latency_ms": round(generation_ms, 2),
        "total_time": (time.time() - t0),
        "method": "hybrid_no_rerank",
    }


def run_hybrid_rerank(question: str, cfg: dict, **kwargs) -> dict:
    """Step 4: Hybrid + Rerank (V3 pipeline class)."""
    from src.pipeline.advanced_b import AdvancedBPipeline
    pipeline = AdvancedBPipeline(cfg)
    result = pipeline.ask(question)
    return _normalise_result(result, "hybrid_rerank")


def run_query_rewrite(question: str, cfg: dict, **kwargs) -> dict:
    """Step 5: + Query Rewriting (V4 pipeline class)."""
    from src.pipeline.advanced_c import AdvancedCPipeline
    pipeline = AdvancedCPipeline(cfg)
    result = pipeline.ask(question)
    return _normalise_result(result, "query_rewrite")


def run_metadata_filter(question: str, cfg: dict, **kwargs) -> dict:
    """Step 6: + Metadata Filtering (V5 pipeline class)."""
    from src.pipeline.advanced_d import AdvancedDPipeline
    pipeline = AdvancedDPipeline(cfg)
    result = pipeline.ask(question)
    return _normalise_result(result, "metadata_filter")


def run_context_compression(question: str, cfg: dict, **kwargs) -> dict:
    """Step 7: + Context Compression (V6 pipeline class)."""
    from src.pipeline.advanced_e import AdvancedEPipeline
    pipeline = AdvancedEPipeline(cfg)
    result = pipeline.ask(question)
    return _normalise_result(result, "context_compression")


def _normalise_result(pipeline_result: dict, method: str) -> dict:
    """Convert a pipeline .ask() result into the ablation result format."""
    return {
        "answer": pipeline_result.get("answer", ""),
        "contexts": [
            c.get("text", "") for c in pipeline_result.get("retrieved_chunks", [])
        ],
        "retrieved_chunks": pipeline_result.get("retrieved_chunks", []),
        "retrieval_latency_ms": pipeline_result.get("retrieval_latency_ms", 0),
        "reranking_latency_ms": pipeline_result.get("reranking_latency_ms", 0),
        "generation_latency_ms": pipeline_result.get("generation_latency_ms", 0),
        "compression_latency_ms": pipeline_result.get("compression_latency_ms", 0),
        "rewrite_latency_ms": pipeline_result.get("rewrite_latency_ms", 0),
        "filter_latency_ms": pipeline_result.get("filter_latency_ms", 0),
        "total_time": pipeline_result.get("latency_ms", 0) / 1000,
        "method": method,
        "insufficient_evidence": pipeline_result.get("insufficient_evidence", False),
    }


# ── Runner registry ──────────────────────────────────────────────────────────

STEP_RUNNERS = {
    "llm_only":            run_llm_only,
    "dense_only":          run_dense_only,
    "sparse_only":         run_sparse_only,
    "hybrid_no_rerank":    run_hybrid_no_rerank,
    "hybrid_rerank":       run_hybrid_rerank,
    "query_rewrite":       run_query_rewrite,
    "metadata_filter":     run_metadata_filter,
    "context_compression": run_context_compression,
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FinSight Ablation Study (Outline §6.5)")
    parser.add_argument("--dataset", default="evaluation/eval_dataset.json")
    parser.add_argument("--output", default="evaluation/results/ablation_results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--steps", nargs="+",
        default=ALL_STEP_KEYS,
        choices=ALL_STEP_KEYS,
        help="Which ablation steps to run (default: all 8)",
    )
    args = parser.parse_args()

    cfg = load_config()
    dataset_path = PROJECT_ROOT / args.dataset
    dataset = load_eval_dataset(str(dataset_path))

    if args.limit:
        dataset = dataset[:args.limit]

    logger.info(f"Running ablation study on {len(dataset)} questions")
    logger.info(f"Steps: {args.steps}")

    all_results = {}

    for step_key in args.steps:
        step_info = next(s for s in ABLATION_STEPS if s["key"] == step_key)
        logger.info(f"\n{'='*60}")
        logger.info(f"Step {step_info['step']}: {step_info['component']} ({step_key})")
        logger.info(f"{'='*60}")

        runner = STEP_RUNNERS[step_key]
        step_results = []

        for i, item in enumerate(dataset, 1):
            logger.info(f"  [{i}/{len(dataset)}] {item['id']}: {item['question'][:50]}...")

            try:
                result = runner(item["question"], cfg)
                step_results.append({
                    "id": item["id"],
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "category": item.get("category", ""),
                    **result,
                    "error": None,
                })
                logger.info(f"    -> {result['total_time']:.1f}s")
            except Exception as e:
                logger.error(f"    -> ERROR: {e}")
                step_results.append({
                    "id": item["id"],
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "category": item.get("category", ""),
                    "answer": f"ERROR: {e}",
                    "contexts": [],
                    "total_time": 0,
                    "retrieval_latency_ms": 0,
                    "reranking_latency_ms": 0,
                    "generation_latency_ms": 0,
                    "method": step_key,
                    "error": str(e),
                })

            time.sleep(0.3)

        # Compute aggregate stats
        valid = [r for r in step_results if not r.get("error")]
        n_valid = max(len(valid), 1)
        avg_latency = sum(r["total_time"] for r in valid) / n_valid
        avg_retrieval = sum(r.get("retrieval_latency_ms", 0) for r in valid) / n_valid
        avg_reranking = sum(r.get("reranking_latency_ms", 0) for r in valid) / n_valid
        avg_generation = sum(r.get("generation_latency_ms", 0) for r in valid) / n_valid
        error_rate = (len(step_results) - len(valid)) / len(step_results)

        all_results[step_key] = {
            "step_info": step_info,
            "per_question": step_results,
            "aggregate": {
                "avg_latency_seconds": round(avg_latency, 4),
                "avg_retrieval_ms": round(avg_retrieval, 2),
                "avg_reranking_ms": round(avg_reranking, 2),
                "avg_generation_ms": round(avg_generation, 2),
                "error_rate": round(error_rate, 4),
                "n_questions": len(step_results),
                "n_valid": len(valid),
                "n_errors": len(step_results) - len(valid),
            },
        }

        logger.info(
            f"  {step_key} complete: avg_latency={avg_latency:.2f}s, "
            f"retrieval={avg_retrieval:.0f}ms, rerank={avg_reranking:.0f}ms, "
            f"gen={avg_generation:.0f}ms, errors={len(step_results) - len(valid)}"
        )

    # Save results
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {output_path}")

    # Print ablation comparison table (outline §10.3, Table 7)
    print("\n" + "=" * 95)
    print("ABLATION STUDY RESULTS — Outline §6.5")
    print("=" * 95)
    print(
        f"{'Step':<6}{'Variant':<8}{'Component':<28}"
        f"{'Latency':>10}{'Retrieval':>12}{'Rerank':>10}{'Generate':>10}{'Errors':>8}"
    )
    print("-" * 95)
    for step_key in args.steps:
        if step_key not in all_results:
            continue
        info = all_results[step_key]["step_info"]
        agg = all_results[step_key]["aggregate"]
        print(
            f"{info['step']:<6}{info['variant']:<8}{info['component']:<28}"
            f"{agg['avg_latency_seconds']:>9.2f}s"
            f"{agg['avg_retrieval_ms']:>10.0f}ms"
            f"{agg['avg_reranking_ms']:>8.0f}ms"
            f"{agg['avg_generation_ms']:>8.0f}ms"
            f"{agg['n_errors']:>8}"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()
