"""
run_query.py
CLI tool to run a single question through the FinSight RAG pipeline
in either "baseline" or "advanced" mode.

Baseline mode:  Dense ChromaDB retrieval only, top-k=5, no reranker, no BM25.
Advanced mode:  Hybrid RRF (dense + BM25) + cross-encoder reranker (existing V3 pipeline).

Usage:
    python scripts/run_query.py "What was Microsoft's revenue in FY2024?"
    python scripts/run_query.py "What was Microsoft's revenue in FY2024?" --mode baseline
    python scripts/run_query.py "What was Microsoft's revenue in FY2024?" --mode advanced
"""

import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb_compat  # noqa: F401  — must precede any chromadb/pipeline import

from src.utils.config_loader import load_config, invalidate_config_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_baseline_pipeline(cfg: dict):
    """Build a baseline pipeline: dense-only retrieval, no reranker."""
    from src.pipeline.baseline import BaselinePipeline

    class BaselineModeWrapper:
        """Wraps BaselinePipeline to force baseline top_k from settings."""

        VARIANT_NAME = "baseline_mode"

        def __init__(self, cfg):
            self._pipeline = BaselinePipeline(cfg)
            self._top_k = cfg["retrieval"].get("baseline_top_k", 5)

        def ask(self, question: str) -> dict:
            t0 = time.time()
            retrieved = self._pipeline.retriever.retrieve(question, top_k=self._top_k)
            gen_result = self._pipeline.generator.generate(question, retrieved)
            from src.generation.citation_formatter import format_citations
            citations = format_citations(gen_result["answer"], retrieved)
            total_latency = (time.time() - t0) * 1000

            return {
                "answer": gen_result["answer"],
                "citations": citations,
                "retrieved_chunks": retrieved,
                "context_used": gen_result.get("context_used", ""),
                "latency_ms": round(total_latency, 2),
                "generation_latency_ms": gen_result.get("latency_ms", 0),
                "variant": "baseline_mode",
                "model": gen_result.get("model", ""),
                "insufficient_evidence": gen_result.get("insufficient_evidence", False),
                "input_tokens": gen_result.get("input_tokens", 0),
                "output_tokens": gen_result.get("output_tokens", 0),
                "total_tokens": gen_result.get("total_tokens", 0),
                "error": gen_result.get("error"),
            }

    return BaselineModeWrapper(cfg)


def build_advanced_pipeline(cfg: dict):
    """Build the advanced pipeline: hybrid RRF + reranker (V3)."""
    from src.pipeline.advanced_b import AdvancedBPipeline
    return AdvancedBPipeline(cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Run a single question through the FinSight RAG pipeline."
    )
    parser.add_argument("question", type=str, help="The question to ask")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "advanced"],
        default=None,
        help="Retrieval mode: 'baseline' (dense-only, top-5, no reranker) or "
             "'advanced' (hybrid RRF + reranker). Defaults to settings.yaml value.",
    )
    args = parser.parse_args()

    cfg = load_config()
    mode = args.mode or cfg["retrieval"].get("mode", "advanced")

    if mode not in ("baseline", "advanced"):
        print(f"Error: Invalid mode '{mode}'. Must be 'baseline' or 'advanced'.")
        sys.exit(1)

    print(f"Mode:     {mode}")
    print(f"Question: {args.question}")
    print("-" * 60)

    if mode == "baseline":
        pipeline = build_baseline_pipeline(cfg)
    else:
        pipeline = build_advanced_pipeline(cfg)

    result = pipeline.ask(args.question)

    if result.get("error"):
        print(f"\nERROR: {result['error']}")

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\n{'─' * 60}")
    print(f"Variant:   {result.get('variant', 'N/A')}")
    print(f"Model:     {result.get('model', 'N/A')}")
    print(f"Latency:   {result.get('latency_ms', 0):.0f} ms")
    print(f"Tokens:    {result.get('total_tokens', 0)}")
    print(f"Chunks:    {len(result.get('retrieved_chunks', []))}")

    citations = result.get("citations", [])
    if citations:
        print(f"\nCitations ({len(citations)}):")
        for cit in citations:
            ref = cit.get("ref", "?")
            period = cit.get("fiscal_period", "N/A")
            doc_type = cit.get("doc_type", "N/A")
            page = cit.get("page_number", "?")
            print(f"  [{ref}] {doc_type} | {period} | Page {page}")


if __name__ == "__main__":
    main()
