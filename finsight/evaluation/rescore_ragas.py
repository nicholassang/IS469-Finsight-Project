"""
rescore_ragas.py
Re-compute RAGAS metrics on previously saved Q&A results.

Run this SEPARATELY from run_evaluation.py so the vLLM server has
time to recover before being used as the RAGAS judge.

If OPENAI_API_KEY is set (in .env or environment), uses gpt-4o-mini as the
judge LLM — much more stable than the local vLLM for 80 sequential calls.
Falls back to local vLLM if no API key is found.

Usage:
    python evaluation/rescore_ragas.py
    python evaluation/rescore_ragas.py --input evaluation/results/eval_results_improved_v2.json
    python evaluation/rescore_ragas.py --input evaluation/results/eval_results_improved_v2.json --output evaluation/results/eval_results_final.json
"""
import sys
import os
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

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


def main():
    _load_env()

    parser = argparse.ArgumentParser(description="Re-score RAGAS metrics on saved results")
    parser.add_argument(
        "--input",
        default="evaluation/results/eval_results_improved_v2.json",
        help="Path to the saved Q&A results JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save scored results (default: overwrites input file)",
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

    for mode in all_results:
        logger.info(f"Re-scoring RAGAS metrics for mode: {mode} ...")
        per_question = all_results[mode]["per_question"]
        ragas_scores = compute_ragas_metrics_stable(per_question, cfg)

        latencies = [r["latency_seconds"] for r in per_question if not r.get("error")]
        avg_latency = sum(latencies) / max(len(latencies), 1)

        all_results[mode]["aggregate"] = {
            **ragas_scores,
            "avg_latency_seconds": round(avg_latency, 4),
        }
        logger.info(f"  {mode}: {json.dumps(ragas_scores)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Scored results saved to {output_path}")

    # Print summary table
    modes = list(all_results.keys())
    metrics_keys = ["faithfulness", "answer_relevancy", "context_recall",
                    "context_precision", "avg_latency_seconds"]
    header = f"{'Metric':<25}" + "".join(f" | {m:>12}" for m in modes)
    print("\n" + "=" * (25 + 15 * len(modes)))
    print("  RAGAS EVALUATION RESULTS — FinSight")
    print("=" * (25 + 15 * len(modes)))
    print(header)
    print("-" * (25 + 15 * len(modes)))
    for k in metrics_keys:
        row = f"{k:<25}"
        for m in modes:
            row += f" | {all_results[m]['aggregate'].get(k, 0):>12.4f}"
        print(row)
    print("=" * (25 + 15 * len(modes)))


if __name__ == "__main__":
    main()
