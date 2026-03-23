"""
category_analysis.py
Category-based evaluation analysis aligned with outline §6.3.

Reads result JSON files (produced by run_eval.py) and breaks down performance
by the four query categories defined in outline.md:
  - factual_retrieval   : Direct lookup (e.g., revenue, income)
  - temporal_reasoning  : Time-based comparisons
  - multi_hop_reasoning : Cross-section synthesis
  - comparative_analysis: Multi-period comparisons

Outputs:
  1. Per-category accuracy table across all variants (V0–V6)
  2. Per-category retrieval metrics (Hit Rate, MRR)
  3. Error / failure breakdown by category and variant
  4. Component impact summary (which component helps which category most)
  5. Ablation step table (outline §6.5)

Usage:
    python evaluation/category_analysis.py --results evaluation/results/
    python evaluation/category_analysis.py --results evaluation/results/ --output evaluation/category_report.json
"""

import sys
import json
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb_compat  # noqa: F401  — must precede any chromadb/pipeline import

from src.utils.logger import get_logger
from evaluation.metrics import (
    compute_rouge_l,
    compute_numeric_match,
    compute_hit_rate,
    compute_mrr,
)

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

CATEGORIES = [
    "factual_retrieval",
    "temporal_reasoning",
    "multi_hop_reasoning",
    "comparative_analysis",
]

CATEGORY_LABELS = {
    "factual_retrieval":   "Factual Retrieval",
    "temporal_reasoning":  "Temporal Reasoning",
    "multi_hop_reasoning": "Multi-hop Reasoning",
    "comparative_analysis":"Comparative Analysis",
}

# Ordered variant list matches outline §2 table
VARIANT_ORDER = [
    "v0_llm_only",
    "v1_baseline",
    "v2_advanced_a",
    "v3_advanced_b",
    "v4_advanced_c",
    "v5_advanced_d",
    "v6_advanced_e",
]

VARIANT_SHORT = {
    "v0_llm_only":    "V0",
    "v1_baseline":    "V1",
    "v2_advanced_a":  "V2",
    "v3_advanced_b":  "V3",
    "v4_advanced_c":  "V4",
    "v5_advanced_d":  "V5",
    "v6_advanced_e":  "V6",
}

FAILURE_TYPES = [
    "retrieval_failure",
    "ranking_failure",
    "chunking_failure",
    "query_understanding_failure",
    "generation_failure",
]

FAILURE_LABELS = {
    "retrieval_failure":           "Retrieval Failure",
    "ranking_failure":             "Ranking Failure",
    "chunking_failure":            "Chunking Failure",
    "query_understanding_failure": "Query Understanding Fail",
    "generation_failure":          "Generation Failure",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_correct(result: dict) -> bool:
    """
    Heuristic correctness check:
    - For RAG variants: answered (not insufficient_evidence, not error)
      AND ROUGE-L ≥ 0.15 against expected summary.
    - For V0 (LLM-only): same but without retrieval filter.
    Falls back to simple answered-flag when no expected answer is available.
    """
    if result.get("error"):
        return False
    if result.get("insufficient_evidence"):
        return False

    ref = result.get("expected_answer_summary", "")
    pred = result.get("answer", "")
    if not ref or not pred:
        return True  # No reference → treat as answered correctly

    rouge = compute_rouge_l(pred, ref)
    numeric_ok = compute_numeric_match(pred, ref)
    # Accept if ROUGE-L is reasonable OR numeric match holds
    return rouge >= 0.15 or numeric_ok


def _classify_failure(result: dict, variant: str) -> Optional[str]:
    """
    Classify a failed (incorrect) result into one of 5 failure categories.
    Priority order: retrieval → ranking → query_understanding → chunking → generation
    """
    if _is_correct(result):
        return None  # Not a failure

    # No citations at all → retrieval failure (except V0 which has no retrieval)
    if variant != "v0_llm_only":
        citations = result.get("citations", [])
        n_chunks = result.get("n_chunks_retrieved", 0)
        if not citations and n_chunks == 0:
            return "retrieval_failure"

        # Low rerank score → ranking failure
        top_score = result.get("top_rerank_score")
        if top_score is not None and top_score < 0.1 and citations:
            return "ranking_failure"

    # Query rewriting available (V4) but insufficient evidence → query understanding
    query_rewrite = result.get("query_rewrite", {})
    if result.get("insufficient_evidence") and not query_rewrite:
        return "query_understanding_failure"

    # Insufficient evidence despite having chunks → chunking failure (info fragmentation)
    if result.get("insufficient_evidence") and result.get("n_chunks_retrieved", 0) > 0:
        return "chunking_failure"

    # Answer given but wrong → generation failure
    if result.get("answer") and not result.get("insufficient_evidence"):
        return "generation_failure"

    return "generation_failure"


# ── Per-category metrics ──────────────────────────────────────────────────────

def compute_category_metrics(results: List[dict], category: str, variant: str) -> Dict:
    """Compute metrics for results filtered to one category."""
    cat_results = [r for r in results if r.get("question_type", "") == category]
    n = len(cat_results)
    if n == 0:
        return {"n": 0, "variant": variant, "category": category}

    correct = [r for r in cat_results if _is_correct(r)]
    accuracy = round(len(correct) / n, 4)

    rouge_scores = []
    for r in correct:
        ref = r.get("expected_answer_summary", "")
        pred = r.get("answer", "")
        if ref and pred:
            rouge_scores.append(compute_rouge_l(pred, ref))

    hit_rate = compute_hit_rate(cat_results, k=3)
    mrr = compute_mrr(cat_results, k=10)

    latencies = [r.get("latency_ms", 0) for r in cat_results]
    avg_latency = round(statistics.mean(latencies), 1) if latencies else 0.0

    # Failure breakdown
    failures = defaultdict(int)
    for r in cat_results:
        ft = _classify_failure(r, variant)
        if ft:
            failures[ft] += 1

    return {
        "variant": variant,
        "category": category,
        "n": n,
        "n_correct": len(correct),
        "accuracy": accuracy,
        "accuracy_pct": round(accuracy * 100, 1),
        "rouge_l_mean": round(statistics.mean(rouge_scores), 4) if rouge_scores else None,
        "top3_hit_rate": hit_rate,
        "mrr": mrr,
        "avg_latency_ms": avg_latency,
        "failures": dict(failures),
    }


# ── Print tables ──────────────────────────────────────────────────────────────

def print_accuracy_table(all_category_metrics: List[Dict], variants: List[str]):
    """Table 3 from outline §10.3: Category-based accuracy across variants."""
    col_w = 10
    print("\n" + "=" * 80)
    print("CATEGORY-BASED ACCURACY (%) — Outline §6.3")
    print("=" * 80)
    header = f"{'Category':<28}" + "".join(f"{VARIANT_SHORT.get(v, v):<{col_w}}" for v in variants)
    print(header)
    print("-" * (28 + col_w * len(variants)))

    for cat in CATEGORIES:
        row = f"{CATEGORY_LABELS[cat]:<28}"
        for v in variants:
            entry = next(
                (m for m in all_category_metrics if m["variant"] == v and m["category"] == cat),
                None,
            )
            if entry and entry.get("n", 0) > 0:
                row += f"{entry['accuracy_pct']:<{col_w}.1f}"
            else:
                row += f"{'N/A':<{col_w}}"
        print(row)
    print("=" * 80)


def print_retrieval_table(all_category_metrics: List[Dict], variants: List[str]):
    """Table 4 from outline §10.3: Retrieval metrics by query type."""
    # Exclude V0 (no retrieval)
    rag_variants = [v for v in variants if v != "v0_llm_only"]
    col_w = 10

    print("\n" + "=" * 80)
    print("RETRIEVAL METRICS BY QUERY TYPE — Outline §6.2 / §10.3")
    print("=" * 80)

    for metric_label, metric_key, fmt in [
        ("Top-3 Hit Rate (%)", "top3_hit_rate", ".1%"),
        ("MRR", "mrr", ".3f"),
    ]:
        print(f"\n  {metric_label}")
        header = f"  {'Category':<26}" + "".join(
            f"{VARIANT_SHORT.get(v, v):<{col_w}}" for v in rag_variants
        )
        print(header)
        print("  " + "-" * (26 + col_w * len(rag_variants)))
        for cat in CATEGORIES:
            row = f"  {CATEGORY_LABELS[cat]:<26}"
            for v in rag_variants:
                entry = next(
                    (m for m in all_category_metrics if m["variant"] == v and m["category"] == cat),
                    None,
                )
                if entry and entry.get("n", 0) > 0:
                    val = entry.get(metric_key, 0)
                    row += f"{format(val, fmt):<{col_w}}"
                else:
                    row += f"{'N/A':<{col_w}}"
            print(row)

    print("=" * 80)


def print_failure_table(all_category_metrics: List[Dict], variants: List[str]):
    """Table 6 from outline §10.3: Failure rates by category and variant."""
    col_w = 10
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS BY CATEGORY — Outline §6.4")
    print("=" * 80)

    for cat in CATEGORIES:
        print(f"\n  {CATEGORY_LABELS[cat]}")
        header = f"  {'Failure Type':<34}" + "".join(
            f"{VARIANT_SHORT.get(v, v):<{col_w}}" for v in variants
        )
        print(header)
        print("  " + "-" * (34 + col_w * len(variants)))

        for ft in FAILURE_TYPES:
            row = f"  {FAILURE_LABELS[ft]:<34}"
            for v in variants:
                entry = next(
                    (m for m in all_category_metrics if m["variant"] == v and m["category"] == cat),
                    None,
                )
                if entry and entry.get("n", 0) > 0:
                    n_total = entry["n"]
                    n_fail = entry.get("failures", {}).get(ft, 0)
                    pct = n_fail / n_total * 100
                    row += f"{pct:<{col_w}.0f}"
                elif v == "v0_llm_only" and ft in ("retrieval_failure", "ranking_failure"):
                    row += f"{'—':<{col_w}}"
                else:
                    row += f"{'N/A':<{col_w}}"
            print(row)

    print("=" * 80)


def print_ablation_table(aggregate_metrics: List[Dict], variants: List[str]):
    """Table 7 from outline §10.3: Ablation study step table."""
    component_map = {
        "v0_llm_only":   "None (LLM-only)",
        "v1_baseline":   "+ Dense Retrieval",
        "v2_advanced_a": "+ Reranking",
        "v3_advanced_b": "+ Hybrid Retrieval",
        "v4_advanced_c": "+ Query Rewriting",
        "v5_advanced_d": "+ Metadata Filtering",
        "v6_advanced_e": "+ Context Compression",
    }
    step_map = {v: i for i, v in enumerate(VARIANT_ORDER)}

    print("\n" + "=" * 80)
    print("ABLATION STUDY — Outline §6.5")
    print("=" * 80)
    print(f"{'Step':<6}{'Variant':<18}{'Component':<28}{'Accuracy':<12}{'Latency P50':<14}")
    print("-" * 78)
    for v in variants:
        m = next((x for x in aggregate_metrics if x.get("variant") == v), None)
        if not m:
            continue
        step = step_map.get(v, "?")
        component = component_map.get(v, v)
        # Aggregate accuracy: correct / total non-error, non-guardrail
        n_ans = m.get("n_answered", 0)
        n_total_eff = m.get("n_total", 1) - m.get("n_errors", 0)
        acc_pct = round(n_ans / max(n_total_eff, 1) * 100, 1)
        lat = m.get("latency", {}).get("median_ms", "N/A")
        lat_str = f"{lat:.0f}ms" if isinstance(lat, (int, float)) else str(lat)
        rouge = m.get("rouge_l_mean")
        rouge_str = f"{rouge:.3f}" if rouge is not None else "N/A"
        print(f"{step:<6}{VARIANT_SHORT.get(v, v):<18}{component:<28}{acc_pct:<12.1f}{lat_str:<14}")
    print("=" * 80)


def print_component_impact_table(all_category_metrics: List[Dict], variants: List[str]):
    """
    Table 8 from outline §10.3: Which component helps which category most.
    Computes accuracy improvement per variant over the previous step.
    """
    col_w = 12
    rag_variants = [v for v in VARIANT_ORDER if v in variants]

    print("\n" + "=" * 80)
    print("COMPONENT IMPACT BY CATEGORY — Outline §7.2")
    print("=" * 80)
    print(f"(Δ accuracy vs previous step)")
    header = f"{'Category':<28}" + "".join(
        f"{VARIANT_SHORT.get(v, v):<{col_w}}" for v in rag_variants[1:]  # skip V0 as baseline
    )
    print(header)
    print("-" * (28 + col_w * (len(rag_variants) - 1)))

    for cat in CATEGORIES:
        row = f"{CATEGORY_LABELS[cat]:<28}"
        prev_acc = None
        for v in rag_variants:
            entry = next(
                (m for m in all_category_metrics if m["variant"] == v and m["category"] == cat),
                None,
            )
            curr_acc = entry["accuracy_pct"] if (entry and entry.get("n", 0) > 0) else None

            if v == rag_variants[0]:
                prev_acc = curr_acc
                continue  # V0 is baseline, skip printing

            if curr_acc is not None and prev_acc is not None:
                delta = curr_acc - prev_acc
                sign = "+" if delta >= 0 else ""
                row += f"{sign}{delta:<{col_w-1}.1f}"
            else:
                row += f"{'N/A':<{col_w}}"
            prev_acc = curr_acc
        print(row)
    print("=" * 80)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FinSight category-based analysis")
    parser.add_argument("--results", default="evaluation/results",
                        help="Directory containing variant JSON result files")
    parser.add_argument("--output", default="evaluation/category_report.json",
                        help="Path to save JSON report")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    result_files = sorted(results_dir.glob("*.json"))
    result_files = [f for f in result_files if f.stem not in ("eval_summary", "category_report")]

    if not result_files:
        logger.error(f"No result files found in {results_dir}")
        sys.exit(1)

    # Load all variant results
    variant_results: Dict[str, List[dict]] = {}
    aggregate_metrics: List[dict] = []

    for rf in result_files:
        variant_key = rf.stem
        with open(rf, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", data) if isinstance(data, dict) else data
        summary = data.get("summary", {}) if isinstance(data, dict) else {}
        variant_results[variant_key] = results

        # Aggregate metrics for ablation table
        from evaluation.metrics import compute_variant_metrics
        agg = compute_variant_metrics(results, variant_key)
        agg.update(summary)  # merge in any extra fields from summary
        aggregate_metrics.append(agg)

    # Order variants
    variants = [v for v in VARIANT_ORDER if v in variant_results]
    # Append any unexpected variant keys
    variants += [v for v in variant_results if v not in variants]

    logger.info(f"Loaded {len(variants)} variants: {variants}")

    # ── Per-category metrics ──────────────────────────────────────────────────
    all_category_metrics: List[Dict] = []

    for variant_key, results in variant_results.items():
        for cat in CATEGORIES:
            cm = compute_category_metrics(results, cat, variant_key)
            all_category_metrics.append(cm)
            if cm["n"] > 0:
                logger.info(
                    f"  {VARIANT_SHORT.get(variant_key, variant_key)} | "
                    f"{CATEGORY_LABELS[cat]}: "
                    f"acc={cm['accuracy_pct']:.1f}% "
                    f"hit={cm['top3_hit_rate']:.1%} "
                    f"mrr={cm['mrr']:.3f}"
                )

    # ── Print all tables ──────────────────────────────────────────────────────
    print_accuracy_table(all_category_metrics, variants)
    print_retrieval_table(all_category_metrics, variants)
    print_failure_table(all_category_metrics, variants)
    print_ablation_table(aggregate_metrics, variants)
    print_component_impact_table(all_category_metrics, variants)

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "variants": variants,
        "categories": CATEGORIES,
        "category_metrics": all_category_metrics,
        "aggregate_metrics": aggregate_metrics,
    }
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nCategory report saved to {output_path}")
    logger.info("Next step: review tables above or open the JSON report for full data.")


if __name__ == "__main__":
    main()
