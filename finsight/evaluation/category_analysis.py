"""
category_analysis.py
Category-based evaluation analysis for FinSight (IS469 Final Project).

Reads eval_results.json (produced by run_evaluation.py) and outputs:
  1. Per-category RAGAS metrics across all variants (V0-V6)
  2. Per-category numerical accuracy across all variants
  3. Per-category failure classification breakdown
  4. Component impact summary (delta vs previous pipeline step)
  5. Latency breakdown by variant and category

Failure types (derived from observable signals in eval_results.json):
  - Retrieval Failure    : answer is a refusal AND variant has retrieval
  - Generation Failure   : answer is wrong/hallucinated despite retrieval
  - Query Understanding  : refusal on multi-period/ambiguous questions in V1
  - Chunking Failure     : low context_recall despite retrieval succeeding

Usage:
    python evaluation/category_analysis.py
    python evaluation/category_analysis.py --results evaluation/results/eval_results.json
    python evaluation/category_analysis.py --output evaluation/category_report.json
"""

import sys
import json
import math
import argparse
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORIES = [
    "factual_retrieval",
    "temporal_reasoning",
    "multi_hop_reasoning",
    "comparative_analysis",
]

CATEGORY_LABELS = {
    "factual_retrieval":    "Factual Retrieval",
    "temporal_reasoning":   "Temporal Reasoning",
    "multi_hop_reasoning":  "Multi-Hop Reasoning",
    "comparative_analysis": "Comparative Analysis",
}

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
    "v0_llm_only":   "V0",
    "v1_baseline":   "V1",
    "v2_advanced_a": "V2",
    "v3_advanced_b": "V3",
    "v4_advanced_c": "V4",
    "v5_advanced_d": "V5",
    "v6_advanced_e": "V6",
}

COMPONENT_ADDED = {
    "v0_llm_only":   "No retrieval (baseline)",
    "v1_baseline":   "+ Dense retrieval",
    "v2_advanced_a": "+ Cross-encoder reranking",
    "v3_advanced_b": "+ Hybrid retrieval (BM25+RRF)",
    "v4_advanced_c": "+ Query rewriting",
    "v5_advanced_d": "+ Metadata filtering",
    "v6_advanced_e": "+ Context compression",
}

# Refusal phrases the generator uses when context is insufficient
REFUSAL_PHRASES = [
    "does not contain",
    "not contain",
    "insufficient",
    "cannot provide",
    "no information",
    "not available",
    "only contains data for",
    "not included",
    "not explicitly stated",
    "cannot be determined",
    "context only contains",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_refusal(answer: str) -> bool:
    """True when the generator declined to answer due to missing context."""
    a = answer.lower()
    return any(p in a for p in REFUSAL_PHRASES)


def safe(val, default=0.0):
    """Return default when val is None or NaN."""
    if val is None:
        return default
    try:
        if math.isnan(val):
            return default
    except TypeError:
        pass
    return val


def fmt(val, decimals=3, default="n/a"):
    """Format a float for display, returning default string for None/NaN."""
    if val is None:
        return default
    try:
        if math.isnan(val):
            return default
    except TypeError:
        pass
    return f"{val:.{decimals}f}"


# ── Failure classification ─────────────────────────────────────────────────────

def classify_failure(q: dict, ragas: Optional[dict], variant: str) -> Optional[str]:
    """
    Classify a question result into one of four failure types using only
    fields that actually exist in eval_results.json.

    Priority order:
      1. Retrieval Failure   — refusal AND no useful context retrieved
                               (context_recall == 0 or no contexts)
      2. Query Understanding — refusal on temporal/comparative in dense-only
                               variants (V1, V2, V5) — signal that query
                               formulation could not discriminate fiscal periods
      3. Chunking Failure    — partial context_recall (0 < recall < 0.5)
                               meaning info was split across chunk boundaries
      4. Generation Failure  — answer was produced but numerical_match=False
                               and answer_relevancy is low

    Returns None if the question is answered correctly (numerical_match=True
    or high RAGAS scores).
    """
    answer = q.get("answer", "")
    numerical_match = q.get("numerical_match", False)
    category = q.get("category", "")
    has_retrieval = variant != "v0_llm_only"

    # Gather RAGAS signals (safe defaults when NaN)
    recall = safe(ragas.get("context_recall") if ragas else None, 0.0)
    relevancy = safe(ragas.get("answer_relevancy") if ragas else None, 0.0)
    faithfulness = safe(ragas.get("faithfulness") if ragas else None, 0.0)
    n_contexts = len(q.get("contexts", []))
    refusal = is_refusal(answer)

    # Consider "correct" if numerical match or very high relevancy
    if numerical_match or relevancy >= 0.85:
        return None

    if not has_retrieval:
        # V0: only generation failure is meaningful
        return "generation_failure"

    # 1. Retrieval Failure: refused AND effectively nothing retrieved
    if refusal and (recall == 0.0 or n_contexts == 0):
        return "retrieval_failure"

    # 2. Query Understanding Failure: refusal on temporal/comparative in
    #    variants without query rewriting (V1, V2, V5)
    if refusal and category in ("temporal_reasoning", "comparative_analysis", "multi_hop_reasoning"):
        if variant in ("v1_baseline", "v2_advanced_a", "v5_advanced_d"):
            return "query_understanding_failure"

    # 3. Chunking Failure: some context retrieved but recall is low,
    #    suggesting relevant info was split across chunk boundaries
    if n_contexts > 0 and 0.0 < recall < 0.5 and refusal:
        return "chunking_failure"

    # 4. Generation Failure: answer produced but wrong
    if not refusal and not numerical_match and relevancy < 0.6:
        return "generation_failure"

    # Default for remaining refusals (retrieval got something, but not enough)
    if refusal:
        return "retrieval_failure"

    # Low relevancy answer that wasn't caught above
    return "generation_failure"


# ── Per-category metrics ───────────────────────────────────────────────────────

def compute_category_metrics(
    per_question: List[dict],
    per_question_ragas: List[dict],
    variant: str,
    category: str,
) -> dict:
    """
    Compute all metrics for one variant × category slice.
    Uses only fields present in the actual eval_results.json.
    """
    # Build ragas lookup by question id
    ragas_map = {r["id"]: r for r in per_question_ragas}

    # Filter to this category
    qs = [q for q in per_question if q.get("category") == category]
    n = len(qs)
    if n == 0:
        return {"variant": variant, "category": category, "n": 0}

    # ── RAGAS metrics (mean, NaN-safe) ────────────────────────────────────────
    def mean_metric(metric_key):
        vals = []
        for q in qs:
            r = ragas_map.get(q["id"])
            if r:
                v = r.get(metric_key)
                if v is not None and not math.isnan(v):
                    vals.append(v)
        return round(statistics.mean(vals), 4) if vals else None

    faithfulness      = mean_metric("faithfulness")
    answer_relevancy  = mean_metric("answer_relevancy")
    context_recall    = mean_metric("context_recall")
    context_precision = mean_metric("context_precision")

    # ── Numerical accuracy ────────────────────────────────────────────────────
    numerical_accuracy = round(
        sum(1 for q in qs if q.get("numerical_match")) / n, 4
    )

    # ── Refusal / answer rates ────────────────────────────────────────────────
    n_refusals = sum(1 for q in qs if is_refusal(q.get("answer", "")))
    answer_rate = round((n - n_refusals) / n, 4)

    # ── Latency ───────────────────────────────────────────────────────────────
    latencies = [q.get("latency_seconds", 0) for q in qs]
    avg_latency = round(statistics.mean(latencies), 3) if latencies else 0.0

    # ── Failure breakdown ─────────────────────────────────────────────────────
    failures: Dict[str, int] = defaultdict(int)
    for q in qs:
        ragas = ragas_map.get(q["id"])
        ft = classify_failure(q, ragas, variant)
        if ft:
            failures[ft] += 1

    return {
        "variant": variant,
        "category": category,
        "n": n,
        "n_refusals": n_refusals,
        "answer_rate": answer_rate,
        "numerical_accuracy": numerical_accuracy,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_recall": context_recall,
        "context_precision": context_precision,
        "avg_latency_s": avg_latency,
        "failures": dict(failures),
    }


# ── Print tables ───────────────────────────────────────────────────────────────

def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_ragas_table(all_metrics: List[dict], variants: List[str]):
    """Table 1: RAGAS metrics by variant × category for each metric."""
    metrics_to_show = [
        ("Faithfulness",       "faithfulness"),
        ("Answer Relevancy",   "answer_relevancy"),
        ("Context Recall",     "context_recall"),
        ("Context Precision",  "context_precision"),
    ]

    for metric_label, metric_key in metrics_to_show:
        print_section(f"RAGAS — {metric_label} by Variant × Category")
        col = 10
        header = f"  {'Category':<26}" + "".join(
            f"{VARIANT_SHORT[v]:<{col}}" for v in variants
        )
        print(header)
        print("  " + "-" * (26 + col * len(variants)))

        for cat in CATEGORIES:
            row = f"  {CATEGORY_LABELS[cat]:<26}"
            for v in variants:
                entry = next(
                    (m for m in all_metrics if m["variant"] == v and m["category"] == cat),
                    None,
                )
                if entry and entry.get("n", 0) > 0:
                    val = entry.get(metric_key)
                    row += f"{fmt(val, 3):<{col}}"
                else:
                    row += f"{'n/a':<{col}}"
            print(row)


def print_numerical_accuracy_table(all_metrics: List[dict], variants: List[str]):
    """Table 2: Numerical accuracy by variant × category."""
    print_section("Numerical Accuracy by Variant × Category")
    col = 10
    header = f"  {'Category':<26}" + "".join(
        f"{VARIANT_SHORT[v]:<{col}}" for v in variants
    )
    print(header)
    print("  " + "-" * (26 + col * len(variants)))

    for cat in CATEGORIES:
        row = f"  {CATEGORY_LABELS[cat]:<26}"
        for v in variants:
            entry = next(
                (m for m in all_metrics if m["variant"] == v and m["category"] == cat),
                None,
            )
            if entry and entry.get("n", 0) > 0:
                val = entry.get("numerical_accuracy", 0.0)
                row += f"{val*100:.0f}%{'':<{col-4}}"
            else:
                row += f"{'n/a':<{col}}"
        print(row)


def print_failure_table(all_metrics: List[dict], variants: List[str]):
    """Table 3: Failure classification by variant × category."""
    failure_types = [
        ("retrieval_failure",         "Retrieval Failure"),
        ("query_understanding_failure","Query Understanding"),
        ("chunking_failure",          "Chunking Failure"),
        ("generation_failure",        "Generation Failure"),
    ]

    print_section("Failure Classification by Variant × Category (count out of 5)")
    col = 10

    for cat in CATEGORIES:
        print(f"\n  — {CATEGORY_LABELS[cat]} —")
        header = f"  {'Failure Type':<30}" + "".join(
            f"{VARIANT_SHORT[v]:<{col}}" for v in variants
        )
        print(header)
        print("  " + "-" * (30 + col * len(variants)))

        for ft_key, ft_label in failure_types:
            row = f"  {ft_label:<30}"
            for v in variants:
                entry = next(
                    (m for m in all_metrics if m["variant"] == v and m["category"] == cat),
                    None,
                )
                if v == "v0_llm_only" and ft_key in (
                    "retrieval_failure", "query_understanding_failure", "chunking_failure"
                ):
                    row += f"{'—':<{col}}"
                elif entry and entry.get("n", 0) > 0:
                    count = entry.get("failures", {}).get(ft_key, 0)
                    row += f"{count:<{col}}"
                else:
                    row += f"{'n/a':<{col}}"
            print(row)


def print_component_impact_table(all_metrics: List[dict], variants: List[str]):
    """
    Table 4: Component impact — delta in answer relevancy vs previous variant.
    Shows which component helps which category most.
    """
    print_section("Component Impact — Δ Answer Relevancy vs Previous Pipeline Step")
    col = 10
    header = f"  {'Category':<26}" + "".join(
        f"{VARIANT_SHORT[v]:<{col}}" for v in variants[1:]  # V1 onward
    )
    print(header)
    print(f"  {'Component':<26}" + "".join(
        f"{COMPONENT_ADDED[v][:col-1]:<{col}}" for v in variants[1:]
    ))
    print("  " + "-" * (26 + col * (len(variants) - 1)))

    for cat in CATEGORIES:
        row = f"  {CATEGORY_LABELS[cat]:<26}"
        prev_val = None
        for v in variants:
            entry = next(
                (m for m in all_metrics if m["variant"] == v and m["category"] == cat),
                None,
            )
            curr_val = entry.get("answer_relevancy") if (entry and entry.get("n", 0) > 0) else None

            if v == variants[0]:
                prev_val = curr_val
                continue  # Skip V0 as it's the delta baseline

            if curr_val is not None and prev_val is not None:
                delta = curr_val - prev_val
                sign = "+" if delta >= 0 else ""
                row += f"{sign}{delta:.3f}{'':<{col-7}}"
            else:
                row += f"{'n/a':<{col}}"
            prev_val = curr_val
        print(row)


def print_latency_table(all_metrics: List[dict], variants: List[str]):
    """Table 5: Average latency by variant × category."""
    print_section("Average Latency (seconds) by Variant × Category")
    col = 10
    header = f"  {'Category':<26}" + "".join(
        f"{VARIANT_SHORT[v]:<{col}}" for v in variants
    )
    print(header)
    print("  " + "-" * (26 + col * len(variants)))

    for cat in CATEGORIES:
        row = f"  {CATEGORY_LABELS[cat]:<26}"
        for v in variants:
            entry = next(
                (m for m in all_metrics if m["variant"] == v and m["category"] == cat),
                None,
            )
            if entry and entry.get("n", 0) > 0:
                val = entry.get("avg_latency_s", 0.0)
                row += f"{val:.2f}s{'':<{col-5}}"
            else:
                row += f"{'n/a':<{col}}"
        print(row)


def print_answer_rate_table(all_metrics: List[dict], variants: List[str]):
    """Table 6: Answer rate (non-refusal) by variant × category."""
    print_section("Answer Rate (non-refusal %) by Variant × Category")
    col = 10
    header = f"  {'Category':<26}" + "".join(
        f"{VARIANT_SHORT[v]:<{col}}" for v in variants
    )
    print(header)
    print("  " + "-" * (26 + col * len(variants)))

    for cat in CATEGORIES:
        row = f"  {CATEGORY_LABELS[cat]:<26}"
        for v in variants:
            entry = next(
                (m for m in all_metrics if m["variant"] == v and m["category"] == cat),
                None,
            )
            if entry and entry.get("n", 0) > 0:
                val = entry.get("answer_rate", 0.0)
                row += f"{val*100:.0f}%{'':<{col-4}}"
            else:
                row += f"{'n/a':<{col}}"
        print(row)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FinSight category-based analysis")
    parser.add_argument(
        "--results",
        default="evaluation/results/eval_results.json",
        help="Path to eval_results.json produced by run_evaluation.py",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/category_report.json",
        help="Path to save the JSON report",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        # Try relative to project root
        results_path = PROJECT_ROOT / args.results
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine variant order from file, respecting VARIANT_ORDER
    available = list(data.keys())
    variants = [v for v in VARIANT_ORDER if v in available]
    variants += [v for v in available if v not in variants]

    print(f"Loaded {len(variants)} variants: {[VARIANT_SHORT.get(v, v) for v in variants]}")
    print(f"Categories: {CATEGORIES}")

    # ── Compute all category metrics ──────────────────────────────────────────
    all_metrics: List[dict] = []

    for variant in variants:
        vdata = data[variant]
        per_question       = vdata.get("per_question", [])
        per_question_ragas = vdata.get("per_question_ragas", [])

        for cat in CATEGORIES:
            m = compute_category_metrics(
                per_question, per_question_ragas, variant, cat
            )
            all_metrics.append(m)

            if m["n"] > 0:
                print(
                    f"  {VARIANT_SHORT.get(variant, variant):<4} | "
                    f"{CATEGORY_LABELS[cat]:<24} | "
                    f"num_acc={m['numerical_accuracy']:.0%} | "
                    f"relevancy={fmt(m['answer_relevancy'])} | "
                    f"refusals={m['n_refusals']}/{m['n']}"
                )

    # ── Print all tables ──────────────────────────────────────────────────────
    print_ragas_table(all_metrics, variants)
    print_numerical_accuracy_table(all_metrics, variants)
    print_answer_rate_table(all_metrics, variants)
    print_failure_table(all_metrics, variants)
    print_component_impact_table(all_metrics, variants)
    print_latency_table(all_metrics, variants)

    # ── Summary: which variant wins each category ─────────────────────────────
    print_section("Best Variant per Category (by Answer Relevancy)")
    for cat in CATEGORIES:
        cat_entries = [
            m for m in all_metrics
            if m["category"] == cat and m.get("answer_relevancy") is not None
        ]
        if not cat_entries:
            continue
        best = max(cat_entries, key=lambda m: safe(m.get("answer_relevancy"), 0.0))
        print(
            f"  {CATEGORY_LABELS[cat]:<26} → "
            f"{VARIANT_SHORT.get(best['variant'], best['variant']):<6} "
            f"(relevancy={fmt(best['answer_relevancy'])}, "
            f"num_acc={best['numerical_accuracy']:.0%})"
        )

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "variants": variants,
        "categories": CATEGORIES,
        "category_metrics": all_metrics,
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nCategory report saved to {output_path}")


if __name__ == "__main__":
    main()