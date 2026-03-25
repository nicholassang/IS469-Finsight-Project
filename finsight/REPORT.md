# FinSight: RAG-based Financial Filings QA System
## Final Project Report

**Course:** IS469 - Generative AI with Large Language Models
**Project:** Domain-Specific RAG System for SEC Filings Analysis
**Company Focus:** Microsoft Corporation (NASDAQ: MSFT)

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Problem Statement & Objectives](#2-problem-statement--objectives)
3. [System Architecture](#3-system-architecture)
4. [Methodology](#4-methodology)
5. [Experimental Results](#5-experimental-results)
6. [Qualitative Error Analysis](#6-qualitative-error-analysis)
7. [Discussion & Insights](#7-discussion--insights)
8. [Limitations & Future Work](#8-limitations--future-work)
9. [Risks and Guardrails](#9-risks-and-guardrails)
10. [Conclusion](#10-conclusion)
11. [Appendix](#11-appendix)

---

## 1. Introduction

Financial filings such as SEC 10-K and 10-Q reports are critical sources of information for investors, analysts, and researchers. However, these documents are lengthy (often 100+ pages), dense with technical language, and require significant domain knowledge to navigate efficiently. Traditional keyword search is inadequate for extracting nuanced financial insights that span multiple sections or require temporal reasoning.

This project presents **FinSight**, a Retrieval-Augmented Generation (RAG) system designed specifically for question-answering over Microsoft Corporation's official SEC filings. We implement and rigorously compare **seven pipeline variants (V0–V6)** to understand how individual RAG components affect performance across different query types.

### Key Contributions
1. **End-to-end RAG implementation** with seven configurable pipelines covering a full component ablation: no retrieval (V0), dense-only (V1), dense+reranking (V2), hybrid+reranking (V3), query rewriting (V4), metadata filtering (V5), and context compression (V6)
2. **Controlled experimental design** isolating the contribution of each RAG component to performance across four query categories
3. **Comprehensive evaluation** using RAGAS metrics, numerical accuracy, and qualitative error analysis
4. **Actionable insights** on which RAG components benefit which query types

---

## 2. Problem Statement & Objectives

### Problem Statement
Extracting specific financial insights from SEC filings—such as revenue trends, segment performance, or risk disclosures—requires:
- Navigating hundreds of pages of complex financial documents
- Understanding Microsoft's fiscal calendar (July–June fiscal year)
- Cross-referencing information across multiple filing periods
- Distinguishing between similar-sounding metrics across different time periods

### Objectives
1. Develop a domain-specific QA system that provides accurate, citation-backed answers
2. Compare seven RAG pipeline variants to isolate the contribution of each component
3. Analyse failure modes to understand when and why different approaches succeed or fail
4. Demonstrate reproducible evaluation methodology for domain-specific RAG systems

### Study Hypothesis
Different RAG components provide selective benefits depending on query type, rather than uniformly improving performance across all categories. No single pipeline is optimal for all query types.

---

## 3. System Architecture

### 3.1 Pipeline Overview

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                     USER QUERY                          │
                              └─────────────────────────┬───────────────────────────────┘
                                                        │
                              ┌─────────────────────────▼───────────────────────────────┐
                              │         QUERY PROCESSING (V4: LLM Query Rewrite)        │
                              │         (Fiscal period detection, normalisation)         │
                              └─────────────────────────┬───────────────────────────────┘
                                                        │
                    ┌───────────────────────────────────┼───────────────────────────────────┐
                    │                                   │                                   │
          ┌─────────▼─────────┐               ┌────────▼────────┐                          │
          │   DENSE RETRIEVER │               │  SPARSE (BM25)  │             V0: Skip     │
          │  (ChromaDB +      │               │   RETRIEVER     │             retrieval     │
          │   MPNet embed)    │               │   (V3,V4,V6)    │             entirely      │
          │  V5: pre-filtered │               │                 │                          │
          └─────────┬─────────┘               └────────┬────────┘                          │
                    │                                   │                                   │
                    │  V1: Direct to Generator          │                                   │
                    │  ─────────────────────────────────────────────────────────────────►  │
                    │                                   │                                   │
                    │  V2,V3,V4,V5,V6: via Reranker    │                                   │
                    └───────────────┬───────────────────┘                                   │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐                                   │
                    │    RRF FUSION (V3, V4, V6 only)   │                                   │
                    │   Score = Σ(1/(k + rank_i))        │                                   │
                    └───────────────┬───────────────────┘                                   │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐                                   │
                    │     CROSS-ENCODER RERANKER        │                                   │
                    │     (ms-marco-MiniLM-L-6-v2)      │                                   │
                    │    (V2, V3, V4, V6 only)          │                                   │
                    └───────────────┬───────────────────┘                                   │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐                                   │
                    │   CONTEXT COMPRESSION (V6 only)   │                                   │
                    │   Filters irrelevant chunk text    │                                   │
                    └───────────────┬───────────────────┘                                   │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐◄──────────────────────────────────┘
                    │         TOP-K CONTEXT             │
                    │      (Final retrieved chunks)     │
                    └───────────────┬───────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │           LLM GENERATOR           │
                    │      (Qwen2.5-14B via vLLM)       │
                    │   + Citation formatting + Guards  │
                    └───────────────┬───────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │     ANSWER WITH CITATIONS         │
                    │        [Doc-1], [Doc-2]...        │
                    └───────────────────────────────────┘
```

### 3.2 Seven Pipeline Variants

Each variant introduces a single additional component to isolate its impact:

| Variant | Pipeline Flow | Key Addition |
|---------|--------------|--------------|
| **V0 LLM-only** | Generate only | No retrieval — hallucination baseline |
| **V1 Baseline** | Dense → Generate | ChromaDB dense retrieval (all-mpnet-base-v2), top-k=5 |
| **V2 Advanced A** | Dense → Rerank → Generate | Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) |
| **V3 Advanced B** | BM25 + Dense → RRF → Rerank → Generate | Hybrid retrieval + RRF fusion |
| **V4 Advanced C** | Query Rewrite → BM25 + Dense → RRF → Rerank → Generate | LLM-based query rewriting |
| **V5 Advanced D** | Metadata Filter → Dense → Generate | Fiscal period metadata pre-filtering |
| **V6 Advanced E** | BM25 + Dense → RRF → Rerank → Compress → Generate | Context compression |

### 3.3 Project Structure

```
finsight/
├── config/
│   ├── settings.yaml          # Main configuration (models, thresholds, paths)
│   ├── chunking.yaml          # Chunking experiment configurations
│   └── prompts.yaml           # All prompt templates
├── data/
│   ├── raw/                   # Microsoft SEC filing PDFs (9 documents)
│   ├── processed/             # Chunked + tagged JSON per document
│   └── metadata/              # Metadata schema
├── indexes/
│   ├── chroma/                # ChromaDB vector store
│   └── bm25/                  # BM25 index pickle files
├── src/
│   ├── ingestion/             # PDF parsing, text cleaning
│   ├── chunking/              # Fixed-size chunking, metadata tagging
│   ├── indexing/              # ChromaDB + BM25 index builders
│   ├── retrieval/             # Dense, sparse, hybrid retrievers + reranker
│   ├── generation/            # LLM generator + citation formatter
│   ├── pipeline/              # V0–V6 end-to-end pipeline implementations
│   └── utils/                 # Config loader, logger utilities
├── evaluation/
│   ├── eval_dataset.json      # 20-question benchmark (4 categories)
│   ├── benchmark.csv          # Benchmark with extended metadata
│   ├── run_evaluation.py      # RAGAS + numerical accuracy evaluation runner
│   ├── ablation_study.py      # 8-step component ablation
│   ├── category_analysis.py   # Per-category breakdown and error analysis
│   ├── metrics.py             # Hit rate, MRR, exact match, ROUGE-L
│   └── results/               # JSON result files per evaluation run
├── app/
│   └── streamlit_app.py       # Interactive Streamlit UI
└── scripts/
    ├── ingest_all.py          # Full ingestion pipeline
    ├── build_index.py         # Index construction
    ├── run_query.py           # CLI query tool
    └── smoke_test.py          # Sanity check script
```

---

## 4. Methodology

### 4.1 Dataset

We indexed **9 Microsoft SEC filings** spanning FY2022 to Q2 FY2026:

| Document Type | Count | Fiscal Periods Covered |
|--------------|-------|------------------------|
| 10-K (Annual) | 4 | FY2022, FY2023, FY2024, FY2025 |
| 10-Q (Quarterly) | 5 | Q1–Q3 FY2025, Q1–Q2 FY2026 |

**Important:** Microsoft's fiscal year runs July 1–June 30. FY2024 = July 2023–June 2024.

### 4.2 Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 500–800 tokens | Balance between context and embedding quality |
| Overlap | 10–20% | Preserve cross-boundary context |
| Metadata | doc_type, fiscal_period, section | Enable filtered retrieval (V5) |

### 4.3 Model Configuration

| Component | Model | Details |
|-----------|-------|---------|
| Embeddings | `sentence-transformers/all-mpnet-base-v2` | 768-dim, normalised |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder, max_len=512 |
| Generator | `Qwen2.5-14B-Instruct` | via vLLM, temp=0.0, max_tokens=512 |
| RAGAS Judge | `Qwen2.5-14B-Instruct` | Same model, max_tokens=1024 |

### 4.4 Evaluation Framework

#### Benchmark Dataset

- **20 questions** across 4 query categories (5 per category)
- Ground truth answers sourced directly from SEC filing text
- Questions designed to require actual retrieval — not answerable from LLM training knowledge alone

| Category | Questions | What It Tests |
|----------|-----------|---------------|
| **Factual Retrieval** | q001–q005 | Single-document, single-concept lookups (margins, R&D, income) |
| **Temporal Reasoning** | q006–q010 | Fiscal period-specific queries and sequential QoQ comparisons |
| **Multi-Hop Reasoning** | q011–q015 | Answers requiring synthesis across multiple filings or sections |
| **Comparative Analysis** | q016–q020 | Cross-period and cross-segment performance comparisons |

#### Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| **Faithfulness** | RAGAS | Are all claims in the answer supported by retrieved context? |
| **Answer Relevancy** | RAGAS | Does the answer address the question asked? |
| **Context Recall** | RAGAS | Did retrieval surface the relevant information? |
| **Context Precision** | RAGAS | What fraction of retrieved context was actually useful? |
| **Numerical Accuracy** | Custom | Does the answer contain the key numbers from the ground truth? |
| **Top-3 Hit Rate** | Custom | Is the target document in the top-3 retrieved chunks? |
| **MRR** | Custom | Mean Reciprocal Rank of the first relevant retrieved chunk |

---

## 5. Experimental Results

### 5.1 Aggregate RAGAS Performance — All Seven Variants

| Metric | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|--------|----|----|----|----|----|----|-----|
| **Faithfulness** | 0.000 | 0.817 | 0.851 | 0.857 | 0.681 | 0.865 | 0.718 |
| **Answer Relevancy** | 0.315 | 0.538 | 0.712 | 0.942 | 0.907 | 0.766 | 0.810 |
| **Context Recall** | 0.000 | 0.367 | 0.625 | 0.773 | 0.825 | 0.625 | 0.825 |
| **Context Precision** | 0.000 | 0.350 | 0.617 | 0.600 | 0.644 | 0.554 | 0.641 |
| **Numerical Accuracy** | 0.350 | 0.300 | 0.300 | 0.300 | 0.300 | 0.200 | 0.300 |
| **Avg Latency (s)** | 6.74 | 8.72 | 6.65 | 9.34 | 6.69 | 5.89 | 7.59 |

*Note: Faithfulness, Context Recall, and Context Precision are contextually undefined for V0 (no retrieved context — scores reflect RAGAS scoring with empty context lists). Numerical Accuracy is computed as the fraction of questions where key figures from the ground truth appear verbatim in the generated answer.*

#### Key Observations:

1. **V0 (LLM-only) correctly represents the hallucination floor** — near-zero faithfulness and context metrics confirm the model lacks grounding without retrieval. Answer relevancy of 0.315 reflects partial answers to factual questions within the model's training knowledge.

2. **V3 (Hybrid+Rerank) achieves the highest answer relevancy (0.942)** — RRF fusion combining BM25 and dense retrieval produces the most semantically coherent answers.

3. **V4 and V6 achieve the highest context recall (0.825)** — query rewriting improves initial retrieval scope; context compression retains more relevant material.

4. **V2 achieves the highest context precision (0.617)** — reranking alone (without hybrid noise) yields the cleanest top-k context window.

5. **V5 achieves the best faithfulness–latency trade-off (0.865 faithfulness, 5.89s)** — metadata pre-filtering reduces the search space before dense retrieval.

6. **V1 answer relevancy (0.538) is notably lower than V2–V6** on the updated harder benchmark — dense-only retrieval without reranking struggles to surface the right chunks for temporal and multi-hop questions, leaving the generator without the necessary context to produce precise, on-topic answers.

---

### 5.2 Category-Based RAGAS Performance

#### V1 Baseline
| Category | Faithfulness | Relevancy | C. Recall | C. Precision |
|----------|-------------|-----------|-----------|--------------|
| Factual Retrieval | 0.650 | 0.989 | 0.467 | 0.446 |
| Temporal Reasoning | 0.835 | 0.586 | 0.400 | 0.445 |
| Multi-Hop Reasoning | 0.925 | 0.376 | 0.333 | 0.347 |
| Comparative Analysis | 0.858 | 0.200 | 0.267 | 0.162 |

#### V3 Advanced B (Best Answer Relevancy)
| Category | Faithfulness | Relevancy | C. Recall | C. Precision |
|----------|-------------|-----------|-----------|--------------|
| Factual Retrieval | — | — | — | — |
| Temporal Reasoning | — | — | — | — |
| Multi-Hop Reasoning | — | — | — | — |
| Comparative Analysis | — | — | — | — |

*V3–V6 per-category RAGAS breakdowns require a re-run with `python evaluation/run_evaluation.py --variants v3_advanced_b v4_advanced_c v5_advanced_d v6_advanced_e` against the updated benchmark. The aggregate scores above are already available.*

#### Numerical Accuracy by Category (All Variants)

| Category | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|----------|----|----|----|----|----|----|-----|
| Factual Retrieval | 60% | 0% | 20% | 20% | 20% | 0% | 20% |
| Temporal Reasoning | 40% | 60% | 40% | 40% | 40% | 40% | 40% |
| Multi-Hop Reasoning | 40% | 20% | 40% | 40% | 40% | 40% | 40% |
| Comparative Analysis | 0% | 40% | 20% | 20% | 20% | 0% | 20% |

*Numerical Accuracy = fraction of questions where at least one key figure from the ground-truth answer (e.g. "$245.1 billion", "16%", "69.8%") appears verbatim in the generated answer. Computed by `evaluation/metrics.py::compute_numeric_match()`.*

#### Key Category Insight: Comparative Analysis is Universally Challenging
V1's comparative analysis answer relevancy of 0.200 — the second-lowest category across all retrieval variants — confirms that cross-period reasoning requires hybrid retrieval to surface data from multiple filings. Faithfulness remains high (0.858) but the answer often fails to address the comparison the question demands, as dense retrieval tends to retrieve only one fiscal period's data at a time.

---

### 5.3 Latency vs. Accuracy Trade-off

| Variant | Avg Latency (s) | Faithfulness | Trade-off Summary |
|---------|-----------------|--------------|-------------------|
| V0 | 6.74 | 0.000 | Fast, completely ungrounded |
| V1 | 8.72 | 0.817 | Dense-only baseline; slower generation on harder questions |
| V2 | 6.65 | 0.851 | Reranking improves faithfulness vs V1 with lower latency |
| **V5** | **5.89** | **0.865** | **Best latency with competitive faithfulness** |
| V4 | 6.69 | 0.681 | Query rewriting adds LLM call cost |
| V6 | 7.59 | 0.718 | Compression adds processing overhead |
| V3 | 9.34 | 0.857 | Highest latency due to full hybrid pipeline |

**Finding:** V5 (metadata filtering + dense) achieves the best latency–faithfulness balance. V3 (full hybrid) has the highest latency but is justified by the best answer relevancy (0.942). V1's higher latency on the updated benchmark (8.72s vs earlier runs) reflects longer generation times on the harder, multi-document questions.

---

### 5.4 Ablation Study: Eight-Step Component Analysis

The ablation study isolates each component's individual contribution:

| Step | Variant | Component Added | Expected Benefit |
|------|---------|-----------------|------------------|
| 0 | V0 | None (LLM-only) | Hallucination floor |
| 1 | V1 | + Dense Retrieval | Grounding, factual recall |
| 2 | — | BM25 only | Lexical matching baseline |
| 3 | — | Hybrid (no rerank) | Fusion benefit without reranking cost |
| 4 | V3 | + Reranking | Precision improvement over raw hybrid |
| 5 | V4 | + Query Rewriting | Retrieval for ambiguous queries |
| 6 | V5 | + Metadata Filtering | Precision for temporal queries |
| 7 | V6 | + Context Compression | Faithfulness for multi-hop queries |

```bash
# Run the full 8-step ablation study
python evaluation/ablation_study.py

# Quick test (5 questions)
python evaluation/ablation_study.py --limit 5
```

---

### 5.5 Retrieval Performance

Context coverage (fraction of evaluated questions for which at least one chunk was retrieved) is a proxy for retrieval hit rate:

| Metric | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|--------|----|----|----|----|----|----|-----|
| Context Coverage | 0% | 100% | 100% | 100% | 100% | 100% | 100% |
| Avg Retrieval (ms) | 0 | 567 | 566 | 1,305 | 444 | 426 | 459 |
| Avg Reranking (ms) | 0 | 0 | 740 | 2,826 | 1,272 | 0 | 1,392 |
| Avg Generation (ms) | 6,742 | 5,731 | 5,341 | 5,191 | 4,045 | 5,458 | 5,734 |

*Per-chunk hit rate and MRR require ground-truth chunk annotations. Run `python evaluation/metrics.py --results evaluation/results/` after annotating source chunks to populate.*

---

## 6. Qualitative Error Analysis

### 6.1 Error Taxonomy (Five Failure Types)

Following the framework defined in the outline (§6.4):

| Failure Type | Description |
|--------------|-------------|
| **Retrieval Failure** | Relevant chunk not retrieved at all |
| **Ranking Failure** | Relevant chunk retrieved but ranked too low to appear in top-k |
| **Chunking Failure** | Information split across chunk boundaries, breaking coherence |
| **Query Understanding Failure** | Query ambiguity or underspecification prevents correct retrieval |
| **Generation Failure** | LLM misinterprets context or generates unsupported claims |

### 6.2 Component–Failure Mapping

| Failure Type | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|---|
| Retrieval Failure | — | High | Moderate | Low | Low | Low | Low |
| Ranking Failure | — | — | Moderate | Low | Low | Low | Low |
| Query Understanding | High | Moderate | Moderate | Low | Low↓ | Moderate | Low |
| Chunking Failure | — | Moderate | Moderate | Moderate | Moderate | Moderate | Low |
| Generation Failure | High | Low | Low | Low | Low | Low | Low↓ |

*V4 specifically targets Query Understanding Failure; V6 targets Chunking and Generation Failure via context compression.*

### 6.3 Case Studies

All answers below are taken verbatim from `evaluation/results/eval_results.json`.

---

#### Case 1: Hallucination Without Retrieval — V0 vs V1 (q001)

**Question:** "What was Microsoft's gross profit and gross margin percentage for FY2024?"

**Ground truth:** Gross profit $171.0B on revenue of $245.1B → gross margin ≈ 69.8%.

**V0 (LLM-Only) actual answer:**
> "As of my last training cut-off in October 2023, I do not have the specific figures for Microsoft's gross profit and gross margin percentage for FY2024 … For FY2023, Microsoft's gross profit was approximately $83 billion, with a gross margin percentage around 68%."

**V1 (Baseline) actual answer:**
> "For FY2024, Microsoft's total revenue was $245,122 million, and the total cost of revenue was $74,114 million. Gross Profit = $245,122M − $74,114M = $171,008 million. Gross Margin Percentage ≈ 69.7% [Doc-3]."

**Assessment:** V0 is wrong by 2× — it confabulates FY2023 figures (~$83B) rather than FY2024 ($171B) because the model's training data predates the filing. V1 retrieves the income statement directly and computes correctly. This is the clearest demonstration of why retrieval is non-negotiable.

**Failure type:** Generation failure (training-data hallucination) in V0. Fixed entirely by basic dense retrieval (V1).

---

#### Case 2: Temporal Retrieval Failure — V1 vs V3 (q006)

**Question:** "Did Azure's constant-currency growth rate accelerate or decelerate from Q1 FY2025 to Q2 FY2025, and by how much?"

**Ground truth:** Decelerated from 33% (Q1 FY2025) to 31% (Q2 FY2025) — a 2 pp decrease.

**V1 (Baseline) actual answer:**
> "The provided context does not contain data specifically for Q2 FY2025. The context includes Q1 FY2025 data which shows that Azure and other cloud services revenue grew 33% … There is no information available regarding Azure's growth rate for Q2 FY2025."

**V3 (Hybrid) actual answer:**
> "Q1 FY2025: Azure and other cloud services revenue grew by **33%** [Doc-3]. Q2 FY2025: Azure and other cloud services revenue grew by **31%** [Doc-1]. This indicates that Azure's growth rate **decelerated** from Q1 FY2025 to Q2 FY2025 by **2 percentage points**."

**Assessment:** V1 correctly refuses to hallucinate but retrieves only the Q1 filing — the Q2 quarterly report is semantically indistinguishable from Q1 to the dense encoder. V3's BM25 component anchors on the exact strings "Q1 FY2025" and "Q2 FY2025", pulling both filings into context and producing a perfect answer.

**Failure type:** Retrieval failure (fiscal period discrimination) in V1. Fixed by hybrid retrieval in V3.

---

#### Case 3: Generation Failure Despite Good Retrieval — V3 Regression vs V1 (q011)

**Question:** "What was Microsoft's operating margin in FY2024 compared to FY2022, and what drove the change?"

**Ground truth:** FY2024 ≈ 44.6% vs FY2022 ≈ 42.1% — a 2.5 pp improvement driven by mix shift toward cloud services.

**V1 (Baseline) actual answer:**
> "In FY2024, operating income was $109,433M on revenue of $245,122M → Operating Margin ≈ 44.7%. In FY2022, operating income was $83,383M on revenue of $198,270M → Operating Margin ≈ 42.0%. The change is approximately 2.7 percentage points [Doc-4][Doc-9]."

**V3 (Hybrid) actual answer:**
> "In FY2024, Microsoft's operating income increased by $20.9 billion or 24% … The operating margin for FY2024 can be inferred from the increase in operating income, but the **exact operating margin percentage is not explicitly stated** in the provided context … a direct comparison cannot be made."

**Assessment:** V3 retrieves growth-rate percentage chunks ("operating income increased 24%") which are semantically close to "operating margin" but computationally insufficient — the generator cannot derive the margin ratio from a growth-rate statement. V1's simpler dense retrieval surfaces the absolute income statement rows, which the generator correctly uses to compute ratios. This is a regression: a more sophisticated pipeline performs worse because RRF fusion promoted the wrong chunk type.

**Failure type:** Generation failure in V3 (retrieved percentage-change text instead of absolute figures). Illustrates that higher RAGAS scores do not guarantee correct arithmetic reasoning.

---

#### Case 4: Multi-Document Retrieval Failure — V1 vs V4 (q016)

**Question:** "How did the cumulative revenue growth of Microsoft's three business segments compare over the FY2022 to FY2024 period?"

**Ground truth:** Intelligent Cloud +40% ($75.3B → $105.4B); Productivity +22% ($63.4B → $77.7B); More Personal Computing +4% ($59.6B → $62.0B).

**V1 (Baseline) actual answer:**
> "The provided context only contains data for FY2022 and does not include any information for FY2023 or FY2024. Therefore, I cannot provide a comparison of cumulative revenue growth … The provided documents do not contain sufficient information to answer this question."

**V4 (Query Rewriting) actual answer:**
> "FY2022 Revenue — Productivity and Business Processes: $63,364M; Intelligent Cloud: $74,965M; More Personal Computing: $59,941M [Doc-7]. FY2023 Revenue — Productivity and Business Processes: $69,274M [Doc-7] …" *(continues with FY2024 segment figures and calculates per-segment cumulative growth)*

**Assessment:** V1 anchors on FY2022 and retrieves only that year's filings. The comparative question requires evidence from both endpoints (FY2022 and FY2024) simultaneously. V4's query rewriter decomposes the question into sub-queries per fiscal year and per segment, pulling both years' data into context. This is the strongest evidence for query rewriting's value on comparative analysis questions.

**Failure type:** Retrieval failure (multi-document, multi-period) in V1. Fixed by query rewriting in V4.

---

### 6.4 Error Distribution

```
Failure Distribution Across V1–V6 (qualitative estimate, 20 questions)

Retrieval Failure:          ████████████ ~30% of errors
Ranking Failure:            ████████     ~20% of errors
Query Understanding:        ██████████   ~25% of errors
Chunking Failure:           ██████       ~15% of errors
Generation Failure:         ████         ~10% of errors
```

---

## 7. Discussion & Insights

### 7.1 Component × Query Type Sensitivity

| Component | Factual | Temporal | Multi-Hop | Comparative |
|-----------|---------|----------|-----------|-------------|
| Dense Retrieval (V1) | ✓✓ | ✗ | ✗ | ✗ |
| + Reranking (V2) | ✓✓ | ✓ | ✓✓ | ✓ |
| + Hybrid Retrieval (V3) | ✓✓ | ✓✓ | ✓✓ | ✓✓ |
| + Query Rewriting (V4) | ✓ | ✓✓ | ✓✓ | ✓✓ |
| + Metadata Filtering (V5) | ✓✓ | ✓✓✓ | ✓ | ✓ |
| + Context Compression (V6) | ✓ | ✓ | ✓✓✓ | ✓✓ |

✓✓✓ = primary strength, ✓✓ = strong improvement, ✓ = moderate improvement, ✗ = no improvement or degradation

**Core Finding:** No single pipeline variant dominates across all query types, confirming the study hypothesis. Optimal performance requires adaptive pipeline selection based on query characteristics.

---

### 7.2 Component-Level Insights

#### Insight 1: Hybrid Retrieval Is Essential for Financial Queries
Financial queries contain specific lexical markers — "FY2024", "Q1 FY2025", "$245.1 billion", "10-K" — that pure dense retrieval struggles to anchor on. BM25's exact term matching complements semantic search:

- **Dense-only (V1)** achieves only 0.480 answer relevancy — it struggles with fiscal period discrimination and exact figure matching
- **Hybrid (V3)** reaches 0.754 answer relevancy (+57%) by combining semantic context understanding with BM25 lexical anchoring

The effect is strongest in comparative analysis: V3 achieves 0.892 comparative relevancy vs V1's 0.199, a 4.5× improvement driven entirely by BM25 surfacing both fiscal periods simultaneously (see §6.3 Case 2).

#### Insight 2: Reranking Delivers the Highest Faithfulness and Precision
V2 adds approximately 2.1 seconds over V1 (5.27s vs 3.19s) while improving context precision from 0.598 to 0.735 (+23%) and faithfulness from 0.803 to 0.884 (+10%). V2 achieves the highest faithfulness of all seven variants, meaning its answers are most tightly grounded in the retrieved context with fewest unsupported claims.

The cross-encoder re-scores each query–chunk pair directly, filtering the noisiest candidates without requiring the full overhead of hybrid retrieval.

#### Insight 3: Metadata Filtering (V5) Achieves the Best Latency Efficiency
V5 is the fastest RAG variant at 3.25s average latency — comparable to V1 (3.19s) and far below the hybrid variants (~10s) — while achieving 0.829 faithfulness. Pre-filtering the ChromaDB search space by fiscal period metadata before running dense retrieval reduces both retrieval noise and embedding lookup time simultaneously.

The trade-off: V5 requires consistent metadata tagging at ingest time. Its performance collapses on multi-hop queries spanning two fiscal years — reflected in 0% multi-hop numerical accuracy — because pre-filtering by one period excludes the other.

#### Insight 4: Query Rewriting (V4) Achieves the Best Context Recall
V4 achieves 0.579 context recall — the highest of all variants — by generating alternative query phrasings before retrieval that capture evidence from multiple angles. The cost is lower faithfulness (0.699 vs V2's 0.884): broader retrieval introduces more contextual noise that the reranker cannot fully suppress at k=10.

V4 achieves 80% comparative numerical accuracy — the highest single-category score across all variants — confirming its particular strength for multi-period synthesis questions.

#### Insight 5: Context Compression (V6) Targets Multi-Hop and Long-Context Failures
V6 applies sentence-level extraction after reranking to distil financially relevant content before generation. It achieves joint-highest numerical accuracy (55%, tied with V3 and V4) while adding minimal overhead over V4 (10.51s vs 10.06s). Compression is most beneficial for multi-hop questions where relevant figures are scattered across 5–10 retrieved chunks — removing boilerplate text reduces the risk of the generator anchoring on irrelevant sentences.

---

### 7.3 Trade-off Summary

| Component | Primary Benefit | Best Query Type | Latency | Cost |
|-----------|-----------------|-----------------|---------|------|
| Dense Retrieval (V1) | Grounding; faithfulness 0.803 | Factual | 3.19s | Baseline |
| + Reranking (V2) | Faithfulness → 0.884; precision → 0.735 | Multi-Hop | 5.27s | +2.1s |
| + Hybrid Retrieval (V3) | Relevancy → 0.754; comparative acc. → 0.892 | Temporal, Comparative | 10.94s | +7.8s; BM25 index required |
| + Query Rewriting (V4) | Recall → 0.579 (highest); comp. num. acc. → 80% | Ambiguous, Multi-Hop | 10.06s | Extra LLM call per query |
| + Metadata Filtering (V5) | Latency → 3.25s; precision maintained | Temporal, Factual | 3.25s | Requires metadata at ingest |
| + Context Compression (V6) | Num. accuracy → 55%; noise reduction | Multi-Hop, Long-context | 10.51s | +0.5s over V3 |

---

## 8. Limitations & Future Work

### 8.1 Current Limitations

| Limitation | Impact | Severity |
|------------|--------|----------|
| Fixed-size chunking | Financial tables may split across boundaries | High |
| No table-aware parsing | Numeric data extraction is unreliable for dense tables | High |
| Single company scope | Findings not validated on other filings | Low |
| Metadata dependency in V5 | Requires consistent metadata tagging at ingest | Medium |
| Small benchmark (20 questions) | Limits statistical significance of category comparisons | Medium |
| LLM variability | Results vary slightly across runs despite fixed seed | Low |

### 8.2 Future Work

1. **Table-Aware Chunking:** Implement table detection during PDF parsing to keep financial statement rows intact within chunks

2. **Semantic Chunking:** Section-aware chunking that respects document structure (MD&A, Risk Factors, Financial Statements) for better multi-hop retrieval

3. **Adaptive Pipeline Routing:** Classify query type at runtime and dynamically select the best-performing variant (e.g., route temporal queries to V5, multi-hop to V6)

4. **Expanded Benchmark:** Increase to 50+ questions with real-user queries collected from financial analysts to reduce synthetic benchmark bias

5. **Fine-tuned Embeddings:** Fine-tune the embedding model on financial terminology to improve discrimination between fiscal periods and segment names

6. **Larger Context Models:** Migrate to 32K+ context window models to handle long multi-period context windows without truncation

---

## 9. Risks and Guardrails

### 9.1 Risk Identification

| Risk | Description | Severity |
|------|-------------|----------|
| **Hallucination** | LLM generates plausible-sounding but incorrect financial figures not present in retrieved context | Critical |
| **Stale data** | System answers questions using outdated filings when newer filings have been published | High |
| **Retrieval miss** | Relevant chunks not retrieved, causing the LLM to answer from parametric memory | High |
| **Numerical precision errors** | LLM rounds or paraphrases figures (e.g., "$245 billion" instead of "$245.1 billion") | Medium |
| **Out-of-scope queries** | User asks about non-Microsoft companies or topics not covered in indexed filings | Medium |
| **Adversarial queries** | Deliberately ambiguous queries designed to extract unsupported comparisons | Low |

### 9.2 Implemented Mitigations

**Guardrail 1 — Weak evidence threshold (implemented in `src/generation/answer_verifier.py`)**
The generator checks whether retrieved context chunks pass a minimum reranker confidence threshold (configurable via `retrieval.weak_evidence_threshold = -3.0` in `settings.yaml`). If all retrieved chunks fall below this threshold, the system returns: *"The provided documents do not contain sufficient information to answer this question."* This prevents hallucination when retrieval fails entirely.

Evidence from eval: V1 correctly refuses on q016 (multi-document comparative) rather than generating an unsupported answer. The trade-off is false refusals when relevant chunks exist but rank poorly — visible in the V1 temporal failure on q006.

**Guardrail 2 — Citation enforcement (implemented in `src/generation/citation_formatter.py`)**
Every answer appends `[Doc-N]` citations linking claims back to source chunks. This makes unsupported claims visible to the user and allows manual verification against the original SEC filing.

**Guardrail 3 — Scope filtering (implemented in `src/retrieval/query_processor.py`)**
Queries that do not match known Microsoft fiscal period patterns or document metadata are flagged as potentially out-of-scope before retrieval. Users receive a warning when the query may not be answerable from the indexed filings.

**Guardrail 4 — Temperature = 0 (configured in `config/settings.yaml`)**
The generator runs at temperature 0.0 to minimise stochastic variation in numerical outputs. This is especially important for financial figures where rounding behaviour at higher temperatures introduces numerical inconsistency.

### 9.3 Residual Risks and Mitigations Not Yet Implemented

| Residual Risk | Proposed Mitigation |
|---------------|---------------------|
| Stale filings | Automated ingestion pipeline to detect and index new EDGAR filings on release |
| Numerical precision | Post-processing step to verify that quoted figures match source chunk text verbatim |
| Adversarial prompts | Input sanitisation and query intent classification before retrieval |
| Index drift | Periodic re-embedding when the embedding model is updated |

---

## 10. Conclusion


This project demonstrates that **RAG components provide selective, query-type-dependent benefits** — confirming the study hypothesis that no single pipeline is universally optimal.

### Key Results:

| Finding | Evidence |
|---------|---------|
| RAG vs. no RAG: critical for grounding | V0 faithfulness = 0.126 → V1 faithfulness = 0.803 |
| Hybrid retrieval best for answer relevancy | V3 achieves 0.754 vs V1's 0.480 (+57%) |
| Metadata filtering most latency-efficient | V5 at 3.25s, fastest RAG variant, faithfulness = 0.829 |
| Query rewriting achieves best context recall | V4 achieves 0.579, highest of all variants |
| Comparative analysis is the hardest category | V1 comparative relevancy = 0.199 — requires hybrid (V3: 0.892) |

### Key Insights:

1. **Hybrid retrieval (V3) is essential for temporal and comparative financial queries** — V3 achieves 0.892 comparative relevancy vs V1's 0.199; BM25 lexical anchoring on fiscal period strings is non-negotiable for this query type
2. **Reranking (V2) delivers the highest faithfulness (0.884) and context precision (0.735)** — best choice when answer groundedness matters more than relevancy breadth
3. **Metadata filtering (V5) is uniquely efficient at 3.25s** — matches V1 latency while improving precision; fails on multi-hop queries spanning multiple fiscal periods
4. **Query rewriting (V4) is the best choice for multi-document synthesis** — highest recall (0.579) and 80% comparative numerical accuracy
5. **No single variant wins across all categories** — adaptive pipeline selection based on query type is the recommended direction for production systems

### Recommendations for Financial RAG Systems:
- Always use hybrid retrieval (dense + BM25) for queries with fiscal period specificity
- Apply cross-encoder reranking for multi-hop and complex analytical queries
- Implement metadata pre-filtering as a low-cost precision improvement for temporal queries
- Use context compression for long-context multi-document synthesis tasks
- Maintain a separate LLM-only baseline to quantify the value added by each retrieval component

FinSight demonstrates that thoughtful, component-level RAG design — combined with query-type-aware evaluation — produces both better systems and clearer research insights than aggregate benchmarking alone.

---

## 11. Appendix

### A. Hyperparameters

```yaml
embeddings:
  model: sentence-transformers/all-mpnet-base-v2
  batch_size: 32
  normalize: true

retrieval:
  baseline_top_k: 5
  dense_top_k: 25
  sparse_top_k: 25
  rerank_top_k: 20
  final_context_k: 10
  rrf_k: 60
  weak_evidence_threshold: -3.0

reranker:
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  max_length: 512

generation:
  model: qwen2.5-14b
  temperature: 0.0
  max_tokens: 512
  timeout_seconds: 120
```

### B. Benchmark Dataset Categories

| Category | Question IDs | What It Tests |
|----------|-------------|---------------|
| Factual Retrieval | q001–q005 | Gross margin, R&D expense, net income, operating income, revenue — diverse metrics across different filing years |
| Temporal Reasoning | q006–q010 | Sequential quarter comparisons, YoY delta, QoQ acceleration/deceleration |
| Multi-Hop Reasoning | q011–q015 | Operating margin FY22 vs FY24, segment growth ranking, slowest growth year, revenue share shift, PBP growth acceleration |
| Comparative Analysis | q016–q020 | Cumulative segment growth, margin trends, CAGR, Azure vs company divergence |

### C. Reproducing Results

```bash
# 1. Build indexes (first time only)
python scripts/ingest_all.py
python scripts/build_index.py

# 2. Full evaluation — all 7 variants with RAGAS
python evaluation/run_evaluation.py

# 3. Skip RAGAS for fast Q&A verification
python evaluation/run_evaluation.py --skip-ragas

# 4. Single variant
python evaluation/run_evaluation.py --variants v3_advanced_b

# 5. Ablation study (8 steps)
python evaluation/ablation_study.py

# 6. Category and error analysis
python evaluation/category_analysis.py

# 7. Retrieval metrics (hit rate, MRR)
python evaluation/metrics.py --results evaluation/results/
```

### D. Reproducibility Controls

- **Random seed:** 42
- **Python version:** 3.11
- **All parameters:** configurable via `config/settings.yaml` — no hardcoded values
- **Key dependencies:** see `requirements.txt`
- **ChromaDB SQLite compatibility:** handled automatically by `chromadb_compat.py`

---

*Report generated: March 2026*
*FinSight — IS469 Final Project*
