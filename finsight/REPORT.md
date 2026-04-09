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
1. **End-to-end RAG implementation** with seven configurable pipelines covering a full component ablation: no retrieval (V0), dense-only (V1), dense + reranking (V2), hybrid+reranking (V3), query rewriting (V4), metadata filtering (V5), and context compression (V6)
2. **Controlled experimental design** isolating the contribution of each RAG component to performance across four query categories
3. **Comprehensive evaluation** using RAGAS metrics, numerical accuracy, and qualitative error analysis
4. **Actionable insights** on which RAG components benefit which query types

---

## 2. Problem Statement & Objectives

### Problem Statement
Extracting specific financial insights from SEC filings requires navigating hundreds of pages of complex financial documents, understanding Microsoft's fiscal calendar (July–June fiscal year), cross-referencing information across multiple filing periods, and distinguishing between similar-sounding metrics across different time periods.

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
                              │         (Fiscal period detection, normalisation)        │
                              └─────────────────────────┬───────────────────────────────┘
                                                        │
                    ┌───────────────────────────────────┼───────────────────────────────────┐
                    │                                   │                                   │
          ┌─────────▼─────────┐                ┌────────▼────────┐                          │
          │   DENSE RETRIEVER │                │  SPARSE (BM25)  │            V0: Skip      │
          │  (ChromaDB +      │                │   RETRIEVER     │            retrieval     │
          │   MPNet embed)    │                │   (V3,V4,V6)    │            entirely      │
          │  V5: pre-filtered │                │                 │                          │
          └─────────┬─────────┘                └────────┬────────┘                          │
                    │                                   │                                   │
                    │  V1: Direct to Generator          │                                   │
                    │──────────────────────────────────────────────────────────────────────►│
                    │                                   │                                   │
                    │  V2,V3,V4,V6: via Reranker        │                                   │
                    └───────────────┬───────────────────┘                                   │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐                                   │
                    │    RRF FUSION (V3, V4, V6 only)   │                                   │
                    │      Score = Σ(1/(k + rank_i))    │                                   │
                    └───────────────┬───────────────────┘                                   │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐                                   │
                    │     CROSS-ENCODER RERANKER        │                                   │
                    │     (ms-marco-MiniLM-L-6-v2)      │                                   │
                    │     (V2, V3, V4, V6 only)         │                                   │
                    └───────────────┬───────────────────┘                                   │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐                                   │
                    │   CONTEXT COMPRESSION (V6 only)   │                                   │
                    │   Filters irrelevant chunk text   │                                   │
                    └───────────────┬───────────────────┘                                   │
                                    │                                                       │
                    ┌───────────────▼───────────────────┐◄──────────────────────────────────┘
                    │         TOP-K CONTEXT             │
                    │     (Final retrieved chunks)      │
                    └───────────────┬───────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │           LLM GENERATOR           │
                    │    (gpt-4o-mini via OpenAI API)   │
                    │   + Citation formatting + Guards  │
                    └───────────────┬───────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │     ANSWER WITH CITATIONS         │
                    │        [Doc-1], [Doc-2]...        │
                    └───────────────────────────────────┘
```

---

### 3.2 Seven Pipeline Variants

Each variant introduces a single additional component to isolate its impact:

| Variant | Pipeline Flow | Key Addition | Purpose |
|---------|--------------|--------------|---------|
| **V0 LLM-only** | Generate only | No retrieval | Hallucination baseline |
| **V1 Baseline** | Dense → Generate | ChromaDB dense retrieval (all-mpnet-base-v2), top-k=5 | Establishes retrieval floor |
| **V2 Advanced A** | Dense → Rerank → Generate | Cross-encoder reranking (ms-marco-MiniLM-L-6-v2), top-k=5 | Improve ranking precision |
| **V3 Advanced B** | BM25 + Dense → RRF → Rerank → Generate | Hybrid retrieval + RRF fusion | Lexical + semantic combination |
| **V4 Advanced C** | Query Rewrite → BM25 + Dense → RRF → Rerank → Generate | LLM-based query rewriting | Handles ambiguous queries |
| **V5 Advanced D** | Metadata Filter → Dense → Generate | Fiscal period metadata pre-filtering | Low-latency precision boost |
| **V6 Advanced E** | BM25 + Dense → RRF → Rerank → Compress → Generate | Context compression | Reduce noise for multi-hop |

---

### 3.3 Model Configuration
| Component | Model | Details |
|-----------|-------|---------|
| Embeddings | sentence-transformers/all-mpnet-base-v2 | 768-dim, normalised |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder, max_len=512 |
| Generator | gpt-4o-mini | via OpenAI-compatible API, temp=0.0, max_tokens=512 |
| RAGAS Judge | gpt-4o-mini | Same configured judge model, max_tokens=1024 |

---

### 3.4 Project Structure

```
finsight/
├── config/
│   ├── settings.yaml          # Main configuration (models, thresholds, paths)
│   ├── chunking.yaml          # Chunking experiment configurations
│   └── prompts.yaml           # All prompt templates
├── data/
│   ├── raw/                   # Optional raw SEC filing PDFs for full rebuilds
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
│   ├── ablation_study.py      # Four-configuration ablation
│   ├── category_analysis.py   # Per-category breakdown and error analysis
│   └── results/               # JSON result files per evaluation run
├── app/
│   └── streamlit_app.py       # Interactive Streamlit UI
├── scripts/
│   ├── ingest_all.py          # Full ingestion pipeline
│   ├── build_index.py         # Index construction
│   ├── run_query.py           # CLI query tool
│   └── smoke_test.py          # Sanity check script
└─ README.md
```

---

## 4. Methodology
### 4.1 Dataset

We indexed **9 Microsoft SEC filings** spanning FY2022 to Q2 FY2026:

| Document Type | Count | Fiscal Periods Covered |
|--------------|-------|------------------------|
| 10-K (Annual) | 4 | FY2022, FY2023, FY2024, FY2025 |
| 10-Q (Quarterly) | 5 | Q1–Q3 FY2025, Q1–Q2 FY2026 |

**Important note:** Microsoft's fiscal year runs July 1–June 30. FY2024 = July 2023–June 2024.

---

### 4.2 Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 500–800 tokens | Balance between context and embedding quality |
| Overlap | 10–20% | Preserve cross-boundary context |
| Metadata | doc_type, fiscal_period, section | Enable filtered retrieval (V5) |

---

### 4.3 Evaluation Framework

#### Benchmark Dataset

- **20 questions** across 4 query categories (5 per category)
- Ground truth answers sourced directly from SEC filing text
- Questions intended to require actual retrieval, though a few remained partially answerable from LLM training knowledge alone

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

---

## 5. Experimental Results

*Source note: Unless otherwise stated, all values in Sections 5.1-5.4 are taken directly from `evaluation/results/eval_results.json` (rounded for display). Section 5.5 uses `evaluation/results/ablation_results.json`. In the tables below, `0.000` means the saved run recorded an actual zero score, while `n/a` means the saved JSON stores `NaN` for that slice, so the metric is unavailable rather than zero.*

### 5.1.1 Aggregate RAGAS Performance — All Seven Variants

| Metric | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|--------|----|----|----|----|----|----|----|
| **Faithfulness** | 0.098 | 0.738 | 0.843 | 0.786 | 0.760 | 0.804 | 0.700 |
| **Answer Relevancy** | 0.360 | 0.483 | 0.478 | 0.711 | 0.784 | 0.483 | 0.738 |
| **Context Recall** | 0.000 | 0.592 | 0.367 | 0.250 | 0.463 | 0.517 | 0.383 |
| **Context Precision** | 0.000 | 0.607 | 0.741 | 0.706 | 0.686 | 0.602 | 0.574 |
| **Numerical Accuracy** | 0.250 | 0.350 | 0.450 | 0.600 | 0.500 | 0.300 | 0.550 |
| **Avg Latency (s)** | 3.94 | 3.46 | 4.50 | 7.71 | 8.51 | 3.63 | 10.52 |

*Note: V0 has no retrieved context, so its context recall and precision are 0.000. Its non-zero faithfulness (0.098) reflects partially verifiable claims from parametric memory rather than grounded retrieval. Numerical Accuracy is defined as the fraction of answers containing at least one ground-truth figure verbatim.*

#### Results Summary

* Faithfulness peaks at V2 (0.843), while answer relevancy peaks at V4 (0.784)
* Multi-stage pipelines (V3, V4, V6) consistently outperform dense-only systems on relevancy and numerical accuracy
* Latency increases with pipeline complexity, ranging from 3.46s (V1) to 10.52s (V6)
* V5 achieves strong faithfulness (0.804) at near-baseline latency (3.63s)

#### Key Observations

* **Retrieval is necessary, but parametric knowledge can confound evaluation** — V0 confirms that retrieval is required for verifiable answers: it achieves near-zero faithfulness (0.098) and no contextual grounding. However, its 0.250 numerical accuracy shows that the model can still answer some questions from parametric memory.
  * This highlights a key evaluation risk: questions answerable from training data weaken the measured impact of retrieval and should be filtered where possible.

* **Precision vs. breadth is the central trade-off** — Two distinct pipeline behaviours emerge:
  * Precision-focused pipelines (V1, V2, V5)
    * High faithfulness (0.738–0.843)
    * Lower relevancy (~0.48)
    * Retrieve limited but well-grounded evidence
  * Breadth-focused pipelines (V3, V4, V6)
    * Higher relevancy (0.711–0.784)
    * Slightly lower faithfulness (0.700–0.786)
    * Retrieve broader but noisier evidence
  * This trade-off is captured by the faithfulness–relevancy gap:
    * V2: +0.365 (high precision, incomplete answers)
    * V4: −0.024, V6: −0.038 (high coverage, mild hallucination risk)
  * **When faithfulness exceeds relevancy**, systems under-retrieve. **When relevancy exceeds faithfulness**, systems over-synthesise beyond the evidence.

* **V2 is the most cost-effective single component addition for faithfulness** — V2 raises faithfulness from 0.738 to 0.843 (+0.105) and context precision from 0.607 to 0.741 (+0.134) at a latency cost of only +1.04s (reranking adds 1,349ms). Expressed as a rate, reranking delivers 0.101 faithfulness units per second of added latency.

* **Reranking improves grounding, not completeness** — Adding a cross-encoder reranker increases:
	* V2 vs V1:
		* +0.105 faithfulness
		* +0.134 context precision
		* ~no change in relevancy (0.478 vs 0.483)
	* Reranking reorders evidence but cannot recover missing information. Relevancy gains therefore require better retrieval coverage, not better ranking alone. 

* **Numerical accuracy is strongest in the multi-stage retrieval pipelines** — V3 leads at 0.600 and V6 follows at 0.550, suggesting that harder numerical questions benefit from richer evidence sets. 
  * Precision-optimising components may help the generator make accurate claims about what it has, but if the retrieved pool is narrow, key numerical figures are absent. 

* **Latency scales with pipeline complexity** 
  * Fastest grounded systems: V1 (3.46s), V5 (3.63s)
  * Moderate cost: V2 (4.50s)
  * High latency: V3–V6 (7.71–10.52s)
  * Each additional retrieval or processing stage introduces measurable latency without guaranteed gains in answer quality.

### 5.1.2 Key Summary

Aggregate results show that no single pipeline dominates across all metrics. Instead:
* Reranking (V2) is the most efficient way to improve faithfulness
* Query expansion and hybrid retrieval (V3–V4) are necessary for higher answer relevancy
* Low-latency pipelines (V1, V5) provide strong grounding but struggle with complex queries

Overall, system performance is governed by a fundamental trade-off between precision (grounded correctness) and breadth (answer completeness), with latency increasing as systems attempt to optimise both simultaneously.

---

### 5.2 Category-based RAGAS Performance

*Note: The category tables below are taken directly from the `aggregate.category_ragas` fields in `evaluation/results/eval_results.json`. A `0.00` score indicates the saved run recorded a complete failure on that dimension. An `n/a` cell indicates the saved JSON contains `NaN` for that metric/category slice, so the score should be treated as unavailable rather than zero.*

---

### 5.2.1 Factual Retrieval RAGAS Performance

| Variant | Faithfulness | Relevancy | Context Recall | Context Precision |
|---------|--------------|-----------|----------------|-------------------|
| V0 | 0.000 | 0.378 | 0.000 | 0.000 |
| V1 | 0.733 | 0.988 | 0.667 | 0.593 |
| V2 | 0.667 | 0.988 | 0.533 | n/a |
| V3 | 0.633 | 0.984 | 0.400 | 0.497 |
| V4 | 0.700 | 0.988 | 0.600 | 0.482 |
| V5 | 0.700 | 0.988 | 0.633 | 0.549 |
| V6 | 0.714 | 0.788 | 0.600 | 0.510 |

#### Key Observations

* **Dense retrieval is sufficient for factual queries**
	* V1, V4, V5 all achieve **0.988 relevancy**
	* Once the correct chunk is retrieved, factual questions are easy to answer. Complex retrieval strategies provide little additional benefit

* **Reranking does not improve factual end-task performance**
	* V2 vs V1:
		* Same relevancy (0.988)
		* Lower recall (0.533 vs 0.667)
		* Context precision unavailable (`n/a`)
	* Reranking does not improve outcomes when retrieval already succeeds

* **Hybrid retrieval helps numerical grounding, not relevance**
	* V3:
		* Slightly lower relevancy (0.984)
		* Higher numerical accuracy (60%)
	* Lexical + semantic signals help recover exact figures, but do not meaningfully improve answer selection

* **Context compression reduces answer completeness**
	* V6:
		* Relevancy drops to **0.788**
		* Recall remains relatively high (0.600)
	* Compression removes useful context needed for full answers, even when key figures are retrieved, answers become incomplete

#### Section Summary 

* Factual queries are **retrieval-light**:
	* Correct answers depend on finding a single relevant chunk
* Dense retrieval alone (V1) is already near-optimal
* Advanced components (reranking, hybrid retrieval, compression):
	* Provide **minimal or negative gains** on this task

**For factual QA, retrieval accuracy matters more than retrieval sophistication**

---

### 5.2.2 Temporal Reasoning RAGAS Performance

| Variant | Faithfulness | Relevancy | Context Recall | Context Precision |
|---------|--------------|-----------|----------------|-------------------|
| V0 | 0.187 | 0.000 | 0.000 | 0.000 |
| V1 | 1.000 | 0.387 | 0.400 | 0.482 |
| V2 | 1.000 | 0.387 | 0.400 | n/a |
| V3 | 0.861 | 0.585 | 0.300 | 0.686 |
| V4 | 0.561 | 0.555 | 0.200 | 0.676 |
| V5 | 1.000 | 0.387 | 0.400 | 0.491 |
| V6 | 0.762 | 0.551 | 0.600 | 0.404 |

#### Key Observations

* **Temporal queries require structured retrieval, not just correct facts**
	* Dense pipelines (V1, V2, V5):
		* High faithfulness (1.000)
		* Low relevancy (0.387)
	* Answers are grounded but incomplete. Systems retrieve *one correct period*, not the full comparison.

* **Hybrid retrieval improves temporal answer completeness**
	* V3 achieves highest relevancy (**0.585**)
	* Combining lexical + semantic retrieval improves coverage across time periods. This better supports multi-period reasoning.

* **Query rewriting improves relevance but weakens grounding**
	* V4:
		* Relevancy: 0.555 (high)
		* Faithfulness: 0.561 (low)
	* Rewriting helps retrieve broader context, but introduces noisier or less reliable evidence

* **More context alone does not solve temporal reasoning**
	* V6:
		* Highest recall (0.600)
		* Relevancy still below V3 (0.551 vs 0.585)
	* Simply retrieving more documents is insufficient, having the correct *alignment* of time-based evidence should also be required.

#### Section Summary 

* Temporal reasoning is **structure-sensitive**, not just retrieval-sensitive:
	* Systems must retrieve **multiple aligned time points**, not isolated facts
* Dense retrieval fails due to **incomplete coverage**
* Hybrid retrieval (V3) provides the best balance of coverage and grounding
* Query rewriting helps, but introduces **faithfulness trade-offs**

Effective temporal QA requires both:
  * **coverage (multiple periods)**
	* **alignment (correct comparison across periods)**

---

### 5.2.3 Multi-Hop Reasoning RAGAS Performance

| Variant | Faithfulness | Relevancy | Context Recall | Context Precision |
|---------|--------------|-----------|----------------|-------------------|
| V0 | 0.000 | 0.521 | 0.000 | 0.000 |
| V1 | n/a | 0.357 | 0.733 | 0.630 |
| V2 | 0.843 | 0.347 | 0.133 | 0.807 |
| V3 | 0.817 | 0.547 | 0.000 | 0.792 |
| V4 | n/a | 0.725 | 0.600 | 0.742 |
| V5 | n/a | 0.357 | 0.667 | 0.645 |
| V6 | 0.730 | 0.729 | 0.133 | 0.628 |

#### Key Observations

* **Multi-hop reasoning requires both coverage and synthesis**
	* High recall alone is insufficient:
		* V1 (0.733 recall) → low relevancy (0.357)
		* V5 (0.667 recall) → low relevancy (0.357)
	* Systems retrieve relevant pieces but fail to combine them. Multi-hop tasks are limited by **reasoning over retrieved evidence**, not just retrieval

* **Reranking over-optimises precision and harms coverage**
	* V2:
		* Precision: **0.807 (highest)**
		* Recall: **0.133 (very low)**
		* Relevancy: **0.347**
	* Aggressive filtering removes necessary intermediate evidence. Multi-hop queries require *multiple supporting documents*, not just top-ranked ones

* **Retrieval breadth significantly improves answer relevancy**
	* V4 and V6:
		* Highest relevancy (0.725, 0.729)
	* Broader retrieval enables the model to access multiple reasoning steps. Coverage is more important than precision for multi-hop tasks

* **Hybrid retrieval alone is insufficient**
	* V3:
		* Improves relevancy (0.547 vs V1)
		* But still far below V4/V6
	* Better retrieval signals help, but do not solve evidence coordination. Additional steps (e.g. query rewriting, restructuring) are needed

#### Section Summary 

* Multi-hop reasoning is **both retrieval- and reasoning-bound**
	* Requires:
		* **high coverage** (multiple documents)
		* **correct synthesis** (linking them together)
* Precision-focused pipelines (V2) fail due to **over-filtering**
* Recall-focused pipelines (V1, V5) fail due to **lack of synthesis**
* Best performance (V4, V6) comes from **broad retrieval + better structuring**

Multi-hop QA cannot be solved by retrieval improvements alone; it requires **coordinated retrieval and reasoning over multiple evidence sources**

---

### 5.2.4 Comparative Analysis RAGAS Performance

| Variant | Faithfulness | Relevancy | Context Recall | Context Precision |
|---------|--------------|-----------|----------------|-------------------|
| V0 | n/a | 0.541 | 0.000 | 0.000 |
| V1 | 0.706 | 0.199 | 0.567 | 0.722 |
| V2 | 0.863 | 0.189 | 0.400 | 0.791 |
| V3 | n/a | 0.726 | 0.300 | 0.848 |
| V4 | n/a | 0.867 | 0.450 | 0.843 |
| V5 | 0.920 | 0.199 | 0.367 | 0.724 |
| V6 | n/a | 0.884 | 0.200 | 0.755 |

#### Key Observations

* **Comparative queries require aligned evidence, not just retrieved facts**
	* Dense pipelines (V1, V2, V5):
		* Low relevancy (~0.19–0.20)
		* Despite strong faithfulness and precision
	* Systems retrieve correct facts but fail to **place them side-by-side**. Comparative tasks require *structured alignment across entities or time periods*.

* **Retrieval breadth is the dominant factor for performance**
	* V4 and V6 achieve highest relevancy:
		* V4: 0.867
		* V6: 0.884 (best overall)
	* Broader retrieval increases likelihood of capturing all comparison targets. Missing even one element breaks the comparison.

* **Hybrid retrieval provides a strong baseline**
	* V3:
		* Relevancy: 0.726 (large jump from V1)
	* Lexical + semantic signals improve retrieval of comparable entities, but still lacks full alignment without additional processing

* **Precision is not the limiting factor**
	* V3–V6 all maintain high context precision (~0.75–0.85)
	  * Yet performance varies significantly in relevancy
	* Having *correct documents* is insufficient, the system must ensure to retrieve the **right set of documents in a comparable structure**

#### Section Summary

* Comparative analysis is **alignment-critical**:
	* Requires:
		* **complete coverage** (all entities/periods)
		* **structured retrieval** (comparable evidence)
* Precision-focused pipelines fail due to **missing comparison targets**
* Hybrid retrieval improves coverage but not alignment
* Best performance (V4, V6) comes from **broad retrieval + query restructuring/compression**

Comparative QA depends less on correctness of individual facts and more on the system’s ability to **retrieve and organise evidence into a comparable form**

---

### 5.2.5 Observation Summary

* **1. Task complexity determines the dominant bottleneck**
	* Factual → retrieval accuracy (find one correct chunk)
	* Temporal → coverage + alignment (multiple time points)
	* Multi-hop → coverage + reasoning (link multiple sources)
	* Comparative → coverage + alignment + structure (organised comparison)
	* As task complexity increases, performance shifts from **retrieval-bound → reasoning-bound**

* **2. Precision vs breadth trade-off generalises across tasks**
	* Precision pipelines (V1, V2, V5):
		* Strong faithfulness
		* Fail on complex tasks (temporal, multi-hop, comparative)
	* Breadth pipelines (V3, V4, V6):
		* Higher relevancy on complex queries
		* Slightly weaker grounding
	* Precision is sufficient for simple queries, but **coverage becomes critical as task complexity increases**

* **3. Retrieval improvements alone are insufficient for complex reasoning**
	* Increasing recall (V1, V5) does not improve multi-hop or comparative performance
	* Increasing precision (V2) can reduce performance by over-filtering
	* Best results (V4, V6) come from:
		* broader retrieval
		* better structuring (rewriting, compression)
	* Complex QA requires **coordination of retrieval and reasoning**, not optimisation of either in isolation

* **4. Alignment is the key missing capability**
	* Temporal → align time periods  
	* Multi-hop → align intermediate facts  
	* Comparative → align entities side-by-side  
	* Failures are rarely due to missing facts, but due to **failure to organise retrieved evidence correctly**

* **5. System design must match task requirements**
	* Simple QA (factual):
		* Dense retrieval (V1) is sufficient
	* Moderate complexity (temporal):
		* Hybrid retrieval (V3) provides best balance
	* High complexity (multi-hop, comparative):
		* Query rewriting + broader retrieval (V4, V6) required
	* There is no universally optimal pipeline; **performance depends on task structure**

#### Overall Takeaway

* RAG system performance is governed by three interacting factors:
	* **Precision** (faithfulness)
	* **Coverage** (recall)
	* **Structure** (alignment of evidence)

* As task complexity increases:
	* Importance shifts from precision → coverage → structure

The primary limitation of current pipelines is not retrieving information, but **structuring retrieved information into usable reasoning chains**

---

### 5.3.1 Numerical Accuracy by Category

| Category | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|----------|----|----|----|----|----|----|----|
| Factual Retrieval | 40% | 20% | 40% | 60% | 40% | 0% | 60% |
| Temporal Reasoning | 20% | 80% | 60% | 80% | 80% | 80% | 60% |
| Multi-Hop Reasoning | 20% | 0% | 40% | 40% | 0% | 0% | 40% |
| Comparative Analysis | 20% | 40% | 40% | 60% | 80% | 40% | 60% |

*Numerical Accuracy = fraction of answers containing at least one ground-truth figure (e.g. "$245.1B", "16%", "69.8%") verbatim.*

*Notes*
* V0 has no retrieved context; scores reflect parametric knowledge only
* Metric is computed via `evaluation/metrics.py::compute_numeric_match()`

#### Key Observations

* **Numerical accuracy is driven by retrieval coverage**
	* Best-performing variants:
		* V3 and V6 (up to 60%–80% depending on task)
	* Correct figures must be *present in retrieved context*; narrow pipelines fail when key values are missing.

* **Factual queries benefit from hybrid retrieval**
	* V3 and V6 achieve highest factual accuracy (60%)
	* V5 drops to 0% despite strong faithfulness
	* Exact numerical values depend on **lexical matching**. Metadata filtering removes useful numeric evidence.

* **Temporal reasoning is relatively easy once data is retrieved**
	* V1, V3, V4, V5 all reach **80% accuracy**
	* Numerical extraction is straightforward if the correct filing is retrieved.*Main challenge is retrieval, not reasoning*.

* **Multi-hop reasoning remains the hardest setting**
	* No variant exceeds **40% accuracy**
	* Retrieval requires combining figures across multiple sources. Retrieval improvements alone are insufficient.

* **Comparative analysis benefits from query structuring**
	* V4 achieves highest accuracy (80%)
	* V3 and V6 follow (60%)
	* Query rewriting improves alignment of comparable figures. Structured retrieval is critical for side-by-side comparisons

#### Section summary

* Numerical accuracy reinforces earlier findings:
	* **Coverage and structure dominate performance on complex tasks**
	* Precision-focused methods (V1, V5) fail outside simple scenarios

* Key patterns:
	* Factual → lexical matching matters most  
	* Temporal → retrieval correctness dominates  
	* Multi-hop → reasoning bottleneck persists  
	* Comparative → alignment + structure required

#### Key Anomaly

* **V0 (no retrieval) achieves 40% factual accuracy**
	* Outperforms V1 (20%) and matches V2/V4
	* Caused by:
		* Questions involving widely known financial metrics
		* Model answering from training data (pre-FY2024)
* Implication:
	* Numerical accuracy can be inflated by **parametric knowledge**
	* → Evaluation datasets must ensure:
		* Questions are **not answerable without retrieval**

---

### 5.3.2 Obeservation Summary

* Numerical accuracy depends on three factors:
	* **Coverage** → are the correct figures retrieved?
	* **Extraction** → can the model identify them?
	* **Alignment** → are they correctly used in context?
* As task complexity increases:
	* Importance shifts from **extraction → coverage → alignment**

Errors in numerical QA are primarily caused by **missing or misaligned evidence**, not failure to generate numbers.

---

### 5.4.1 Latency vs. Accuracy Trade-off

| Variant | Avg Latency (s) | Faithfulness | Trade-off Summary |
|---------|-----------------|--------------|-------------------|
| V0 | 3.94 | 0.098 | Fast, completely ungrounded |
| V1 | 3.46 | 0.738 | Fastest RAG baseline; moderate grounding |
| V2 | 4.50 | 0.843 | Higher faithfulness with modest latency increase (reranking) |
| V3 | 7.71 | 0.786 | Higher latency from hybrid retrieval; moderate faithfulness |
| V4 | 8.51 | 0.760 | Query rewriting + hybrid retrieval; highest answer relevancy |
| V5 | 3.63 | 0.804 | Metadata filtering; strong low-latency grounding without reranking |
| V6 | 10.52 | 0.700 | Highest latency; compression overhead with reduced faithfulness |

#### Key Observations

* **V2 offers the best faithfulness-latency trade-off**
	* Highest faithfulness (0.843)
	* Only ~1s slower than V1
	* Efficient improvement via reranking.

* **V5 is optimal for low-latency deployments**
	* 3.63s latency with 0.804 faithfulness
	* Metadata filtering provides cheap grounding
	* Reliable for fast real-time queries, but brittle for cross-period comparisons.

* **V4 prioritizes answer quality over speed**
	* Highest answer relevancy (0.784)
	* Latency rises to 8.51s
	* Query rewriting + hybrid retrieval improves completeness at a performance cost.

* **V3 and V6 show the cost of broader retrieval**
	* Latencies: 7.71s (V3) and 10.52s (V6)
	* Faithfulness does not surpass V2
	* Multi-stage and compressed pipelines are expensive for only modest gains.

* **V1 and V0 illustrate baseline extremes**
	* V1: fastest grounded pipeline (3.46s, 0.738 faithfulness)
	* V0: fast but ungrounded (3.94s, 0.098 faithfulness)
	* Retrieval remains critical for trustworthy outputs.

---

### 5.4.2 Observation Summary

* Precision-enhancing components (reranking, query rewriting):
	* Provide consistent faithfulness improvements
	* Slight latency penalties

* Recall-expanding components (hybrid retrieval, compression):
	* Significantly increase latency
	* Gains in faithfulness are inconsistent

* Practical deployment:
	* Choose V2 for **balanced performance**
	* Choose V5 for **fast low-latency needs**
	* Avoid overly complex pipelines unless full coverage is required.

---

### 5.5.1 Ablation Study: Four-Configuration Component Analysis

```bash
# Run the full 4-configuration ablation study
python evaluation/ablation_study.py

# Quick test (5 questions)
python evaluation/ablation_study.py --limit 5
``` 

The saved ablation run compares four retrieval configurations on a single benchmark question (`q001`) to isolate each component's effect:

| Method | Retrieval Strategy | Avg Latency (s) | Answer Quality | Key Observation |
|--------|--------------------|-----------------|----------------|-----------------|
| Dense-only | Dense (embedding-based) | 9.73 | High (correct + derived) | Strong semantic retrieval but inefficient and verbose|
| Sparse-only | BM25 (lexical) | 2.47 | Low (missing key figures) | Fast but fails on semantic matching |
| Hybrid (no rerank) | BM25 + Dense (RRF) | 7.23 | High (concise + correct) | Combines lexical + semantic strengths effectively |
| Hybrid + Rerank | Hybrid + Cross-encoder | 11.35 | High (slightly less precise numerically) | Improved ranking but diminishing returns vs cost |

#### Key Observations

* **Dense retrieval ensures completeness but is inefficient**
	* Retrieves semantically relevant chunks for correct derivations
	* Latency: 9.73s
	* Over-retrieval leads to verbose answers.
	* Dense retrieval alone lacks early precision.

* **Sparse retrieval is fast but insufficient**
	* Latency: 2.47s
	* Fails on semantic matching, misses exact figures. Lexical matching alone cannot handle paraphrased or numerically grounded queries.

* **Hybrid retrieval (no rerank) balances accuracy and efficiency**
	* Latency: 7.23s
	* Retrieves exact figures (lexical) + semantic context (embedding)
	* Concise and correct answers. Reranking not required to maintain quality in this setup

* **Adding reranking improves order but with diminishing returns**
	* Latency: 11.35s
	* Slight drop in numerical accuracy (~70%)
	* Better-ranked context does not always yield better generated answers.
	* Highlights trade-off: retrieval optimization ≠ answer correctness.

* **Lexical signals are critical for numerical extraction**
	* Hybrid setups outperform dense-only for concise, correct numerical answers.
	* Sparse-only fails due to missing exact tokens.
	* Lexical + semantic fusion is essential for financial QA tasks.

---

### 5.5.2 Observation Summary

Hybrid retrieval without reranking is the most cost-effective configuration. It captures multi-retriever benefits while avoiding high computational overhead.
* Strong baseline for production deployment

---

## 6. Qualitative Error Analysis

### 6.1 Error Taxonomy (Five Failure Types)

Following the framework defined in the outline (§6.4):

| Failure Type | Description |
|--------------|-------------|
| **Retrieval Failure** | Relevant chunk not retrieved at all |
| **Ranking Failure** | Relevant chunk retrieved but ranked too low to appear in top-k |
| **Chunking Failure** | Information split across chunk boundaries, breaking coherence |
| **Query Understanding Failure** | Query ambiguity or under-specification prevents correct retrieval |
| **Generation Failure** | LLM misinterprets context or generates unsupported claims |

### 6.2 Component–Failure Mapping

The earlier qualitative matrix overstated some failure types. The table below is **derived from** `evaluation/results/eval_results.json` by running the heuristic classifier in `evaluation/category_analysis.py`; the saved materialised summary lives in [evaluation/results/category_report.json](evaluation/results/category_report.json). Under that derived analysis, the per-variant mapping is:

| Failure Type | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|---|
| Retrieval Failure | 0 | 2 | 6 | 2 | 2 | 3 | 3 |
| Query Understanding Failure | 0 | 5 | 1 | 0 | 0 | 4 | 0 |
| Generation Failure | 11 | 0 | 1 | 0 | 1 | 0 | 0 |
| Ranking Failure | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Chunking Failure | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

This makes the component pattern clearer, but it should still be read as **heuristic failure labeling rather than definitive root-cause attribution**. All 11 V0 failures were classified as generation failures; V1 and V5 are dominated by query-understanding failures; and V2, V3, V4, and V6 are dominated by retrieval failures in the saved classifier output.

Two caveats matter. First, **ranking failure** and **chunking failure** appear as zero not because those issues are impossible, but because the current heuristic in [evaluation/category_analysis.py](evaluation/category_analysis.py) rarely surfaces them as separate labels; most such cases are absorbed into retrieval or generation failure. Second, the table should be read as a mapping of *observed dominant failure behaviour in this experiment*, not as a universal statement about what each component can or cannot do.

From a systems perspective, the main takeaway is that adding retrieval components reduced pure hallucination sharply after V0, but the new bottleneck became **evidence selection and evidence alignment**, not raw answer generation. That shift is exactly what the case studies show: the stronger pipelines usually fail by missing or misassembling evidence, not by inventing unsupported claims from scratch.

### 6.3 Case Studies

All answers below are abridged excerpts from `evaluation/results/eval_results.json`, with truncation only for readability.

---

#### Case 1: Knowledge-Cutoff Failure Without Retrieval — V0 vs V1 (q001)

**Question:** "What was Microsoft's gross profit and gross margin percentage for FY2024?"

**Ground truth:** Gross profit $171.0B on revenue of $245.1B → gross margin ≈ 69.8%.

**V0 (LLM-Only) actual answer:**
> "As of my last knowledge update in October 2023, I do not have the specific figures for Microsoft's gross profit and gross margin percentage for FY2024, as that fiscal year had not concluded and the financial results had not been publicly reported at that time. For the most accurate and up-to-date information, I recommend checking Microsoft's official financial statements or their investor relations website."

**V1 (Baseline) actual answer:**
> "For FY2024, Microsoft's total revenue was $245,122 million, and the total cost of revenue was $74,114 million. Gross Profit = $245,122M − $74,114M = $171,008 million. Gross Margin Percentage ≈ 69.7% [Doc-3]."

**Assessment:** V0 does not hallucinate a number here; instead, it refuses because its parametric knowledge predates the FY2024 filing. V1 retrieves the income statement directly and computes the answer correctly. This is still a clear demonstration of why retrieval is non-negotiable for post-cutoff financial questions.

**Failure type:** Generation failure in the saved classifier output; operationally, this case is a stale-knowledge refusal that is fixed entirely by basic dense retrieval (V1).

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

#### Case 3: Wrong-Evidence Retrieval Regression — V3 vs V1 (q011)

**Question:** "What was Microsoft's operating margin in FY2024 compared to FY2022, and what drove the change?"

**Ground truth:** FY2024 ≈ 44.6% vs FY2022 ≈ 42.1% — a 2.5 pp improvement driven by mix shift toward cloud services.

**V1 (Baseline) actual answer:**
> "In FY2024, operating income was $109,433M on revenue of $245,122M → Operating Margin ≈ 44.7%. In FY2022, operating income was $83,383M on revenue of $198,270M → Operating Margin ≈ 42.0%. The change is approximately 2.7 percentage points [Doc-4][Doc-9]."

**V3 (Hybrid) actual answer:**
> "In FY2024, Microsoft's operating income increased by $20.9 billion or 24% … The operating margin for FY2024 can be inferred from the increase in operating income, but the **exact operating margin percentage is not explicitly stated** in the provided context … a direct comparison cannot be made."

**Assessment:** V3 retrieves growth-rate percentage chunks ("operating income increased 24%") which are semantically close to "operating margin" but computationally insufficient — the generator cannot derive the margin ratio from a growth-rate statement. V1's simpler dense retrieval surfaces the absolute income statement rows, which the generator correctly uses to compute ratios. This is a regression: a more sophisticated pipeline performs worse because RRF fusion promoted the wrong chunk type.

**Failure type:** Retrieval failure in the saved heuristic classifier output. Qualitatively, this is a wrong-evidence-type regression: V3 retrieved percentage-change text instead of the absolute figures needed for margin calculation, and the answer failed downstream as a result.

---

#### Case 4: Comparative Analysis Remained Brittle — V1 vs V3 vs V4 (q016)

**Question:** "How did the cumulative revenue growth of Microsoft's three business segments compare over the FY2022 to FY2024 period?"

**Ground truth:** Intelligent Cloud +40% ($75.3B → $105.4B); Productivity +22% ($63.4B → $77.7B); More Personal Computing +4% ($59.6B → $62.0B).

**V1 (Baseline) actual answer:**
> "The provided context only contains data for FY2022 and does not include any information for FY2023 or FY2024. Therefore, I cannot provide a comparison of cumulative revenue growth … The provided documents do not contain sufficient information to answer this question."

**V3 (Hybrid) actual answer:**
> "FY2022 showed a higher growth rate of 18% compared to FY2024's growth rate of 12%. This indicates that while all segments continued to grow, the rate of growth slowed from FY2022 to FY2024."

**V4 (Query Rewriting) actual answer:**
> "FY2022 Revenue: Productivity and Business Processes $69,274M, Intelligent Cloud $87,907M, More Personal Computing $54,734M … FY2024 Revenue: Productivity and Business Processes $85,000M, Intelligent Cloud $137.4B, More Personal Computing $57,000M … cumulative growth 31.8%."

**Assessment:** This question is harder than simple multi-document retrieval because it requires the system to align the same segment table across two fiscal endpoints and then compute per-segment cumulative growth. V1 fails first at retrieval coverage, surfacing only one endpoint and therefore refusing. V3 retrieves more evidence but answers the wrong question: it compares overall company growth rates in FY2022 and FY2024 instead of segment-level cumulative growth from FY2022 to FY2024. V4 is worse in a different way: it produces a confident structured answer, but mixes incompatible figures, treats Microsoft Cloud totals as if they were segment revenue, and invents intermediate values such as "$85,000M" and "$57,000M." In other words, more aggressive retrieval expansion did not solve the task; it increased the opportunity for structured hallucination once evidence alignment broke down.

**Failure type:** Manual qualitative interpretation rather than the saved heuristic classifier output: V1 shows a retrieval-coverage failure; V3 answers the wrong comparison despite broader evidence; and V4 generates a confident but unsupported synthesis on noisy multi-document evidence. This case explains why comparative analysis remained brittle even after adding hybrid retrieval and query rewriting.

---

### 6.4 Error Distribution

Using the automatic classifier in [evaluation/category_analysis.py](evaluation/category_analysis.py) on `evaluation/results/eval_results.json`, the derived analysis assigns **30 total failures across V1–V6**. The distribution is more concentrated than the earlier qualitative estimate suggested:

| Failure Type | Count | Share of Classified Failures | Main Concentration |
|--------------|-------|------------------------------|--------------------|
| **Retrieval Failure** | 18 | 60.0% | Temporal reasoning (8), comparative analysis (5), multi-hop reasoning (4) |
| **Query Understanding Failure** | 10 | 33.3% | Multi-hop reasoning (6), comparative analysis (4) |
| **Generation Failure** | 2 | 6.7% | Multi-hop reasoning only |
| **Ranking Failure** | 0 | 0.0% | Not surfaced separately by the current heuristic |
| **Chunking Failure** | 0 | 0.0% | Not surfaced separately by the current heuristic |

Two patterns stand out. First, **retrieval failure is the dominant bottleneck**, accounting for three-fifths of all classified errors. This is especially visible in temporal questions, where systems often retrieved only one fiscal period and then refused rather than hallucinating, as in q006. Second, **query-understanding failure is concentrated in multi-hop and comparative questions**, where the model must interpret an underspecified request, align multiple years or segments, and retrieve the right evidence for each sub-part of the comparison.

The near-zero counts for ranking and chunking should not be interpreted as proof that those problems never occurred. Rather, the current rule-based classifier in [evaluation/category_analysis.py](evaluation/category_analysis.py) tended to absorb those cases into retrieval or generation failure unless there was a very clear observable signal. In other words, the automatic counts are most reliable for identifying the dominant failure families, not for perfectly separating every low-level cause.

At the variant level, the weakest pipelines were **V2** and **V5** for retrieval-related misses, while **V1** and **V5** accounted for most query-understanding failures. By contrast, **V3**, **V4**, and **V6** reduced query-understanding failures substantially, but they did not eliminate retrieval misses entirely. This matches the case studies above: better retrieval breadth helps, but complex comparative questions still fail when evidence must be aligned and reasoned over consistently.

---

## 7. Discussion & Insights

### 7.1 Component × Query Type Sensitivity

| Component | Factual | Temporal | Multi-Hop | Comparative |
|-----------|---------|----------|-----------|-------------|
| Dense Retrieval (V1) | ✓✓✓ | ✗ | ✗ | ✗ |
| + Reranking (V2) | ✓ | ✗ | ✗ | ✗ |
| + Hybrid Retrieval (V3) | ✓ | ✓✓✓ | ✓ | ✓✓ |
| + Query Rewriting (V4) | ✗ | ✓✓ | ✓✓ | ✓✓✓ |
| + Metadata Filtering (V5) | ✓ | ✓ | ✗ | ✗ |
| + Context Compression (V6) | ✗ | ✓ | ✓✓✓ | ✓✓✓ |

✓✓✓ = primary strength, ✓✓ = strong improvement, ✓ = moderate improvement, ✗ = no improvement or degradation

These labels are based on the saved category-level outcomes in Section 5, prioritising answer relevancy and numerical accuracy over intuition about what a component "should" help.

**Core Finding:** No single pipeline variant dominates across all query types, confirming the study hypothesis. Optimal performance requires adaptive pipeline selection based on query characteristics.

---

### 7.2 Component-Level Insights

#### Insight 1: Hybrid Retrieval Is Essential for Financial Queries
Financial queries contain specific lexical markers — "FY2024", "Q1 FY2025", "$245.1 billion", "10-K" — that pure dense retrieval struggles to anchor on. BM25's exact term matching complements semantic search:

- **Dense-only (V1)** achieves 0.483 answer relevancy — it struggles with fiscal period discrimination and exact figure matching
- **Hybrid (V3)** reaches 0.711 answer relevancy, a roughly 47% improvement, by combining semantic context understanding with BM25 lexical anchoring

The effect is strongest in comparative analysis: V3 achieves 0.726 comparative relevancy vs V1's 0.199, a 3.6× improvement driven by BM25 surfacing both fiscal periods simultaneously.

#### Insight 2: Reranking Delivers the Highest Faithfulness and Precision
V2 adds about 1.04 seconds over V1 (4.50s vs 3.46s) while improving context precision from 0.607 to 0.741 and faithfulness from 0.738 to 0.843. V2 achieves the highest faithfulness of all seven variants, meaning its answers are the most tightly grounded in the retrieved context.

The cross-encoder re-scores each query–chunk pair directly, filtering the noisiest candidates without requiring the full overhead of hybrid retrieval.

#### Insight 3: Metadata Filtering (V5) Achieves the Best Latency Efficiency
V5 is the second-fastest RAG variant at 3.63s average latency — just behind V1 (3.46s) and far below the hybrid variants — while maintaining 0.804 faithfulness. Pre-filtering the ChromaDB search space by fiscal period metadata before running dense retrieval reduces retrieval noise without introducing reranking cost.

Notably, **V5 intentionally omits the cross-encoder reranker** — the metadata filter is designed as a lower-cost substitute, confirmed by reranking latency = 0ms across all 20 evaluated questions.

The trade-off: V5 requires consistent metadata tagging at ingest time. Its performance collapses on multi-hop queries spanning two fiscal years — reflected in 0% multi-hop numerical accuracy — because pre-filtering by one period excludes the other.

#### Insight 4: Query Rewriting (V4) Strengthens Comparative Coverage
V4 raises aggregate answer relevancy to 0.784 and delivers 80% comparative numerical accuracy, one of the joint-highest category-specific numerical accuracy scores in the study. Its context recall (0.463) is solid but not the highest overall, and the added query-rewrite + hybrid pipeline raises latency to 8.51s.

V4 achieves 80% comparative numerical accuracy — tied with the best temporal results in the saved run — confirming its particular strength for multi-period synthesis questions.

#### Insight 5: Context Compression (V6) Targets Multi-Hop and Long-Context Failures
V6 applies sentence-level extraction after reranking to distil financially relevant content before generation. It reaches 0.729 multi-hop relevancy, 0.884 comparative relevancy, and 55% aggregate numerical accuracy, but it is also the slowest variant at 10.52s. Compression is most beneficial when relevant evidence is scattered across many chunks and the generator needs a cleaner final context.

---

### 7.3 Query Type Conclusion Summary

| Variant | Primary Benefit (Query-Type Specific) | Best Query Type | Justification (Based on Pipeline) |
|---------|--------------------------------------|-----------------|-----------------------------------|
| V0 | Baseline for simple factual recall without grounding | Factual (limited) | No retrieval → relies on parametric knowledge; can partially answer common factual queries but fails on structured or time-specific questions |
| V1 | Reliable single-document grounding for direct lookups | Factual | Dense retrieval surfaces semantically relevant chunks, which is sufficient for single-hop factual queries located within one document |
| V2 | High-faithfulness grounding for precision-focused queries | Factual / Grounding | Reranking improves context precision and faithfulness most strongly on queries where the main challenge is selecting the cleanest evidence, not assembling many pieces across periods |
| V3 | Strong cross-period retrieval for temporal and comparative queries | Temporal, Comparative | BM25 captures exact fiscal terms (e.g., “Q1 FY2025”) while dense retrieval captures semantics; RRF fusion ensures both periods are retrieved together |
| V4 | Structured retrieval for comparative and multi-document queries | Comparative, Multi-Hop | Query rewriting expands ambiguous questions into clearer retrieval targets, which is especially helpful for cross-period comparisons and multi-document synthesis |
| V5 | Fast and precise retrieval for single-period temporal queries | Temporal, Factual | Metadata filtering restricts search to a specific fiscal period, improving precision and speed when the query targets a known timeframe |
| V6 | Noise-reduced context for complex multi-hop and comparative reasoning | Multi-Hop, Comparative | Compression removes irrelevant text after retrieval, helping the model focus on key facts needed to synthesise answers across multiple sources |

Each variant’s benefit is most pronounced when its pipeline modification directly addresses the dominant challenge of the query type — whether it is locating the correct document (V3, V5), structuring the query (V4), selecting precise evidence (V2), or filtering noise during synthesis (V6).

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

6. **Retrieval-level metrics:** Implement chunk-level annotation to measure metrics like MRR and top-k hit rate

---

## 9. Risks and Guardrails

### 9.1 Risk Identification

| Risk | Description | Severity |
|------|-------------|----------|
| **Hallucination** | LLM generates plausible-sounding but incorrect financial figures not found in retrieved context | Critical |
| **Stale data** | System answers questions using outdated filings when newer filings have been published | High |
| **Retrieval miss** | Relevant chunks not retrieved, causing the LLM to answer from parametric memory | High |
| **Numerical precision errors** | LLM rounds or paraphrases figures (e.g., "$245 billion" instead of "$245.1 billion") | Medium |
| **Out-of-scope queries** | User asks about non-Microsoft companies or topics not covered in indexed filings | Medium |
| **Adversarial queries** | Deliberately ambiguous queries designed to extract unsupported comparisons | Low |

### 9.2 Implemented Mitigations

1. Prompt-level insufficient-evidence handling (implemented in `config/prompts.yaml` and `src/generation/generator.py`)
* The system prompt explicitly instructs the generator not to guess and to return an insufficient-evidence answer when the retrieved excerpts do not support the requested claim.
  * This behavior is visible in evaluation cases such as V1 on q016, where the model refuses rather than fabricating a cross-period comparison.

2. Citation requirement and citation formatting (implemented in `config/prompts.yaml`, `src/generation/generator.py`, and `src/generation/citation_formatter.py`)
* The prompt requires inline `[Doc-N]` citations for factual claims, and the citation formatter maps those references back to chunk metadata for inspection.
  * This makes grounding auditable even when an answer is only partially correct.

3. Scope filtering (implemented in `src/generation/generator.py` with topic controls from `config/settings.yaml`)
* Questions that do not match the allowed Microsoft-focused topic list are rejected during answer generation, reducing the risk of unsupported answers on out-of-scope prompts.

4. Temperature = 0 (configured in `config/settings.yaml`)
* The generator runs at temperature 0.0 to minimise stochastic variation in numerical outputs. 
  * This is especially important for financial figures where rounding behaviour at higher temperatures introduces numerical inconsistency.

### 9.3 Residual Risks and Mitigations Not Yet Implemented

| Residual Risk | Proposed Mitigation |
|---------------|---------------------|
| Stale filings | Automated ingestion pipeline to detect and index new EDGAR filings on release |
| Numerical precision | Post-processing step to verify that quoted figures match source chunk text verbatim |
| Adversarial prompts | Input sanitisation and query intent classification before retrieval |
| Index drift | Periodic re-embedding when the embedding model is updated |

---

## 10. Conclusion

### Study Hypothesis

This project demonstrates that **RAG components provide selective, query-type-dependent benefits** — confirming the study hypothesis that no single pipeline is universally optimal.

### Overall Results

| Finding | Evidence |
|---------|----------|
| RAG vs. no RAG: critical for grounding | V0 faithfulness = 0.098 → V1 faithfulness = 0.738 |
| Query rewriting achieves the highest overall answer relevancy | V4 = 0.784 vs V1 = 0.483 and V3 = 0.711 |
| Reranking delivers the strongest grounding quality | V2 has the highest faithfulness (0.843) and context precision (0.741) |
| Hybrid retrieval achieves the best aggregate numerical accuracy | V3 = 0.600, ahead of V6 = 0.550 and V4 = 0.500 |
| Dense-only retrieval remains the fastest RAG baseline | V1 = 3.46s, slightly faster than V5 = 3.63s |
| Comparative and multi-hop questions remain the most demanding | V1 scores only 0.199 on comparative relevancy and 0.357 on multi-hop relevancy, while advanced variants are required to recover performance |

### Recommendations for Financial RAG Systems

1. Use hybrid retrieval (dense + BM25) for fiscal-period-specific and cross-period comparison queries, where lexical anchors such as fiscal years and quarter labels matter.
2. Apply cross-encoder reranking for highest context precision (V2, 0.741).
3. Implement metadata pre-filtering as a low-latency precision improvement for narrowly scoped temporal queries with a clearly identifiable fiscal period.
4. Use query rewriting (V4) or context compression (V6) for comparative and multi-hop questions.
5. Maintain an LLM-only baseline to quantify the value added by each retrieval component, and ensure benchmark questions are genuinely unanswerable from model training data alone.

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
  model: gpt-4o-mini
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
# 1. Fast path: rebuild indexes from the committed processed artefacts if needed
python scripts/build_index.py

# 1b. Full raw-to-index rebuild (requires placing the 9 PDFs in data/raw/ first)
python scripts/ingest_all.py --force
python scripts/build_index.py --reset

# 2. Full evaluation — all 7 variants with RAGAS
python evaluation/run_evaluation.py

# 3. Skip RAGAS for fast Q&A verification
python evaluation/run_evaluation.py --skip-ragas

# 4. Single variant
python evaluation/run_evaluation.py --variants v3_advanced_b

# 5. Ablation study
python evaluation/ablation_study.py

# 6. Category and error analysis
python evaluation/category_analysis.py

```

### D. Reproducibility Controls

- **Random seed:** 42
- **Python version:** 3.11
- **Runtime configuration:** core parameters are set via `config/settings.yaml`, `config/prompts.yaml`, and `config/chunking.yaml`, though a small number of helper defaults remain hardcoded in code
- **Key dependencies:** see `requirements.txt`
- **ChromaDB SQLite compatibility:** handled automatically by `chromadb_compat.py`

---

*Report generated: April 2026*
*FinSight — IS469 Final Project*
