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
9. [Conclusion](#9-conclusion)
10. [Appendix](#10-appendix)

---

## 1. Introduction

Financial filings such as SEC 10-K and 10-Q reports are critical sources of information for investors, analysts, and researchers. However, these documents are lengthy (often 100+ pages), dense with technical language, and require significant domain knowledge to navigate efficiently. Traditional keyword search is inadequate for extracting nuanced financial insights that span multiple sections or require temporal reasoning. Extracting precise financial insights requires both domain knowledge and significant manual effort.

This project presents **FinSight**, a Retrieval-Augmented Generation (RAG) system designed specifically for question-answering over Microsoft Corporation's official SEC filings. We implement and rigorously compare **seven pipeline variants (V0–V6)** to understand how individual RAG components affect performance across different query types.

### Key Contributions
* **End-to-end RAG implementation** with seven (7) configurable pipelines covering a full component ablation: 
    1. No retrieval; LLM-only (V0), 
    2. Dense-only (V1), 
    3. Dense + reranking (V2), 
    4. Hybrid + reranking (V3),
    5. Query rewriting (V4),
    6. Metadata filtering (V5), and 
    7. Context compression (V6)

* **Controlled experimental design** isolating the contribution of each RAG component to performance across four query categories
* **Controlled component-level evaluation** using RAGAS metrics, numerical accuracy, and qualitative error analysis
* **Actionable insights from query-type based performance analysis** on which RAG components benefit which query types
* **Empirical validation of RAG behaviour** under different conditions

---

## 2. Problem Statement & Objectives

### Problem Statement
Extracting specific financial insights from SEC filings—such as revenue trends, segment performance, or risk disclosures—requires:
* Navigating hundreds of pages of complex financial documents
* Temporal inconsistencies when understanding Microsoft's fiscal calendar (July–June fiscal year)
* Cross-document dependencies across multiple filing periods
* Distinguishing between similar-sounding metrics across different time periods

### Objectives
1. Develop a domain-specific QA system that provides accurate, citation-backed answers (citation-grounded)
2. Compare seven RAG pipeline variants to isolate the contribution of each component across qeury types
3. Analyse failure modes to understand when and why different approaches succeed or fail, and trade-offs when prioritising a certain domain
4. Demonstrate reproducible evaluation methodology for domain-specific RAG systems

### Study Hypothesis
Different RAG components provide selective benefits depending on query type, rather than uniformly improving performance across all categories. 

Key idea: **No single pipeline is optimal for all query types.**

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
**Note:** Microsoft's fiscal year runs July 1–June 30. FY2024 = July 2023–June 2024.

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
**20 questions** across 4 query categories (5 per category)
* Ground truth answers sourced directly from SEC filing text
* Questions designed to require actual retrieval (not simple retrieval; not answerable from LLM training knowledge alone)

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
### 5.1 Aggregate RAGAS Performance
