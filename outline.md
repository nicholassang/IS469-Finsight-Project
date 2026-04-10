# FinSight: RAG-based Financial Filings QA System
## 1. Problem Motivations & Objectives
### Problem Statement
Financial filings (e.g., SEC 10-K and 10-Q reports) are lengthy, complex, and difficult for users to navigate efficiently. Extracting specific financial insights—such as revenue trends, segment performance, or risk disclosures—requires significant manual effort and domain knowledge.

### Objective
To develop FinSight, a domain-specific question-answering (QA) system over Microsoft SEC filings using Retrieval-Augmented Generation (RAG) techniques. The system aims to:
- Provide accurate, citation-backed answers to financial questions
- Compare multiple RAG architectures to evaluate performance trade-offs under varying query types

### Study Hypothesis
A systematic study of how different RAG components behave under different query types


* Different RAG components provide selective benefits depending on query types, rather than uniformly improving performance

## 2. Project Scope
This project implements & evaluates 1 base LLM and 6 RAG variants.
| Variant | Pipeline | Description |
|---------|----------|-------------|
| **V0 LLM-only** | User Query → Generate | No retrieval (LLM-only baseline); tests hallucination and lack of grounding |
| **V1 Baseline** | Dense → Generate | Fixed-size chunking + embedding retrieval; basic RAG baseline |
| **V2 Advanced A** | Dense → Rerank → Generate | Dense retrieval + cross-encoder reranking; improves precision at top-k |
| **V3 Advanced B** | BM25 + Dense → RRF → Rerank → Generate | Hybrid retrieval + RRF fusion + reranking; improves recall and overall robustness |
| **V4 Advanced C** | Query Rewrite → BM25 + Dense → RRF → Rerank → Generate | LLM-based query rewriting to improve retrieval for ambiguous or underspecified queries |
| **V5 Advanced D** | Metadata Filter → Dense → Generate | Metadata-aware filtering (e.g., year, document type) before retrieval; improves precision for structured queries |
| **V6 Advanced E** | BM25 + Dense → RRF → Rerank → Context Compression → Generate | Context filtering/compression to remove irrelevant information before generation; improves faithfulness and reduces noise |

Each variant introduces a single additional capability to isolate its impact on retrieval quality, answer accuracy, and robustness across query types.


## 3. System Architecture
### Project Structure
```
finsight/
├── config/
│   ├── settings.yaml        # Main configuration
│   ├── chunking.yaml        # Chunking experiment configs
│   └── prompts.yaml         # All prompt templates
├── data/
│   ├── raw/                 # Microsoft SEC filing PDFs (not committed to git)
│   ├── processed/           # Chunked + tagged JSON per document
│   └── metadata/            # Metadata schema
├── indexes/
│   ├── chroma/              # ChromaDB vector store (not committed)
│   └── bm25/                # BM25 index pickle files (not committed)
├── src/
│   ├── ingestion/           # PDF parsing, cleaning
│   ├── chunking/            # Chunking strategies, metadata tagging
│   ├── indexing/            # ChromaDB + BM25 index builders
│   ├── retrieval/           # Dense, sparse, hybrid retrievers + reranker
│   ├── generation/          # LLM generator + citation formatter
│   ├── pipeline/            # V1/V2/V3 end-to-end pipelines
│   └── utils/               # Config loader, logger
├── evaluation/
│   ├── eval_dataset.json    # 20-question benchmark (4 categories)
│   ├── run_evaluation.py    # RAGAS evaluation runner
│   └── results/             # JSON result files per run
├── app/
│   └── streamlit_app.py     # Streamlit UI
├── scripts/
│   ├── ingest_all.py        # Full ingestion pipeline
│   ├── build_index.py       # Index builder
│   ├── run_query.py         # CLI query tool
│   └── smoke_test.py        # Quick sanity check
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_chunking_experiment.ipynb
    └── 03_retrieval_debug.ipynb
```

### Pipeline Overview
```
User Query
   ↓
Query Processing
   ↓
Retriever
   ├── Dense (V1, V2)
   └── Hybrid BM25 + Dense (V3)
   ↓
Fusion (RRF, V3 only)
   ↓
Reranker (V2, V3)
   ↓
Top-k Retrieved Chunks
   ↓
LLM Generator
   ↓
Answer + Citations
```

### Data Flow
1. Ingestion: Parse SEC filing PDFs into structured texts
2. Chunking: Split documents into fixed-sized segments with metadata
3. Indexing:
    * Dense embeddings -> vector store (ChromaDB)
    * Sparse Indexing -> BM25
4. Retrieval: Dense / Sparse / Hybrid retrieval
5. Generation: LLM produces answers using retrieved content
6. Output: Answers with source citations

## 4. Dataset
### Training Data
The following Microsoft SEC filings are indexed (already downloaded to `data/raw/`):

| Filename | Document | Period |
|----------|----------|--------|
| `msft_10k_fy2022.pdf` | Annual Report (10-K) | FY2022 (ended Jun 30, 2022) |
| `msft_10k_fy2023.pdf` | Annual Report (10-K) | FY2023 (ended Jun 30, 2023) |
| `msft_10k_fy2024.pdf` | Annual Report (10-K) | FY2024 (ended Jun 30, 2024) |
| `msft_10k_fy2025.pdf` | Annual Report (10-K) | FY2025 (ended Jun 30, 2025) |
| `msft_10q_q1_fy2025.pdf` | Quarterly Report (10-Q) | Q1 FY2025 (Sep 2024) |
| `msft_10q_q2_fy2025.pdf` | Quarterly Report (10-Q) | Q2 FY2025 (Dec 2024) |
| `msft_10q_q3_fy2025.pdf` | Quarterly Report (10-Q) | Q3 FY2025 (Mar 2025) |
| `msft_10q_q1_fy2026.pdf` | Quarterly Report (10-Q) | Q1 FY2026 (Sep 2025) |
| `msft_10q_q2_fy2026.pdf` | Quarterly Report (10-Q) | Q2 FY2026 (Dec 2025) |

Source: [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=789019) / [Microsoft Investor Relations](https://investor.microsoft.com/sec-filings)

**Note:** Microsoft fiscal year ends June 30. Q1=Jul–Sep, Q2=Oct–Dec, Q3=Jan–Mar, Q4=Apr–Jun.

Characteristics:
* Long-form financial documents
* Structured but noisy text (tables, footnotes)
* Requires numerical reasoning, and cross-section referencing


### Testing Data
????

## **5. Methodology (Detailed Design & Rationale)**

### **5.1 Study Design Overview**
Framed as a controlled experiment to evaluate how different RAG components affect performance across varying query types. Each variant introduces one additional capability to isolate its effects on:
* Retrival quality
* Answer accuracy
* Robustness across query types

| Component          | Variants Tested              |
| ------------------ | ---------------------------- |
| Retrieval Strategy | Dense, Hybrid (BM25 + Dense) |
| Query Processing   | Raw, Query Rewriting         |
| Context Selection  | Top-k, Reranked              |
| Context Quality    | Raw, Compressed              |

### **5.2 Document Processing & Chunking Strategy**
The ingestion pipeline converts SEC filing PDFs into structured text using a parsing module that removes boilerplate artifacts (headers, footers, page numbers) and preserves section-level structure where possible.

**a. Fixed-size Chunking (Baseline)**
* Chunk-size: 500–800 tokens with 10-20% overlap to preserve context continuity
* Metadata tagging:
  * Document type (10-K / 10-Q)
  * Fiscal year / quarter
  * Section labels (if identifiable, e.g., “Risk Factors”, “MD&A”)
* Rationale:
   * Fixed-size chunking ensures consistent embedding quality
   * Retrieval efficiency for large-scale indexing
   * Overlap reduces boundary fragmentation (important for financial narratives)

**b. Semantic Chunking (Experimental)**
* Split documents based on logical sections and semantic boundaries
* Rationale:
   * Preserves contextual coherence
   * Better supports multi-hop reasoning

**Trade-offs**
| Approach   | Strength           | Weakness                    |
|------------|--------------------|-----------------------------|
| Fixed-size | Efficient, uniform | May break context           |
| Semantic   | Coherent reasoning | Less consistent chunk sizes |
* May split tables or financial statements incorrectly
* May reduce coherence for multi-paragraph reasoning

---

### **5.3 Retrieval Strategies**

#### **a. Dense Retrieval (V1, V2, V5)**
* Uses embedding model to encode query and document chunks
* Retrieves top-k semantically similar chunks via vector similarity
* Strengths:
   * Captures semantic meaning (e.g., “cloud revenue” ≈ “Azure growth”)
* Weaknesses:
   * May miss exact keyword matches (e.g., specific financial terms or figures)

---

#### **b. Sparse Retrieval (BM25) (V3, V4, V5)**
* Lexical matching based on term frequency and inverse document frequency
* Strengths:
   * Strong for keyword-heavy or precise queries (e.g., “FY2024 revenue”)
* Weaknesses:
   * Cannot capture semantic relationships

---

#### **c. Hybrid Retrieval with RRF (V3, V4, V6)**
* Combines Dense + BM25 results
* Uses **Reciprocal Rank Fusion (RRF)** to merge rankings
   * Score = Σ (1 / (k + rank_i))
* Rationale:
   * Improves recall by combining semantic + lexical signals
   * Reduces risk of missing relevant chunks
   * Better performacne on multi-hop queries and ambiguous queries
* Trade-off:
   * Introduces noise which requires reranking

---

#### **d. Metadata-Aware Retrieval (V5)**
* Applies structured filtering before retrieval:
   * Fiscal year
   * Section
   * Document type
* Rationale:
   * Reduce irrelevant search space
   * Improves precision for temporal and structured queries

---

### **5.4 Query Processing (V4)**
**Purpose**
* Improve retrieval for:
   * Ambiguous queries
   * Underspecified queries

**Expected Behaviour**
| Query Type | Effect                  |
|------------|-------------------------|
| Factual    | Minimal improvement     |
| Ambiguous  | Significant improvement |
| Multi-hop  | Moderate improvement    |

---

### **5.5 Reranking Module (V2, V3, V4, V6)**
* Cross-encoder model evaluates query–chunk pairs
* Produces relevance scores for final ranking

**Purpose:**
* Improve **precision at top-k**
* Filter noisy results from hybrid retrieval
* Analyse trade-offs: increased latency and additional compute cost

**Expected Behaviour**
| Query Type | Effect   |
|------------|----------|
| Factual    | Moderate |
| Multi-hop  | High     |
| Ambiguous  | High     |

---

### **5.6 Context Processing (V6)**
**Context Compression**
* Filters or summarises retrieved chunks before generation
* Rationale:
   * Financial filings contain redundant and noisy text
   * Reducing context improves signal-to-noise ratio

**Expected Behaviour**
| Query Type | Effect   |
|------------|----------|
| Factual    | Minimal  |
| Multi-hop  | High     |
| Ambiguous  | Moderate |

---

### **5.7 Generation Module**
* LLM generates answers conditioned on retrieved context

**Key Design Choices:**
* Low temperature → reduce hallucination
* Structured prompts → enforce answer format
* Context-limited answering → ensure grounding

**Output Requirements:**
* Direct answer
* Supporting citations (document + section (if any))

---

### **5.8 Guardrails & Safety Mechanisms**

**Implemented Controls:**
* Out-of-scope detection (reject unrelated queries)
* Mandatory citation enforcement
* Context-limited answering (no external knowledge)
* Rationale:
   * Financial QA requires high factual reliability
   * Prevent hallucinated financial figures
   * Strong traceability

---

## **6. Evaluation Plan (Comprehensive Framework)**
### **6.1 Evaludation Design Overview**
The evaluation is structured to systematically analyse how different RAG components (retrieval, reranking, query processing, and context handling) behave across different query types.

Rather than reporting only aggregate performance, results are analysed along two axes:
* Pipeline Variants (V0–V6) → component-level differences
* Query Categories → task-level differences

--- 

### **6.2 Quantitative Evaluation**
Evaluation will be conducted using **RAGAS metrics**, including:
* **Faithfulness**: Alignment of generated answer with retrieved context
* **Answer Relevance**: Relevance of answer to the query
* **Context Precision**: Quality (proportion that are relevant) of retrieved chunks
* **Context Recall**: Coverage of all relevant information

**Additional Metrics**
* Exact Match / Numerical Accuracy (%) → correctness of financial values
* Top-k Hit Rate (%) → whether relevant chunk appears in top-k
* MPR (Mean Reciprocal Rank) → ranking effectiveness

**Experimental Setup:**
* 20 benchmark questions across 4 categories
* Same dataset across all variants
* Controlled parameters
   * top-k
   * temperature
   * prompt template

---

### **6.3 Category-Based Evaluation (Core Study)**
| Category                 | Description                           |
|--------------------------|---------------------------------------|
| **Factual Retrieval**    | Direct lookup (e.g., revenue, income) |
| **Temporal Reasoning**   | Time-based comparisons                |
| **Multi-hop Reasoning**  | Cross-section synthesis               |
| **Comparative Analysis** | Multi-period comparisons              |

**Evaluation Objective**
For each category:
* Identify which RAG component contributes most
* Analyse performance differences across variants

**Insight Mapping**
| Component           | Expected Strength             |
|---------------------|-------------------------------|
| Dense retrieval     | Semantic queries              |
| Hybrid retrieval    | Keyword + ambiguous queries   |
| Reranking           | Multi-hop queries             |
| Query rewriting     | Ambiguous queries             |
| Metadata filtering  | Temporal / structured queries |
| Context compression | Multi-hop / noisy contexts    |

---

### **6.4 Qualitative Evaluation**
A structured **error analysis framework** is used to diagnose system behaviour.

**Failure Categories**
1. **Retrieval Failure:** → Relevant chunk not retrieved
2. **Ranking Failure:** → Relevant chunk retrieved but ranked too low
3. **Chunking Failure:** → Information split across chunks
4. **Query Understanding Failure** (NEW) → Query 
5. **Generation Failure:** → LLM misinterprets context or hallucinates

| Failure Type             | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|--------------------------|----|----|----|----|----|----|----|
| Retrieval Failure        | —  | %  | %  | %  | %  | %  | %  |
| Ranking Failure          | —  | %  | %  | %  | %  | %  | %  |
| Query Understanding Fail | %  | %  | %  | %  | %↓ | %  | %  |
| Generation Failure       | %  | %  | %  | %  | %  | %  | %↓ |

#### **Analysis Method**
For each failure:
* Compare ground truth vs generated answer
* Inspect retrieved chunks and rankings
* Identify root cause (component responsible)
* Map failure → pipeline design
* Propose targeted improvement

---

### **6.5 Component-Level Ablation Study**
To isolate impact of components:
| Experiment            | Component Tested      |
|-----------------------|-----------------------|
| LLM-only (V0)         | No retrieval baseline |
| Dense only            | Semantic retrieval    |
| BM25 only             | Lexical retrieval     |
| Hybrid (no rerank)    | Fusion effect         |
| Hybrid + rerank       | Reranking effect      |
| + Query rewriting     | Query understanding   |
| + Metadata filtering  | Structured retrieval  |
| + Context compression | Context quality       |

**Metrics Compared:**
* RAGAS scores
* Retrieval precision (MRR, hit rate)
* Answer accuracy (%)

**Goal**
* Quantify incremental gains from each component
* Identify diminishing returns or trade-offs (if any)

---

### **6.6 Latency & Efficiency Analysis**
**Measure:**
* End-to-end response time per query
* Retrieval vs reranking vs generation time

**Purpose:** Evaluate trade-offs between:
  * Accuracy
  * Speed
  * Computational cost

**Key question:** Do advanced RAG components provide sufficient accuracy gains to justify added complexity?

---

## 7. Expected Insights & Study Outcomes
### **7.1 Core Study Objective**
To determine which RAG component(s) are most effective for which query types

---

### **7.2 Expected Findings (Component x Query Types)**
1. **Retrieval Strategy**
* Hybrid retrieval (V3-V6):
   * Strong improvement in context recall
   * Especially effective for:
      * Keyword-heavy queries
      * Ambiguous queries

2. **Reranking**
* Improves:
   * Context precision
   * Faithfulness
* Beneficial for:
   * Multi-hop reasoning
   * Complex queries

3. **Query Rewriting (V4)**
* Improves retrieval quality for ambiguous queries
* Limited impact on siple factual queries

4. **Metadata Filtering (V5)**
* Improves precision for temporal and structured queries
* Reduces retrieval noise

5. **Context Compression (V6)**
* Improves:
   * Faithfulness
   * Reduces hallucination
* Most effective for:
   * Long-context and multi-hop queries

6. **Chunking Strategy**
* Semantic chunking: improves multi-hop reasoning
* Fixed chunking: more effecient but less coherent

---

### **7.3 Cross-cutting Insight**
Key expected outcome: No single RAG pipeline performs best across all query types
* Different components provide targeted benefits
* Optimal performance requires adaptive pipeline design

---

### **7.4 Trade-off Insights**
| Component           | Benefit              | Cost                         |
|---------------------|----------------------|------------------------------|
| Hybrid retrieval    | Higher recall        | More noise                   |
| Reranking           | Higher precision     | Increased latency            |
| Query rewriting     | Better understanding | Extra LLM cost               |
| Metadata filtering  | Better precision     | Requires structured metadata |
| Context compression | Better faithfulness  | Additional processing        |

---

### **7.5 Final Expected Conclusion**
The study demonstrates that: Effective RAG systems should not rely on a single pipeline, but instead adapt retrieval, ranking, and context strategies based on query characteristics

---

## **8. Risks, Limitations & Mitigation**

### **8.1 Key Risks**
1. **Hallucinated financial values (V0, weak retrieval cases)**
* More likely in LLM-only or low-recall retrieval settings

2. **Component Misalignment with query type**
* Certain components may degrade performance for specific query categories (eg. hybrid retrieval introducing noise for simple factual queries)

3. **Retrieval noise from hybrid and expanded pipelines**
* Higher recall may introduce irrelevant chunks, affecting downstream generation 

4. **Query ambiguity affecting retrieval quality**
* Underspecified queries may lead to incorrect or incomplete context retrieval

5. **Chunking-induced information fragmentation**
* Important financial information split across chunks may hinder multi-hop reasoning

---

### **8.2 Mitigation Strategies**
1. Citation enforcement + grounded generation → Reduces hallucination and ensures traceability

2. Reranking (V2–V6) → Filters noisy retrieval results, improving precision

3. Query rewriting (V4) → Improves retrieval quality for ambiguous queries

4. Metadata filtering (V5) → Reduces irrelevant search space for structured queries

5. Context compression (V6) → Improves signal-to-noise ratio before generation

6. Controlled evaluation across query types → Ensures failures are analysed systematically rather than hidden in aggregate metrics

---

### **8.3 System Limitations**
* Fixed chunking may break semantic structure
* Limited ability to interpret tables or structured data
* Restricted to single-company dataset (Microsoft)
* Limited multi-document long-range reasoning capability

---

### **8.4 Study-Level Limitations**
* Small evaluation dataset (20 questions) → May limit statistical generalisability
* Synthetic benchmark design bias → Questions may not fully reflect real-world user queries
* Component interaction effects not fully isolated → Some improvements may overlap (e.g., reranking + compression)
* LLM variability → Results may vary slightly across runs despite fixed seeds

---

## **9. Reproducibility & Experimental Control**

* Fixed random seed (42) ensures consistent results
* All parameters configurable via YAML
* Full pipeline reproducible via scripts:
  * ingestion
  * indexing
  * querying
  * evaluation
* Version-pinned dependencies ensure environment stability

---

## **10. Deliverables & Success Criteria**
### **10.1 Deliverables**
* Functional QA system (CLI + Streamlit UI)
* Code repository with documentation
* Modular RAG pipeline supporting V0-V6
* Evaluation framework & results (quantitative + qualitative)
* Benchmark dataset (query-type categorised)
* Final report:
   * Component-level analysis
   * Query-type insights
   * Trade-off evaluation

---

### **10.2 Success Criteria**
1. Component effectiveness
* Performance of each RAG component on:
   * Faithfulness
   * Retrieval quality
   * Accuracy

2. Query-type Sensitivity
* Variation in performance across:
   * Factual
   * Temporal
   * Multi-hop
   * Comparative queries

3. Trade-off analysis
* Clear explanation of:
   * Accuracy vs latency
   * Recall vs precision
   * Complexity vs performance

Key insight: Which RAG component(s) work best for which query types, and why?

---

### **10.3 Quantitative Results (EXAMPLE)**
1. RAG Performance Metrics
| Metric            | Unit | V0 (LLM-only) | V1   | V2   | V3       | V4   | V5       | V6       |
| ----------------- | ---- | ------------- | ---- | ---- | -------- | ---- | -------- | -------- |
| Faithfulness      | 0–1  | 0.58          | 0.72 | 0.81 | 0.86     | 0.87 | 0.85     | **0.89** |
| Answer Relevance  | 0–1  | 0.62          | 0.75 | 0.83 | 0.88     | 0.89 | 0.87     | **0.90** |
| Context Precision | 0–1  | —             | 0.68 | 0.79 | 0.85     | 0.86 | **0.88** | 0.87     |
| Context Recall    | 0–1  | —             | 0.70 | 0.76 | **0.91** | 0.92 | 0.88     | 0.89     |


2. Accuracy
| Metric              | Unit | V0  | V1  | V2  | V3  | V4  | V5  | V6      |
| ------------------- | ---- | --- | --- | --- | --- | --- | --- | ------- |
| Correct Answer Rate | %    | 48% | 65% | 78% | 85% | 87% | 86% | **89%** |


3. Category-based Accuracy
| Category    | V0 | V1 | V2 | V3 | V4 | V5     | V6     |
| ----------- | -- | -- | -- | -- | -- | ------ | ------ |
| Factual     | 65 | 80 | 88 | 90 | 90 | **93** | 92     |
| Temporal    | 40 | 60 | 72 | 85 | 86 | **88** | 87     |
| Multi-hop   | 30 | 50 | 68 | 82 | 84 | 80     | **88** |
| Comparative | 35 | 55 | 70 | 84 | 85 | 82     | **87** |


4. Latency Trade-offs
| Variant | Latency (s) | Key Trade-off                 |
| ------- | ----------- | ----------------------------- |
| V0      | 1.0         | Fast, low accuracy            |
| V1      | 1.8         | Efficient baseline            |
| V2      | 2.9         | Precision ↑, latency ↑        |
| V3      | 3.6         | Recall ↑, noise ↑             |
| V4      | 4.2         | Better query understanding    |
| V5      | 3.0         | Precision ↑ with minimal cost |
| V6      | 4.5         | Best quality, highest cost    |


4. Retrieval Performance 
**By RAG variation**
| Metric                     | Unit        | V0 | V1  | V2  | V3  | V4  | V5  | V6  |
|----------------------------|-------------|----|-----|-----|-----|-----|-----|-----|
| Top-3 Hit Rate             | %           | —  | 60% | 72% | 88% | 90% | 86% | 89% |
| MRR (Reciprocal Rank)      | Score (0–1) | —  | 0.55| 0.69| 0.81| 0.83| 0.80| 0.82|
| Context Recall (RAGAS)     | Score (0–1) | —  | 0.70| 0.76| 0.91| 0.92| 0.88| 0.91|
| Context Precision (RAGAS)  | Score (0–1) | —  | 0.68| 0.79| 0.85| 0.86| 0.88| 0.87|

**By Query Type**
| Query Type         | Metric       | V1  | V2  | V3  | V4  | V5  | V6  |
|--------------------|--------------|-----|-----|-----|-----|-----|-----|
| Factual            | Hit Rate (%) | 80  | 88  | 90  | 90  | 93  | 92  |
|                    | MRR          | 0.70| 0.82| 0.85| 0.86| 0.88| 0.87|
| Temporal           | Hit Rate (%) | 60  | 72  | 85  | 86  | 88  | 87  |
|                    | MRR          | 0.50| 0.65| 0.80| 0.82| 0.84| 0.83|
| Multi-hop          | Hit Rate (%) | 50  | 68  | 82  | 84  | 80  | 88  |
|                    | MRR          | 0.40| 0.60| 0.75| 0.78| 0.72| 0.81|
| Comparative        | Hit Rate (%) | 55  | 70  | 84  | 85  | 82  | 87  |
|                    | MRR          | 0.45| 0.62| 0.78| 0.80| 0.76| 0.82|


6. Error / Failure Metrics
**By RAG variation**
| Failure Type              | Unit | V0  | V1  | V2  | V3  | V4  | V5  | V6  |
|---------------------------|------|-----|-----|-----|-----|-----|-----|-----|
| Retrieval Failure         | %    | —   | 25% | 15% | 8%  | 7%  | 10% | 8%  |
| Ranking Failure           | %    | —   | —   | 18% | 10% | 9%  | 11% | 9%  |
| Query Understanding Fail  | %    | 22% | 20% | 15% | 12% | 6%  | 14% | 10% |
| Chunking Failure          | %    | —   | 12% | 12% | 11% | 11% | 10% | 9%  |
| Generation Failure        | %    | 35% | 18% | 10% | 8%  | 7%  | 9%  | 6%  |

**By Query Type**
| Query Type  | Failure Type       | V1  | V2  | V3  | V4  | V5  | V6  |
|-------------|--------------------|-----|-----|-----|-----|-----|-----|
| Factual     | Retrieval Failure  | 15% | 8%  | 5%  | 5%  | 3%  | 4%  |
|             | Generation Failure | 12% | 6%  | 5%  | 5%  | 4%  | 3%  |
| Temporal    | Retrieval Failure  | 30% | 18% | 10% | 9%  | 7%  | 8%  |
|             | Query Failure      | 18% | 12% | 10% | 5%  | 10% | 7%  |
| Multi-hop   | Ranking Failure    | —   | 22% | 12% | 10% | 14% | 9%  |
|             | Chunking Failure   | 15% | 15% | 14% | 13% | 12% | 10% |
| Comparative | Retrieval Failure  | 28% | 16% | 9%  | 8%  | 10% | 9%  |
|             | Generation Failure | 20% | 12% | 9%  | 8%  | 10% | 7%  |


7. Ablation Study Metrics
**By Component**
| Step | Variant| Component Added        | Faithfulness | Accuracy (%) | Latency (s) |
|------|--------|------------------------|--------------|--------------|-------------|
| 0    | V0     | None (LLM-only)        | 0.58         | 48%          | 1.0         |
| 1    | V1     | + Dense Retrieval      | 0.72         | 65%          | 1.8         |
| 2    | V2     | + Reranking            | 0.81         | 78%          | 2.9         |
| 3    | V3     | + Hybrid Retrieval     | 0.86         | 85%          | 3.6         |
| 4    | V4     | + Query Rewriting      | 0.87         | 87%          | 4.2         |
| 5    | V5     | + Metadata Filtering   | 0.85         | 86%          | 3.0         |
| 6    | V6     | + Context Compression  | 0.89         | 89%          | 4.5         |


8. Component impact
| Component              | Primary Benefit        | Best Query Type     | Trade-off              |
|------------------------|------------------------|---------------------|------------------------|
| Dense Retrieval        | Baseline retrieval     | Factual             | Limited recall         |
| Reranking              | Precision ↑            | Multi-hop           | Latency ↑              |
| Hybrid Retrieval       | Recall ↑               | Temporal/Ambiguous  | Noise ↑                |
| Query Rewriting        | Query clarity ↑        | Ambiguous           | Extra LLM cost         |
| Metadata Filtering     | Precision ↑            | Temporal/Factual    | Requires metadata      |
| Context Compression    | Faithfulness ↑         | Multi-hop           | Extra processing       |


### 10.4 Mapping to Overall Success Criteria
| Success Criterion       | Metric Used                        | Unit             |
| ----------------------- | ---------------------------------- | ---------------- |
| Improved answer quality | Faithfulness, relevance            | 0–1 score        |
| Better retrieval        | Recall, hit rate                   | %                |
| Trade-offs explained    | Latency, cost                      | seconds, USD/SGD |
| Robust evaluation       | Accuracy, failure rate             | %                |
| Insight generation      | Category breakdown, V0 vs RAG gap  | %                |

| Component           | Best For                    | Impact         |
| ------------------- | --------------------------- | -------------- |
| Hybrid retrieval    | Ambiguous / keyword queries | Recall ↑       |
| Reranking           | Multi-hop queries           | Precision ↑    |
| Query rewriting     | Ambiguous queries           | Retrieval ↑    |
| Metadata filtering  | Temporal queries            | Precision ↑    |
| Context compression | Multi-hop queries           | Faithfulness ↑ |


---

## **11. Conclusion**
This project aims to bridge **practical system development** and **research-driven evaluation** in the domain of financial QA. We systematically studied how RAG components behave under different query types and derived actionable design insights to obtain the most ideal RAG pipeline in accordance to the results.

---
