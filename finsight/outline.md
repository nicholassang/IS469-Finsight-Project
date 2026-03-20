# FinSight: RAG-based Financial Filings QA System
## 1. Problem Motivations & Objectives
### Problem Statement
Financial filings (e.g., SEC 10-K and 10-Q reports) are lengthy, complex, and difficult for users to navigate efficiently. Extracting specific financial insights—such as revenue trends, segment performance, or risk disclosures—requires significant manual effort and domain knowledge.

### Objective
To develop FinSight, a domain-specific question-answering (QA) system over Microsoft SEC filings using Retrieval-Augmented Generation (RAG) techniques. The system aims to:
- Provide accurate, citation-backed answers to financial questions
- Compare multiple RAG architectures to evaluate performance trade-offs
- Analyse when advanced retrieval strategies improve answer quality

## 2. Project Scope
This project implements & evaluates 3 RAG variants
| Variant | Pipeline | Description |
|---------|----------|-------------|
| **V1 Baseline** | Dense → Generate | Fixed-size chunking + embedding retrieval |
| **V2 Advanced A** | Dense → Rerank → Generate | Dense retrieval + cross-encoder reranking |
| **V3 Advanced B** | BM25 + Dense → RRF → Rerank → Generate | Hybrid retrieval + RRF fusion + reranking |

The project addresses a real-world financial QA task, aligning with domain-specific RAG system requirements.

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



## **5. Methodology (Detailed Design & Rationale)**

### **5.1 Document Processing & Chunking Strategy**
The ingestion pipeline converts SEC filing PDFs into structured text using a parsing module that removes boilerplate artifacts (headers, footers, page numbers) and preserves section-level structure where possible.

**Chunking Approach (Current):**
* Fixed-size chunking (e.g., 500–800 tokens with overlap)
* Overlap (~10–20%) to preserve context continuity
* Metadata tagging:
  * Document type (10-K / 10-Q)
  * Fiscal year / quarter
  * Section (if identifiable, e.g., “Risk Factors”, “MD&A”)

**Rationale:**
* Fixed-size chunking ensures consistent embedding quality and retrieval efficiency
* Overlap reduces boundary fragmentation (important for financial narratives)

**Known Trade-offs:**
* May split tables or financial statements incorrectly
* May reduce coherence for multi-paragraph reasoning

**Planned Improvement:**
* Compare against **semantic chunking** (section-based splitting)
* Evaluate impact on retrieval precision and answer faithfulness

---

### **5.2 Retrieval Strategies**

#### **a. Dense Retrieval (V1, V2)**
* Uses embedding model to encode query and document chunks
* Retrieves top-k semantically similar chunks via vector similarity

**Strengths:**
* Captures semantic meaning (e.g., “cloud revenue” ≈ “Azure growth”)

**Weaknesses:**
* May miss exact keyword matches (e.g., specific financial terms or figures)

---

#### **b. Sparse Retrieval (BM25)**
* Lexical matching based on term frequency and inverse document frequency

**Strengths:**
* Strong for keyword-heavy queries (e.g., “FY2024 revenue”)

**Weaknesses:**
* Cannot capture semantic relationships

---

#### **c. Hybrid Retrieval with RRF (V3)**

* Combines Dense + BM25 results
* Uses **Reciprocal Rank Fusion (RRF)** to merge rankings

**RRF Formula:**

* Score = Σ (1 / (k + rank_i))

**Rationale:**

* Improves recall by combining semantic + lexical signals
* Reduces risk of missing relevant chunks

**Expected Behavior:**

* Better performance on:
  * Multi-hop queries
  * Ambiguous queries

* Potential downside:
  * Introduces noise → requires reranking

---

### **5.3 Reranking Module (V2, V3)**
* Cross-encoder model evaluates query–chunk pairs
* Produces relevance scores for final ranking

**Purpose:**
* Improve **precision at top-k**
* Filter noisy results from hybrid retrieval

**Trade-offs:**
* Increased latency
* Additional compute cost

**Hypothesis:**
* Reranking significantly improves answer faithfulness by ensuring only highly relevant context is passed to the generator

---

### **5.4 Generation Module**
* LLM generates answers conditioned on retrieved context
* Prompt includes:
  * Retrieved chunks
  * Instructions for grounded answering
  * Citation formatting requirements

**Key Design Choices:**
* Low temperature → reduce hallucination
* Structured prompts → enforce answer format

**Output Requirements:**
* Direct answer
* Supporting citations (document + section)

---

### **5.5 Guardrails & Safety Mechanisms**

**Implemented Controls:**
* Out-of-scope detection (reject unrelated queries)
* Mandatory citation enforcement
* Context-limited answering (no external knowledge)

**Rationale:**
* Financial QA requires high factual reliability
* Prevent hallucinated financial figures

---

## **6. Evaluation Plan (Comprehensive Framework)**

### **6.1 Quantitative Evaluation**
Evaluation will be conducted using **RAGAS metrics**, including:
* **Faithfulness**: Alignment with retrieved context
* **Answer Relevance**: Relevance to the question
* **Context Precision**: Quality of retrieved chunks
* **Context Recall**: Coverage of relevant information

**Experimental Setup:**
* 20 benchmark questions
* Same dataset across all variants
* Controlled parameters (top-k, temperature)

---

### **6.2 Qualitative Evaluation (Critical Component)**

A structured **error analysis framework** with case categories will be applied:
1. **Retrieval Failure:** Relevant chunk not retrieved
2. **Ranking Failure:** Relevant chunk retrieved but ranked too low
3. **Chunking Failure:** Information split across chunks
4. **Generation Failure:** LLM misinterprets context or hallucinates

---

#### **Analysis Method**
For each failure:
* Compare expected vs generated answer
* Inspect retrieved chunks
* Identify root cause
* Propose fix

---

### **6.3 Category-Based Evaluation**

Questions will be grouped into:
* **Factual Retrieval** (e.g., revenue figures)
* **Temporal Reasoning** (e.g., quarter-over-quarter growth)
* **Multi-hop Reasoning** (cross-section synthesis)
* **Comparative Analysis** (e.g., year-over-year changes)

**Goal:**
* Identify which pipeline performs best per category
* Understand strengths & downsides of each retrieval strategy

---

### **6.4 Ablation Study (Component-Level Analysis)**
To isolate impact of components:
```
| Experiment         | Description               |
| ------------------ | ------------------------- |
| Dense only         | Baseline retrieval        |
| BM25 only          | Sparse retrieval          |
| Hybrid (no rerank) | Fusion without refinement |
| Hybrid + rerank    | Full pipeline             |
```
**Metrics Compared:**
* RAGAS scores
* Retrieval precision
* Answer accuracy

---

### **6.5 Latency & Efficiency Analysis**
**Measure:**
* End-to-end response time per query
* Retrieval vs reranking vs generation time

**Purpose:** Evaluate trade-offs between:
  * Accuracy
  * Speed
  * Computational cost

---

## **7. Expected Insights & Hypotheses**
The study aims to validate:

1. **Hybrid retrieval improves recall**
   * Especially for complex or ambiguous queries

2. **Reranking improves precision**
   * Reduces noise introduced by hybrid retrieval

3. **Trade-off exists between latency and performance**
   * Advanced pipelines yield better accuracy but slower responses

4. **Chunking strategy impacts answer quality**
   * Poor chunking leads to incomplete or incorrect answers

---

## **8. Risks, Limitations & Mitigation**

### **8.1 Key Risks**
1. **Hallucinated financial values**
2. **Incorrect retrieval leading to misleading answers**
3. **Loss of context due to chunk boundaries**
4. **Ambiguity in financial language**

---

### **8.2 Mitigation Strategies**
* Enforce citation-based responses
* Restrict knowledge to verified SEC filings
* Use reranking to improve retrieval accuracy
* Apply conservative generation settings

---

### **8.3 System Limitations**
* Fixed chunking may break semantic structure
* Limited ability to interpret tables or structured data
* Restricted to single-company dataset
* Limited multi-document reasoning capability

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

### **Deliverables**
* Functional QA system (CLI + Streamlit UI)
* Code repository with documentation
* Evaluation results (quantitative + qualitative)
* Final report with analysis and insights

---

### **Success Criteria**
* Demonstrated improvement of advanced RAG over baseline
* Clear explanation of trade-offs
* Robust evaluation with both quantitative and qualitative evidence
* Reproducible and well-documented system

**Examples of Quantitative Results from Project Study**
1. RAG Performance Metrics
```
| Metric            | Unit        | V1 Baseline | V2 (Rerank) | V3 (Hybrid) | Interpretation              |
| ----------------- | ----------- | ----------- | ----------- | ----------- | --------------------------- |
| Faithfulness      | Score (0–1) | 0.72        | 0.81        | **0.86**    | Higher = less hallucination |
| Answer Relevance  | Score (0–1) | 0.75        | 0.83        | **0.88**    | Better alignment to query   |
| Context Precision | Score (0–1) | 0.68        | 0.79        | **0.85**    | Better chunk selection      |
| Context Recall    | Score (0–1) | 0.70        | 0.76        | **0.91**    | Hybrid improves recall      |
```

2. Latency & Efficiency Metrics
```
| Metric            | Unit        | V1   | V2   | V3   |
| ----------------- | ----------- | ---- | ---- | ---- |
| Avg Response Time | seconds (s) | 1.8s | 2.9s | 3.6s |
| Retrieval Time    | seconds (s) | 0.6s | 0.6s | 1.2s |
| Reranking Time    | seconds (s) | —    | 1.1s | 1.3s |
| Generation Time   | seconds (s) | 1.2s | 1.2s | 1.1s |
```

3. Accuracy Success Metrics
```
| Metric              | Unit | V1  | V2  | V3      |
| ------------------- | ---- | --- | --- | ------- |
| Correct Answer Rate | %    | 65% | 78% | **85%** |
```

4. Retrieval Quality metrics
```
| Metric                     | Unit        | V1   | V2   | V3       |
| -------------------------- | ----------- | ---- | ---- | -------- |
| Top-3 Hit Rate             | %           | 60%  | 72%  | **88%**  |
| MRR (Mean Reciprocal Rank) | Score (0–1) | 0.55 | 0.69 | **0.81** |
```

5. Category-based Performance
```
| Category             | Unit      | V1  | V2  | V3      |
| -------------------- | --------- | --- | --- | ------- |
| Factual Questions    | % correct | 80% | 88% | **90%** |
| Temporal Reasoning   | % correct | 60% | 72% | **85%** |
| Multi-hop Reasoning  | % correct | 50% | 68% | **82%** |
| Comparative Analysis | % correct | 55% | 70% | **84%** |
```

6. Error / Failure Metrics
```
| Failure Type       | Unit | V1  | V2  | V3     |
| ------------------ | ---- | --- | --- | ------ |
| Retrieval Failures | %    | 25% | 15% | **8%** |
| Hallucination Rate | %    | 18% | 10% | **6%** |
| Chunking Errors    | %    | 12% | 12% | 11%    |
```

7. Ablation Study Metrics
```
| Setup              | Faithfulness | Recall   | Latency (s) |
| ------------------ | ------------ | -------- | ----------- |
| Dense only         | 0.72         | 0.70     | 1.8         |
| BM25 only          | 0.68         | 0.75     | 1.6         |
| Hybrid (no rerank) | 0.80         | 0.90     | 2.5         |
| Hybrid + rerank    | **0.86**     | **0.91** | 3.6         |
```

### Mapping to Overall Success Criteria
```
| Success Criterion       | Metric Used             | Unit         |
| ----------------------- | ----------------------- | ------------ |
| Improved answer quality | Faithfulness, relevance | 0–1 score    |
| Better retrieval        | Recall, hit rate        | %            |
| Trade-offs explained    | Latency, cost           | seconds, USD |
| Robust evaluation       | Accuracy, failure rate  | %            |
| Insight generation      | Category breakdown      | %            |
```
---

## **12. Conclusion**
This project aims to bridge **practical system development** and **research-driven evaluation** in the domain of financial QA. By systematically comparing multiple RAG architectures and analyzing their behavior, FinSight will provide meaningful insights into the effectiveness of advanced retrieval strategies in real-world applications.

---
