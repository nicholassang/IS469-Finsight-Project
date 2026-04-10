# FinSight: RAG-Based Financial Filings QA — Microsoft Corporation (NASDAQ: MSFT)

A question-answering system over **official Microsoft Corporation SEC filings only**, implementing and comparing **seven pipeline variants (V0–V6)**:

| Variant | Pipeline | Description |
|---------|----------|-------------|
| **V0 LLM-Only** | Generate (no retrieval) | Hallucination baseline — no RAG |
| **V1 Baseline** | Dense → Generate | Fixed-size chunking + embedding retrieval |
| **V2 Advanced A** | Dense → Rerank → Generate | Dense retrieval + cross-encoder reranking |
| **V3 Advanced B** | BM25 + Dense → RRF → Rerank → Generate | Hybrid retrieval + RRF fusion + reranking |
| **V4 Advanced C** | Query Rewrite → Hybrid → Rerank → Generate | LLM query rewriting before hybrid retrieval |
| **V5 Advanced D** | Metadata Filter → Dense → Generate | Fiscal-period pre-filtering before dense search |
| **V6 Advanced E** | Hybrid → Rerank → Compress → Generate | Context compression after reranking |

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/cig-masteracc/finsight.git
cd finsight
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure your LLM backend
cp .env.example .env
# Then update config/settings.yaml for one of these options:
#   A. Direct OpenAI API: set generation.api_key to "" and put OPENAI_API_KEY in .env
#   B. Local/HPC vLLM: set generation.base_url/model to your local endpoint and keep api_key=dummy
# See cluster/README.md for the vLLM setup path.

# 3. Verify the local artefacts and wiring
python scripts/smoke_test.py

# 4. Launch the app
streamlit run app/streamlit_app.py
```

This repository snapshot already includes:
- processed filing artefacts in `data/processed/`
- built index artefacts in `indexes/chroma/` and `indexes/bm25/`

That means you can run the app and evaluation without first downloading the raw PDFs. Download the PDFs only if you want to reproduce the ingestion step from scratch.

---

## Data Acquisition

There are two supported starting points:

1. **Fast path (already included in this repo):**
   - `data/processed/` contains chunked JSON artefacts for all 9 filings
   - `indexes/chroma/` and `indexes/bm25/` contain built retrieval indexes
   - this is enough to run the app, CLI, and evaluation scripts immediately

2. **Full rebuild from raw PDFs:**
   - download the following Microsoft SEC filings
   - place them in `data/raw/` using the exact filenames below
   - rerun `python scripts/ingest_all.py --force` and `python scripts/build_index.py --reset`

Raw PDFs are **not committed** in the current repo snapshot. The expected filenames are:

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

---

## Rebuild From Raw PDFs

```bash
# 1. Download the 9 PDFs above into data/raw/

# 2. Recreate processed chunks from the raw filings
python scripts/ingest_all.py --force

# 3. Rebuild both retrieval indexes from the processed JSON files
python scripts/build_index.py --reset
```

If you do **not** need to regenerate `data/processed/`, you can skip ingestion and rebuild indexes directly from the committed processed artefacts.

---

## Build Indexes

```bash
# Full rebuild from data/processed/
python scripts/build_index.py --reset

# Build using existing processed JSON without deleting existing indexes first
python scripts/build_index.py

# Build only the dense index
python scripts/build_index.py --dense-only

# Build only the BM25 sparse index
python scripts/build_index.py --sparse-only
```

---

## Run the App

```bash
streamlit run app/streamlit_app.py

# App will be available at http://localhost:8501
```

---

## CLI Query Tool

```bash
# Run a question using the default mode from settings.yaml
python scripts/run_query.py "What was Microsoft's total revenue in FY2024?"

# Explicitly choose a mode
python scripts/run_query.py "How did Azure grow in Q1 FY2025?" --mode baseline
python scripts/run_query.py "How did Azure grow in Q1 FY2025?" --mode advanced
```

---

## Run Evaluation

```bash
# Full RAGAS evaluation — all 7 variants (V0–V6), 20 questions
python evaluation/run_evaluation.py

# Quick test — first 5 questions only
python evaluation/run_evaluation.py --limit 5

# Single variant
python evaluation/run_evaluation.py --variants v1_baseline
python evaluation/run_evaluation.py --variants v3_advanced_b

# Custom output path
python evaluation/run_evaluation.py --output evaluation/results/my_run.json
```

Results are saved to `evaluation/results/eval_results.json` and a comparison table is printed.

---

## Run Analysis Scripts

```bash
# Re-score / back-fill saved evaluation outputs
python evaluation/rescore_ragas.py --input evaluation/results/eval_results.json

# Category breakdown + heuristic error analysis
python evaluation/category_analysis.py --results evaluation/results/eval_results.json

# Retrieval ablation study
python evaluation/ablation_study.py
```

---

## Reproducibility

- Project seed is `42`, configured in `config/settings.yaml` and applied by the main CLI, evaluation, and app entry points via `src/utils/seeding.py`
- Core runtime parameters live in `config/settings.yaml`, `config/prompts.yaml`, and `config/chunking.yaml`
- `data/processed/` contains committed chunked artefacts for the 9 Microsoft filings used in the project
- Retrieval indexes can be regenerated from `data/processed/` with `python scripts/build_index.py --reset`
- Evaluation outputs are saved under `evaluation/results/` as JSON artefacts
- `requirements.txt` pins most Python package versions used by the project
- Full raw-to-index reproduction is possible if you download the 9 PDFs into `data/raw/` and rerun ingestion + indexing

---

## Project Structure

```
finsight/
├── config/
│   ├── settings.yaml        # Main configuration
│   ├── chunking.yaml        # Chunking experiment configs
│   └── prompts.yaml         # All prompt templates
├── data/
│   ├── raw/                 # Optional raw SEC PDFs for full ingestion rebuilds
│   ├── processed/           # Committed chunked JSON artefacts for 9 filings
│   └── metadata/            # Metadata schema
├── indexes/
│   ├── chroma/              # ChromaDB vector store (rebuildable)
│   └── bm25/                # BM25 index pickle files (rebuildable)
├── src/
│   ├── ingestion/           # PDF parsing, cleaning
│   ├── chunking/            # Chunking strategies, metadata tagging
│   ├── indexing/            # ChromaDB + BM25 index builders
│   ├── retrieval/           # Dense, sparse, hybrid retrievers + reranker
│   ├── generation/          # LLM generator + citation formatter
│   ├── pipeline/            # V0–V6 end-to-end pipeline implementations
│   └── utils/               # Config loader, logger
├── evaluation/
│   ├── eval_dataset.json    # 20-question benchmark (4 categories)
│   ├── benchmark.csv        # Extended benchmark metadata
│   ├── run_evaluation.py    # Main evaluation runner
│   ├── ablation_study.py    # Retrieval ablation experiments
│   ├── category_analysis.py # Category and failure analysis
│   ├── metrics.py           # Derived evaluation metrics
│   ├── rescore_ragas.py     # Re-score/back-fill saved runs
│   └── results/             # JSON result files per run
├── app/
│   └── streamlit_app.py     # Streamlit UI
├── scripts/
│   ├── ingest_all.py        # Full ingestion pipeline
│   ├── build_index.py       # Index builder
│   ├── run_query.py         # CLI query tool
│   ├── diagnose.py          # Backend/index diagnostics
│   └── smoke_test.py        # Quick sanity check
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_chunking_experiment.ipynb
    ├── 03_retrieval_debug.ipynb
    └── 04_evaluation_analysis.ipynb
```

---

## Configuration

Edit `config/settings.yaml` to change:
- Seed (`project.seed`)
- Embedding model (`embeddings.model`)
- LLM backend, model, endpoint, and temperature (`generation.backend`, `generation.model`, `generation.base_url`, `generation.temperature`)
- Retrieval top-k values (`retrieval.dense_top_k`, etc.)
- Retrieval mode (`retrieval.mode`: `baseline` or `advanced`)
- Guardrails (`guardrails.out_of_scope_check`, `guardrails.require_citations`, etc.)

---

## System Requirements

- Python 3.11 (recommended; numpy 1.26.4 incompatible with 3.14)
- 8GB RAM minimum (16GB recommended for all models loaded simultaneously)
- ~2GB disk space for indexes
- An OpenAI-compatible chat backend for answer generation and RAGAS judging
- The checked-in config currently points to `gpt-4o-mini` via the OpenAI API
- For local/HPC vLLM setup, see `cluster/README.md`

---

## Team

| Role | Responsibilities |
|------|-----------------|
| **Data & Retrieval Lead** | Ingestion, chunking, BM25/Chroma indexing, retrieval modules |
| **Model & App Lead** | Embeddings, reranker, generation, citation, Streamlit app |
| **Evaluation & Report Lead** | Benchmark, RAGAS metrics, GPT-judge, qualitative analysis, report |

### Individual Contributions

**Data & Retrieval Lead**
- Designed and implemented the PDF ingestion pipeline (`src/ingestion/`) including parsing, cleaning, and fiscal metadata tagging
- Built fixed-size and semantic chunking strategies (`src/chunking/`)
- Implemented ChromaDB dense indexer and BM25 sparse indexer (`src/indexing/`)
- Implemented dense retriever, sparse retriever, and hybrid retriever with RRF fusion (`src/retrieval/`)
- Built the metadata-aware retrieval flow used by V5 (`src/retrieval/query_processor.py`, `src/pipeline/advanced_d.py`)
- Ran ingestion pipeline on cluster; generated `data/processed/` JSON files and BM25/Chroma indexes

*Contributions of other members observed: Model & App Lead integrated the retrieval modules into end-to-end pipelines and resolved ChromaDB SQLite compatibility issues. Evaluation & Report Lead used the retrieval logs in `indexes/retrieval_logs/` to compute per-stage latency breakdowns in the evaluation.*

---

**Model & App Lead**
- Integrated embedding model (`sentence-transformers/all-mpnet-base-v2`) into the indexing and retrieval pipeline
- Implemented cross-encoder reranker (`src/retrieval/reranker.py`) using `ms-marco-MiniLM-L-6-v2`
- Implemented LLM generator with citation formatting and guardrail logic (`src/generation/`)
- Built all six advanced pipeline variants (V1–V6) in `src/pipeline/`
- Developed the Streamlit application (`app/streamlit_app.py`)
- Set up vLLM server on the cluster (`cluster/serve_vllm.sh`) and resolved ChromaDB compatibility (`chromadb_compat.py`)

*Contributions of other members observed: Data & Retrieval Lead provided chunked JSON files and index artifacts that the pipelines depend on. Evaluation & Report Lead identified the V3 generation regression during evaluation and raised it as a bug, which was traced to RRF ranking of percentage-change chunks over absolute-value chunks.*

---

**Evaluation & Report Lead**
- Designed the 20-question benchmark dataset across four query categories (`evaluation/eval_dataset.json`)
- Implemented RAGAS evaluation runner with multi-variant support (`evaluation/run_evaluation.py`)
- Implemented numerical accuracy metric and category-level breakdown (`evaluation/metrics.py`)
- Built ablation study and category analysis scripts (`evaluation/ablation_study.py`, `evaluation/category_analysis.py`)
- Ran full evaluation across all 7 variants on the cluster and committed results (`evaluation/results/eval_results.json`)
- Authored the full project report (`REPORT.md`) including all tables, case studies, and analysis

*Contributions of other members observed: Data & Retrieval Lead's metadata tagging enabled the per-category retrieval analysis. Model & App Lead's pipeline implementations produced the diverse results that made the comparative analysis meaningful.*

---

## License

Academic use only. All Microsoft Corporation documents are the property of Microsoft Corporation.
This system is built for research and educational purposes.
