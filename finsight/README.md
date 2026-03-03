# FinSight: RAG-Based Financial Filings QA — Grab Holdings (NASDAQ: GRAB)

A question-answering system over **official Grab Holdings financial disclosures only**, implementing and comparing three RAG variants:

| Variant | Pipeline | Description |
|---------|----------|-------------|
| **V1 Baseline** | Dense → Generate | Fixed-size chunking + embedding retrieval |
| **V2 Advanced A** | Dense → Rerank → Generate | Dense retrieval + cross-encoder reranking |
| **V3 Advanced B** | BM25 + Dense → RRF → Rerank → Generate | Hybrid retrieval + RRF fusion + reranking |

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/<your-org>/finsight.git
cd finsight
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Place Grab PDF files in data/raw/
# See "Data Acquisition" section below

# 4. Run ingestion pipeline
python scripts/ingest_all.py

# 5. Build vector and BM25 indexes
python scripts/build_index.py

# 6. Verify everything works
python scripts/smoke_test.py

# 7. Launch the app
streamlit run app/streamlit_app.py
```

---

## Data Acquisition

Download the following official Grab Holdings documents and place them in `data/raw/` with the exact filenames listed:

| Filename | Document | Source |
|----------|----------|--------|
| `grab_20f_fy2023.pdf` | FY2023 Annual Report (20-F) | [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=GRAB) |
| `grab_20f_fy2022.pdf` | FY2022 Annual Report (20-F) | SEC EDGAR |
| `grab_6k_q3_2024.pdf` | Q3 2024 Results (6-K) | SEC EDGAR |
| `grab_6k_q2_2024.pdf` | Q2 2024 Results (6-K) | SEC EDGAR |
| `grab_6k_q1_2024.pdf` | Q1 2024 Results (6-K) | SEC EDGAR |
| `grab_earnings_pr_q4_fy2023.pdf` | Q4 FY2023 Earnings Release | [ir.grab.com](https://ir.grab.com) |
| `grab_investor_day_2023.pdf` | Investor Day 2023 Presentation | [ir.grab.com](https://ir.grab.com) |
| `grab_earnings_deck_q2_2024.pdf` | Q2 2024 Earnings Presentation | ir.grab.com |
| `grab_earnings_deck_q3_2024.pdf` | Q3 2024 Earnings Presentation | ir.grab.com |

**Verification:** After placing files, run `python scripts/ingest_all.py --doc grab_20f_fy2023` for a single-document test.

---

## Build Indexes

```bash
# Full build (first time)
python scripts/build_index.py

# Force rebuild from scratch (after adding new documents)
python scripts/build_index.py --reset

# Build only the dense index
python scripts/build_index.py --dense-only

# Build only the BM25 sparse index
python scripts/build_index.py --sparse-only

# Use a different chunking configuration
python scripts/ingest_all.py --chunking experiment_A
python scripts/build_index.py --reset
```

---

## Run the App

```bash
streamlit run app/streamlit_app.py

# App will be available at http://localhost:8501
```

---

## Run Evaluation

```bash
# Full evaluation — all 80 questions across all 3 variants
python evaluation/run_eval.py

# Quick test — first 10 questions only
python evaluation/run_eval.py --limit 10

# Single variant
python evaluation/run_eval.py --variants v1_baseline

# Skip investment advice guardrail tests
python evaluation/run_eval.py --skip-guardrail-tests

# Compute metrics from saved results
python evaluation/metrics.py --results evaluation/results/

# Skip GPT-judge (to avoid API calls)
python evaluation/metrics.py --skip-gpt-judge
```

---

## Reproducibility

- Random seed is `42`, set in `config/settings.yaml`
- All model versions are pinned in `requirements.txt`
- Index artefacts can be fully regenerated with `python scripts/build_index.py --reset`
- Evaluation results are saved as JSON in `evaluation/results/` — commit these for reproducibility
- Config parameters are read from YAML files only — nothing is hardcoded in `src/`

---

## Project Structure

```
finsight/
├── config/
│   ├── settings.yaml        # Main configuration
│   ├── chunking.yaml        # Chunking experiment configs
│   └── prompts.yaml         # All prompt templates
├── data/
│   ├── raw/                 # Downloaded PDFs (not committed to git)
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
│   ├── benchmark.csv        # 80-question benchmark
│   ├── run_eval.py          # Evaluation runner
│   ├── metrics.py           # Metrics computation
│   └── results/             # JSON result files per variant
├── app/
│   └── streamlit_app.py     # Streamlit UI
├── scripts/
│   ├── ingest_all.py        # Full ingestion pipeline
│   ├── build_index.py       # Index builder
│   └── smoke_test.py        # Quick sanity check
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_chunking_experiment.ipynb
    └── 03_retrieval_debug.ipynb
```

---

## Configuration

Edit `config/settings.yaml` to change:
- Embedding model (`embeddings.model`)
- LLM model and temperature (`generation.model`, `generation.temperature`)
- Retrieval top-k values (`retrieval.dense_top_k`, etc.)
- Chunking parameters (set in `config/chunking.yaml`, select in `settings.yaml`)

---

## System Requirements

- Python 3.10+
- 8GB RAM minimum (16GB recommended for all models loaded simultaneously)
- ~2GB disk space for indexes
- OpenAI API key

---

## Team

| Role | Responsibilities |
|------|-----------------|
| **Data & Retrieval Lead** | Ingestion, chunking, BM25/Chroma indexing, retrieval modules |
| **Model & App Lead** | Embeddings, reranker, generation, citation, Streamlit app |
| **Evaluation & Report Lead** | Benchmark, metrics, GPT-judge, qualitative analysis, report |

---

## License

Academic use only. All Grab Holdings documents are the property of Grab Holdings Limited.
This system is built for research and educational purposes.
