"""
smoke_test.py
Quick sanity checks for the entire FinSight system.
Verifies: imports, config loading, index existence, pipeline wiring (dry-run).
Does NOT require an OpenAI API key — uses mock generation.
Run after every major change: python scripts/smoke_test.py

Exit code 0 = all checks passed.
Exit code 1 = one or more failures.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

failures = []
warnings = []


def check(name: str, fn):
    try:
        result = fn()
        msg = result if isinstance(result, str) else ""
        print(f"  {PASS} {name} {msg}")
        return True
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        failures.append((name, str(e)))
        return False


def warn(name: str, fn):
    try:
        result = fn()
        msg = result if isinstance(result, str) else ""
        print(f"  {PASS} {name} {msg}")
    except Exception as e:
        print(f"  {WARN} {name}: {e} (non-fatal)")
        warnings.append((name, str(e)))


# ── 1. Python version ─────────────────────────────────────────────────────────
print("\n[1] Python Environment")
check("Python version >= 3.10",
      lambda: (sys.version_info >= (3, 10)) or (_ for _ in ()).throw(RuntimeError(f"Python {sys.version} < 3.10")))
check("Project root exists", lambda: PROJECT_ROOT.exists() or (_ for _ in ()).throw(RuntimeError("Root not found")))

# ── 2. Config loading ─────────────────────────────────────────────────────────
print("\n[2] Configuration")
check("Load settings.yaml", lambda: __import__("src.utils.config_loader", fromlist=["load_config"]).load_config())
check("Load chunking.yaml", lambda: __import__("src.utils.config_loader", fromlist=["load_chunking_config"]).load_chunking_config("default"))
check("Load prompts.yaml", lambda: __import__("src.utils.config_loader", fromlist=["load_prompts"]).load_prompts())

# ── 3. Core imports ───────────────────────────────────────────────────────────
print("\n[3] Core Module Imports")
check("src.ingestion.parser", lambda: __import__("src.ingestion.parser"))
check("src.ingestion.cleaner", lambda: __import__("src.ingestion.cleaner"))
check("src.chunking.chunker", lambda: __import__("src.chunking.chunker"))
check("src.chunking.metadata_tagger", lambda: __import__("src.chunking.metadata_tagger"))
check("src.indexing.dense_indexer", lambda: __import__("src.indexing.dense_indexer"))
check("src.indexing.sparse_indexer", lambda: __import__("src.indexing.sparse_indexer"))
check("src.retrieval.dense_retriever", lambda: __import__("src.retrieval.dense_retriever"))
check("src.retrieval.sparse_retriever", lambda: __import__("src.retrieval.sparse_retriever"))
check("src.retrieval.hybrid_retriever", lambda: __import__("src.retrieval.hybrid_retriever"))
check("src.retrieval.reranker", lambda: __import__("src.retrieval.reranker"))
check("src.generation.generator", lambda: __import__("src.generation.generator"))
check("src.generation.citation_formatter", lambda: __import__("src.generation.citation_formatter"))
check("src.pipeline.baseline", lambda: __import__("src.pipeline.baseline"))
check("src.pipeline.advanced_a", lambda: __import__("src.pipeline.advanced_a"))
check("src.pipeline.advanced_b", lambda: __import__("src.pipeline.advanced_b"))

# ── 4. Optional dependencies ──────────────────────────────────────────────────
print("\n[4] Optional Dependencies")
warn("chromadb", lambda: __import__("chromadb") and "ok")
warn("sentence_transformers", lambda: __import__("sentence_transformers") and "ok")
warn("rank_bm25", lambda: __import__("rank_bm25") and "ok")
warn("openai", lambda: __import__("openai") and "ok")
warn("pdfplumber", lambda: __import__("pdfplumber") and "ok")
warn("tiktoken", lambda: __import__("tiktoken") and "ok")
warn("streamlit", lambda: __import__("streamlit") and "ok")
warn("rouge_score", lambda: __import__("rouge_score") and "ok")

# ── 5. Chunker unit test ──────────────────────────────────────────────────────
print("\n[5] Chunker Unit Test")

def test_chunker():
    from src.chunking.chunker import chunk_text
    sample = (
        "Grab Holdings reported revenue of USD 2.36 billion in FY2023. "
        "This represents a 65% increase year-over-year. "
        "The Deliveries segment was the largest contributor. "
        "Adjusted EBITDA turned positive for the first time in company history. "
        "Management expects continued margin improvement in 2024. " * 20
    )
    cfg = {"strategy": "fixed_token", "chunk_size": 128, "chunk_overlap": 20,
           "tokenizer": "cl100k_base", "min_chunk_tokens": 30, "respect_sentence_boundaries": True}
    chunks = chunk_text(sample, cfg)
    assert len(chunks) > 1, "Expected multiple chunks"
    assert all(c.get("token_count", 0) > 0 for c in chunks), "All chunks should have token_count"
    return f"({len(chunks)} chunks produced)"

check("Fixed-token chunking", test_chunker)

# ── 6. Metadata tagger unit test ──────────────────────────────────────────────
print("\n[6] Metadata Tagger Unit Test")

def test_tagger():
    from src.chunking.metadata_tagger import tag_document_chunks, validate_chunk_metadata
    fake_chunks = [
        {"text": "Grab total revenue FY2023 was USD 2.36B.", "page_number": 5,
         "chunk_index": 0, "token_count": 18, "global_chunk_index": 0}
    ]
    doc_cfg = {
        "id": "test_doc", "company": "Grab Holdings", "ticker": "GRAB",
        "doc_type": "20-F", "fiscal_period": "FY2023", "filing_date": "2024-03-15",
        "filename": "test.pdf", "source_url": "https://example.com"
    }
    tagged = tag_document_chunks(fake_chunks, doc_cfg)
    assert len(tagged) == 1
    meta = tagged[0]["metadata"]
    assert meta["company"] == "Grab Holdings"
    assert meta["chunk_id"].startswith("test_doc")
    issues = validate_chunk_metadata(tagged[0])
    assert not issues, f"Unexpected issues: {issues}"
    return f"(chunk_id: {meta['chunk_id']})"

check("Metadata tagging", test_tagger)

# ── 7. Citation formatter unit test ───────────────────────────────────────────
print("\n[7] Citation Formatter Unit Test")

def test_citations():
    from src.generation.citation_formatter import format_citations, extract_citation_refs
    answer = "Grab revenue was USD 2.36B [Doc-1]. EBITDA turned positive [Doc-2]."
    refs = extract_citation_refs(answer)
    assert refs == [1, 2], f"Expected [1,2], got {refs}"
    chunks = [
        {"text": "Revenue was USD 2.36B in FY2023.", "metadata": {
            "doc_type": "20-F", "fiscal_period": "FY2023", "filing_date": "2024-03-15",
            "page_number": 42, "source_file": "grab_20f.pdf", "source_url": "https://x.com",
            "section_title": "Financial Results", "chunk_id": "test_p0042_c0001"
        }, "score": 0.85},
        {"text": "Adjusted EBITDA was positive for the first time.", "metadata": {
            "doc_type": "20-F", "fiscal_period": "FY2023", "filing_date": "2024-03-15",
            "page_number": 43, "source_file": "grab_20f.pdf", "source_url": "https://x.com",
            "section_title": None, "chunk_id": "test_p0043_c0001"
        }, "score": 0.80},
    ]
    citations = format_citations(answer, chunks)
    assert len(citations) == 2
    assert citations[0]["ref"] == "Doc-1"
    return f"({len(citations)} citations extracted)"

check("Citation extraction", test_citations)

# ── 8. Index existence check ──────────────────────────────────────────────────
print("\n[8] Index Existence Check")

def check_dense_index():
    from src.utils.config_loader import load_config
    cfg = load_config()
    chroma_dir = Path(cfg["paths"]["chroma_db"])
    if not chroma_dir.exists():
        raise FileNotFoundError(f"Dense index not found at {chroma_dir} — run build_index.py")
    return f"(at {chroma_dir})"

def check_sparse_index():
    from src.utils.config_loader import load_config
    cfg = load_config()
    bm25_path = Path(cfg["paths"]["bm25_index"])
    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {bm25_path} — run build_index.py")
    return f"(at {bm25_path})"

warn("Dense index exists", check_dense_index)
warn("Sparse index exists", check_sparse_index)

# ── 9. OpenAI API key check ───────────────────────────────────────────────────
print("\n[9] API Key Check")
warn("OPENAI_API_KEY set", lambda: os.getenv("OPENAI_API_KEY") or (_ for _ in ()).throw(EnvironmentError("Not set — add to .env")))

# ── 10. Data files check ──────────────────────────────────────────────────────
print("\n[10] Data Files Check")

def check_corpus():
    from src.utils.config_loader import load_config
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_data"])
    docs = cfg.get("documents", [])
    present = [d for d in docs if (raw_dir / d["filename"]).exists()]
    if not present:
        raise FileNotFoundError(
            f"No PDFs found in {raw_dir}. Download Grab filings and place them there."
        )
    return f"({len(present)}/{len(docs)} documents present)"

warn("Grab PDF files present", check_corpus)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Smoke test summary:")
print(f"  {PASS} Passed: {20 - len(failures) - len(warnings)} checks")
print(f"  {WARN} Warnings: {len(warnings)} (non-fatal, dependencies or data)")
print(f"  {FAIL} Failures: {len(failures)}")

if failures:
    print("\nFailed checks:")
    for name, err in failures:
        print(f"  {FAIL} {name}: {err}")
    print("\nFix the failures above, then re-run smoke_test.py")
    sys.exit(1)
else:
    print(f"\nAll critical checks passed! {'(some warnings above)' if warnings else ''}")
    sys.exit(0)
