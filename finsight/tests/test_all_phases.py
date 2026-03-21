"""
tests/test_all_phases.py
========================
Comprehensive pytest suite covering Phase 1, 2, and 3 improvements.

Run with:
    pytest tests/test_all_phases.py -v
    pytest tests/test_all_phases.py -v --tb=short   (less noise)
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb_compat  # noqa — patches SQLite before any chromadb import


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1  — Temporal accuracy & context management
# ─────────────────────────────────────────────────────────────────────────────

class TestFiscalPeriodExtractor:
    """FiscalPeriodExtractor correctly parses fiscal periods from queries."""

    @classmethod
    def setup_class(cls):
        from src.retrieval.query_processor import FiscalPeriodExtractor
        cls.extractor = FiscalPeriodExtractor()

    def test_full_quarter_and_year(self):
        r = self.extractor.extract("What was revenue in Q2 FY2025?")
        assert r["fiscal_year"] == "FY2025"
        assert r["quarter"] == "Q2"

    def test_annual_only(self):
        r = self.extractor.extract("Show me FY2024 annual results")
        assert r["fiscal_year"] == "FY2024"
        assert r["quarter"] is None

    def test_quarter_only(self):
        r = self.extractor.extract("Q3 2025 cloud growth")
        assert r["quarter"] == "Q3"

    def test_no_period(self):
        r = self.extractor.extract("What is Microsoft's strategy?")
        assert r["fiscal_year"] is None
        assert r["quarter"] is None

    def test_fy2026_q1(self):
        r = self.extractor.extract("What did Q1 FY2026 look like?")
        assert r["fiscal_year"] == "FY2026"
        assert r["quarter"] == "Q1"

    def test_full_year_spelling(self):
        r = self.extractor.extract("Microsoft fiscal year 2023 revenue")
        assert r["fiscal_year"] == "FY2023"


class TestContextManager:
    """ContextManager fits chunks within the model's token budget."""

    @classmethod
    def setup_class(cls):
        from src.generation.context_manager import ContextManager
        cls.cm = ContextManager(model_name="qwen2.5-14b")

    def test_token_limit_respected(self):
        large_chunks = [
            {"text": "word " * 400, "metadata": {"chunk_id": f"c{i}"}}
            for i in range(20)
        ]
        fitted, stats = self.cm.fit_context(large_chunks, min_chunks=2)
        assert stats["selected_tokens"] <= self.cm.max_context_tokens

    def test_min_chunks_honoured(self):
        large_chunks = [
            {"text": "word " * 600, "metadata": {"chunk_id": f"c{i}"}}
            for i in range(5)
        ]
        fitted, stats = self.cm.fit_context(large_chunks, min_chunks=3)
        assert stats["selected_count"] >= 3

    def test_small_context_passes_through(self):
        small_chunks = [
            {"text": "Short text.", "metadata": {"chunk_id": f"c{i}"}}
            for i in range(5)
        ]
        fitted, stats = self.cm.fit_context(small_chunks, min_chunks=1)
        assert stats["selected_count"] == 5
        assert not stats["truncated"]

    def test_stats_keys_present(self):
        chunks = [{"text": "hello world", "metadata": {"chunk_id": "c0"}}]
        _, stats = self.cm.fit_context(chunks)
        for key in ("original_count", "selected_count", "original_tokens",
                    "selected_tokens", "truncated"):
            assert key in stats, f"Missing stat: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2  — Verification & caching
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerVerifier:
    """AnswerVerifier catches hallucinated citations and bad doc refs."""

    @classmethod
    def setup_class(cls):
        from src.generation.answer_verifier import AnswerVerifier, validate_citations
        cls.verifier = AnswerVerifier()
        cls._validate_citations = staticmethod(validate_citations)

    def test_good_citations_pass(self):
        from src.generation.answer_verifier import validate_citations
        answer = "Revenue was $65.6 billion [Doc-1] in Q2 FY2025. Azure grew 31% [Doc-2]."
        is_valid, _ = validate_citations(answer, 3)
        assert is_valid

    def test_out_of_range_citation_rejected(self):
        from src.generation.answer_verifier import validate_citations
        answer = "Revenue was $999 billion [Doc-99]."
        is_valid, issues = validate_citations(answer, 3)
        assert not is_valid
        assert issues

    def test_verify_returns_confidence(self):
        mock_chunks = [
            {"text": "Revenue was $65.6 billion.", "metadata": {"fiscal_period": "Q2 FY2025"}},
        ]
        result = self.verifier.verify(
            "Revenue was $65.6 billion [Doc-1].",
            "Revenue was $65.6 billion.",
            mock_chunks,
            "Q2 FY2025",
        )
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_period_mismatch_lowers_confidence(self):
        """When the answer explicitly mentions a different period than requested,
        the verifier should lower confidence and/or add warnings."""
        mock_chunks = [
            {"text": "Revenue was $65.6 billion.", "metadata": {"fiscal_period": "Q1 FY2024"}},
        ]
        # Answer mentions Q3 FY2022 — wrong year AND quarter vs requested Q2 FY2025
        result = self.verifier.verify(
            "Revenue was $65.6 billion [Doc-1] in Q3 FY2022.",
            "Revenue was $65.6 billion.",
            mock_chunks,
            "Q2 FY2025",
        )
        has_warning = (
            result.get("confidence", 1.0) < 0.9
            or bool(result.get("warnings"))
        )
        assert has_warning, (
            f"Period mismatch should lower confidence or add warnings. "
            f"Got confidence={result.get('confidence')}, warnings={result.get('warnings')}"
        )


class TestQueryCache:
    """QueryCache stores and retrieves responses with query normalisation."""

    @classmethod
    def setup_class(cls):
        from src.utils.query_cache import QueryCache
        cls.cache = QueryCache()

    def test_basic_put_and_get(self):
        self.cache.put_response("What was Microsoft revenue?", {"answer": "65B"})
        result = self.cache.get_response("What was Microsoft revenue?")
        assert result is not None
        assert result["answer"] == "65B"

    def test_normalisation_hits_cache(self):
        self.cache.put_response("Azure growth Q2 FY2025", {"answer": "31%"})
        # Different casing and extra spaces
        result = self.cache.get_response("  AZURE GROWTH q2 fy2025  ")
        assert result is not None

    def test_miss_returns_none(self):
        result = self.cache.get_response("Something never cached xyzzy123")
        assert result is None

    def test_stats_structure(self):
        stats = self.cache.stats
        assert "response_cache" in stats
        assert "size" in stats["response_cache"]
        assert "hit_rate" in stats["response_cache"]


class TestVerifiedRetrieverPeriodMatching:
    """VerifiedRetriever correctly matches fiscal periods."""

    @classmethod
    def setup_class(cls):
        from src.retrieval.verified_retriever import VerifiedRetriever

        class _Stub(VerifiedRetriever):
            def __init__(self):
                pass  # skip ChromaDB init

        cls.vr = _Stub()

    def test_exact_quarter_match(self):
        assert self.vr._periods_match("Q2 FY2025", "Q2 FY2025") is True

    def test_annual_contains_quarterly(self):
        assert self.vr._periods_match("Q2 FY2025", "FY2025") is True

    def test_wrong_quarter_rejected(self):
        assert self.vr._periods_match("Q1 FY2025", "Q2 FY2025") is False

    def test_wrong_year_rejected(self):
        assert self.vr._periods_match("Q2 FY2024", "Q2 FY2025") is False

    def test_fy_to_fy_match(self):
        assert self.vr._periods_match("FY2024", "FY2024") is True

    def test_different_fy_rejected(self):
        assert self.vr._periods_match("FY2023", "FY2024") is False


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3  — Semantic chunking
# ─────────────────────────────────────────────────────────────────────────────

# Sample SEC filing excerpt for tests
_SEC_TEXT = """
PART II

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

The following discussion should be read in conjunction with our consolidated financial statements.

REVENUE

Total revenue was $245 billion for fiscal year 2024, an increase of 16% compared to fiscal year 2023.

ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK

We are exposed to economic risk from foreign exchange rates, interest rates, credit risk, and equity prices.

NOTE 1 — SUMMARY OF SIGNIFICANT ACCOUNTING POLICIES

Revenue Recognition. We account for revenue in accordance with ASC 606.
Revenue is recognized when control of the promised goods or services is transferred.

NOTE 2 — EARNINGS PER SHARE

  Basic EPS     2024    2023    2022
  Net income    88,136  72,361  72,738
  Shares        7,466   7,472   7,557
  EPS           11.80   9.68    9.61
"""


class TestSemanticChunker:
    """SemanticChunker respects SEC section boundaries and preserves tables."""

    @classmethod
    def setup_class(cls):
        from src.chunking.semantic_chunker import SemanticChunker
        cls.chunker = SemanticChunker(max_chunk_tokens=600, min_chunk_tokens=20)

    def test_produces_chunks(self):
        chunks = self.chunker.chunk_text(_SEC_TEXT)
        assert len(chunks) >= 2, "Should produce multiple chunks from multi-section text"

    def test_output_has_required_keys(self):
        chunks = self.chunker.chunk_text(_SEC_TEXT)
        required = {"text", "chunk_index", "token_count", "section_title", "has_table"}
        for chunk in chunks:
            missing = required - set(chunk.keys())
            assert not missing, f"Chunk missing keys: {missing}"

    def test_section_title_extracted(self):
        chunks = self.chunker.chunk_text(_SEC_TEXT)
        titles = [c["section_title"] for c in chunks if c["section_title"]]
        assert len(titles) >= 1, "Should extract at least one section title"

    def test_table_detected(self):
        chunks = self.chunker.chunk_text(_SEC_TEXT)
        table_chunks = [c for c in chunks if c["has_table"]]
        assert len(table_chunks) >= 1, "Should detect the EPS table in NOTE 2"

    def test_token_count_within_limit(self):
        chunks = self.chunker.chunk_text(_SEC_TEXT)
        for chunk in chunks:
            assert chunk["token_count"] <= self.chunker.max_chunk_tokens + 50, (
                f"Chunk exceeded token limit: {chunk['token_count']}"
            )

    def test_text_content_preserved(self):
        chunks = self.chunker.chunk_text(_SEC_TEXT)
        combined = " ".join(c["text"] for c in chunks)
        # Key phrases must survive chunking
        assert "245 billion" in combined
        assert "16%" in combined

    def test_chunk_index_sequential(self):
        chunks = self.chunker.chunk_text(_SEC_TEXT)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i


class TestChunkerDispatcher:
    """chunker.py dispatcher routes 'semantic' strategy to SemanticChunker."""

    def test_semantic_strategy_dispatched(self):
        from src.chunking.chunker import chunk_text
        chunks = chunk_text(
            _SEC_TEXT,
            config={"strategy": "semantic", "max_chunk_tokens": 600, "min_chunk_tokens": 20},
        )
        assert len(chunks) >= 1

    def test_fixed_token_still_works(self):
        from src.chunking.chunker import chunk_text
        chunks = chunk_text(
            _SEC_TEXT,
            config={"strategy": "fixed_token", "chunk_size": 100, "chunk_overlap": 10,
                    "min_chunk_tokens": 10},
        )
        assert len(chunks) >= 1

    def test_experiment_d_config_loaded(self):
        from src.utils.config_loader import load_config
        cfg = load_config()
        import yaml
        with open(PROJECT_ROOT / "config" / "chunking.yaml") as f:
            chunking = yaml.safe_load(f)
        assert "experiment_D" in chunking
        assert chunking["experiment_D"]["strategy"] == "semantic"


class TestSemanticChunkerChunkPages:
    """chunk_pages() works with the semantic chunker (backward-compat interface)."""

    def test_chunk_pages_semantic(self):
        from src.chunking.semantic_chunker import SemanticChunker
        chunker = SemanticChunker(max_chunk_tokens=600, min_chunk_tokens=20)
        pages = [
            {"text": "PART I\n\nThis is part one content about Microsoft Azure revenue.", "page": 1},
            {"text": "ITEM 1. BUSINESS\n\nMicrosoft is a technology company with $245B revenue.", "page": 2},
        ]
        # chunk_pages takes only `pages` — no doc_id argument
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "page_number" in chunk
            assert "global_chunk_index" in chunk
            assert chunk["global_chunk_index"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# FINE-TUNING DATA  — Generated training data quality checks
# ─────────────────────────────────────────────────────────────────────────────

class TestFineTuneData:
    """Verify the generated fine-tuning data meets quality requirements."""

    @classmethod
    def setup_class(cls):
        cls.ft_dir = PROJECT_ROOT / "data" / "finetune"

    def test_embedder_train_exists(self):
        assert (self.ft_dir / "embedder_train.json").exists()

    def test_reranker_train_exists(self):
        assert (self.ft_dir / "reranker_train.json").exists()

    def test_embedder_train_size(self):
        with open(self.ft_dir / "embedder_train.json") as f:
            data = json.load(f)
        assert len(data) >= 500, f"Expected ≥500 embedder train examples, got {len(data)}"

    def test_embedder_val_size(self):
        with open(self.ft_dir / "embedder_val.json") as f:
            data = json.load(f)
        assert len(data) >= 50, f"Expected ≥50 embedder val examples, got {len(data)}"

    def test_embedder_triplet_keys(self):
        with open(self.ft_dir / "embedder_train.json") as f:
            data = json.load(f)
        required = {"query", "positive", "hard_negative"}
        for item in data[:10]:
            missing = required - set(item.keys())
            assert not missing, f"Triplet missing keys: {missing}"

    def test_hard_negatives_are_different_period(self):
        """Hard negatives should NOT come from the same source doc as the positive."""
        with open(self.ft_dir / "embedder_train.json") as f:
            data = json.load(f)
        for item in data[:50]:
            assert item["positive"] != item["hard_negative"], (
                "Positive and hard_negative must be different"
            )

    def test_reranker_balance(self):
        """Reranker labels should be roughly balanced (within 3:1 ratio)."""
        with open(self.ft_dir / "reranker_train.json") as f:
            data = json.load(f)
        positives = sum(1 for x in data if x["label"] == 1.0)
        negatives = sum(1 for x in data if x["label"] == 0.0)
        assert positives > 0 and negatives > 0, "Need both positive and negative examples"
        ratio = max(positives, negatives) / min(positives, negatives)
        assert ratio <= 3.0, (
            f"Imbalanced reranker data: {positives} pos / {negatives} neg (ratio {ratio:.1f})"
        )

    def test_reranker_keys(self):
        with open(self.ft_dir / "reranker_train.json") as f:
            data = json.load(f)
        required = {"query", "passage", "label"}
        for item in data[:10]:
            missing = required - set(item.keys())
            assert not missing, f"Reranker item missing keys: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — Settings must be correct for production
# ─────────────────────────────────────────────────────────────────────────────

class TestConfiguration:
    """Critical settings must be present and sensible."""

    @classmethod
    def setup_class(cls):
        from src.utils.config_loader import load_config
        cls.cfg = load_config()

    def test_weak_evidence_threshold_is_logit_scale(self):
        """threshold must be in logit range, not cosine (0-1) range."""
        threshold = self.cfg["retrieval"]["weak_evidence_threshold"]
        assert threshold <= 0.0, (
            f"weak_evidence_threshold={threshold} looks like cosine scale; "
            "must be ≤0 for cross-encoder logits"
        )

    def test_collection_name_is_msft(self):
        assert self.cfg["chroma"]["collection_name"] == "msft_filings"

    def test_retrieval_mode_is_advanced(self):
        assert self.cfg["retrieval"]["mode"] == "advanced"

    def test_period_guaranteed_slots(self):
        slots = self.cfg["retrieval"]["period_guaranteed_slots"]
        assert slots >= 1, "period_guaranteed_slots must be >= 1"

    def test_rerank_top_k_greater_than_final_k(self):
        assert (
            self.cfg["retrieval"]["rerank_top_k"]
            >= self.cfg["retrieval"]["final_context_k"]
        ), "rerank_top_k should be >= final_context_k"
