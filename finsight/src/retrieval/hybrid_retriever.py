"""
hybrid_retriever.py
Merges dense and sparse retrieval results using Reciprocal Rank Fusion (RRF).
Formula: RRF_score(d) = Σ 1/(k + rank(d)) across all ranked lists
Standard k=60 avoids large penalties for documents ranked in the top positions.

Enhanced with improved fiscal period detection using FiscalPeriodExtractor.
"""

import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.query_processor import FiscalPeriodExtractor, QueryPreprocessor
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Mapping from year mention → fiscal_period metadata values to boost.
# Values must exactly match the fiscal_period field in settings.yaml / chunk metadata.
# Annual 10-K format: "FY2024". Quarterly 10-Q format: "Q1 FY2025", "Q2 FY2025", etc.
_FY_MAP = {
    "2022": ["FY2022"],
    "2023": ["FY2023"],
    "2024": ["FY2024"],
    "2025": ["FY2025", "Q1 FY2025", "Q2 FY2025", "Q3 FY2025", "Q4 FY2025"],
    "2026": ["FY2026", "Q1 FY2026", "Q2 FY2026"],
    "q1 2025": ["Q1 FY2025"], "q2 2025": ["Q2 FY2025"],
    "q3 2025": ["Q3 FY2025"], "q4 2025": ["Q4 FY2025"],
    "q1 2024": ["Q1 FY2024"], "q2 2024": ["Q2 FY2024"],
    "q3 2024": ["Q3 FY2024"], "q4 2024": ["Q4 FY2024"],
    "q1 2026": ["Q1 FY2026"], "q2 2026": ["Q2 FY2026"],
}


def _detect_fiscal_periods(query: str) -> List[str]:
    """Return fiscal_period metadata values mentioned in the query.

    Detection order (most specific first):
      1. Quarter mentions:  Q1 FY2025, Q3 2024, Q1 FY 2025
      2. Full-year mentions (only if no quarters found): FY2024, fiscal year 2023
      3. Bare year mentions (last resort): "revenue in 2023"
    """
    q = query.lower()
    periods = []

    # 1. Quarter mentions: Q3 FY2024, Q1 FY2025, Q1 2025 (FY is optional)
    for m in re.finditer(r'q([1-4])\s+(?:fy\s*)?(\d{4})', q):
        key = f"q{m.group(1)} {m.group(2)}"
        periods.extend(_FY_MAP.get(key, [f"Q{m.group(1)} FY{m.group(2)}"]))

    # 2. Full year mentions (only when no specific quarters detected)
    if not periods:
        for m in re.finditer(r'(?:fy\s*|fiscal\s+year\s*)(\d{4})', q):
            key = m.group(1)
            periods.extend(_FY_MAP.get(key, [f"FY{key}"]))

    # 3. Bare year mentions (last resort): "revenue in 2023"
    if not periods:
        for m in re.finditer(r'\b(20\d{2})\b', q):
            key = m.group(1)
            periods.extend(_FY_MAP.get(key, []))

    return list(dict.fromkeys(periods))  # deduplicate, preserve order


class HybridRetriever:
    """
    Combines BM25 and dense retrieval with RRF fusion.
    Query → dense candidates + sparse candidates → merged by RRF → sorted.

    Enhanced with improved fiscal period detection and filtering.
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self.dense = DenseRetriever(self.cfg)
        self.sparse = SparseRetriever(self.cfg)
        self.k = self.cfg["retrieval"]["rrf_k"]  # RRF constant
        self.fiscal_extractor = FiscalPeriodExtractor()
        self.query_preprocessor = QueryPreprocessor()

    def retrieve(self, query: str, top_k: int = None, skip_rerank: bool = False) -> List[Dict]:
        """
        Retrieve candidates from both dense and sparse retrievers,
        merge with RRF, return fused ranked list.

        When the query mentions a specific fiscal year/quarter, injects a
        targeted metadata-filtered dense search for that period so that
        table-heavy chunks with poor embeddings are guaranteed to enter
        the candidate pool.

        Args:
            query: The search query
            top_k: Maximum results to return (uses config default if None)
            skip_rerank: If True, skip any reranking step (for ablation studies)

        Returns:
            List of chunk dicts with 'rrf_score', 'retriever'='hybrid'
            Sorted by RRF score descending.
        """
        dense_top_k = self.cfg["retrieval"]["dense_top_k"]
        sparse_top_k = self.cfg["retrieval"]["sparse_top_k"]

        t0 = time.time()

        # Extract fiscal info using new extractor
        fiscal_info = self.fiscal_extractor.extract(query)

        # Get candidates from each retriever
        # Pass fiscal filtering to dense retriever
        d_results = self.dense.retrieve(query, top_k=dense_top_k, use_fiscal_filtering=True)
        s_results = self.sparse.retrieve(query, top_k=sparse_top_k)

        # ── Fiscal-period boost ───────────────────────────────────────────
        # When a specific year/quarter is mentioned, pull top-15 chunks
        # from the matching fiscal period via Chroma metadata filter.
        # These are assigned a guaranteed high RRF rank (rank 1) so they
        # always enter the reranker candidate pool.
        boost_results: List[Dict] = []
        detected_periods = _detect_fiscal_periods(query)
        if detected_periods:
            boost_results = self._retrieve_by_period(query, detected_periods, top_k=15)
            logger.debug(
                f"HybridRetriever: period boost for {detected_periods} → "
                f"{len(boost_results)} extra candidates"
            )

        # RRF fusion
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, dict] = {}

        for rank, chunk in enumerate(d_results, start=1):
            cid = chunk["metadata"].get("chunk_id", f"dense_{rank}")
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.k + rank)
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(s_results, start=1):
            cid = chunk["metadata"].get("chunk_id", f"sparse_{rank}")
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = chunk  # prefer dense version if duplicate

        # Inject boosted period chunks at rank=1 in an extra virtual list
        for rank, chunk in enumerate(boost_results, start=1):
            cid = chunk["metadata"].get("chunk_id", f"boost_{rank}")
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        merged = []
        for cid in sorted_ids:
            chunk = chunk_map[cid].copy()
            chunk["score"] = round(rrf_scores[cid], 8)
            chunk["rrf_score"] = round(rrf_scores[cid], 8)
            chunk["retriever"] = "hybrid"
            # Record which source(s) found this chunk
            in_dense = any(c["metadata"].get("chunk_id") == cid for c in d_results)
            in_sparse = any(c["metadata"].get("chunk_id") == cid for c in s_results)
            chunk["found_by"] = (
                "both" if in_dense and in_sparse
                else "dense" if in_dense
                else "sparse"
            )
            merged.append(chunk)

        latency_ms = (time.time() - t0) * 1000
        self._log_retrieval(query, merged, latency_ms, len(d_results), len(s_results), fiscal_info)
        logger.debug(
            f"HybridRetriever: {len(merged)} merged chunks "
            f"(dense={len(d_results)}, sparse={len(s_results)}) in {latency_ms:.0f}ms"
            f" | fiscal_period={fiscal_info.get('raw')}"
        )

        return merged

    def _log_retrieval(
        self,
        query: str,
        chunks: List[dict],
        latency_ms: float,
        n_dense: int,
        n_sparse: int,
        fiscal_info: Optional[Dict] = None,
    ):
        log_dir = Path(self.cfg["paths"].get("retrieval_logs", "indexes/retrieval_logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "hybrid_retrieval.jsonl"

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "method": "hybrid",
            "n_dense_candidates": n_dense,
            "n_sparse_candidates": n_sparse,
            "n_merged": len(chunks),
            "top_chunk_ids": [c["metadata"].get("chunk_id", "") for c in chunks[:5]],
            "top_rrf_scores": [c.get("rrf_score", 0) for c in chunks[:5]],
            "found_by": [c.get("found_by", "") for c in chunks[:5]],
            "latency_ms": round(latency_ms, 2),
            "fiscal_period_detected": fiscal_info.get("raw") if fiscal_info else None,
            "fiscal_year": fiscal_info.get("fiscal_year") if fiscal_info else None,
            "fiscal_quarter": fiscal_info.get("quarter") if fiscal_info else None,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _retrieve_by_period(
        self, query: str, periods: List[str], top_k: int = 15
    ) -> List[Dict]:
        """
        Run a Chroma dense retrieval restricted to the given fiscal_period(s).
        Returns up to top_k chunks, tagged as retriever='period_boost'.
        """
        try:
            normalize = self.cfg["embeddings"].get("normalize", True)
            q_vec = self.dense.model.encode(
                query, normalize_embeddings=normalize
            ).tolist()

            where_filter = (
                {"fiscal_period": {"$in": periods}}
                if len(periods) > 1
                else {"fiscal_period": periods[0]}
            )

            results = self.dense.collection.query(
                query_embeddings=[q_vec],
                n_results=min(top_k, self.dense.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            chunks = []
            if results and results["documents"]:
                for text, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    chunks.append({
                        "text": text,
                        "metadata": meta,
                        "score": round(1.0 - float(dist), 6),
                        "retriever": "period_boost",
                    })
            return chunks
        except Exception as e:
            logger.warning(f"Period boost failed for {periods}: {e}")
            return []
