"""
hybrid_retriever.py
Merges dense and sparse retrieval results using Reciprocal Rank Fusion (RRF).
Formula: RRF_score(d) = Σ 1/(k + rank(d)) across all ranked lists
Standard k=60 avoids large penalties for documents ranked in the top positions.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Combines BM25 and dense retrieval with RRF fusion.
    Query → dense candidates + sparse candidates → merged by RRF → sorted.
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self.dense = DenseRetriever(self.cfg)
        self.sparse = SparseRetriever(self.cfg)
        self.k = self.cfg["retrieval"]["rrf_k"]  # RRF constant

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve candidates from both dense and sparse retrievers,
        merge with RRF, return fused ranked list.

        Returns:
            List of chunk dicts with 'rrf_score', 'retriever'='hybrid'
            Sorted by RRF score descending.
        """
        dense_top_k = self.cfg["retrieval"]["dense_top_k"]
        sparse_top_k = self.cfg["retrieval"]["sparse_top_k"]

        t0 = time.time()

        # Get candidates from each retriever
        d_results = self.dense.retrieve(query, top_k=dense_top_k)
        s_results = self.sparse.retrieve(query, top_k=sparse_top_k)

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
        self._log_retrieval(query, merged, latency_ms, len(d_results), len(s_results))
        logger.debug(
            f"HybridRetriever: {len(merged)} merged chunks "
            f"(dense={len(d_results)}, sparse={len(s_results)}) in {latency_ms:.0f}ms"
        )

        return merged

    def _log_retrieval(
        self,
        query: str,
        chunks: List[dict],
        latency_ms: float,
        n_dense: int,
        n_sparse: int,
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
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
