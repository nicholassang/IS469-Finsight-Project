"""
sparse_retriever.py
BM25-based keyword retrieval. Loads pre-built index from disk.
Can run completely independently of ChromaDB.
"""

import re
import pickle
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Import tokenizer from sparse_indexer to ensure consistency
try:
    from src.indexing.sparse_indexer import tokenize
except ImportError:
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())


class SparseRetriever:
    """
    Retrieves top-k chunks using BM25 keyword scoring.
    """

    def __init__(self, cfg: dict = None):
        if not BM25_AVAILABLE:
            raise RuntimeError("rank-bm25 not installed. Run: pip install rank-bm25")
        self.cfg = cfg or load_config()
        self._bm25 = None
        self._corpus = None

    def _load(self):
        if self._bm25 is not None:
            return
        index_path = Path(self.cfg["paths"]["bm25_index"])
        corpus_path = Path(self.cfg["paths"]["bm25_corpus"])

        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_path}. "
                f"Run: python scripts/build_index.py"
            )

        logger.info("SparseRetriever: loading BM25 index ...")
        with open(index_path, "rb") as f:
            self._bm25 = pickle.load(f)
        with open(corpus_path, "rb") as f:
            self._corpus = pickle.load(f)
        logger.info(f"SparseRetriever: loaded {len(self._corpus)} documents")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve top-k chunks by BM25 score.

        Returns:
            List of dicts with keys: text, metadata, score, retriever
        """
        self._load()
        top_k = top_k or self.cfg["retrieval"]["sparse_top_k"]

        t0 = time.time()
        tokens = tokenize(query)

        if not tokens:
            logger.warning("SparseRetriever: empty token list after tokenization")
            return []

        scores = self._bm25.get_scores(tokens)

        # Get indices of top-k by score
        top_k_actual = min(top_k, len(self._corpus))
        top_indices = scores.argsort()[::-1][:top_k_actual]

        latency_ms = (time.time() - t0) * 1000

        chunks = []
        for idx in top_indices:
            if float(scores[idx]) <= 0:
                continue   # BM25 score of 0 = no term overlap at all
            record = self._corpus[idx]
            chunks.append({
                "text": record["text"],
                "metadata": record["metadata"],
                "score": round(float(scores[idx]), 6),
                "retriever": "sparse",
            })

        self._log_retrieval(query, chunks, latency_ms)
        logger.debug(f"SparseRetriever: {len(chunks)} chunks in {latency_ms:.0f}ms (tokens: {tokens[:5]})")
        return chunks

    def _log_retrieval(self, query: str, chunks: List[dict], latency_ms: float):
        log_dir = Path(self.cfg["paths"].get("retrieval_logs", "indexes/retrieval_logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "sparse_retrieval.jsonl"

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "method": "sparse",
            "num_returned": len(chunks),
            "top_chunk_ids": [c["metadata"].get("chunk_id", "") for c in chunks[:5]],
            "top_scores": [c["score"] for c in chunks[:5]],
            "latency_ms": round(latency_ms, 2),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
