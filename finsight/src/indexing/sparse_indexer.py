"""
sparse_indexer.py
Builds and manages the BM25 sparse index.
Serialises the index to disk with pickle for fast loading.
"""

import re
import pickle
import time
from pathlib import Path
from typing import List, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed — sparse indexing unavailable")


# Financial stopwords (very common terms in filings that reduce retrieval quality)
FINANCIAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "that", "this",
    "these", "those", "it", "its", "we", "our", "us", "they", "their",
    "which", "who", "whom", "what", "when", "where", "how", "all", "any",
    "each", "such", "if", "not", "no", "than", "then", "also", "more",
    "per", "net", "total", "including", "excluding", "approximately",
}


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25 indexing.
    Keeps alphanumeric tokens, lowercases, removes stopwords.
    Preserves numbers and financial abbreviations (USD, GMV, EBITDA, etc.)
    """
    tokens = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]\b|\b[a-zA-Z0-9]\b", text.lower())
    return [t for t in tokens if t not in FINANCIAL_STOPWORDS and len(t) > 1]


class SparseIndexer:
    """Manages the BM25 index for FinSight."""

    def __init__(self, cfg: dict = None):
        if not BM25_AVAILABLE:
            raise RuntimeError("rank-bm25 not installed. Run: pip install rank-bm25")
        self.cfg = cfg or load_config()

    def build_index(self, chunks: List[dict]) -> "SparseIndexer":
        """
        Build BM25 index from a list of tagged chunks.
        Stores (bm25 object, corpus list) to disk.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list")

        logger.info(f"Building BM25 index from {len(chunks)} chunks ...")
        t0 = time.time()

        # Build tokenized corpus
        tokenized_corpus = []
        corpus_records = []  # Stores text + metadata for retrieval

        for chunk in chunks:
            text = chunk.get("text", "")
            tokens = tokenize(text)
            tokenized_corpus.append(tokens)
            corpus_records.append({
                "text": text,
                "metadata": chunk.get("metadata", {}),
                "chunk_id": chunk.get("chunk_id", ""),
            })

        bm25 = BM25Okapi(tokenized_corpus)

        # Save to disk
        index_path = Path(self.cfg["paths"]["bm25_index"])
        corpus_path = Path(self.cfg["paths"]["bm25_corpus"])
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "wb") as f:
            pickle.dump(bm25, f)
        with open(corpus_path, "wb") as f:
            pickle.dump(corpus_records, f)

        elapsed = time.time() - t0
        logger.info(
            f"BM25 index built: {len(chunks)} documents, {elapsed:.1f}s\n"
            f"  Index saved: {index_path}\n"
            f"  Corpus saved: {corpus_path}"
        )
        return self

    def verify_index(self) -> bool:
        """Quick sanity check."""
        try:
            index_path = Path(self.cfg["paths"]["bm25_index"])
            corpus_path = Path(self.cfg["paths"]["bm25_corpus"])

            if not index_path.exists() or not corpus_path.exists():
                logger.warning("BM25 index files not found")
                return False

            with open(corpus_path, "rb") as f:
                corpus = pickle.load(f)

            logger.info(f"BM25 index OK: {len(corpus)} documents")
            return True
        except Exception as e:
            logger.error(f"BM25 index verification failed: {e}")
            return False

    def get_index_stats(self) -> dict:
        corpus_path = Path(self.cfg["paths"]["bm25_corpus"])
        if corpus_path.exists():
            with open(corpus_path, "rb") as f:
                corpus = pickle.load(f)
            return {"document_count": len(corpus), "corpus_path": str(corpus_path)}
        return {"document_count": 0, "corpus_path": str(corpus_path)}
