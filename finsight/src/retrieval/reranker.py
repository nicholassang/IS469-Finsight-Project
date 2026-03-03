"""
reranker.py
Cross-encoder reranker for V2 and V3 pipelines.
Takes (query, chunk_text) pairs and produces relevance scores.
Returns top-k reranked chunks sorted by cross-encoder score.
"""

import time
from typing import List, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers CrossEncoder not available")


class Reranker:
    """
    Cross-encoder reranker.
    Scores (query, passage) pairs with a bi-directional attention model.
    Much more accurate than bi-encoder cosine similarity but slower — only run on top candidates.
    """

    def __init__(self, cfg: dict = None):
        if not CROSS_ENCODER_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        self.cfg = cfg or load_config()
        self._model = None

    @property
    def model(self) -> "CrossEncoder":
        if self._model is None:
            model_name = self.cfg["reranker"]["model"]
            max_length = self.cfg["reranker"]["max_length"]
            logger.info(f"Reranker: loading cross-encoder {model_name}")
            self._model = CrossEncoder(model_name, max_length=max_length)
        return self._model

    def rerank(self, query: str, chunks: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Score all (query, chunk_text) pairs and return top-k by score.

        Args:
            query: the user question
            chunks: list of chunk dicts from any retriever
            top_k: number of chunks to return (default from config)

        Returns:
            List of chunk dicts with 'rerank_score' added, sorted descending.
            Length = min(top_k, len(chunks))
        """
        if not chunks:
            return []

        top_k = top_k or self.cfg["retrieval"]["rerank_top_k"]
        t0 = time.time()

        # Prepare (query, passage) pairs
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Batch scoring with cross-encoder
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach rerank_score to each chunk
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            c = chunk.copy()
            c["rerank_score"] = round(float(score), 6)
            scored_chunks.append(c)

        # Sort by rerank_score descending
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        result = scored_chunks[:top_k]

        latency_ms = (time.time() - t0) * 1000
        logger.debug(
            f"Reranker: {len(chunks)} → {len(result)} chunks in {latency_ms:.0f}ms "
            f"(top score: {result[0]['rerank_score']:.4f} if result else 'N/A')"
        )

        return result

    def get_top_score(self, chunks: List[dict]) -> float:
        """Return the highest rerank_score in the chunk list (for weak-evidence detection)."""
        if not chunks:
            return 0.0
        scores = [c.get("rerank_score", c.get("score", 0.0)) for c in chunks]
        return max(scores)
