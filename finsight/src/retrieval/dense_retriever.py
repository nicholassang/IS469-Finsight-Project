"""
dense_retriever.py
Query-time dense retrieval using ChromaDB + sentence-transformers.
Can run completely independently of BM25.
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class DenseRetriever:
    """
    Retrieves top-k semantically similar chunks from ChromaDB.
    Thread-safe for concurrent Streamlit sessions (model is stateless at query time).
    """

    def __init__(self, cfg: dict = None):
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb not installed")
        if not ST_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")

        self.cfg = cfg or load_config()
        self._model = None
        self._collection = None

    @property
    def model(self):
        if self._model is None:
            model_name = self.cfg["embeddings"]["model"]
            logger.info(f"DenseRetriever: loading model {model_name}")
            self._model = SentenceTransformer(model_name)
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            persist_dir = self.cfg["paths"]["chroma_db"]
            client = chromadb.PersistentClient(path=persist_dir)
            collection_name = self.cfg["chroma"]["collection_name"]
            self._collection = client.get_collection(collection_name)
        return self._collection

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve top-k chunks most similar to the query.

        Returns:
            List of dicts with keys: text, metadata, score, retriever
            Sorted by score descending.
        """
        top_k = top_k or self.cfg["retrieval"]["dense_top_k"]
        normalize = self.cfg["embeddings"].get("normalize", True)

        t0 = time.time()
        q_vec = self.model.encode(query, normalize_embeddings=normalize).tolist()

        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        latency_ms = (time.time() - t0) * 1000

        chunks = []
        if results and results["documents"]:
            for text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance → similarity score (1 - distance)
                score = 1.0 - float(dist)
                chunks.append({
                    "text": text,
                    "metadata": meta,
                    "score": round(score, 6),
                    "retriever": "dense",
                })

        # Sort by score descending (should already be, but be explicit)
        chunks.sort(key=lambda x: x["score"], reverse=True)

        self._log_retrieval(query, chunks, latency_ms, method="dense")
        logger.debug(f"DenseRetriever: {len(chunks)} chunks in {latency_ms:.0f}ms")
        return chunks

    def _log_retrieval(self, query: str, chunks: List[dict], latency_ms: float, method: str):
        """Append retrieval log entry to indexes/retrieval_logs/."""
        log_dir = Path(self.cfg["paths"].get("retrieval_logs", "indexes/retrieval_logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{method}_retrieval.jsonl"

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "method": method,
            "num_returned": len(chunks),
            "top_chunk_ids": [c["metadata"].get("chunk_id", "") for c in chunks[:5]],
            "top_scores": [c["score"] for c in chunks[:5]],
            "latency_ms": round(latency_ms, 2),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
