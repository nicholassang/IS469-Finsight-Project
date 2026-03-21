"""
dense_retriever.py
Query-time dense retrieval using ChromaDB + sentence-transformers.
Can run completely independently of BM25.

Enhanced with fiscal period filtering to prevent temporal retrieval errors.
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.retrieval.query_processor import FiscalPeriodExtractor, QueryPreprocessor

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

    Enhanced with fiscal period filtering to address temporal retrieval errors.
    """

    def __init__(self, cfg: dict = None):
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb not installed")
        if not ST_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")

        self.cfg = cfg or load_config()
        self._model = None
        self._collection = None
        self.query_preprocessor = QueryPreprocessor()
        self.fiscal_extractor = FiscalPeriodExtractor()

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

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        fiscal_filter: Optional[Dict] = None,
        use_fiscal_filtering: bool = True,
    ) -> List[Dict]:
        """
        Retrieve top-k chunks most similar to the query.

        Args:
            query: The search query
            top_k: Number of chunks to retrieve
            fiscal_filter: Optional explicit metadata filter (overrides auto-detection)
            use_fiscal_filtering: Whether to auto-detect and filter by fiscal period

        Returns:
            List of dicts with keys: text, metadata, score, retriever
            Sorted by score descending.
        """
        top_k = top_k or self.cfg["retrieval"]["dense_top_k"]
        normalize = self.cfg["embeddings"].get("normalize", True)

        t0 = time.time()
        q_vec = self.model.encode(query, normalize_embeddings=normalize).tolist()

        # Determine metadata filter
        where_clause = None
        fiscal_info = None

        if fiscal_filter:
            where_clause = fiscal_filter
        elif use_fiscal_filtering:
            fiscal_info = self.fiscal_extractor.extract(query)
            where_clause = self.fiscal_extractor.to_metadata_filter(fiscal_info)

        # Query ChromaDB with optional filtering
        try:
            if where_clause:
                # Over-retrieve when filtering (some may be filtered out)
                results = self.collection.query(
                    query_embeddings=[q_vec],
                    n_results=min(top_k * 3, self.collection.count()),
                    where=where_clause,
                    include=["documents", "metadatas", "distances"],
                )
                logger.debug(f"DenseRetriever: filtered by {where_clause}")
            else:
                results = self.collection.query(
                    query_embeddings=[q_vec],
                    n_results=min(top_k, self.collection.count()),
                    include=["documents", "metadatas", "distances"],
                )
        except Exception as e:
            # Fallback to unfiltered if filter fails
            logger.warning(f"Filtered query failed ({e}), falling back to unfiltered")
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
                    "fiscal_filtered": where_clause is not None,
                })

        # If fiscal filtering returned too few results, fallback to relaxed filter
        if where_clause and len(chunks) < top_k // 2 and fiscal_info:
            logger.info(f"DenseRetriever: only {len(chunks)} results with strict filter, trying relaxed")
            relaxed_filter = self.fiscal_extractor.to_relaxed_filter(fiscal_info)
            if relaxed_filter and relaxed_filter != where_clause:
                # Re-query with relaxed filter
                try:
                    relaxed_results = self.collection.query(
                        query_embeddings=[q_vec],
                        n_results=min(top_k * 2, self.collection.count()),
                        where=relaxed_filter,
                        include=["documents", "metadatas", "distances"],
                    )
                    # Add new results not already in chunks
                    existing_ids = {c["metadata"].get("chunk_id") for c in chunks}
                    for text, meta, dist in zip(
                        relaxed_results["documents"][0],
                        relaxed_results["metadatas"][0],
                        relaxed_results["distances"][0],
                    ):
                        if meta.get("chunk_id") not in existing_ids:
                            score = 1.0 - float(dist)
                            chunks.append({
                                "text": text,
                                "metadata": meta,
                                "score": round(score, 6),
                                "retriever": "dense",
                                "fiscal_filtered": True,
                                "relaxed_match": True,
                            })
                except Exception as e:
                    logger.warning(f"Relaxed filter query failed: {e}")

        # Sort by score descending (should already be, but be explicit)
        chunks.sort(key=lambda x: x["score"], reverse=True)

        # Trim to top_k
        chunks = chunks[:top_k]

        self._log_retrieval(query, chunks, latency_ms, method="dense", fiscal_info=fiscal_info)
        logger.debug(f"DenseRetriever: {len(chunks)} chunks in {latency_ms:.0f}ms")
        return chunks

    def _log_retrieval(
        self,
        query: str,
        chunks: List[dict],
        latency_ms: float,
        method: str,
        fiscal_info: Optional[Dict] = None,
    ):
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
            "fiscal_period_detected": fiscal_info.get("raw") if fiscal_info else None,
            "fiscal_filtered": any(c.get("fiscal_filtered") for c in chunks),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
