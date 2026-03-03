"""
dense_indexer.py
Builds and manages the ChromaDB vector index.
Encodes chunk texts with sentence-transformers, upserts into a persistent Chroma collection.
"""

import json
import time
from pathlib import Path
from typing import List, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed — dense indexing unavailable")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed — dense indexing unavailable")


def _check_deps():
    if not CHROMA_AVAILABLE:
        raise RuntimeError("chromadb not installed. Run: pip install chromadb")
    if not ST_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")


class DenseIndexer:
    """
    Manages the ChromaDB collection for FinSight.
    Loads the embedding model once, then handles build/update operations.
    """

    def __init__(self, cfg: dict = None):
        _check_deps()
        self.cfg = cfg or load_config()
        self._model = None
        self._client = None
        self._collection = None

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            model_name = self.cfg["embeddings"]["model"]
            logger.info(f"Loading embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            persist_dir = self.cfg["paths"]["chroma_db"]
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)
            collection_name = self.cfg["chroma"]["collection_name"]
            distance_fn = self.cfg["chroma"]["distance_function"]

            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_fn},
            )
        return self._collection

    def reset_collection(self):
        """Delete and recreate the Chroma collection (full rebuild)."""
        persist_dir = self.cfg["paths"]["chroma_db"]
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        collection_name = self.cfg["chroma"]["collection_name"]
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass
        self._collection = None
        self._client = None
        logger.info("Collection reset — ready for rebuild")

    def index_chunks(self, chunks: List[dict], batch_size: int = None) -> int:
        """
        Encode and upsert a list of tagged chunks into ChromaDB.
        Uses upsert (not add) so it's safe to re-run.

        Returns: number of chunks indexed
        """
        if not chunks:
            logger.warning("index_chunks called with empty list")
            return 0

        batch_size = batch_size or self.cfg["embeddings"]["batch_size"]
        normalize = self.cfg["embeddings"].get("normalize", True)
        total = len(chunks)
        indexed = 0

        logger.info(f"Indexing {total} chunks into ChromaDB ...")
        t0 = time.time()

        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start: batch_start + batch_size]
            texts = [c["text"] for c in batch]
            ids = [c["metadata"]["chunk_id"] for c in batch]
            metadatas = []
            for c in batch:
                # ChromaDB metadata values must be str, int, float, or bool
                meta = {}
                for k, v in c["metadata"].items():
                    if v is None:
                        meta[k] = ""
                    elif isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                metadatas.append(meta)

            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            ).tolist()

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            indexed += len(batch)
            elapsed = time.time() - t0
            logger.info(
                f"  Indexed {indexed}/{total} chunks "
                f"({100*indexed/total:.1f}%) — {elapsed:.1f}s elapsed"
            )

        logger.info(f"Dense indexing complete: {indexed} chunks in {time.time()-t0:.1f}s")
        return indexed

    def get_collection_stats(self) -> dict:
        """Return basic stats about the current collection."""
        count = self.collection.count()
        return {
            "collection_name": self.cfg["chroma"]["collection_name"],
            "vector_count": count,
            "persist_dir": self.cfg["paths"]["chroma_db"],
        }

    def verify_index(self) -> bool:
        """Quick sanity check: verify the collection is non-empty and queryable."""
        try:
            count = self.collection.count()
            if count == 0:
                logger.warning("Dense index is empty")
                return False
            # Try a dummy query
            sample = self.model.encode(["test query"], normalize_embeddings=True).tolist()
            results = self.collection.query(query_embeddings=sample, n_results=1)
            logger.info(f"Dense index OK: {count} vectors, test query successful")
            return True
        except Exception as e:
            logger.error(f"Dense index verification failed: {e}")
            return False
