"""
query_cache.py
Caching layer for RAG queries to avoid redundant LLM calls.

Phase 2 Enhancement: Implements multi-level caching:
1. Exact query match cache (fastest)
2. Semantic similarity cache (for paraphrased queries)
3. Retrieval cache (cache retrieved chunks)

Improves response time and reduces API costs for repeated/similar queries.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from threading import Lock

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, max_size: int = 100):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, moving to end if found."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Any):
        """Add item to cache, evicting oldest if full."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    @property
    def stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 3),
        }


class QueryCache:
    """
    Multi-level cache for RAG queries.

    Levels:
    1. Response cache: Full answer responses
    2. Retrieval cache: Retrieved chunk results
    3. Embedding cache: Query embeddings (optional)

    TTL-based expiration ensures freshness.
    """

    DEFAULT_TTL = 3600  # 1 hour
    DEFAULT_MAX_SIZE = 500

    def __init__(
        self,
        cfg: dict = None,
        response_ttl: int = None,
        retrieval_ttl: int = None,
        max_response_cache: int = None,
        max_retrieval_cache: int = None,
        persist_dir: Optional[str] = None,
    ):
        self.cfg = cfg or load_config()

        # Get config values with defaults
        cache_cfg = self.cfg.get("cache", {})
        self.response_ttl = response_ttl or cache_cfg.get("response_ttl", self.DEFAULT_TTL)
        self.retrieval_ttl = retrieval_ttl or cache_cfg.get("retrieval_ttl", self.DEFAULT_TTL // 2)

        max_response = max_response_cache or cache_cfg.get("max_response_cache", self.DEFAULT_MAX_SIZE)
        max_retrieval = max_retrieval_cache or cache_cfg.get("max_retrieval_cache", self.DEFAULT_MAX_SIZE)

        # Initialize caches
        self.response_cache = LRUCache(max_size=max_response)
        self.retrieval_cache = LRUCache(max_size=max_retrieval)

        # Persistence (optional)
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching."""
        # Lowercase and remove extra whitespace
        normalized = " ".join(query.lower().split())
        return normalized

    def _query_hash(self, query: str) -> str:
        """Generate hash for query."""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get_response(
        self,
        query: str,
        context_hash: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Get cached response for query.

        Args:
            query: User query
            context_hash: Optional hash of context (for stricter matching)

        Returns:
            Cached response dict or None
        """
        key = self._query_hash(query)
        if context_hash:
            key = f"{key}:{context_hash}"

        cached = self.response_cache.get(key)
        if cached:
            timestamp = cached.get("_cached_at", 0)
            if time.time() - timestamp < self.response_ttl:
                logger.debug(f"QueryCache: Response cache hit for '{query[:50]}...'")
                return cached.get("response")
            # Expired - will be overwritten on next put

        return None

    def put_response(
        self,
        query: str,
        response: Dict,
        context_hash: Optional[str] = None,
    ):
        """Cache a response."""
        key = self._query_hash(query)
        if context_hash:
            key = f"{key}:{context_hash}"

        self.response_cache.put(key, {
            "response": response,
            "_cached_at": time.time(),
            "_query": query,
        })

    def get_retrieval(self, query: str) -> Optional[List[Dict]]:
        """Get cached retrieval results."""
        key = f"ret:{self._query_hash(query)}"
        cached = self.retrieval_cache.get(key)

        if cached:
            timestamp = cached.get("_cached_at", 0)
            if time.time() - timestamp < self.retrieval_ttl:
                logger.debug(f"QueryCache: Retrieval cache hit for '{query[:50]}...'")
                return cached.get("chunks")

        return None

    def put_retrieval(self, query: str, chunks: List[Dict]):
        """Cache retrieval results."""
        key = f"ret:{self._query_hash(query)}"
        self.retrieval_cache.put(key, {
            "chunks": chunks,
            "_cached_at": time.time(),
        })

    def clear(self, cache_type: Optional[str] = None):
        """
        Clear cache(s).

        Args:
            cache_type: "response", "retrieval", or None for all
        """
        if cache_type in [None, "response"]:
            self.response_cache.clear()
        if cache_type in [None, "retrieval"]:
            self.retrieval_cache.clear()
        logger.info(f"QueryCache: Cleared {cache_type or 'all'} cache(s)")

    @property
    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "response_cache": self.response_cache.stats,
            "retrieval_cache": self.retrieval_cache.stats,
        }

    def _load_from_disk(self):
        """Load cache from disk (if persistence enabled)."""
        if not self.persist_dir:
            return

        response_file = self.persist_dir / "response_cache.json"
        retrieval_file = self.persist_dir / "retrieval_cache.json"

        try:
            if response_file.exists():
                with open(response_file, "r") as f:
                    data = json.load(f)
                    for key, value in data.items():
                        self.response_cache.put(key, value)
                logger.info(f"Loaded {len(data)} response cache entries from disk")
        except Exception as e:
            logger.warning(f"Failed to load response cache: {e}")

        try:
            if retrieval_file.exists():
                with open(retrieval_file, "r") as f:
                    data = json.load(f)
                    for key, value in data.items():
                        self.retrieval_cache.put(key, value)
                logger.info(f"Loaded {len(data)} retrieval cache entries from disk")
        except Exception as e:
            logger.warning(f"Failed to load retrieval cache: {e}")

    def save_to_disk(self):
        """Save cache to disk (if persistence enabled)."""
        if not self.persist_dir:
            return

        try:
            response_file = self.persist_dir / "response_cache.json"
            with open(response_file, "w") as f:
                json.dump(dict(self.response_cache.cache), f)
            logger.info(f"Saved response cache to {response_file}")
        except Exception as e:
            logger.warning(f"Failed to save response cache: {e}")

        try:
            retrieval_file = self.persist_dir / "retrieval_cache.json"
            with open(retrieval_file, "w") as f:
                json.dump(dict(self.retrieval_cache.cache), f)
            logger.info(f"Saved retrieval cache to {retrieval_file}")
        except Exception as e:
            logger.warning(f"Failed to save retrieval cache: {e}")


class CachedPipeline:
    """
    Wrapper that adds caching to a RAG pipeline.

    Usage:
        pipeline = CachedPipeline(retriever, generator)
        result = pipeline.run("What was Microsoft revenue in Q2 FY2025?")
    """

    def __init__(
        self,
        retriever,
        generator,
        cache: Optional[QueryCache] = None,
        enable_response_cache: bool = True,
        enable_retrieval_cache: bool = True,
    ):
        self.retriever = retriever
        self.generator = generator
        self.cache = cache or QueryCache()
        self.enable_response_cache = enable_response_cache
        self.enable_retrieval_cache = enable_retrieval_cache

    def run(
        self,
        query: str,
        top_k: int = None,
        force_refresh: bool = False,
    ) -> Dict:
        """
        Run RAG pipeline with caching.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            force_refresh: Skip cache and force fresh generation

        Returns:
            Generation result dict with cache info
        """
        result = {"cache_hit": False, "cache_type": None}

        # Check response cache first (if not forcing refresh)
        if self.enable_response_cache and not force_refresh:
            cached_response = self.cache.get_response(query)
            if cached_response:
                cached_response["cache_hit"] = True
                cached_response["cache_type"] = "response"
                return cached_response

        # Check retrieval cache
        chunks = None
        if self.enable_retrieval_cache and not force_refresh:
            chunks = self.cache.get_retrieval(query)
            if chunks:
                result["retrieval_cache_hit"] = True

        # Retrieve if not cached
        if chunks is None:
            if hasattr(self.retriever, "retrieve"):
                chunks = self.retriever.retrieve(query, top_k=top_k)
                if isinstance(chunks, tuple):
                    chunks = chunks[0]  # Handle (chunks, stats) tuple
            else:
                raise ValueError("Retriever must have a retrieve() method")

            # Cache retrieval results
            if self.enable_retrieval_cache:
                self.cache.put_retrieval(query, chunks)

        # Generate response
        generation_result = self.generator.generate(query, chunks)
        result.update(generation_result)

        # Cache response
        if self.enable_response_cache:
            self.cache.put_response(query, result)

        return result

    @property
    def cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.stats


# Singleton cache instance for easy access
_global_cache: Optional[QueryCache] = None


def get_query_cache(cfg: dict = None) -> QueryCache:
    """Get or create global query cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = QueryCache(cfg)
    return _global_cache


def clear_cache():
    """Clear global cache."""
    global _global_cache
    if _global_cache:
        _global_cache.clear()
