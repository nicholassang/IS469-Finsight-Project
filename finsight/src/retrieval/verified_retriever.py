"""
verified_retriever.py
Multi-stage verified retrieval that ensures temporal accuracy.

Phase 2 Enhancement: When a query specifies a fiscal period, this retriever:
1. Retrieves initial candidates using hybrid retrieval
2. Verifies that top results match the requested period
3. If temporal mismatch detected, re-retrieves with strict filtering
4. Returns verified, temporally-accurate chunks

This addresses q007-style failures where wrong quarter data was retrieved.
"""

import time
from typing import List, Dict, Optional, Tuple
from collections import Counter

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import FiscalPeriodExtractor
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VerifiedRetriever:
    """
    Multi-stage retrieval with temporal verification.

    Workflow:
    1. Extract fiscal period from query
    2. Perform initial hybrid retrieval
    3. Analyze temporal distribution of results
    4. If majority doesn't match requested period, re-retrieve with strict filter
    5. Return verified results with confidence score
    """

    # Minimum percentage of results that should match requested period
    TEMPORAL_THRESHOLD = 0.4  # At least 40% of top results should match

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self.hybrid_retriever = HybridRetriever(self.cfg)
        self.fiscal_extractor = FiscalPeriodExtractor()

        # Override threshold from config if provided
        retrieval_cfg = self.cfg.get("retrieval", {})
        self.temporal_threshold = retrieval_cfg.get(
            "temporal_verification_threshold",
            self.TEMPORAL_THRESHOLD
        )

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        verify_temporal: bool = True,
    ) -> Tuple[List[Dict], Dict]:
        """
        Retrieve and verify results match requested fiscal period.

        Args:
            query: Search query
            top_k: Maximum results to return
            verify_temporal: Whether to perform temporal verification

        Returns:
            Tuple of (chunks, verification_stats)
        """
        top_k = top_k or self.cfg["retrieval"].get("rerank_top_k", 10)
        t0 = time.time()

        # Extract fiscal period from query
        fiscal_info = self.fiscal_extractor.extract(query)
        requested_period = fiscal_info.get("raw")  # e.g., "Q2 FY2025"

        stats = {
            "requested_period": requested_period,
            "verification_performed": False,
            "re_retrieval_triggered": False,
            "initial_match_rate": None,
            "final_match_rate": None,
            "method": "hybrid",
        }

        # Initial retrieval
        # Fetch extra candidates for verification (2x top_k)
        initial_chunks = self.hybrid_retriever.retrieve(query, top_k=top_k * 2)

        # If no specific period requested or verification disabled, return as-is
        if not requested_period or not verify_temporal:
            stats["latency_ms"] = (time.time() - t0) * 1000
            return initial_chunks[:top_k], stats

        stats["verification_performed"] = True

        # Analyze temporal distribution of initial results
        initial_match_rate = self._analyze_temporal_match(
            initial_chunks[:top_k],
            requested_period
        )
        stats["initial_match_rate"] = initial_match_rate

        # If sufficient match rate, return initial results
        if initial_match_rate >= self.temporal_threshold:
            # Sort to prioritize matching periods
            sorted_chunks = self._sort_by_temporal_match(
                initial_chunks,
                requested_period
            )
            stats["final_match_rate"] = initial_match_rate
            stats["latency_ms"] = (time.time() - t0) * 1000
            logger.info(
                f"VerifiedRetriever: Initial retrieval sufficient "
                f"({initial_match_rate:.0%} match for {requested_period})"
            )
            return sorted_chunks[:top_k], stats

        # Temporal mismatch - trigger re-retrieval with strict filtering
        logger.warning(
            f"VerifiedRetriever: Temporal mismatch detected. "
            f"Only {initial_match_rate:.0%} match for {requested_period}. "
            f"Triggering strict re-retrieval."
        )
        stats["re_retrieval_triggered"] = True

        # Get strictly-filtered chunks via period boost
        strict_chunks = self._strict_period_retrieve(
            query,
            requested_period,
            top_k=top_k
        )

        # Merge: strict results first, then non-duplicate initial results
        merged = self._merge_results(strict_chunks, initial_chunks, top_k)

        final_match_rate = self._analyze_temporal_match(merged, requested_period)
        stats["final_match_rate"] = final_match_rate
        stats["method"] = "verified_hybrid"
        stats["latency_ms"] = (time.time() - t0) * 1000

        logger.info(
            f"VerifiedRetriever: Re-retrieval improved match rate "
            f"from {initial_match_rate:.0%} to {final_match_rate:.0%}"
        )

        return merged, stats

    def _analyze_temporal_match(
        self,
        chunks: List[Dict],
        requested_period: str
    ) -> float:
        """Calculate what fraction of chunks match the requested period."""
        if not chunks:
            return 0.0

        matches = 0
        for chunk in chunks:
            chunk_period = chunk.get("metadata", {}).get("fiscal_period", "")
            if self._periods_match(chunk_period, requested_period):
                matches += 1

        return matches / len(chunks)

    def _periods_match(self, chunk_period: str, requested_period: str) -> bool:
        """
        Check if chunk period matches requested period.

        Handles partial matches:
        - "Q2 FY2025" matches "Q2 FY2025" (exact)
        - "FY2025" matches "Q2 FY2025" (annual contains quarterly)
        - "Q2 FY2025" partially matches "FY2025" (quarterly is part of annual)
        """
        if not chunk_period or not requested_period:
            return False

        chunk_lower = chunk_period.lower()
        requested_lower = requested_period.lower()

        # Exact match
        if chunk_lower == requested_lower:
            return True

        # Extract fiscal year from both (the 4-digit year part)
        import re
        chunk_year_match = re.search(r'(20\d{2})', chunk_period)
        requested_year_match = re.search(r'(20\d{2})', requested_period)

        if chunk_year_match and requested_year_match:
            chunk_year = chunk_year_match.group(1)
            requested_year = requested_year_match.group(1)

            # Must be same fiscal year
            if chunk_year != requested_year:
                return False

            # If chunk is full-year (FY20XX without Q), it contains all quarters
            chunk_has_quarter = re.search(r'q[1-4]', chunk_lower)
            if not chunk_has_quarter:
                return True  # FY2025 matches Q2 FY2025

            # If both have quarters, they must match
            requested_quarter = re.search(r'q([1-4])', requested_lower)
            chunk_quarter = re.search(r'q([1-4])', chunk_lower)
            if requested_quarter and chunk_quarter:
                return requested_quarter.group(1) == chunk_quarter.group(1)

        # Check if chunk contains the requested period
        if requested_lower in chunk_lower:
            return True

        return False

    def _sort_by_temporal_match(
        self,
        chunks: List[Dict],
        requested_period: str
    ) -> List[Dict]:
        """Sort chunks to prioritize those matching requested period."""
        def sort_key(chunk):
            chunk_period = chunk.get("metadata", {}).get("fiscal_period", "")
            matches = self._periods_match(chunk_period, requested_period)
            # Matching chunks get priority (sort first)
            # Within each group, maintain original order (by index)
            return (0 if matches else 1, chunks.index(chunk))

        return sorted(chunks, key=sort_key)

    def _strict_period_retrieve(
        self,
        query: str,
        period: str,
        top_k: int
    ) -> List[Dict]:
        """
        Retrieve chunks strictly filtered to the specified period.
        Uses the hybrid retriever's period boost mechanism.
        """
        # Use _retrieve_by_period from hybrid retriever
        return self.hybrid_retriever._retrieve_by_period(
            query,
            periods=[period],
            top_k=top_k
        )

    def _merge_results(
        self,
        primary: List[Dict],
        secondary: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Merge primary and secondary results, avoiding duplicates.
        Primary results take precedence.
        """
        seen_ids = set()
        merged = []

        # Add all primary results first
        for chunk in primary:
            chunk_id = chunk.get("metadata", {}).get("chunk_id", id(chunk))
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(chunk)

        # Fill remaining slots with secondary results
        for chunk in secondary:
            if len(merged) >= top_k:
                break
            chunk_id = chunk.get("metadata", {}).get("chunk_id", id(chunk))
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(chunk)

        return merged[:top_k]


class RetrieverWithFallback:
    """
    Retriever wrapper that handles various failure modes gracefully.

    Provides:
    - Automatic fallback from verified to standard retrieval on errors
    - Configurable minimum result threshold
    - Logging for debugging retrieval issues
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self.verified_retriever = VerifiedRetriever(self.cfg)
        self.hybrid_retriever = self.verified_retriever.hybrid_retriever

        self.min_results = self.cfg.get("retrieval", {}).get(
            "min_results_threshold", 3
        )

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_verification: bool = True,
    ) -> Tuple[List[Dict], Dict]:
        """
        Retrieve with automatic fallback.

        Args:
            query: Search query
            top_k: Maximum results
            use_verification: Whether to use verified retrieval

        Returns:
            Tuple of (chunks, stats)
        """
        top_k = top_k or self.cfg["retrieval"].get("rerank_top_k", 10)

        try:
            if use_verification:
                chunks, stats = self.verified_retriever.retrieve(
                    query,
                    top_k=top_k
                )

                # Check if we got enough results
                if len(chunks) >= self.min_results:
                    return chunks, stats

                # Insufficient results - try without verification
                logger.warning(
                    f"RetrieverWithFallback: Verified retrieval returned only "
                    f"{len(chunks)} results. Falling back to standard retrieval."
                )

            # Standard hybrid retrieval (fallback)
            chunks = self.hybrid_retriever.retrieve(query, top_k=top_k)
            stats = {
                "method": "hybrid_fallback",
                "verification_performed": False,
            }
            return chunks, stats

        except Exception as e:
            logger.error(f"RetrieverWithFallback: Error during retrieval: {e}")
            # Emergency fallback
            try:
                chunks = self.hybrid_retriever.retrieve(query, top_k=top_k)
                return chunks, {"method": "emergency_fallback", "error": str(e)}
            except Exception as e2:
                logger.error(f"RetrieverWithFallback: All retrieval failed: {e2}")
                return [], {"method": "failed", "error": str(e2)}
