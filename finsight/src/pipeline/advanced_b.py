"""
advanced_b.py — Variant 3: Hybrid Retrieval + Cross-Encoder Reranker
Pipeline: BM25(top-20) + Dense(top-20) → RRF Merge → Reranker → Generator (top-5)

Best of both worlds:
- BM25 handles keyword-rich queries (company names, exact metrics, tickers)
- Dense handles semantic/conceptual queries (risks, strategy, outlook)
- RRF fusion deduplicates and ranks without score normalisation
- Reranker then refines the merged candidate set
"""

import time
from typing import Dict

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.generator import Generator
from src.generation.citation_formatter import format_citations
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedBPipeline:
    """
    V3 — Hybrid retrieval (BM25 + dense + RRF) with cross-encoder reranking.
    Best for: cross-document synthesis, trend queries, risk factor questions,
              questions mixing exact financial terms with conceptual context.
    """

    VARIANT_NAME = "v3_advanced_b"
    DESCRIPTION = "Hybrid retrieval (BM25 + dense + RRF) + cross-encoder reranking"

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self._retriever = None
        self._reranker = None
        self._generator = None

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(self.cfg)
        return self._retriever

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.cfg)
        return self._reranker

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(self.cfg)
        return self._generator

    def ask(self, question: str) -> Dict:
        """
        Answer a question using hybrid retrieval + reranking.
        """
        t0 = time.time()
        rerank_top_k = self.cfg["retrieval"]["rerank_top_k"]

        # Step 1: Hybrid retrieval — BM25 + dense, fused by RRF
        candidates = self.retriever.retrieve(question)

        # Step 2: Rerank the merged candidates
        reranked = self.reranker.rerank(question, candidates, top_k=rerank_top_k)

        # Step 3: Generate
        gen_result = self.generator.generate(question, reranked)

        # Step 4: Citations
        citations = format_citations(gen_result["answer"], reranked)

        total_latency = (time.time() - t0) * 1000
        top_rerank_score = reranked[0]["rerank_score"] if reranked else 0.0

        # Count how many candidates came from each source
        found_by_both = sum(1 for c in candidates if c.get("found_by") == "both")
        found_by_dense = sum(1 for c in candidates if c.get("found_by") == "dense")
        found_by_sparse = sum(1 for c in candidates if c.get("found_by") == "sparse")

        result = {
            "answer": gen_result["answer"],
            "citations": citations,
            "retrieved_chunks": reranked,
            "all_candidates": candidates,
            "context_used": gen_result.get("context_used", ""),
            "latency_ms": round(total_latency, 2),
            "generation_latency_ms": gen_result.get("latency_ms", 0),
            "variant": self.VARIANT_NAME,
            "model": gen_result.get("model", ""),
            "insufficient_evidence": gen_result.get("insufficient_evidence", False),
            "input_tokens": gen_result.get("input_tokens", 0),
            "output_tokens": gen_result.get("output_tokens", 0),
            "total_tokens": gen_result.get("total_tokens", 0),
            "top_rerank_score": top_rerank_score,
            "n_candidates": len(candidates),
            "n_reranked": len(reranked),
            "fusion_stats": {
                "found_by_both": found_by_both,
                "found_by_dense_only": found_by_dense,
                "found_by_sparse_only": found_by_sparse,
            },
            "error": gen_result.get("error"),
        }

        logger.info(
            f"[V3 Adv-B] Q: '{question[:60]}...' | "
            f"Candidates: {len(candidates)} (both={found_by_both}, d={found_by_dense}, s={found_by_sparse}) "
            f"→ Reranked: {len(reranked)} | "
            f"Top rerank: {top_rerank_score:.4f} | "
            f"Latency: {total_latency:.0f}ms"
        )
        return result
