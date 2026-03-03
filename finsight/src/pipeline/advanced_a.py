"""
advanced_a.py — Variant 2: Dense RAG + Cross-Encoder Reranker
Pipeline: DenseRetriever (top-20) → Reranker → Generator (top-5)

Retrieves more candidates than V1, then re-scores with a cross-encoder
to promote the most relevant chunks into the LLM context.
This significantly reduces hallucinations on numeric/factual questions.
"""

import time
from typing import Dict

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import Reranker
from src.generation.generator import Generator
from src.generation.citation_formatter import format_citations
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedAPipeline:
    """
    V2 — Dense retrieval with cross-encoder reranking.
    Best for: factual questions, numeric extraction, single-document queries.
    """

    VARIANT_NAME = "v2_advanced_a"
    DESCRIPTION = "Dense retrieval + cross-encoder reranking"

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self._retriever = None
        self._reranker = None
        self._generator = None

    @property
    def retriever(self) -> DenseRetriever:
        if self._retriever is None:
            self._retriever = DenseRetriever(self.cfg)
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
        Answer a question using dense retrieval + reranking.
        """
        t0 = time.time()
        dense_top_k = self.cfg["retrieval"]["dense_top_k"]
        rerank_top_k = self.cfg["retrieval"]["rerank_top_k"]

        # Step 1: Dense retrieval — wider net
        candidates = self.retriever.retrieve(question, top_k=dense_top_k)

        # Step 2: Rerank — precision cut
        reranked = self.reranker.rerank(question, candidates, top_k=rerank_top_k)

        # Step 3: Generate from reranked top-k
        gen_result = self.generator.generate(question, reranked)

        # Step 4: Format citations
        citations = format_citations(gen_result["answer"], reranked)

        total_latency = (time.time() - t0) * 1000

        top_rerank_score = (
            reranked[0]["rerank_score"] if reranked else 0.0
        )

        result = {
            "answer": gen_result["answer"],
            "citations": citations,
            "retrieved_chunks": reranked,
            "all_candidates": candidates,      # full candidate set for inspection
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
            "error": gen_result.get("error"),
        }

        logger.info(
            f"[V2 Adv-A] Q: '{question[:60]}...' | "
            f"Candidates: {len(candidates)} → Reranked: {len(reranked)} | "
            f"Top rerank score: {top_rerank_score:.4f} | "
            f"Latency: {total_latency:.0f}ms"
        )
        return result
