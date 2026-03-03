"""
baseline.py — Variant 1: Dense Vector RAG (Baseline)
Pipeline: DenseRetriever → Generator (no reranking)

This is the simplest variant and the performance floor.
top-k chunks retrieved by cosine similarity, passed directly to the LLM.
"""

import time
from typing import Dict, List

from src.retrieval.dense_retriever import DenseRetriever
from src.generation.generator import Generator
from src.generation.citation_formatter import format_citations
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaselinePipeline:
    """
    V1 — Baseline dense RAG pipeline.
    Suitable for: factual lookups, simple extractions.
    Weakness: no reranking may allow less-relevant chunks into context.
    """

    VARIANT_NAME = "v1_baseline"
    DESCRIPTION = "Dense vector retrieval (no reranking)"

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self._retriever = None
        self._generator = None

    @property
    def retriever(self) -> DenseRetriever:
        if self._retriever is None:
            self._retriever = DenseRetriever(self.cfg)
        return self._retriever

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(self.cfg)
        return self._generator

    def ask(self, question: str) -> Dict:
        """
        Answer a question using baseline dense RAG.

        Returns:
            dict with: answer, citations, retrieved_chunks, context_used,
                       latency_ms, variant, model, insufficient_evidence,
                       input_tokens, output_tokens
        """
        t0 = time.time()
        top_k = self.cfg["retrieval"]["final_context_k"]

        # Retrieve
        retrieved = self.retriever.retrieve(question, top_k=top_k)

        # Generate
        gen_result = self.generator.generate(question, retrieved)

        # Format citations
        citations = format_citations(gen_result["answer"], retrieved)

        total_latency = (time.time() - t0) * 1000

        result = {
            "answer": gen_result["answer"],
            "citations": citations,
            "retrieved_chunks": retrieved,
            "context_used": gen_result.get("context_used", ""),
            "latency_ms": round(total_latency, 2),
            "generation_latency_ms": gen_result.get("latency_ms", 0),
            "variant": self.VARIANT_NAME,
            "model": gen_result.get("model", ""),
            "insufficient_evidence": gen_result.get("insufficient_evidence", False),
            "input_tokens": gen_result.get("input_tokens", 0),
            "output_tokens": gen_result.get("output_tokens", 0),
            "total_tokens": gen_result.get("total_tokens", 0),
            "error": gen_result.get("error"),
        }

        logger.info(
            f"[V1 Baseline] Q: '{question[:60]}...' | "
            f"Chunks: {len(retrieved)} | "
            f"Latency: {total_latency:.0f}ms | "
            f"Tokens: {result['total_tokens']}"
        )
        return result
