"""
generator.py
Assembles prompts from retrieved chunks and calls the configured LLM backend.

Supports two backends — set in config/settings.yaml under generation.backend:
  - "openai"  : OpenAI API (gpt-4o-mini, gpt-4o, etc.) — requires OPENAI_API_KEY
  - "ollama"  : Local Ollama server (llama3.2, mistral, qwen2.5, etc.) — free, no API key

Enforces all guardrails: grounded answers only, no investment advice, citations required.

Enhanced with ContextManager for dynamic context truncation to prevent token limit errors.
"""

import os
import time
from typing import List, Dict, Optional, Tuple

from src.utils.config_loader import load_config, load_prompts
from src.utils.logger import get_logger
from src.generation.context_manager import ContextManager

logger = get_logger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


_OUT_OF_SCOPE_RESPONSE = (
    "I can only answer questions about Microsoft Corporation financial "
    "performance, business segments, and strategy. Please ask a "
    "relevant question."
)

_CITATION_SYSTEM_ADDENDUM = (
    "\n\nYou must cite the source document for every factual claim using "
    "[Source: document_name] format. If the context does not contain "
    "enough information to answer, say 'The provided documents do not "
    "contain sufficient information to answer this question' rather "
    "than guessing."
)


def _is_investment_advice_request(question: str, keywords: List[str]) -> bool:
    q_lower = question.lower()
    return any(kw.lower() in q_lower for kw in keywords)


def _is_out_of_scope(question: str, allowed_topics: List[str]) -> bool:
    """Check if the question is related to any allowed topic."""
    q_lower = question.lower()
    return not any(topic.lower() in q_lower for topic in allowed_topics)


def format_context(chunks: List[Dict], max_chunk_chars: int = 1500, include_truncation_warning: bool = True) -> str:
    """
    Format chunks into a context string for the prompt.

    Args:
        chunks: List of chunk dictionaries
        max_chunk_chars: Maximum characters per chunk (legacy, ContextManager handles this better)
        include_truncation_warning: Whether to note truncated chunks

    Returns:
        Formatted context string with [Doc-N] headers
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        m = chunk.get("metadata", {})
        header = (
            f"[Doc-{i}] "
            f"{m.get('doc_type', 'Filing')} | "
            f"Period: {m.get('fiscal_period', 'N/A')} | "
            f"Filed: {m.get('filing_date', 'N/A')} | "
            f"Page: {m.get('page_number', '?')} | "
            f"File: {m.get('source_file', 'N/A')}"
        )

        # Add truncation indicator if chunk was truncated by ContextManager
        if chunk.get("truncated") and include_truncation_warning:
            header += " [TRUNCATED]"

        text = chunk.get("text", "").strip()
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars] + " ...[truncated]"
        lines.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(lines)


class OllamaBackend:
    """
    Calls a locally running Ollama server.
    Install:  https://ollama.com/download
    Pull model: ollama pull llama3.2
    Start:    ollama serve
    """

    def __init__(self, cfg: dict):
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests not installed. Run: pip install requests")
        self.model = cfg["generation"]["ollama_model"]
        self.base_url = cfg["generation"].get("ollama_base_url", "http://localhost:11434")
        self.temperature = cfg["generation"]["temperature"]
        self.max_tokens = cfg["generation"]["max_tokens"]

    def chat(self, system_prompt: str, user_prompt: str) -> Dict:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        t0 = time.time()
        try:
            resp = _requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.time() - t0) * 1000
            answer = data.get("message", {}).get("content", "").strip()
            input_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)
            return {
                "answer": answer,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_ms": round(latency_ms, 2),
                "model": self.model,
                "error": None,
            }
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            logger.error(f"Ollama request failed: {e}")
            return {
                "answer": f"Error contacting Ollama: {e}",
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "latency_ms": round(latency_ms, 2),
                "model": self.model,
                "error": str(e),
            }

    def is_running(self) -> bool:
        try:
            resp = _requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            resp = _requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []


class OpenAIBackend:
    def __init__(self, cfg: dict):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai not installed. Run: pip install openai")
        # Prefer api_key from config (e.g. "dummy" for local vLLM), fall back to env var
        api_key = cfg["generation"].get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "No API key found. Set 'api_key' in settings.yaml or OPENAI_API_KEY env var."
            )
        base_url = cfg["generation"].get("base_url")  # None → real OpenAI
        timeout = cfg["generation"].get("timeout_seconds", 120)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = cfg["generation"]["model"]
        self.temperature = cfg["generation"]["temperature"]
        self.max_tokens = cfg["generation"]["max_tokens"]
        logger.info(
            f"OpenAIBackend: base_url={base_url or 'api.openai.com'} "
            f"model={self.model} timeout={timeout}s"
        )

    def chat(self, system_prompt: str, user_prompt: str) -> Dict:
        t0 = time.time()
        logger.info(f"OpenAIBackend: sending request to {self.client.base_url} model={self.model}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            latency_ms = (time.time() - t0) * 1000
            answer = response.choices[0].message.content.strip()
            usage = response.usage
            return {
                "answer": answer,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "latency_ms": round(latency_ms, 2),
                "model": self.model,
                "error": None,
            }
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            logger.error(f"OpenAI API call failed: {e}")
            return {
                "answer": f"Error calling OpenAI: {e}",
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "latency_ms": round(latency_ms, 2),
                "model": self.model,
                "error": str(e),
            }


class Generator:
    """
    Handles prompt assembly and LLM calls.
    Backend selected from config: generation.backend = "openai" | "ollama"

    Enhanced with ContextManager for automatic context truncation to prevent token limit errors.
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self.prompts = load_prompts()
        backend = self.cfg["generation"].get("backend", "openai").lower()

        # Initialize ContextManager for dynamic truncation
        model_name = self.cfg["generation"].get("model", "qwen2.5-14b")
        self.context_manager = ContextManager(
            model_name=model_name,
            reserved_for_output=self.cfg["generation"].get("max_tokens", 512),
            reserved_for_prompt=800,  # Approximate prompt template size
        )

        if backend == "ollama":
            self._backend = OllamaBackend(self.cfg)
            logger.info(
                f"Generator: Ollama backend — "
                f"model={self.cfg['generation']['ollama_model']} "
                f"@ {self.cfg['generation'].get('ollama_base_url', 'http://localhost:11434')}"
            )
        elif backend == "openai":
            self._backend = OpenAIBackend(self.cfg)
            logger.info(f"Generator: OpenAI backend — model={self.cfg['generation']['model']}")
        else:
            raise ValueError(
                f"Unknown backend: '{backend}'. Use 'openai' or 'ollama' in settings.yaml"
            )

    def generate(self, question: str, chunks: List[Dict]) -> Dict:
        guardrails = self.cfg.get("guardrails", {})

        # Guardrail 1: Investment advice check (existing)
        invest_keywords = self.prompts.get("investment_advice_keywords", [])
        if _is_investment_advice_request(question, invest_keywords):
            return self._investment_advice_refusal()

        # Guardrail 2: Out-of-scope check
        if guardrails.get("out_of_scope_check", False):
            allowed_topics = guardrails.get("allowed_topics", [])
            if allowed_topics and _is_out_of_scope(question, allowed_topics):
                return self._out_of_scope_response()

        if not chunks:
            return self._no_context_response()

        # Use ContextManager to fit chunks within token budget
        fitted_chunks, context_stats = self.context_manager.fit_context(
            chunks,
            min_chunks=3,  # Always try to include at least 3 chunks
        )

        if context_stats.get('truncated'):
            logger.info(
                f"Generator: Context truncated from {context_stats['original_count']} to "
                f"{context_stats['selected_count']} chunks "
                f"({context_stats['original_tokens']} → {context_stats['selected_tokens']} tokens)"
            )

        context = format_context(
            fitted_chunks,
            max_chunk_chars=self.cfg["generation"].get("max_chunk_chars", 1800)
        )
        system_prompt = self.prompts["qa_system"]

        # Guardrail 3: Append citation requirement to system prompt
        if guardrails.get("require_citations", False):
            system_prompt = system_prompt + _CITATION_SYSTEM_ADDENDUM

        user_prompt = self.prompts["qa_user"].format(context=context, question=question)

        raw = self._backend.chat(system_prompt, user_prompt)
        insufficient = (
            self.prompts.get("insufficient_evidence_phrase", "Insufficient evidence")
            in raw.get("answer", "")
        )
        return {
            "answer": raw["answer"],
            "raw_response": raw["answer"],
            "model": raw["model"],
            "input_tokens": raw["input_tokens"],
            "output_tokens": raw["output_tokens"],
            "total_tokens": raw["total_tokens"],
            "latency_ms": raw["latency_ms"],
            "context_used": context,
            "insufficient_evidence": insufficient,
            "error": raw.get("error"),
            "context_stats": context_stats,  # Include truncation stats
        }

    def _investment_advice_refusal(self) -> Dict:
        answer = (
            "This tool is for financial research purposes only and cannot provide "
            "investment advice, stock recommendations, or price targets. "
            "Please consult a qualified financial advisor for investment decisions."
        )
        model = getattr(self._backend, "model", "local")
        return {
            "answer": answer, "raw_response": answer, "model": model,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "latency_ms": 0.0, "context_used": "",
            "insufficient_evidence": False, "refused_investment_advice": True, "error": None,
        }

    def _out_of_scope_response(self) -> Dict:
        model = getattr(self._backend, "model", "local")
        return {
            "answer": _OUT_OF_SCOPE_RESPONSE,
            "raw_response": _OUT_OF_SCOPE_RESPONSE,
            "model": model,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "latency_ms": 0.0, "context_used": "",
            "insufficient_evidence": False, "out_of_scope": True, "error": None,
        }

    def _no_context_response(self) -> Dict:
        answer = "Insufficient evidence in the provided filings to answer this question."
        model = getattr(self._backend, "model", "local")
        return {
            "answer": answer, "raw_response": answer, "model": model,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "latency_ms": 0.0, "context_used": "",
            "insufficient_evidence": True, "error": None,
        }
