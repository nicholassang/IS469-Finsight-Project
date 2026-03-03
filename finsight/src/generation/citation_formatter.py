"""
citation_formatter.py
Parses [Doc-N] markers from generated answers and maps them to chunk metadata.
Returns structured citation objects for display in the Streamlit app.
"""

import re
from typing import List, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Matches [Doc-1], [Doc-12], etc. — case-insensitive
CITATION_PATTERN = re.compile(r"\[Doc-(\d+)\]", re.IGNORECASE)

# Length of snippet to show in citations panel
SNIPPET_MAX_CHARS = 300


def extract_citation_refs(answer: str) -> List[int]:
    """
    Extract all [Doc-N] references from an answer string.
    Returns sorted unique list of 1-based indices.
    """
    matches = CITATION_PATTERN.findall(answer)
    return sorted(set(int(m) for m in matches))


def format_citations(
    answer: str,
    chunks: List[Dict],
) -> List[Dict]:
    """
    Build structured citation objects from an answer + its retrieved chunks.

    Args:
        answer: generated answer text containing [Doc-N] markers
        chunks: list of retrieved chunk dicts (order matches Doc-N numbering)

    Returns:
        List of citation dicts, one per unique [Doc-N] reference in the answer.
        Each dict contains: ref, doc_type, fiscal_period, filing_date, page_number,
                           source_file, source_url, section_title, snippet, chunk_id
    """
    cited_indices = extract_citation_refs(answer)
    citations = []

    for idx in cited_indices:
        # idx is 1-based; chunks is 0-based
        chunk_pos = idx - 1
        if chunk_pos < 0 or chunk_pos >= len(chunks):
            logger.warning(f"Citation [Doc-{idx}] references a chunk that doesn't exist (only {len(chunks)} chunks)")
            continue

        chunk = chunks[chunk_pos]
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "")

        # Truncate snippet cleanly at a sentence boundary if possible
        snippet = _make_snippet(text, SNIPPET_MAX_CHARS)

        citations.append({
            "ref": f"Doc-{idx}",
            "doc_type": meta.get("doc_type", ""),
            "fiscal_period": meta.get("fiscal_period", ""),
            "filing_date": meta.get("filing_date", ""),
            "page_number": meta.get("page_number", ""),
            "source_file": meta.get("source_file", ""),
            "source_url": meta.get("source_url", ""),
            "section_title": meta.get("section_title") or "",
            "snippet": snippet,
            "chunk_id": meta.get("chunk_id", ""),
            "score": chunk.get("rerank_score", chunk.get("score", 0.0)),
        })

    return citations


def _make_snippet(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, trying to end at a sentence boundary."""
    if len(text) <= max_chars:
        return text.strip()

    truncated = text[:max_chars]
    # Find last period, exclamation, or question mark
    last_period = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )
    if last_period > max_chars * 0.5:
        return truncated[:last_period + 1].strip()
    return truncated.strip() + "..."


def annotate_answer_html(answer: str, citations: List[Dict]) -> str:
    """
    Replace [Doc-N] markers in the answer with styled HTML spans.
    For use in the Streamlit app (via st.markdown with unsafe_allow_html).
    """
    # Build a lookup for tooltip content
    cite_map = {c["ref"]: c for c in citations}

    def replace_citation(match):
        ref = f"Doc-{match.group(1)}"
        c = cite_map.get(ref, {})
        tooltip = f"{c.get('doc_type','')} | {c.get('fiscal_period','')} | p.{c.get('page_number','')}"
        return f'<span title="{tooltip}" style="background:#e8f4fd;padding:1px 4px;border-radius:3px;font-size:0.85em;font-weight:bold;color:#1a6396">[{ref}]</span>'

    return CITATION_PATTERN.sub(replace_citation, answer)
