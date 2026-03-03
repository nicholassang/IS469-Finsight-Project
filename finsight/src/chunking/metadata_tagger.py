"""
metadata_tagger.py
Attaches document-level and chunk-level metadata to each chunk dict.
Output chunks are fully self-describing — all 12 schema fields populated.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Metadata schema ───────────────────────────────────────────────────────────
# Reference for all consumers (indexer, retriever, app)
METADATA_SCHEMA = {
    "chunk_id": "str — unique identifier: {company}_{doctype}_{period}_p{page}_c{chunk}",
    "company": "str — always 'Grab Holdings'",
    "ticker": "str — 'GRAB'",
    "doc_type": "str — enum: 20-F, 6-K, earnings_pr, investor_deck, earnings_deck",
    "source_file": "str — filename in data/raw/",
    "source_url": "str — official URL",
    "filing_date": "str — ISO 8601 YYYY-MM-DD",
    "fiscal_period": "str — e.g. 'FY2023', 'Q2 2024'",
    "page_number": "int — 1-indexed",
    "section_title": "str | null — best-effort heading detection",
    "chunk_index": "int — position within document",
    "token_count": "int — approximate token count",
}

# Section heading heuristic: lines that are short, mostly capitalised, no period at end
HEADING_PATTERN_MIN_CAPS_RATIO = 0.5
MAX_HEADING_LINE_LEN = 100


def _make_chunk_id(doc_id: str, page_number: int, chunk_index: int) -> str:
    """Generate a stable, readable chunk ID."""
    # Sanitise doc_id for use in an ID string
    safe_doc = doc_id.replace(" ", "_").replace("/", "_").lower()
    return f"{safe_doc}_p{page_number:04d}_c{chunk_index:04d}"


def _detect_section_title(text: str) -> str | None:
    """
    Heuristic: check if the first non-empty line looks like a section heading.
    Returns the heading or None.
    """
    lines = text.strip().split("\n")
    for line in lines[:3]:  # Only check first 3 lines
        line = line.strip()
        if not line or len(line) < 3 or len(line) > MAX_HEADING_LINE_LEN:
            continue
        # Avoid detecting sentence starts as headings
        if line.endswith(".") or line.endswith(","):
            continue
        # Check capitalisation ratio
        alpha_chars = [c for c in line if c.isalpha()]
        if not alpha_chars:
            continue
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if caps_ratio >= HEADING_PATTERN_MIN_CAPS_RATIO and len(line.split()) <= 12:
            return line
    return None


def tag_chunk(
    chunk: dict,
    doc_config: dict,
    chunk_index_in_doc: int,
) -> dict:
    """
    Attach all metadata fields to a single chunk dict.

    Args:
        chunk: dict from chunker.py (has 'text', 'page_number', 'token_count', etc.)
        doc_config: one entry from settings.yaml documents list
        chunk_index_in_doc: position of this chunk within the document (0-indexed)

    Returns:
        chunk dict with all 12 metadata schema fields added under 'metadata' key
        AND at the top level (for ChromaDB which stores metadata separately)
    """
    page_number = chunk.get("page_number", 0)
    doc_id = doc_config.get("id", "unknown")
    text = chunk.get("text", "")

    chunk_id = _make_chunk_id(doc_id, page_number, chunk_index_in_doc)
    section_title = _detect_section_title(text)

    metadata = {
        "chunk_id": chunk_id,
        "company": doc_config.get("company", "Grab Holdings"),
        "ticker": doc_config.get("ticker", "GRAB"),
        "doc_type": doc_config.get("doc_type", "unknown"),
        "source_file": doc_config.get("filename", ""),
        "source_url": doc_config.get("source_url", ""),
        "filing_date": doc_config.get("filing_date", ""),
        "fiscal_period": doc_config.get("fiscal_period", ""),
        "page_number": page_number,
        "section_title": section_title,
        "chunk_index": chunk_index_in_doc,
        "token_count": chunk.get("token_count", 0),
    }

    tagged = {**chunk, "metadata": metadata, "chunk_id": chunk_id}
    return tagged


def tag_document_chunks(chunks: List[dict], doc_config: dict) -> List[dict]:
    """
    Tag all chunks from one document with metadata.

    Args:
        chunks: list from chunker.chunk_pages()
        doc_config: document entry from settings.yaml

    Returns:
        List of tagged chunk dicts
    """
    tagged = []
    for i, chunk in enumerate(chunks):
        tagged.append(tag_chunk(chunk, doc_config, chunk_index_in_doc=i))

    logger.debug(
        f"Tagged {len(tagged)} chunks for {doc_config.get('id')} "
        f"({doc_config.get('doc_type')}, {doc_config.get('fiscal_period')})"
    )
    return tagged


def save_metadata_schema(output_dir: Path):
    """Save the metadata schema to data/metadata/schema.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(METADATA_SCHEMA, f, indent=2)
    logger.info(f"Metadata schema saved to {schema_path}")


def validate_chunk_metadata(chunk: dict) -> List[str]:
    """
    Check that all required metadata fields are present and non-null.
    Returns list of missing/invalid field names.
    """
    meta = chunk.get("metadata", {})
    issues = []

    required_non_null = [
        "chunk_id", "company", "ticker", "doc_type",
        "source_file", "filing_date", "fiscal_period",
        "page_number", "chunk_index", "token_count"
    ]
    for field in required_non_null:
        val = meta.get(field)
        if val is None or val == "":
            issues.append(f"Missing: {field}")

    if meta.get("token_count", 0) < 10:
        issues.append("token_count too low (< 10)")

    return issues
