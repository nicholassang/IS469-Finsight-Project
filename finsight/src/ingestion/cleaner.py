"""
cleaner.py
Text cleaning utilities for Grab financial filings.
Removes headers/footers, normalises whitespace, strips boilerplate legal text.
All functions are pure (str -> str) and independently testable.
"""

import re
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Patterns specific to Grab filings ─────────────────────────────────────────

# Common footer patterns seen in 20-F / 6-K filings
FOOTER_PATTERNS = [
    r"Grab Holdings Limited\s*\|?\s*(?:Annual Report|Form 20-F|Form 6-K)[^\n]*\d{4}",
    r"CONFIDENTIAL[^\n]*PAGE\s+\d+",
    r"(?:Page\s+)?\d+\s+of\s+\d+\s+pages?",
    r"Copyright\s+©?\s*\d{4}\s+Grab(?:\s+Holdings)?[^\n]*",
    r"This\s+(?:document|presentation|report)\s+contains\s+forward[- ]looking\s+statements[^\n]*",
    r"GRAB HOLDINGS LIMITED[^\n]*EDGAR[^\n]*",
]

# Table of contents entries (page reference lines) — remove these
TOC_LINE_PATTERN = re.compile(r"^[A-Z][^\n]{5,80}\.{3,}\s*\d{1,4}\s*$", re.MULTILINE)

# Repeated header lines that appear on every page
RUNNING_HEADER_PATTERN = re.compile(
    r"(?:^|\n)(?:Grab Holdings|GRAB HOLDINGS)[^\n]{0,60}\n",
    re.IGNORECASE,
)

# Form feed characters (page break artifacts)
FORM_FEED_PATTERN = re.compile(r"\f")

# Multiple consecutive blank lines → single blank line
MULTI_BLANK_PATTERN = re.compile(r"\n{3,}")

# Multiple spaces / tabs → single space
MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")

# Unicode non-breaking spaces, zero-width spaces, etc.
UNICODE_JUNK_PATTERN = re.compile(r"[\u00a0\u200b\u200c\u200d\ufeff]")

# Isolated single characters on their own line (OCR noise)
ISOLATED_CHAR_PATTERN = re.compile(r"(?:^|\n)[^a-zA-Z0-9\n]{0,2}[a-zA-Z]{1}[^a-zA-Z0-9\n]{0,2}(?=\n|$)")


def remove_footers(text: str) -> str:
    """Remove known Grab filing footer / header boilerplate."""
    for pat in FOOTER_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def remove_toc_lines(text: str) -> str:
    """Remove table-of-contents style lines (text .... 42)."""
    return TOC_LINE_PATTERN.sub("", text)


def normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs; reduce excessive blank lines."""
    text = FORM_FEED_PATTERN.sub("\n", text)
    text = UNICODE_JUNK_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    text = MULTI_BLANK_PATTERN.sub("\n\n", text)
    return text.strip()


def fix_hyphenation(text: str) -> str:
    """
    Rejoin words that were hyphenated across line breaks.
    e.g. "contribut-\ning" → "contributing"
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def normalise_numbers(text: str) -> str:
    """
    Normalise common financial notation variants:
    - "US$1,234.5 million" → keep as-is (don't destroy numbers)
    - Fix spacing around $ and % signs
    """
    # Fix "$1 23.4" space-corrupted numbers (common OCR artifact)
    text = re.sub(r"\$\s+(\d)", r"$\1", text)
    # Fix "1 ,234" → "1,234"
    text = re.sub(r"(\d)\s+,(\d)", r"\1,\2", text)
    return text


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a page or chunk of text.
    Apply in order — each step builds on the previous.
    """
    text = remove_footers(text)
    text = remove_toc_lines(text)
    text = fix_hyphenation(text)
    text = normalise_numbers(text)
    text = normalise_whitespace(text)
    return text


def clean_pages(pages: List[dict]) -> List[dict]:
    """
    Apply clean_text() to each page dict from parser.extract_pages().
    Drops pages that become empty after cleaning.
    """
    cleaned = []
    for page in pages:
        raw = page.get("text", "")
        cleaned_text = clean_text(raw)
        if len(cleaned_text.strip()) >= 30:
            cleaned.append({**page, "text": cleaned_text})
        else:
            logger.debug(f"Page {page.get('page_number')} dropped after cleaning (too short)")
    logger.debug(f"Cleaning: {len(pages)} pages in → {len(cleaned)} pages out")
    return cleaned


def detect_scanned_page(text: str) -> bool:
    """
    Heuristic: if extracted text is very short for what should be a full page,
    it may be a scanned image. Flag it for manual review.
    """
    return len(text.strip()) < 100
