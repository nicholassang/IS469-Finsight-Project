"""
semantic_chunker.py
Semantic + table-aware chunking for SEC financial filings.

Replaces fixed-token splitting with structure-aware splitting that:
  1. Detects SEC section headers (PART I, ITEM 1A, Notes to Financial Statements…)
     and starts a new chunk at each boundary.
  2. Detects financial tables (lines of numbers separated by whitespace) and
     keeps them intact — never splits a table mid-row.
  3. Falls back to paragraph splitting for large sections, and merges
     undersized sections with their neighbour.

Output dict format matches chunker.py exactly so downstream code is unaffected:
  {
    "text":               str,
    "chunk_index":        int,
    "token_count":        int,
    "start_token":        int  (-1, position not tracked),
    "end_token":          int  (-1),
    "page_number":        int  (set by chunk_pages caller),
    "global_chunk_index": int  (set by chunk_pages caller),
    "section_title":      str  (NEW — name of the SEC section this chunk is from),
    "has_table":          bool (NEW — True if chunk contains a financial table),
  }
"""

import re
from typing import List, Dict, Tuple

from src.utils.logger import get_logger
from src.chunking.chunker import count_tokens, split_sentences

logger = get_logger(__name__)


# ── SEC filing section-header patterns ───────────────────────────────────────
# Ordered from most specific to least specific.
_SEC_PATTERNS = [
    re.compile(r'^PART\s+[IVX]+[\s\.\-—]*', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^ITEM\s+\d+[A-Z]?[\s\.\-—]+', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^(NOTES?\s+TO\s+(CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^NOTE\s+\d+\s*[—\-–]', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^(MANAGEMENT.S\s+DISCUSSION|MD&A)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^(RISK\s+FACTORS)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^(QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^(CONSOLIDATED\s+STATEMENTS?\s+OF)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^(SELECTED\s+FINANCIAL\s+DATA)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^(CRITICAL\s+ACCOUNTING)', re.MULTILINE | re.IGNORECASE),
]

# Table detection — matches common SEC filing table formats:
#   "Total revenue    245,122    211,915    16%"   (label + numeric columns)
#   "245,122    211,915    88,523"                 (pure numeric row)
_TABLE_ROW = re.compile(
    r'^\s*'
    r'(?:[\w\s\(\),\.\-–—]+\s+)?'              # optional text label at start
    r'(?:[\$\(]?\s*[\d,]+\.?\d*\s*[%\)BbMmKk]?\s+)+'   # one or more numeric columns
    r'[\$\(]?\s*[\d,]+\.?\d*\s*[%\)BbMmKk]?\s*$',      # final numeric column
    re.MULTILINE
)
_TABLE_HEADER = re.compile(
    r'^\s*(\w[\w\s]+\s{3,}){2,}',   # 2+ words each followed by 3+ spaces
    re.MULTILINE
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_section_header(line: str) -> str:
    """Return normalised header label if line is a section header, else ''."""
    line_stripped = line.strip()
    if not line_stripped:
        return ""
    for pat in _SEC_PATTERNS:
        if pat.match(line_stripped):
            return line_stripped[:80]   # truncate very long headers
    return ""


def _is_table_block(text: str) -> bool:
    """Return True if text contains a financial table.

    Uses per-line matching so a multi-line table block is counted correctly
    (findall with MULTILINE can collapse the whole table into one match).

    Lines longer than 200 chars are skipped — real table rows are always
    short, and the _TABLE_ROW regex has catastrophic backtracking on long
    lines that mix prose text with numbers.
    """
    matching_lines = sum(
        1 for line in text.splitlines()
        if len(line) <= 200 and (_TABLE_ROW.match(line) or _TABLE_HEADER.match(line))
    )
    return matching_lines >= 2  # at least 2 numeric/header lines = likely a table


def _split_into_table_and_prose(text: str) -> List[Tuple[str, bool]]:
    """
    Split text into alternating (block_text, is_table) segments.
    Tables are kept as atomic units; prose can be further split.
    """
    lines = text.split("\n")
    segments: List[Tuple[str, bool]] = []
    buffer: List[str] = []
    in_table = False

    def flush(is_tbl: bool):
        nonlocal buffer
        if buffer:
            segments.append(("\n".join(buffer).strip(), is_tbl))
            buffer = []

    for line in lines:
        # Skip long lines before applying the regex — _TABLE_ROW has catastrophic
        # backtracking on lines > ~200 chars that mix prose and numbers.
        line_looks_like_table = (
            len(line) <= 200
            and (bool(_TABLE_ROW.match(line)) or bool(_TABLE_HEADER.match(line)))
        )

        if line_looks_like_table and not in_table:
            flush(False)          # flush prose
            in_table = True
        elif not line_looks_like_table and in_table:
            # Allow one blank line inside a table without breaking it
            if line.strip() == "" and buffer:
                buffer.append(line)
                continue
            flush(True)           # flush table
            in_table = False

        buffer.append(line)

    flush(in_table)
    return segments


def _split_by_paragraphs(text: str, max_tokens: int, min_tokens: int,
                          tokenizer: str = "cl100k_base") -> List[str]:
    """
    Split text into paragraph-sized pieces no larger than max_tokens.
    Merges short paragraphs with the next one.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    result: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        pt = count_tokens(para, tokenizer)
        if pt > max_tokens:
            # Para itself is too large — split by sentences
            if current:
                result.append("\n\n".join(current))
                current, current_tokens = [], 0
            sents = split_sentences(para)
            sent_buf: List[str] = []
            sent_tokens = 0
            for s in sents:
                st = count_tokens(s, tokenizer)
                if sent_tokens + st > max_tokens and sent_buf:
                    result.append(" ".join(sent_buf))
                    sent_buf, sent_tokens = [], 0
                sent_buf.append(s)
                sent_tokens += st
            if sent_buf:
                result.append(" ".join(sent_buf))
        elif current_tokens + pt > max_tokens and current:
            result.append("\n\n".join(current))
            current, current_tokens = [para], pt
        else:
            current.append(para)
            current_tokens += pt

    if current:
        result.append("\n\n".join(current))

    # Merge tiny tail into previous
    if len(result) >= 2 and count_tokens(result[-1], tokenizer) < min_tokens:
        result[-2] = result[-2] + "\n\n" + result[-1]
        result.pop()

    return result


# ── Main class ────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Chunk SEC filings by semantic structure rather than fixed token counts.

    Respects:
      - Section headers  (PART / ITEM / NOTE …)
      - Financial tables (kept as atomic units)
      - Paragraph breaks (fallback within large sections)
    """

    def __init__(
        self,
        max_chunk_tokens: int = 600,
        min_chunk_tokens: int = 80,
        tokenizer: str = "cl100k_base",
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.tokenizer = tokenizer

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk_text(self, text: str) -> List[Dict]:
        """Chunk a single page/document string. Returns list of chunk dicts."""
        sections = self._split_by_sections(text)
        chunks: List[Dict] = []
        idx = 0

        for section_title, section_text in sections:
            if not section_text.strip():
                continue

            tc = count_tokens(section_text, self.tokenizer)

            if tc <= self.max_chunk_tokens:
                # Section fits in one chunk
                if tc >= self.min_chunk_tokens:
                    chunks.append(self._make_chunk(section_text, idx, section_title))
                    idx += 1
                else:
                    # Too small — merge into previous if possible
                    if chunks:
                        chunks[-1]["text"] += "\n\n" + section_text
                        chunks[-1]["token_count"] = count_tokens(
                            chunks[-1]["text"], self.tokenizer)
                    else:
                        chunks.append(self._make_chunk(section_text, idx, section_title))
                        idx += 1
            else:
                # Section too large — split into table + prose segments first
                segments = _split_into_table_and_prose(section_text)
                for seg_text, is_table in segments:
                    if not seg_text.strip():
                        continue
                    seg_tc = count_tokens(seg_text, self.tokenizer)

                    if is_table or seg_tc <= self.max_chunk_tokens:
                        # Tables are atomic; small prose goes in as-is
                        if seg_tc >= self.min_chunk_tokens:
                            c = self._make_chunk(seg_text, idx, section_title)
                            c["has_table"] = is_table
                            chunks.append(c)
                            idx += 1
                        elif chunks:
                            chunks[-1]["text"] += "\n\n" + seg_text
                            chunks[-1]["token_count"] = count_tokens(
                                chunks[-1]["text"], self.tokenizer)
                    else:
                        # Large prose — split by paragraph
                        pieces = _split_by_paragraphs(
                            seg_text,
                            max_tokens=self.max_chunk_tokens,
                            min_tokens=self.min_chunk_tokens,
                            tokenizer=self.tokenizer,
                        )
                        for piece in pieces:
                            if not piece.strip():
                                continue
                            ptc = count_tokens(piece, self.tokenizer)
                            if ptc >= self.min_chunk_tokens:
                                chunks.append(self._make_chunk(piece, idx, section_title))
                                idx += 1

        return chunks

    def chunk_pages(self, pages: List[dict]) -> List[Dict]:
        """
        Chunk a list of page dicts (same interface as chunker.chunk_pages).
        Adds page_number and global_chunk_index to each chunk.
        """
        all_chunks: List[Dict] = []
        global_idx = 0

        for page in pages:
            text = page.get("text", "")
            if not text.strip():
                continue
            page_chunks = self.chunk_text(text)
            for c in page_chunks:
                c["page_number"] = page.get("page_number", 0)
                c["global_chunk_index"] = global_idx
                global_idx += 1
                all_chunks.append(c)

        logger.debug(f"SemanticChunker: {len(pages)} pages → {len(all_chunks)} chunks")
        return all_chunks

    # ── Private helpers ────────────────────────────────────────────────────────

    def _split_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split document into (section_title, section_text) pairs at SEC headers.
        Anything before the first header is labelled 'Preamble'.
        """
        lines = text.split("\n")
        sections: List[Tuple[str, str]] = []
        current_title = "Preamble"
        current_lines: List[str] = []

        for line in lines:
            header = _detect_section_header(line)
            if header:
                # Flush current section
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines)))
                current_title = header
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

        return sections

    @staticmethod
    def _make_chunk(text: str, idx: int, section_title: str = "") -> Dict:
        return {
            "text":          text.strip(),
            "chunk_index":   idx,
            "token_count":   count_tokens(text),
            "start_token":   -1,
            "end_token":     -1,
            "section_title": section_title,
            "has_table":     _is_table_block(text),
        }
