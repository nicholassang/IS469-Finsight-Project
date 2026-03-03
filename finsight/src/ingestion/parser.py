"""
parser.py
Extracts page-by-page text from PDF files.
Primary: pdfplumber (handles text-layer PDFs well).
Fallback: PyMuPDF (fitz) — better on complex layouts.
Returns list of page dicts with page_number and raw text.
"""

from pathlib import Path
from typing import List, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed — will try PyMuPDF fallback")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# Minimum character count per page to be considered non-empty
MIN_PAGE_CHARS = 50


def _extract_with_pdfplumber(pdf_path: Path) -> List[Dict]:
    """Extract text page-by-page using pdfplumber."""
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                # Also try table extraction for pages with heavy tables
                tables = page.extract_tables()
                if tables:
                    table_text = _flatten_tables(tables)
                    if table_text not in text:
                        text = text + "\n\n[TABLE DATA]\n" + table_text
            except Exception as e:
                logger.debug(f"pdfplumber failed on page {i}: {e}")
                text = ""

            if len(text.strip()) >= MIN_PAGE_CHARS:
                pages.append({"page_number": i, "text": text, "parser": "pdfplumber"})
            else:
                logger.debug(f"Page {i}: skipped (only {len(text.strip())} chars)")

    return pages


def _extract_with_pymupdf(pdf_path: Path) -> List[Dict]:
    """Extract text page-by-page using PyMuPDF (fallback)."""
    pages = []
    doc = fitz.open(str(pdf_path))
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        if len(text.strip()) >= MIN_PAGE_CHARS:
            pages.append({"page_number": i, "text": text, "parser": "pymupdf"})
        else:
            logger.debug(f"PyMuPDF page {i}: skipped ({len(text.strip())} chars)")
    doc.close()
    return pages


def _flatten_tables(tables: List) -> str:
    """Convert pdfplumber table structures to plain text rows."""
    lines = []
    for table in tables:
        for row in table:
            if row:
                cells = [str(c).strip() if c else "" for c in row]
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_pages(pdf_path: Path, force_fallback: bool = False) -> List[Dict]:
    """
    Extract all pages from a PDF.

    Returns:
        List of dicts: {page_number: int, text: str, parser: str}
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Parsing {pdf_path.name} ...")

    # Try pdfplumber first
    if PDFPLUMBER_AVAILABLE and not force_fallback:
        try:
            pages = _extract_with_pdfplumber(pdf_path)
            if pages:
                logger.info(f"  pdfplumber: {len(pages)} pages extracted")
                return pages
            else:
                logger.warning(f"  pdfplumber returned 0 usable pages — trying fallback")
        except Exception as e:
            logger.warning(f"  pdfplumber failed: {e} — trying fallback")

    # Fallback: PyMuPDF
    if PYMUPDF_AVAILABLE:
        pages = _extract_with_pymupdf(pdf_path)
        logger.info(f"  PyMuPDF: {len(pages)} pages extracted")
        return pages

    raise RuntimeError(
        f"Cannot parse {pdf_path.name}: neither pdfplumber nor PyMuPDF available. "
        f"Install: pip install pdfplumber PyMuPDF"
    )


def get_pdf_metadata(pdf_path: Path) -> Dict:
    """Extract basic PDF metadata (title, author, pages, etc.)."""
    pdf_path = Path(pdf_path)
    meta = {"filename": pdf_path.name, "total_pages": 0}

    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                meta["total_pages"] = len(pdf.pages)
                meta["pdf_metadata"] = pdf.metadata or {}
        except Exception:
            pass
    elif PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(str(pdf_path))
            meta["total_pages"] = doc.page_count
            meta["pdf_metadata"] = doc.metadata or {}
            doc.close()
        except Exception:
            pass

    return meta
