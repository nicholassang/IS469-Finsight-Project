"""
downloader.py
Downloads Grab PDF filings from configured URLs to data/raw/.
Skips files that already exist (idempotent).
"""

import time
import hashlib
from pathlib import Path
from typing import List, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed — downloader will be non-functional. pip install requests")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_document(doc: dict, raw_dir: Path, force: bool = False) -> dict:
    """
    Download a single document dict (from config documents list).
    Returns status dict: {filename, status, path, error}.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = doc["filename"]
    url = doc.get("source_url", "")
    dest = raw_dir / filename

    if dest.exists() and not force:
        logger.info(f"[SKIP] {filename} — already exists at {dest}")
        return {"filename": filename, "status": "skipped", "path": str(dest), "error": None}

    if not REQUESTS_AVAILABLE:
        logger.error("requests library not available — cannot download")
        return {"filename": filename, "status": "error", "path": None, "error": "requests not installed"}

    if not url or url.startswith("https://ir.grab.com"):
        # Placeholder URL — team must replace with real direct PDF links
        logger.warning(
            f"[MANUAL] {filename} — URL '{url}' is a landing page, not a direct PDF link.\n"
            f"  Please download manually from {url} and place at {dest}"
        )
        return {
            "filename": filename,
            "status": "manual_required",
            "path": str(dest),
            "error": "URL is a landing page; direct PDF URL needed",
        }

    logger.info(f"[DOWNLOAD] {filename} from {url}")
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "FinSight-Research/1.0"})
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            logger.info(f"[OK] {filename} saved ({len(resp.content):,} bytes)")
            return {"filename": filename, "status": "downloaded", "path": str(dest), "error": None}
        except Exception as e:
            logger.warning(f"Attempt {attempt}/3 failed for {filename}: {e}")
            time.sleep(2 ** attempt)

    return {"filename": filename, "status": "error", "path": None, "error": "Download failed after 3 attempts"}


def download_all(force: bool = False) -> List[dict]:
    """Download all documents listed in config/settings.yaml."""
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_data"])
    docs = cfg.get("documents", [])

    if not docs:
        logger.warning("No documents configured in settings.yaml")
        return []

    results = []
    for doc in docs:
        result = download_document(doc, raw_dir, force=force)
        results.append(result)

    ok = [r for r in results if r["status"] in ("downloaded", "skipped")]
    manual = [r for r in results if r["status"] == "manual_required"]
    errors = [r for r in results if r["status"] == "error"]

    logger.info(f"\nDownload summary: {len(ok)} ok, {len(manual)} manual required, {len(errors)} errors")
    if manual:
        logger.info("Files requiring manual download:")
        for r in manual:
            logger.info(f"  → {r['filename']}: {r['error']}")
    return results


def check_corpus(raw_dir: Path = None) -> List[dict]:
    """Verify which configured documents are present on disk."""
    cfg = load_config()
    raw_dir = raw_dir or Path(cfg["paths"]["raw_data"])
    docs = cfg.get("documents", [])

    status = []
    for doc in docs:
        path = raw_dir / doc["filename"]
        present = path.exists()
        size = path.stat().st_size if present else 0
        status.append({
            "filename": doc["filename"],
            "doc_type": doc["doc_type"],
            "fiscal_period": doc["fiscal_period"],
            "present": present,
            "size_kb": round(size / 1024, 1),
        })

    present_count = sum(1 for s in status if s["present"])
    logger.info(f"Corpus check: {present_count}/{len(docs)} documents present in {raw_dir}")
    for s in status:
        mark = "✓" if s["present"] else "✗"
        logger.info(f"  [{mark}] {s['filename']} ({s['doc_type']}, {s['fiscal_period']}) — {s['size_kb']} KB")

    return status


if __name__ == "__main__":
    results = download_all()
