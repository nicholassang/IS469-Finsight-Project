"""
ingest_all.py
Runs the complete ingestion pipeline for all configured Grab documents:
  1. Check which documents are present in data/raw/
  2. Parse PDF → pages
  3. Clean pages
  4. Chunk pages
  5. Tag chunks with metadata
  6. Save processed chunks to data/processed/<doc_id>.json

Run this before build_index.py.
Usage:
    python scripts/ingest_all.py
    python scripts/ingest_all.py --doc grab_20f_fy2023   # single document
    python scripts/ingest_all.py --chunking experiment_A  # different chunking config
"""

import sys
import json
import argparse
import time
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, load_chunking_config
from src.utils.logger import get_logger
from src.ingestion.parser import extract_pages
from src.ingestion.cleaner import clean_pages
from src.chunking.chunker import chunk_pages
from src.chunking.metadata_tagger import tag_document_chunks, validate_chunk_metadata, save_metadata_schema

logger = get_logger(__name__)


def ingest_document(doc_config: dict, cfg: dict, chunking_config: dict) -> list:
    """
    Full ingestion pipeline for a single document.
    Returns list of tagged chunk dicts, also saves to data/processed/.
    """
    raw_dir = Path(cfg["paths"]["raw_data"])
    processed_dir = Path(cfg["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    doc_id = doc_config["id"]
    filename = doc_config["filename"]
    pdf_path = raw_dir / filename

    if not pdf_path.exists():
        logger.warning(
            f"[SKIP] {filename} not found in {raw_dir}. "
            f"Download it manually and place at {pdf_path}"
        )
        return []

    logger.info(f"\n{'='*60}")
    logger.info(f"Ingesting: {doc_id} ({doc_config['doc_type']}, {doc_config['fiscal_period']})")
    logger.info(f"{'='*60}")

    t0 = time.time()

    # Step 1: Parse
    pages = extract_pages(pdf_path)
    logger.info(f"  Parsed: {len(pages)} pages")

    # Step 2: Clean
    cleaned = clean_pages(pages)
    logger.info(f"  Cleaned: {len(cleaned)} pages (dropped {len(pages)-len(cleaned)} empty/short)")

    # Step 3: Chunk
    chunks = chunk_pages(cleaned, config=chunking_config)
    logger.info(f"  Chunked: {len(chunks)} chunks")

    if not chunks:
        logger.warning(f"  No chunks produced for {doc_id} — check PDF quality")
        return []

    # Step 4: Tag with metadata
    tagged = tag_document_chunks(chunks, doc_config)

    # Step 5: Validate
    issues_found = 0
    for chunk in tagged:
        issues = validate_chunk_metadata(chunk)
        if issues:
            issues_found += 1
            logger.debug(f"  Metadata issues in {chunk.get('chunk_id', '?')}: {issues}")

    if issues_found:
        logger.warning(f"  {issues_found} chunks have metadata issues — check debug logs")

    # Step 6: Save to data/processed/<doc_id>.json
    output_path = processed_dir / f"{doc_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tagged, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    avg_tokens = sum(c.get("token_count", 0) for c in tagged) / max(len(tagged), 1)
    logger.info(
        f"  Saved: {output_path.name} "
        f"({len(tagged)} chunks, avg {avg_tokens:.0f} tokens, {elapsed:.1f}s)"
    )

    return tagged


def load_all_processed_chunks(cfg: dict) -> list:
    """Load all processed chunks from data/processed/*.json"""
    processed_dir = Path(cfg["paths"]["processed_data"])
    all_chunks = []
    for json_file in sorted(processed_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
            logger.info(f"  Loaded {len(chunks)} chunks from {json_file.name}")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="FinSight ingestion pipeline")
    parser.add_argument("--doc", help="Ingest only this document ID (default: all)")
    parser.add_argument("--chunking", default="default",
                        help="Chunking experiment name from chunking.yaml (default: 'default')")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest even if processed file already exists")
    args = parser.parse_args()

    cfg = load_config()
    chunking_config = load_chunking_config(args.chunking)
    docs = cfg.get("documents", [])

    if args.doc:
        docs = [d for d in docs if d["id"] == args.doc]
        if not docs:
            logger.error(f"Document ID '{args.doc}' not found in config")
            sys.exit(1)

    # Save metadata schema
    save_metadata_schema(Path(cfg["paths"]["metadata"]))

    logger.info(f"FinSight Ingestion Pipeline")
    logger.info(f"  Documents to process: {len(docs)}")
    logger.info(f"  Chunking config: {args.chunking}")
    logger.info(f"  Strategy: {chunking_config.get('strategy')} | "
                f"size={chunking_config.get('chunk_size', '?')} | "
                f"overlap={chunking_config.get('chunk_overlap', chunking_config.get('overlap_sentences', '?'))}")

    processed_dir = Path(cfg["paths"]["processed_data"])
    total_chunks = 0
    skipped = 0

    for doc in docs:
        doc_id = doc["id"]
        output_path = processed_dir / f"{doc_id}.json"

        if output_path.exists() and not args.force:
            with open(output_path) as f:
                existing = json.load(f)
            logger.info(f"[CACHED] {doc_id}: {len(existing)} chunks already processed (use --force to re-ingest)")
            total_chunks += len(existing)
            skipped += 1
            continue

        chunks = ingest_document(doc, cfg, chunking_config)
        total_chunks += len(chunks)

    logger.info(f"\n{'='*60}")
    logger.info(f"Ingestion complete: {total_chunks:,} total chunks ({skipped} cached)")
    logger.info(f"Next step: python scripts/build_index.py")


if __name__ == "__main__":
    main()
