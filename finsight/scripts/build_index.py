"""
build_index.py
One-command entry point to rebuild both dense (ChromaDB) and sparse (BM25) indexes.
Reads all processed chunks from data/processed/*.json

Usage:
    python scripts/build_index.py            # build if index doesn't exist
    python scripts/build_index.py --reset    # delete and rebuild from scratch
    python scripts/build_index.py --dense-only
    python scripts/build_index.py --sparse-only
"""

import sys
import json
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.indexing.dense_indexer import DenseIndexer
from src.indexing.sparse_indexer import SparseIndexer

logger = get_logger(__name__)


def load_all_chunks(cfg: dict) -> list:
    processed_dir = Path(cfg["paths"]["processed_data"])
    all_chunks = []

    json_files = sorted(processed_dir.glob("*.json"))
    if not json_files:
        logger.error(
            f"No processed chunks found in {processed_dir}. "
            f"Run: python scripts/ingest_all.py first."
        )
        sys.exit(1)

    logger.info(f"Loading chunks from {len(json_files)} processed files ...")
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)
        logger.info(f"  {jf.name}: {len(chunks)} chunks")

    logger.info(f"Total chunks loaded: {len(all_chunks):,}")
    return all_chunks


def build_dense_index(chunks: list, cfg: dict, reset: bool = False):
    logger.info("\n--- Building Dense (ChromaDB) Index ---")
    indexer = DenseIndexer(cfg)

    if reset:
        indexer.reset_collection()

    t0 = time.time()
    n = indexer.index_chunks(chunks)
    elapsed = time.time() - t0

    stats = indexer.get_collection_stats()
    logger.info(f"Dense index complete: {n} vectors in {elapsed:.1f}s")
    logger.info(f"  Collection: {stats['collection_name']} at {stats['persist_dir']}")

    ok = indexer.verify_index()
    if not ok:
        logger.error("Dense index verification FAILED")
        sys.exit(1)
    return n


def build_sparse_index(chunks: list, cfg: dict):
    logger.info("\n--- Building Sparse (BM25) Index ---")
    indexer = SparseIndexer(cfg)

    t0 = time.time()
    indexer.build_index(chunks)
    elapsed = time.time() - t0

    stats = indexer.get_index_stats()
    logger.info(f"Sparse index complete: {stats['document_count']} documents in {elapsed:.1f}s")

    ok = indexer.verify_index()
    if not ok:
        logger.error("Sparse index verification FAILED")
        sys.exit(1)
    return stats["document_count"]


def main():
    parser = argparse.ArgumentParser(description="FinSight index builder")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing indexes and rebuild from scratch")
    parser.add_argument("--dense-only", action="store_true", help="Only build dense index")
    parser.add_argument("--sparse-only", action="store_true", help="Only build sparse index")
    args = parser.parse_args()

    cfg = load_config()

    if args.reset:
        logger.info("RESET mode: will delete and rebuild all indexes")
        # Remove BM25 files
        import shutil
        bm25_dir = Path(cfg["paths"]["bm25_index"]).parent
        if bm25_dir.exists():
            shutil.rmtree(bm25_dir)
            logger.info(f"  Deleted: {bm25_dir}")

    chunks = load_all_chunks(cfg)

    total_t0 = time.time()

    if not args.sparse_only:
        build_dense_index(chunks, cfg, reset=args.reset)

    if not args.dense_only:
        build_sparse_index(chunks, cfg)

    total_elapsed = time.time() - total_t0
    logger.info(f"\n{'='*60}")
    logger.info(f"Index build complete in {total_elapsed:.1f}s")
    logger.info(f"Next step: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
