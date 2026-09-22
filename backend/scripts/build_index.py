"""CLI driver: build the BM25 and dense (FAISS) indices from
data/processed/chunks.jsonl and persist them under data/processed/index/.

Usage:
    python -m backend.scripts.build_index
"""
from __future__ import annotations

import logging
import shutil

from backend.config.settings import INDEX_DIR, PROCESSED_DIR, settings
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.dense import DenseIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# When EMBEDDING_PROVIDER=remote, a committed precomputed index here is used
# instead of re-embedding the whole corpus via the API on every deploy -
# avoids burning the free-tier embeddings quota (and its rate limits) on
# every build. Built once locally via `build_index.py --precompute-remote`.
PRECOMPUTED_REMOTE_DENSE_DIR = PROCESSED_DIR / "index_remote_precomputed" / "dense"


def main() -> None:
    import sys

    precompute = "--precompute-remote" in sys.argv

    if not CHUNKS_PATH.exists():
        raise SystemExit("data/processed/chunks.jsonl not found - run build_corpus first.")

    logger.info("Building BM25 index...")
    bm25 = BM25Index.from_jsonl(CHUNKS_PATH)
    bm25.save(INDEX_DIR / "bm25.pkl")
    logger.info("BM25 index built over %d chunks", len(bm25.chunks))
    chunks = bm25.chunks

    dense_out = INDEX_DIR / "dense"
    if settings.embedding_provider == "none":
        logger.info("embedding_provider=none - skipping dense index entirely (BM25 only)")
    elif precompute:
        target = PRECOMPUTED_REMOTE_DENSE_DIR
        logger.info("Precomputing remote dense index (one-time) -> %s", target)
        dense = DenseIndex.build(chunks)
        dense.save(target)
        dense.save(dense_out)
        logger.info("Precomputed remote index saved. Commit %s to git.", target)
    elif settings.embedding_provider == "remote" and PRECOMPUTED_REMOTE_DENSE_DIR.exists():
        logger.info("Using precomputed remote dense index from %s (no API calls needed)", PRECOMPUTED_REMOTE_DENSE_DIR)
        shutil.copytree(PRECOMPUTED_REMOTE_DENSE_DIR, dense_out, dirs_exist_ok=True)
    else:
        logger.info("Building dense (FAISS) index... (this downloads/calls the embedding provider)")
        dense = DenseIndex.build(chunks)
        dense.save(dense_out)
        logger.info("Dense index built with model '%s' over %d chunks", dense.model_name, len(chunks))

    logger.info("Done. Indices written to %s", INDEX_DIR)


if __name__ == "__main__":
    main()
