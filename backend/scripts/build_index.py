"""CLI driver: build the BM25 and dense (FAISS) indices from
data/processed/chunks.jsonl and persist them under data/processed/index/.

Usage:
    python -m backend.scripts.build_index
"""
from __future__ import annotations

import logging

from backend.config.settings import INDEX_DIR, PROCESSED_DIR
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.dense import DenseIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise SystemExit("data/processed/chunks.jsonl not found - run build_corpus first.")

    logger.info("Building BM25 index...")
    bm25 = BM25Index.from_jsonl(CHUNKS_PATH)
    bm25.save(INDEX_DIR / "bm25.pkl")
    logger.info("BM25 index built over %d chunks", len(bm25.chunks))

    logger.info("Building dense (FAISS) index... (this downloads the embedding model on first run)")
    chunks = bm25.chunks
    dense = DenseIndex.build(chunks)
    dense.save(INDEX_DIR / "dense")
    logger.info("Dense index built with model '%s' over %d chunks", dense.model_name, len(chunks))

    logger.info("Done. Indices written to %s", INDEX_DIR)


if __name__ == "__main__":
    main()
