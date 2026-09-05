"""CLI driver: fetch/cache the PubMed corpus, chunk it, and write
data/processed/chunks.jsonl (spec section 5 pipeline entry point).

Usage:
    python -m backend.scripts.build_corpus [--force-refetch] [--retmax N]
"""
from __future__ import annotations

import argparse
import json
import logging

from backend.config.settings import PROCESSED_DIR
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.loader import (
    DEFAULT_TOPIC_QUERIES,
    fetch_and_cache_corpus,
    load_cached_documents,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
DOCS_PATH = PROCESSED_DIR / "documents.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-refetch", action="store_true")
    parser.add_argument("--retmax", type=int, default=20)
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip PubMed network calls entirely; use whatever is already cached.",
    )
    args = parser.parse_args()

    if args.offline:
        logger.info("Offline mode: loading only from local cache (data/raw/pubmed/)")
        docs = load_cached_documents()
    else:
        docs = fetch_and_cache_corpus(
            topics=DEFAULT_TOPIC_QUERIES,
            retmax_per_topic=args.retmax,
            force_refetch=args.force_refetch,
        )

    if not docs:
        raise SystemExit(
            "No documents available (no network and no local cache in data/raw/pubmed/). "
            "Run without --offline at least once to populate the cache."
        )

    logger.info("Total unique documents: %d", len(docs))

    with DOCS_PATH.open("w") as f:
        for d in docs:
            f.write(json.dumps(d.model_dump()) + "\n")

    chunks = chunk_documents(docs)
    logger.info("Total chunks: %d", len(chunks))

    with CHUNKS_PATH.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump()) + "\n")

    logger.info("Wrote %s and %s", DOCS_PATH, CHUNKS_PATH)


if __name__ == "__main__":
    main()
