"""Ablation study (spec section 18).

Usage:
    python -m backend.evaluation.ablation [--force]

Writes experiments/results/ablation_summary.json comparing the full system
against each component removed one at a time - all measured, not assumed.
"""
from __future__ import annotations

import argparse
import json
import logging

from backend.config.settings import RESULTS_DIR
from backend.evaluation.runner import run_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ABLATION_LABELS = {
    "LLM only (no retrieval)": "llm_only",
    "Dense RAG (dense only, no rerank)": "dense_rag",
    "BM25 RAG (BM25 only, no rerank)": "bm25_rag",
    "Hybrid RAG (BM25+Dense, no rerank)": "hybrid_rag",
    "Hybrid + Reranker (no claim verification)": "hybrid_reranker_no_verification",
    "Full System": "full_system",
    "Full System - remove BM25": "remove_bm25",
    "Full System - remove dense retrieval": "remove_dense",
    "Full System - remove patient-context augmentation": "remove_patient_context",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-run even if cached results exist")
    args = parser.parse_args()

    results = run_all(list(ABLATION_LABELS.values()), force=args.force)

    table = {name: results[label] for name, label in ABLATION_LABELS.items()}
    out_path = RESULTS_DIR / "ablation_summary.json"
    out_path.write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
