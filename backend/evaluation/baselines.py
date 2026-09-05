"""Baseline experiments (spec section 17).

Usage:
    python -m backend.evaluation.baselines [--force]

Writes experiments/results/baselines_summary.json with the measured
Recall@5, Recall@10, citation accuracy, supported-claim rate, and
hallucination rate for each baseline plus the final system - all computed
from real pipeline runs over data/evaluation/benchmark.json, never assumed.
"""
from __future__ import annotations

import argparse
import json
import logging

from backend.config.settings import RESULTS_DIR
from backend.evaluation.runner import run_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASELINE_LABELS = {
    "Baseline A: LLM only": "llm_only",
    "Baseline B: Dense + LLM": "dense_rag",
    "Baseline C: BM25 + LLM": "bm25_rag",
    "Baseline D: Hybrid (BM25+Dense) + LLM": "hybrid_rag",
    "Final System: Hybrid + Reranker + Verification": "full_system",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-run even if cached results exist")
    args = parser.parse_args()

    results = run_all(list(BASELINE_LABELS.values()), force=args.force)

    table = {name: results[label] for name, label in BASELINE_LABELS.items()}
    out_path = RESULTS_DIR / "baselines_summary.json"
    out_path.write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
