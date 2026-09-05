"""Shared evaluation runner used by both baselines.py and ablation.py.

Defines one canonical set of pipeline configurations (so a config that is
identical between the baseline comparison and the ablation study - e.g.
"hybrid, no reranker" is both Baseline D and the "remove reranker" ablation -
is only ever executed once) and computes real, measured metrics per
benchmark item and in aggregate. Nothing here is a placeholder: every result
file written under experiments/results/ comes from an actual run of
`backend.pipeline.run_pipeline` against the benchmark in
data/evaluation/benchmark.json.
"""
from __future__ import annotations

import json
import logging
import time

from backend.config.settings import RESULTS_DIR
from backend.evaluation.benchmark import BenchmarkItem, load_benchmark, load_patient
from backend.evaluation.metrics import (
    citation_accuracy_from_result,
    hallucination_rate_from_result,
    recall_at_k,
    summarize_run,
    supported_claim_rate_from_result,
)
from backend.models.result import PipelineConfig, PipelineResult
from backend.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Canonical configuration set. Labels double as both the baseline names
# (spec section 17) and the ablation names (spec section 18) wherever the
# underlying toggles are identical.
CANONICAL_CONFIGS: dict[str, PipelineConfig] = {
    "llm_only": PipelineConfig(
        use_bm25=False, use_dense=False, use_reranker=False,
        use_patient_context=True, use_verification=True, label="llm_only",
    ),
    "dense_rag": PipelineConfig(
        use_bm25=False, use_dense=True, use_reranker=False,
        use_patient_context=True, use_verification=True, label="dense_rag",
    ),
    "bm25_rag": PipelineConfig(
        use_bm25=True, use_dense=False, use_reranker=False,
        use_patient_context=True, use_verification=True, label="bm25_rag",
    ),
    "hybrid_rag": PipelineConfig(  # == ablation "remove reranker"
        use_bm25=True, use_dense=True, use_reranker=False,
        use_patient_context=True, use_verification=True, label="hybrid_rag",
    ),
    "hybrid_reranker_no_verification": PipelineConfig(  # == ablation "remove claim verification"
        use_bm25=True, use_dense=True, use_reranker=True,
        use_patient_context=True, use_verification=False, label="hybrid_reranker_no_verification",
    ),
    "full_system": PipelineConfig(
        use_bm25=True, use_dense=True, use_reranker=True,
        use_patient_context=True, use_verification=True, label="full_system",
    ),
    "remove_bm25": PipelineConfig(
        use_bm25=False, use_dense=True, use_reranker=True,
        use_patient_context=True, use_verification=True, label="remove_bm25",
    ),
    "remove_dense": PipelineConfig(
        use_bm25=True, use_dense=False, use_reranker=True,
        use_patient_context=True, use_verification=True, label="remove_dense",
    ),
    "remove_patient_context": PipelineConfig(
        use_bm25=True, use_dense=True, use_reranker=True,
        use_patient_context=False, use_verification=True, label="remove_patient_context",
    ),
}


def run_single_item(item: BenchmarkItem, config: PipelineConfig) -> dict:
    patient = load_patient(item.patient_id)
    t0 = time.time()
    result = run_pipeline(patient, item.question, config=config)
    wall_time = time.time() - t0

    # `result.reranked` is the final evidence set handed to the generator
    # regardless of config - when reranking is disabled, run_pipeline
    # already populates it with fused[:top_k] (see pipeline.py), so this is
    # always the right list to score recall@k against.
    retrieved_ids = [e.chunk_id for e in result.reranked]
    recall5 = recall_at_k(retrieved_ids, item.gold_evidence, 5)
    recall10 = recall_at_k(retrieved_ids, item.gold_evidence, 10)

    return {
        "item_id": item.item_id,
        "recall_at_5": recall5,
        "recall_at_10": recall10,
        "citation_accuracy_counts": citation_accuracy_from_result(result),
        "supported_claim_counts": supported_claim_rate_from_result(result),
        "hallucination_counts": hallucination_rate_from_result(result),
        "latency_retrieval_s": result.latency_retrieval_s,
        "latency_generation_s": result.latency_generation_s,
        "latency_verification_s": result.latency_verification_s,
        "wall_time_s": wall_time,
        "num_claims": len(result.claims),
        "num_citations_total": sum(len(c.citation_ids) for c in result.claims),
    }


def run_config(label: str, force: bool = False) -> dict:
    """Run one canonical config over the whole benchmark, cache to
    experiments/results/<label>.json, and return the summary dict. Cached
    results are reused unless force=True - this makes long evaluation runs
    resumable across multiple invocations.
    """
    out_path = RESULTS_DIR / f"{label}.json"
    if out_path.exists() and not force:
        logger.info("Using cached result for '%s' at %s", label, out_path)
        return json.loads(out_path.read_text())["summary"]

    config = CANONICAL_CONFIGS[label]
    benchmark = load_benchmark()
    logger.info("Running config '%s' over %d benchmark items...", label, len(benchmark))

    per_item = []
    for item in benchmark:
        logger.info("  [%s] item %s: %s", label, item.item_id, item.question[:60])
        per_item.append(run_single_item(item, config))

    summary = summarize_run(per_item)
    payload = {"label": label, "config": config.model_dump(), "per_item": per_item, "summary": summary}
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s -- summary: %s", out_path, summary)
    return summary


def run_all(labels: list[str], force: bool = False) -> dict[str, dict]:
    return {label: run_config(label, force=force) for label in labels}
