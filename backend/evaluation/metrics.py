"""Evaluation metrics (spec section 16). All functions operate on the
already-computed pipeline output (or lighter retrieval-only output) - none
of them assume or hardcode a result; every number is computed from what was
actually retrieved/generated/verified.
"""
from __future__ import annotations

from backend.models.result import PipelineResult
from backend.verification.claim_verifier import Verdict


def recall_at_k(retrieved_chunk_ids: list[str], gold_chunk_ids: list[str], k: int) -> float:
    """Fraction of gold_chunk_ids that appear anywhere in the top-k retrieved
    chunk ids. Returns 1.0 if gold_chunk_ids is empty (nothing to miss)."""
    if not gold_chunk_ids:
        return 1.0
    top_k_ids = set(retrieved_chunk_ids[:k])
    hit = sum(1 for g in gold_chunk_ids if g in top_k_ids)
    return hit / len(gold_chunk_ids)


def citation_accuracy_from_result(result: PipelineResult) -> dict:
    """supported citations / total citations, where "supported" means the
    citation is structurally VALID (points to a retrieved chunk) AND the
    claim-level verifier judged that citation's claim as SUPPORTED.
    """
    total = 0
    supported = 0
    for claim in result.claims:
        for status in claim.citation_statuses:
            total += 1
            if status == "VALID" and claim.verdict == Verdict.SUPPORTED.value:
                supported += 1
    return {"supported": supported, "total": total}


def supported_claim_rate_from_result(result: PipelineResult) -> dict:
    total = len([c for c in result.claims if c.verdict is not None])
    supported = len([c for c in result.claims if c.verdict == Verdict.SUPPORTED.value])
    return {"supported": supported, "total": total}


def hallucination_rate_from_result(result: PipelineResult) -> dict:
    total = len([c for c in result.claims if c.verdict is not None])
    bad = len(
        [c for c in result.claims if c.verdict in (Verdict.UNSUPPORTED.value, Verdict.CONTRADICTED.value)]
    )
    return {"unsupported_or_contradicted": bad, "total": total}


def aggregate_rate(counts: list[dict], numerator_key: str, denominator_key: str = "total") -> float | None:
    num = sum(c[numerator_key] for c in counts)
    den = sum(c[denominator_key] for c in counts)
    return (num / den) if den else None


def summarize_run(per_query_results: list[dict]) -> dict:
    """per_query_results: list of dicts each containing keys
    'recall_at_5', 'recall_at_10', 'citation_accuracy_counts',
    'supported_claim_counts', 'hallucination_counts',
    'latency_retrieval_s', 'latency_generation_s', 'latency_verification_s'.
    """
    n = len(per_query_results)
    if n == 0:
        return {"num_queries": 0}

    recall5 = [r["recall_at_5"] for r in per_query_results if r.get("recall_at_5") is not None]
    recall10 = [r["recall_at_10"] for r in per_query_results if r.get("recall_at_10") is not None]

    citation_counts = [r["citation_accuracy_counts"] for r in per_query_results]
    supported_counts = [r["supported_claim_counts"] for r in per_query_results]
    hallucination_counts = [r["hallucination_counts"] for r in per_query_results]

    return {
        "num_queries": n,
        "recall_at_5": sum(recall5) / len(recall5) if recall5 else None,
        "recall_at_10": sum(recall10) / len(recall10) if recall10 else None,
        "citation_accuracy": aggregate_rate(citation_counts, "supported"),
        "supported_claim_rate": aggregate_rate(supported_counts, "supported"),
        "hallucination_rate": aggregate_rate(hallucination_counts, "unsupported_or_contradicted"),
        "avg_latency_retrieval_s": sum(r["latency_retrieval_s"] for r in per_query_results) / n,
        "avg_latency_generation_s": sum(r["latency_generation_s"] for r in per_query_results) / n,
        "avg_latency_verification_s": sum(r["latency_verification_s"] for r in per_query_results) / n,
        "total_citations": sum(c["total"] for c in citation_counts),
        "total_claims": sum(c["total"] for c in supported_counts),
    }
