"""Hybrid retrieval fusion (spec sections 6-7): combine BM25 and dense
candidate lists into one ranked candidate pool using either weighted score
fusion (min-max normalized scores, configurable alpha) or Reciprocal Rank
Fusion. Both retrieved (pre-fusion) and fused evidence are exposed so the
caller (and the UI) can show retrieval provenance for debugging/evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backend.config.settings import settings
from backend.models.evidence import EvidenceChunk


@dataclass
class FusionCandidate:
    chunk: EvidenceChunk
    bm25_score: float | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    hybrid_score: float = 0.0


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def merge_candidate_lists(
    *result_lists: list[tuple[EvidenceChunk, float]], top_k: int | None = None
) -> list[tuple[EvidenceChunk, float]]:
    """Union multiple same-scorer ranked lists (e.g. BM25 run on the raw
    question and BM25 run on a patient-context-augmented query), keeping
    each chunk's best score across lists. Used to let patient-context
    augmentation ADD candidates the raw question alone would miss, without
    being able to displace/dilute the raw question's own top hits the way a
    single concatenated query can - see RESULTS.md "patient-context
    augmentation hurts retrieval" ablation finding.
    """
    best: dict[str, tuple[EvidenceChunk, float]] = {}
    for results in result_lists:
        for chunk, score in results:
            current = best.get(chunk.chunk_id)
            if current is None or score > current[1]:
                best[chunk.chunk_id] = (chunk, score)
    merged = sorted(best.values(), key=lambda x: x[1], reverse=True)
    return merged[:top_k] if top_k is not None else merged


def fuse(
    bm25_results: list[tuple[EvidenceChunk, float]],
    dense_results: list[tuple[EvidenceChunk, float]],
    alpha: float | None = None,
    method: str | None = None,
) -> list[FusionCandidate]:
    """Fuse two ranked lists into one candidate pool.

    hybrid_score = alpha * normalized_bm25_score + (1-alpha) * normalized_dense_score
    (or Reciprocal Rank Fusion when method == "rrf").
    """
    alpha = settings.hybrid_alpha if alpha is None else alpha
    method = settings.fusion_method if method is None else method

    by_id: dict[str, FusionCandidate] = {}

    for rank, (chunk, score) in enumerate(bm25_results):
        cand = by_id.setdefault(chunk.chunk_id, FusionCandidate(chunk=chunk))
        cand.bm25_score = score
        cand.bm25_rank = rank

    for rank, (chunk, score) in enumerate(dense_results):
        cand = by_id.setdefault(chunk.chunk_id, FusionCandidate(chunk=chunk))
        cand.dense_score = score
        cand.dense_rank = rank

    if method == "rrf":
        k = settings.rrf_k
        for cand in by_id.values():
            score = 0.0
            if cand.bm25_rank is not None:
                score += 1.0 / (k + cand.bm25_rank + 1)
            if cand.dense_rank is not None:
                score += 1.0 / (k + cand.dense_rank + 1)
            cand.hybrid_score = score
    else:  # weighted score fusion
        bm25_norm = _min_max_normalize(
            {cid: c.bm25_score for cid, c in by_id.items() if c.bm25_score is not None}
        )
        dense_norm = _min_max_normalize(
            {cid: c.dense_score for cid, c in by_id.items() if c.dense_score is not None}
        )
        for cid, cand in by_id.items():
            b = bm25_norm.get(cid, 0.0)
            d = dense_norm.get(cid, 0.0)
            cand.hybrid_score = alpha * b + (1 - alpha) * d

    return sorted(by_id.values(), key=lambda c: c.hybrid_score, reverse=True)
