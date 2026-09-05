"""End-to-end RAG pipeline orchestrator: patient-aware query construction ->
BM25 + dense retrieval -> hybrid fusion -> cross-encoder reranking ->
LLM generation -> claim extraction -> claim/citation verification ->
reliability gate -> final answer.

This is the single place that wires every module together, and is used
directly by the FastAPI routes AND by the evaluation/baseline/ablation
harness (via `PipelineConfig` toggles), so the live app and the measured
experiments are guaranteed to run the exact same code path.
"""
from __future__ import annotations

import logging
import re
import time

from backend.config.settings import INDEX_DIR, settings
from backend.generation.generator import get_generator
from backend.generation.prompts import build_generation_prompt
from backend.models.evidence import EvidenceChunk, RetrievedEvidence
from backend.models.patient import Patient
from backend.models.result import (
    ClaimResult,
    EvidenceSnapshot,
    PatientContextFieldOut,
    PipelineConfig,
    PipelineResult,
)
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.dense import DenseIndex
from backend.retrieval.hybrid import fuse, merge_candidate_lists
from backend.retrieval.query_builder import build_augmented_query
from backend.retrieval.reranker import CrossEncoderReranker
from backend.verification.citation_verifier import CitationStatus, verify_citations
from backend.verification.claim_extractor import extract_claims
from backend.verification.claim_verifier import Verdict, verify_claims

logger = logging.getLogger(__name__)

_engine_singleton: "RetrievalEngine | None" = None


class RetrievalEngine:
    """Holds the loaded BM25 + dense indices and reranker (expensive to
    construct, so this is a process-wide singleton)."""

    def __init__(self):
        bm25_path = INDEX_DIR / "bm25.pkl"
        dense_path = INDEX_DIR / "dense"
        if not bm25_path.exists() or not dense_path.exists():
            raise RuntimeError(
                "Indices not found under data/processed/index/. Run "
                "`python -m backend.scripts.build_corpus` then "
                "`python -m backend.scripts.build_index` first."
            )
        logger.info("Loading BM25 index from %s", bm25_path)
        self.bm25 = BM25Index.load(bm25_path)
        logger.info("Loading dense index from %s", dense_path)
        self.dense = DenseIndex.load(dense_path)
        self._reranker: CrossEncoderReranker | None = None

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def chunk_by_id(self, chunk_id: str) -> EvidenceChunk | None:
        for c in self.bm25.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None


def get_engine() -> RetrievalEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = RetrievalEngine()
    return _engine_singleton


def _to_snapshot(chunk: EvidenceChunk, **scores) -> EvidenceSnapshot:
    return EvidenceSnapshot(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=chunk.title,
        section=chunk.section,
        pmid=chunk.pmid,
        url=chunk.url,
        publication_year=chunk.publication_year,
        text=chunk.text,
        **scores,
    )


def run_pipeline(patient: Patient, question: str, config: PipelineConfig | None = None) -> PipelineResult:
    config = config or PipelineConfig()
    engine = get_engine()

    t0 = time.time()

    # --- 1. Patient-aware query construction ---
    if config.use_patient_context:
        augmented_query, context_fields = build_augmented_query(patient, question)
    else:
        augmented_query, context_fields = question, []

    # --- 2. Retrieval ---
    # Patient-context terms are UNIONED into the candidate pool from a
    # separate search rather than concatenated into one query string: a
    # single concatenated query lets verbose patient-context text dilute
    # the BM25/dense signal for the actual question (measured to cost ~33
    # points of Recall@5 - see RESULTS.md ablation). Running the raw
    # question and the augmented query as two searches and merging
    # (keeping each chunk's best score) lets patient context ADD relevant
    # candidates the question alone would miss, without ever displacing the
    # question-only retrieval's own top hits.
    bm25_k = config.bm25_candidate_k or settings.bm25_candidate_k
    dense_k = config.dense_candidate_k or settings.dense_candidate_k
    context_augmented = config.use_patient_context and augmented_query != question

    if config.use_bm25:
        bm25_results = engine.bm25.search(question, top_k=bm25_k)
        if context_augmented:
            bm25_results = merge_candidate_lists(
                bm25_results, engine.bm25.search(augmented_query, top_k=bm25_k), top_k=bm25_k
            )
    else:
        bm25_results = []

    if config.use_dense:
        dense_results = engine.dense.search(question, top_k=dense_k)
        if context_augmented:
            dense_results = merge_candidate_lists(
                dense_results, engine.dense.search(augmented_query, top_k=dense_k), top_k=dense_k
            )
    else:
        dense_results = []

    # --- 3. Hybrid fusion ---
    fused = fuse(bm25_results, dense_results, alpha=config.hybrid_alpha)

    # --- 4. Cross-encoder reranking (or direct truncation if disabled/ablated) ---
    # Reranked against the raw question (not the augmented query) for the
    # same reason: the reranker should judge relevance to what was actually
    # asked, not to a query bloated with patient-record text.
    top_k = config.rerank_top_k or settings.rerank_top_k
    if config.use_reranker and fused:
        reranked_pairs = engine.reranker.rerank(question, fused, top_k=top_k)
        reranked = [
            RetrievedEvidence(
                chunk=c.chunk, bm25_score=c.bm25_score, dense_score=c.dense_score,
                hybrid_score=c.hybrid_score, rerank_score=score,
            )
            for c, score in reranked_pairs
        ]
    else:
        reranked = [
            RetrievedEvidence(
                chunk=c.chunk, bm25_score=c.bm25_score, dense_score=c.dense_score,
                hybrid_score=c.hybrid_score, rerank_score=None,
            )
            for c in fused[:top_k]
        ]

    latency_retrieval = time.time() - t0

    # --- 5. Generation ---
    t1 = time.time()
    system_prompt, user_prompt = build_generation_prompt(question, context_fields, reranked)
    generator = get_generator()
    gen_result = generator.generate(system_prompt, user_prompt)
    latency_generation = time.time() - t1

    # --- 6/7. Claim extraction + verification, 8. citation verification ---
    t2 = time.time()
    claims = extract_claims(gen_result.text)
    evidence_by_citation = {ev.citation_id: ev.chunk.text for ev in reranked if ev.citation_id}
    citation_reports = verify_citations(claims, reranked)

    claim_results: list[ClaimResult] = []

    if config.use_verification:
        verifications = verify_claims(claims, evidence_by_citation)
        for claim, verification, citation_report in zip(claims, verifications, citation_reports):
            statuses = [c.status.value for c in citation_report.citation_checks]
            marker, included = _reliability_marker(
                verification.verdict, config.reliability_gate_mode or settings.reliability_gate_mode
            )
            claim_results.append(
                ClaimResult(
                    position=claim.position,
                    text=claim.text,
                    raw_text=claim.raw_text,
                    citation_ids=claim.citation_ids,
                    verdict=verification.verdict.value,
                    nli_entailment=verification.nli_entailment,
                    nli_contradiction=verification.nli_contradiction,
                    citation_statuses=statuses,
                    display_marker=marker,
                    included_in_final=included,
                )
            )
    else:
        for claim, citation_report in zip(claims, citation_reports):
            statuses = [c.status.value for c in citation_report.citation_checks]
            claim_results.append(
                ClaimResult(
                    position=claim.position,
                    text=claim.text,
                    raw_text=claim.raw_text,
                    citation_ids=claim.citation_ids,
                    verdict=None,
                    citation_statuses=statuses,
                    display_marker="(not verified - verification disabled)",
                    included_in_final=True,
                )
            )

    latency_verification = time.time() - t2

    final_answer = _apply_reliability_gate(
        gen_result.text, claim_results,
        config.reliability_gate_mode or settings.reliability_gate_mode,
        verification_enabled=config.use_verification,
    )

    return PipelineResult(
        patient_id=patient.patient_id,
        question=question,
        augmented_query=augmented_query,
        patient_context_fields=[
            PatientContextFieldOut(field_type=f.field_type, text=f.text, similarity=f.similarity)
            for f in context_fields
        ],
        retrieved_bm25=[_to_snapshot(c, bm25_score=s) for c, s in bm25_results],
        retrieved_dense=[_to_snapshot(c, dense_score=s) for c, s in dense_results],
        fused=[_to_snapshot(c.chunk, bm25_score=c.bm25_score, dense_score=c.dense_score, hybrid_score=c.hybrid_score) for c in fused],
        reranked=[EvidenceSnapshot.from_retrieved(ev) for ev in reranked],
        raw_answer=gen_result.text,
        final_answer=final_answer,
        llm_provider=gen_result.provider,
        llm_model=gen_result.model,
        claims=claim_results,
        reliability_gate_mode=config.reliability_gate_mode or settings.reliability_gate_mode,
        latency_retrieval_s=latency_retrieval,
        latency_generation_s=latency_generation,
        latency_verification_s=latency_verification,
        config=config,
    )


def _reliability_marker(verdict: Verdict, mode: str) -> tuple[str, bool]:
    if verdict == Verdict.SUPPORTED:
        return "✓ Supported", True
    if verdict == Verdict.CONTRADICTED:
        return "✗ Contradicted by evidence", mode != "remove"
    return "⚠ Unsupported by retrieved evidence", mode != "remove"


def _apply_reliability_gate(
    raw_answer: str, claim_results: list[ClaimResult], mode: str, verification_enabled: bool
) -> str:
    """Reliability gate (spec section 14): by default ("flag") the answer
    text is kept intact and claim-level statuses are surfaced alongside it
    (see the UI / API claims list) rather than silently edited - this keeps
    the research prototype transparent. In "remove" mode, unsupported/
    contradicted sentences are stripped from the displayed answer text.
    """
    if not verification_enabled or mode == "flag":
        return raw_answer

    if mode == "remove":
        dropped = [c for c in claim_results if not c.included_in_final and c.raw_text]
        if not dropped:
            return raw_answer
        result = raw_answer
        for c in dropped:
            # raw_text is the exact sentence substring extract_claims() split
            # out of raw_answer (citation markers included), so an exact
            # substring removal is precise - no fragile line-based guessing.
            result = result.replace(c.raw_text, "")
        result = re.sub(r"[ \t]{2,}", " ", result)
        result = re.sub(r"\n{3,}", "\n\n", result).strip()
        if not any(c.included_in_final for c in claim_results):
            result += (
                "\n\nThe available evidence does not provide enough support to "
                "answer this confidently."
            )
        return result

    # mode == "rewrite": append an uncertainty note rather than silently deleting
    unsupported = [c for c in claim_results if not c.included_in_final]
    if not unsupported:
        return raw_answer
    note = "\n\n[Reliability note] The following statement(s) were not well-supported by the retrieved evidence and should be treated with caution: " + \
        " / ".join(c.text for c in unsupported)
    return raw_answer + note
