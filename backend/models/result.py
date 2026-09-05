"""Pydantic models for the full pipeline result, shared by the API and the
evaluation harness so the UI, ad-hoc queries, and metrics all consume the
exact same shape.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from backend.models.evidence import RetrievedEvidence


class PipelineConfig(BaseModel):
    """Toggles used by the baseline/ablation harness (spec sections 17-18).
    Defaults reproduce the full final system."""

    use_bm25: bool = True
    use_dense: bool = True
    use_reranker: bool = True
    use_patient_context: bool = True
    use_verification: bool = True
    hybrid_alpha: Optional[float] = None
    rerank_top_k: Optional[int] = None
    bm25_candidate_k: Optional[int] = None
    dense_candidate_k: Optional[int] = None
    reliability_gate_mode: Optional[str] = None
    label: str = "full_system"


class EvidenceSnapshot(BaseModel):
    citation_id: Optional[str] = None
    chunk_id: str
    document_id: str
    title: str
    section: str
    pmid: Optional[str] = None
    url: Optional[str] = None
    publication_year: Optional[int] = None
    text: str
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @classmethod
    def from_retrieved(cls, ev: RetrievedEvidence) -> "EvidenceSnapshot":
        return cls(
            citation_id=ev.citation_id,
            chunk_id=ev.chunk.chunk_id,
            document_id=ev.chunk.document_id,
            title=ev.chunk.title,
            section=ev.chunk.section,
            pmid=ev.chunk.pmid,
            url=ev.chunk.url,
            publication_year=ev.chunk.publication_year,
            text=ev.chunk.text,
            bm25_score=ev.bm25_score,
            dense_score=ev.dense_score,
            hybrid_score=ev.hybrid_score,
            rerank_score=ev.rerank_score,
        )


class ClaimResult(BaseModel):
    position: int
    text: str
    raw_text: str = ""  # original sentence as it appeared in raw_answer, citation markers included
    citation_ids: list[str]
    verdict: Optional[str] = None
    nli_entailment: float = 0.0
    nli_contradiction: float = 0.0
    citation_statuses: list[str] = []
    display_marker: str = ""
    included_in_final: bool = True


class PatientContextFieldOut(BaseModel):
    field_type: str
    text: str
    similarity: float


class PipelineResult(BaseModel):
    patient_id: str
    question: str
    augmented_query: str
    patient_context_fields: list[PatientContextFieldOut]

    retrieved_bm25: list[EvidenceSnapshot]
    retrieved_dense: list[EvidenceSnapshot]
    fused: list[EvidenceSnapshot]
    reranked: list[EvidenceSnapshot]

    raw_answer: str
    final_answer: str
    llm_provider: str
    llm_model: str

    claims: list[ClaimResult]

    reliability_gate_mode: str
    disclaimer: str = (
        "Research prototype - not a medical device. Synthetic patient data. "
        "Not a substitute for professional medical advice."
    )

    latency_retrieval_s: float = 0.0
    latency_generation_s: float = 0.0
    latency_verification_s: float = 0.0

    config: PipelineConfig
