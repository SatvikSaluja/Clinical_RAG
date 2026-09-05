"""Evidence document and chunk models, with full provenance (spec section 4/5)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class EvidenceDocument(BaseModel):
    document_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    source: str = "PubMed"
    pmid: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    text: str  # full abstract / article text
    query_topic: Optional[str] = None  # which ingestion topic query retrieved this doc


class EvidenceChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    title: str
    section: str = "abstract"
    publication_year: Optional[int] = None
    source: str = "PubMed"
    pmid: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None


class RetrievedEvidence(BaseModel):
    """A chunk plus retrieval/rerank scores, as surfaced to the UI/API."""

    chunk: EvidenceChunk
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None
    citation_id: Optional[str] = None  # e.g. "E1", assigned at prompt-build time
