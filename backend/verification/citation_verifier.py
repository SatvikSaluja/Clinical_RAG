"""Citation verification (spec section 13): independent of claim-level NLI
verification, this checks the structural integrity of every citation - does
it point to a real, indexed evidence chunk that was actually retrieved for
this query? This catches fabricated/invalid/missing citations even before
asking whether the cited passage semantically supports the claim.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from backend.models.evidence import RetrievedEvidence
from backend.verification.claim_extractor import ExtractedClaim


class CitationStatus(str, Enum):
    VALID = "VALID"               # points to a real, retrieved chunk
    MISSING = "MISSING"           # claim has no citation at all
    INVALID = "INVALID"           # citation ID doesn't match any retrieved chunk (fabricated)
    UNSUPPORTING = "UNSUPPORTING"  # valid chunk, but NLI verifier found it doesn't support the claim


class CitationCheck(BaseModel):
    citation_id: str
    status: CitationStatus
    document_id: str | None = None
    chunk_id: str | None = None
    pmid: str | None = None


class ClaimCitationReport(BaseModel):
    claim_text: str
    citation_checks: list[CitationCheck]
    has_any_valid_citation: bool
    fully_traceable: bool  # every citation on this claim resolves to a real chunk


def verify_citations(
    claims: list[ExtractedClaim],
    retrieved_evidence: list[RetrievedEvidence],
) -> list[ClaimCitationReport]:
    """Trace every citation marker back to the actual retrieved evidence set.
    A citation ID that was never handed to the generator (i.e. doesn't
    appear in `retrieved_evidence`) is INVALID - this is exactly what would
    happen if the model fabricated a reference.
    """
    index = {ev.citation_id: ev for ev in retrieved_evidence if ev.citation_id}

    reports = []
    for claim in claims:
        if not claim.citation_ids:
            reports.append(
                ClaimCitationReport(
                    claim_text=claim.text,
                    citation_checks=[],
                    has_any_valid_citation=False,
                    fully_traceable=False,
                )
            )
            continue

        checks = []
        for cid in claim.citation_ids:
            ev = index.get(cid)
            if ev is None:
                checks.append(CitationCheck(citation_id=cid, status=CitationStatus.INVALID))
            else:
                checks.append(
                    CitationCheck(
                        citation_id=cid,
                        status=CitationStatus.VALID,
                        document_id=ev.chunk.document_id,
                        chunk_id=ev.chunk.chunk_id,
                        pmid=ev.chunk.pmid,
                    )
                )
        reports.append(
            ClaimCitationReport(
                claim_text=claim.text,
                citation_checks=checks,
                has_any_valid_citation=any(c.status == CitationStatus.VALID for c in checks),
                fully_traceable=all(c.status == CitationStatus.VALID for c in checks),
            )
        )
    return reports


def citation_accuracy(
    reports: list[ClaimCitationReport],
    verified_supported_citation_ids: set[str] | None = None,
) -> dict:
    """Compute citation accuracy = supported citations / total citations.

    A citation counts as "supported" if it is structurally VALID and (when
    `verified_supported_citation_ids` is provided, keyed as
    f"{claim_index}:{citation_id}") the claim-level verifier found it
    actually entails the claim. If no verification set is given, VALID alone
    counts (structural accuracy only).
    """
    total = 0
    supported = 0
    for i, report in enumerate(reports):
        for check in report.citation_checks:
            total += 1
            if check.status != CitationStatus.VALID:
                continue
            if verified_supported_citation_ids is None:
                supported += 1
            elif f"{i}:{check.citation_id}" in verified_supported_citation_ids:
                supported += 1
    accuracy = supported / total if total else None
    return {"supported": supported, "total": total, "accuracy": accuracy}
