from backend.models.evidence import EvidenceChunk, RetrievedEvidence
from backend.verification.citation_verifier import CitationStatus, verify_citations
from backend.verification.claim_extractor import ExtractedClaim


def _evidence(citation_id, chunk_id):
    chunk = EvidenceChunk(chunk_id=chunk_id, document_id=chunk_id, text="evidence text", title="t")
    return RetrievedEvidence(chunk=chunk, citation_id=citation_id)


def test_valid_citation_traces_to_retrieved_chunk():
    retrieved = [_evidence("E1", "chunk_1")]
    claims = [ExtractedClaim(position=0, text="a claim", raw_text="a claim [E1]", citation_ids=["E1"])]
    reports = verify_citations(claims, retrieved)
    assert reports[0].citation_checks[0].status == CitationStatus.VALID
    assert reports[0].citation_checks[0].chunk_id == "chunk_1"
    assert reports[0].fully_traceable is True


def test_fabricated_citation_id_is_invalid():
    retrieved = [_evidence("E1", "chunk_1")]
    claims = [ExtractedClaim(position=0, text="a claim", raw_text="a claim [E9]", citation_ids=["E9"])]
    reports = verify_citations(claims, retrieved)
    assert reports[0].citation_checks[0].status == CitationStatus.INVALID
    assert reports[0].has_any_valid_citation is False


def test_claim_with_no_citation_marker_reports_empty_checks():
    claims = [ExtractedClaim(position=0, text="a claim", raw_text="a claim", citation_ids=[])]
    reports = verify_citations(claims, [])
    assert reports[0].citation_checks == []
    assert reports[0].has_any_valid_citation is False
    assert reports[0].fully_traceable is False
