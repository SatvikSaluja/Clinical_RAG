from backend.verification.claim_extractor import extract_claims


def test_extracts_citations_and_strips_markers_from_claim_text():
    answer = (
        "Answer\n"
        "Elevated HbA1c indicates higher average blood glucose [E1]. "
        "Metformin is commonly used to improve glycemic control [E2][E3].\n\n"
        "Limitations / uncertainty\n"
        "The evidence does not address this patient's specific dose.\n"
    )
    claims = extract_claims(answer)
    assert len(claims) == 2
    assert claims[0].citation_ids == ["E1"]
    assert "[E1]" not in claims[0].text
    assert claims[1].citation_ids == ["E2", "E3"]


def test_claim_with_no_citation_has_empty_citation_list():
    answer = "Answer\nThis statement has no citation at all.\n"
    claims = extract_claims(answer)
    assert len(claims) == 1
    assert claims[0].citation_ids == []


def test_limitations_section_excluded_from_claims():
    answer = (
        "Answer\nMetformin lowers glucose [E1].\n\n"
        "Limitations / uncertainty\nThis claim should not be extracted as a factual claim.\n"
    )
    claims = extract_claims(answer)
    assert len(claims) == 1
    assert "should not be extracted" not in claims[0].text
