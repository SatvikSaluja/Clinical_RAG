from backend.verification.claim_extractor import ExtractedClaim
from backend.verification.claim_verifier import NLIVerifier, Verdict, verify_claims


class _FakeNLI(NLIVerifier):
    """Skips loading the real NLI model; returns pre-scripted scores keyed
    by a substring of the premise, so verify_against_passages logic is
    tested in isolation from the actual entailment model."""

    def __init__(self, script):
        self._script = script  # dict: passage_substring -> (entail, contra, neutral)

    def _score(self, premise, hypothesis):
        for key, (e, c, n) in self._script.items():
            if key in premise:
                return {"entailment": e, "neutral": n, "contradiction": c}
        return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}


def test_supported_verdict_when_entailment_above_threshold(monkeypatch):
    fake = _FakeNLI({"supporting passage": (0.9, 0.02, 0.08)})
    monkeypatch.setattr("backend.verification.claim_verifier.NLIVerifier", lambda: fake)
    monkeypatch.setattr("backend.verification.claim_verifier.settings.use_llm_secondary_verifier", False)

    claims = [ExtractedClaim(position=0, text="claim text", raw_text="claim text [E1]", citation_ids=["E1"])]
    results = verify_claims(claims, {"E1": "this is the supporting passage"}, use_llm_secondary=False)
    assert results[0].verdict == Verdict.SUPPORTED


def test_contradicted_verdict_when_contradiction_dominates(monkeypatch):
    fake = _FakeNLI({"contradicting passage": (0.05, 0.9, 0.05)})
    monkeypatch.setattr("backend.verification.claim_verifier.NLIVerifier", lambda: fake)

    claims = [ExtractedClaim(position=0, text="claim text", raw_text="claim text [E1]", citation_ids=["E1"])]
    results = verify_claims(claims, {"E1": "this is the contradicting passage"}, use_llm_secondary=False)
    assert results[0].verdict == Verdict.CONTRADICTED


def test_unsupported_when_no_citation_present(monkeypatch):
    fake = _FakeNLI({})
    monkeypatch.setattr("backend.verification.claim_verifier.NLIVerifier", lambda: fake)

    claims = [ExtractedClaim(position=0, text="claim text", raw_text="claim text", citation_ids=[])]
    results = verify_claims(claims, {}, use_llm_secondary=False)
    assert results[0].verdict == Verdict.UNSUPPORTED
    assert "no citation marker" in results[0].reason
