"""Claim-level verification (spec section 12): classify each extracted claim
as SUPPORTED / CONTRADICTED / UNSUPPORTED using an NLI entailment model
against the specific evidence passage(s) the claim cited - the generator
never gets to judge its own output.

Primary: a general-purpose NLI cross-encoder (`MoritzLaurer/DeBERTa-v3-base-
mnli-fever-anli`). No off-the-shelf biomedical-specific NLI model is readily
available as a pip-installable model, so this is documented (README/RESULTS)
as the practical choice rather than claimed to be biomedical-specific.

Secondary (optional, spec-sanctioned "LLM-based verifier as a secondary
evaluation mechanism"): the same configurable LLM provider is asked, in a
strict forced-choice format, to judge entailment independently. When the two
disagree, the NLI verdict is treated as primary but the disagreement is
recorded on the result for transparency.
"""
from __future__ import annotations

import logging
import re
from enum import Enum

import torch
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config.settings import settings
from backend.generation.generator import get_generator
from backend.verification.claim_extractor import ExtractedClaim

logger = logging.getLogger(__name__)

_nli_cache: dict[str, tuple] = {}


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"


class ClaimVerification(BaseModel):
    claim: ExtractedClaim
    verdict: Verdict
    nli_entailment: float = 0.0
    nli_contradiction: float = 0.0
    nli_neutral: float = 0.0
    best_evidence_citation: str | None = None
    llm_verdict: Verdict | None = None
    disagreement: bool = False
    reason: str = ""


def _load_nli(model_name: str):
    if model_name in _nli_cache:
        return _nli_cache[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    id2label = {i: label.lower() for i, label in model.config.id2label.items()}
    _nli_cache[model_name] = (tokenizer, model, id2label)
    return tokenizer, model, id2label


class NLIVerifier:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.nli_model
        self.tokenizer, self.model, self.id2label = _load_nli(self.model_name)

    @torch.no_grad()
    def _score(self, premise: str, hypothesis: str) -> dict[str, float]:
        inputs = self.tokenizer(
            premise, hypothesis, truncation=True, max_length=512, return_tensors="pt"
        )
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        result = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        for i, p in enumerate(probs.tolist()):
            label = self.id2label.get(i, "")
            for key in result:
                if key in label:
                    result[key] = p
        return result

    def verify_against_passages(
        self, claim_text: str, cited_passages: dict[str, str]
    ) -> tuple[Verdict, float, float, float, str | None]:
        """Score the claim against each cited passage individually and take
        the passage with the strongest signal (max entailment, unless a
        stronger contradiction exists)."""
        if not cited_passages:
            return Verdict.UNSUPPORTED, 0.0, 0.0, 0.0, None

        best_citation = None
        best_entail, best_contra, best_neutral = 0.0, 0.0, 0.0
        for citation_id, passage in cited_passages.items():
            scores = self._score(premise=passage, hypothesis=claim_text)
            if scores["contradiction"] > best_contra:
                best_contra = scores["contradiction"]
            if scores["entailment"] > best_entail:
                best_entail = scores["entailment"]
                best_neutral = scores["neutral"]
                best_citation = citation_id

        if best_contra >= settings.nli_contradiction_threshold and best_contra > best_entail:
            return Verdict.CONTRADICTED, best_entail, best_contra, best_neutral, best_citation
        if best_entail >= settings.nli_support_threshold:
            return Verdict.SUPPORTED, best_entail, best_contra, best_neutral, best_citation
        return Verdict.UNSUPPORTED, best_entail, best_contra, best_neutral, best_citation


_LLM_VERDICT_RE = re.compile(r"\b(SUPPORTED|CONTRADICTED|UNSUPPORTED)\b", re.IGNORECASE)


def _llm_secondary_verify(claim_text: str, cited_passages: dict[str, str]) -> Verdict | None:
    if not cited_passages:
        return None
    evidence_text = "\n\n".join(f"[{cid}] {text}" for cid, text in cited_passages.items())
    system = (
        "You are a strict fact-checking verifier. Given an EVIDENCE passage and a "
        "CLAIM, decide if the evidence SUPPORTS, CONTRADICTS, or is UNSUPPORTED "
        "(does not address) the claim. Respond with exactly one word: SUPPORTED, "
        "CONTRADICTED, or UNSUPPORTED."
    )
    user = f"EVIDENCE:\n{evidence_text}\n\nCLAIM:\n{claim_text}\n\nVerdict (one word):"
    try:
        generator = get_generator()
        result = generator.generate(system, user)
        match = _LLM_VERDICT_RE.search(result.text.upper())
        if match:
            return Verdict(match.group(1).upper())
    except Exception as exc:  # pragma: no cover - defensive, secondary is optional
        logger.warning("LLM secondary verifier failed: %s", exc)
    return None


def verify_claims(
    claims: list[ExtractedClaim],
    evidence_by_citation: dict[str, str],
    use_llm_secondary: bool | None = None,
) -> list[ClaimVerification]:
    use_llm_secondary = settings.use_llm_secondary_verifier if use_llm_secondary is None else use_llm_secondary
    nli = NLIVerifier()
    results: list[ClaimVerification] = []

    for claim in claims:
        cited_passages = {
            cid: evidence_by_citation[cid] for cid in claim.citation_ids if cid in evidence_by_citation
        }
        verdict, entail, contra, neutral, best_cid = nli.verify_against_passages(
            claim.text, cited_passages
        )

        llm_verdict = None
        disagreement = False
        if use_llm_secondary and cited_passages:
            llm_verdict = _llm_secondary_verify(claim.text, cited_passages)
            if llm_verdict is not None and llm_verdict != verdict:
                disagreement = True

        reason_bits = []
        if not claim.citation_ids:
            reason_bits.append("claim has no citation marker")
        elif not cited_passages:
            reason_bits.append("cited ID(s) not found in retrieved evidence")
        else:
            reason_bits.append(
                f"NLI entailment={entail:.2f} contradiction={contra:.2f} vs. {best_cid}"
            )
        if disagreement:
            reason_bits.append(f"LLM secondary verifier disagreed (said {llm_verdict})")

        results.append(
            ClaimVerification(
                claim=claim,
                verdict=verdict,
                nli_entailment=entail,
                nli_contradiction=contra,
                nli_neutral=neutral,
                best_evidence_citation=best_cid,
                llm_verdict=llm_verdict,
                disagreement=disagreement,
                reason="; ".join(reason_bits),
            )
        )
    return results
