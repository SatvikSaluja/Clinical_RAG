"""Claim extraction (spec section 11): split a generated answer into
individual factual-claim sentences, each retaining its citation marker(s)
and its position in the answer.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from backend.ingestion.chunker import split_sentences

_CITATION_RE = re.compile(r"\[E(\d+)\]")


class ExtractedClaim(BaseModel):
    position: int
    text: str          # claim text with citation markers stripped
    raw_text: str       # original sentence, markers included
    citation_ids: list[str]  # e.g. ["E1", "E2"]


def _strip_section_headers(answer: str) -> str:
    """Drop the literal 'Answer' / 'Limitations / uncertainty' / 'Evidence
    Sources' headers so they aren't misparsed as claims."""
    lines = []
    skip_evidence_sources = False
    for line in answer.split("\n"):
        stripped = line.strip()
        low = stripped.lower()
        if low in ("answer", "limitations / uncertainty", "limitations", "uncertainty"):
            continue
        if low.startswith("evidence sources"):
            skip_evidence_sources = True
            continue
        if skip_evidence_sources:
            # "E1. ..." reference-list lines are not claims either.
            if re.match(r"^E\d+[.:]", stripped):
                continue
            skip_evidence_sources = False
        lines.append(line)
    return "\n".join(lines)


def extract_claims(answer_text: str, only_limitations: bool = False) -> list[ExtractedClaim]:
    """Extract one ExtractedClaim per sentence found in the answer body
    (excludes the "Limitations / uncertainty" section by default, since
    those are hedge statements, not factual claims to verify)."""
    body = answer_text
    if "Limitations" in answer_text:
        body = answer_text.split("Limitations")[0]
    if "Evidence Sources" in body:
        body = body.split("Evidence Sources")[0]

    cleaned = _strip_section_headers(body)
    sentences = split_sentences(cleaned.replace("\n", " "))

    claims: list[ExtractedClaim] = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue
        citation_ids = [f"E{m}" for m in _CITATION_RE.findall(sent)]
        claim_text = _CITATION_RE.sub("", sent).strip()
        claim_text = re.sub(r"\s+", " ", claim_text)
        if len(claim_text) < 3:
            continue  # not a substantive claim (stray punctuation etc.)
        claims.append(
            ExtractedClaim(position=i, text=claim_text, raw_text=sent, citation_ids=citation_ids)
        )
    return claims
