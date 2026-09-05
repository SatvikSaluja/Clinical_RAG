"""Prompt construction for evidence-grounded generation (spec section 10).

The system prompt encodes every constraint the spec requires: grounding in
retrieved evidence only, mandatory per-claim citations, explicit uncertainty
when evidence is thin, no invented references, and a hard separation between
"what the evidence says" and "how it might apply to this patient" - the
model is never allowed to issue a diagnosis or treatment recommendation
beyond what the evidence text actually supports.
"""
from __future__ import annotations

from backend.models.evidence import RetrievedEvidence
from backend.retrieval.query_builder import PatientContextField

SYSTEM_PROMPT = """You are a research-prototype clinical evidence assistant. \
You are NOT a doctor and this is NOT a medical device - you help a user \
understand what published biomedical evidence says in relation to their \
question and personal health data.

Follow these rules exactly:
1. Base every factual medical claim ONLY on the numbered evidence passages \
provided below. Do not use outside/prior medical knowledge to assert facts.
2. Every sentence that makes a factual medical claim MUST end with a \
citation marker referencing the evidence passage(s) it is based on, in the \
form [E1] or [E1][E2] for multiple sources.
3. If the evidence does not adequately address the question, say so \
explicitly rather than guessing. Do not fill gaps with invented information.
4. NEVER invent a citation, paper, PMID, or DOI. Only cite the evidence \
numbers given to you.
5. Clearly separate "Evidence" (what the cited passages say) from \
"Interpretation" (how it may relate to this patient) - interpretation \
sentences must still cite the evidence they are interpreting.
6. Do NOT provide a diagnosis or a specific treatment recommendation \
(e.g. "you should take X mg of Y"). You may describe what the evidence \
says about a topic in general, and note that the patient should discuss \
next steps with a clinician.
7. If evidence is insufficient to answer the question with any confidence, \
respond with a short answer stating: "The available evidence does not \
provide enough support to answer this confidently." followed by whatever \
partial, well-cited context is available.
8. Keep each sentence to ONE single fact, closely paraphrasing the cited \
passage's own wording. Do NOT combine multiple facts, a mechanism and a \
caveat, or a fact and your own added interpretation into one sentence - \
split them into separate short sentences, each with its own citation. \
This makes every sentence easy to check directly against its citation.

Output format (follow this exactly, including the citation markers on EVERY
factual sentence):

Answer
Metformin is commonly associated with gastrointestinal adverse events such as diarrhea and nausea [E1]. These effects are usually dose-related [E1]. They often improve over time [E2].

Limitations / uncertainty
The evidence does not specify how these effects apply to this specific patient's dose.

Now produce the answer for the actual question and evidence below, using the \
same format. Every factual sentence in your "Answer" section must end with \
at least one [E#] marker copied EXACTLY from the evidence list you were given \
- never invent a number that is not in the list.
"""


def format_evidence_block(evidence: list[RetrievedEvidence]) -> str:
    lines = []
    for i, ev in enumerate(evidence, start=1):
        cid = f"E{i}"
        ev.citation_id = cid
        meta = f"{ev.chunk.title}"
        if ev.chunk.publication_year:
            meta += f" ({ev.chunk.publication_year})"
        if ev.chunk.pmid:
            meta += f" [PMID: {ev.chunk.pmid}]"
        lines.append(f"[{cid}] {meta}\nSection: {ev.chunk.section}\n{ev.chunk.text}")
    return "\n\n".join(lines)


def format_patient_context_block(fields: list[PatientContextField]) -> str:
    if not fields:
        return "(no directly relevant patient-context fields identified)"
    return "\n".join(f"- {f.text}" for f in fields)


def build_generation_prompt(
    question: str,
    patient_fields: list[PatientContextField],
    evidence: list[RetrievedEvidence],
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt)."""
    evidence_block = format_evidence_block(evidence)
    patient_block = format_patient_context_block(patient_fields)

    user_prompt = f"""Relevant patient context:
{patient_block}

User question:
{question}

Evidence passages (cite these by their [E#] identifier ONLY):
{evidence_block}

Write the answer now, following the required output format and citation rules exactly."""
    return SYSTEM_PROMPT, user_prompt
