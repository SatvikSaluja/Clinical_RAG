"""Patient-aware query construction (spec section 9).

Clinical Query Understanding pipeline:

    User Question -> embed against every patient-context field ->
    keep fields above a similarity threshold (capped to a max count) ->
    Augmented Retrieval Query (question + only the relevant fields)

This avoids dumping the entire patient record into the retrieval query
(which would blur BM25/dense signal and unnecessarily expose the full
record) while still surfacing the labs/medications/history actually
relevant to what was asked. The relevance decision is transparent and
inspectable (each field's similarity score is returned alongside it) rather
than an opaque LLM call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.config.settings import settings
from backend.models.patient import Patient
from backend.retrieval.bm25 import tokenize
from backend.retrieval.dense import encode_texts


@dataclass
class PatientContextField:
    field_type: str  # "lab" | "medication" | "history"
    text: str        # human-readable rendering, e.g. "Lab: HbA1c 7.4% (high)"
    similarity: float


def _render_lab(lab) -> str:
    return f"Lab result: {lab.test} = {lab.value} {lab.unit} (reference {lab.reference_range}, status: {lab.status}, {lab.date})"


def _render_medication(med) -> str:
    indication = f" for {med.indication}" if med.indication else ""
    return f"Medication: {med.name} {med.dose} {med.frequency}{indication}"


def _render_history(ev) -> str:
    return f"History ({ev.category}, {ev.date}): {ev.event}"


def render_all_fields(patient: Patient) -> list[PatientContextField]:
    fields: list[PatientContextField] = []
    for lab in patient.labs:
        fields.append(PatientContextField("lab", _render_lab(lab), 0.0))
    for med in patient.medications:
        fields.append(PatientContextField("medication", _render_medication(med), 0.0))
    for ev in patient.history:
        fields.append(PatientContextField("history", _render_history(ev), 0.0))
    return fields


def _keyword_overlap_similarities(question: str, texts: list[str]) -> list[float]:
    """Jaccard token overlap - a zero-dependency fallback relevance signal
    when no embedding provider is available (settings.embedding_provider ==
    "none"). Reuses the BM25 tokenizer so hyphenated/numeric clinical terms
    (e.g. "HbA1c", "7.4") still match correctly."""
    q_tokens = set(tokenize(question))
    sims = []
    for text in texts:
        t_tokens = set(tokenize(text))
        union = q_tokens | t_tokens
        sims.append(len(q_tokens & t_tokens) / len(union) if union else 0.0)
    return sims


def select_relevant_context(
    patient: Patient, question: str, max_fields: int | None = None, threshold: float | None = None
) -> list[PatientContextField]:
    """Rank every patient-context field by relevance to the question and
    keep the ones above `threshold`, capped at `max_fields`. Always returns
    at least one field (the single most relevant) so the generator never
    runs with zero patient context.
    """
    max_fields = settings.max_patient_context_fields if max_fields is None else max_fields
    threshold = settings.patient_context_similarity_threshold if threshold is None else threshold

    fields = render_all_fields(patient)
    if not fields:
        return []

    field_texts = [f.text for f in fields]
    if settings.embedding_provider == "none":
        sims = _keyword_overlap_similarities(question, field_texts)
        # Jaccard overlap runs much lower than cosine similarity (rarely
        # exceeds ~0.3 even for a strong match) - the configured threshold
        # is tuned for embedding similarity, so cap it for this fallback.
        threshold = min(threshold, 0.1)
    else:
        embeddings = encode_texts([question] + field_texts)
        q_emb = embeddings[0]
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        sims = []
        for emb in embeddings[1:]:
            e_norm = emb / (np.linalg.norm(emb) + 1e-9)
            sims.append(float(np.dot(q_norm, e_norm)))

    for f, s in zip(fields, sims):
        f.similarity = s

    ranked = sorted(fields, key=lambda f: f.similarity, reverse=True)
    selected = [f for f in ranked if f.similarity >= threshold][:max_fields]
    if not selected and ranked:
        selected = [ranked[0]]
    return selected


def build_augmented_query(patient: Patient, question: str) -> tuple[str, list[PatientContextField]]:
    """Construct the final retrieval query: the question plus the relevant
    patient-context fields, rendered as plain text for BM25/dense retrieval.
    """
    relevant = select_relevant_context(patient, question)
    context_str = " ".join(f.text for f in relevant)
    augmented_query = f"{question} {context_str}".strip()
    return augmented_query, relevant
