import numpy as np

from backend.models.patient import HistoryEvent, LabResult, Medication, Patient
from backend.retrieval import query_builder


class _FakeModel:
    """Deterministic bag-of-words embedder so tests don't need to download a
    real model - similarity is just normalized word overlap, which is enough
    to exercise the relevance-ranking/threshold logic under test."""

    def encode(self, texts, convert_to_numpy=True):
        vocab = {}

        def vec(text):
            v = np.zeros(64)
            for word in text.lower().split():
                idx = hash(word) % 64
                v[idx] += 1.0
            return v

        return np.array([vec(t) for t in texts])


def _patient():
    return Patient(
        patient_id="p1", name="Test Patient", age=50, sex="male",
        labs=[LabResult(test="HbA1c", value=7.4, unit="%", reference_range="4.0-5.6", date="2026-01-01", status="high")],
        medications=[Medication(name="metformin", dose="500mg", frequency="BID", indication="type 2 diabetes")],
        history=[HistoryEvent(date="2025-01-01", event="Diagnosed with hypertension", category="diagnosis")],
    )


def test_select_relevant_context_prefers_matching_field(monkeypatch):
    monkeypatch.setattr(query_builder, "get_embedding_model", lambda *a, **k: (_FakeModel(), "fake"))
    patient = _patient()
    selected = query_builder.select_relevant_context(
        patient, "What does my HbA1c result mean?", max_fields=5, threshold=0.0
    )
    assert any("HbA1c" in f.text for f in selected)


def test_select_relevant_context_always_returns_at_least_one_field(monkeypatch):
    monkeypatch.setattr(query_builder, "get_embedding_model", lambda *a, **k: (_FakeModel(), "fake"))
    patient = _patient()
    selected = query_builder.select_relevant_context(
        patient, "completely unrelated gibberish xyzzy", max_fields=5, threshold=0.99
    )
    assert len(selected) == 1


def test_build_augmented_query_includes_question(monkeypatch):
    monkeypatch.setattr(query_builder, "get_embedding_model", lambda *a, **k: (_FakeModel(), "fake"))
    patient = _patient()
    query, fields = query_builder.build_augmented_query(patient, "What does my HbA1c mean?")
    assert "HbA1c" in query
