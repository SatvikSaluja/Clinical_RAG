"""Evaluation benchmark loader (spec section 15)."""
from __future__ import annotations

import json

from pydantic import BaseModel

from backend.config.settings import EVALUATION_DIR, PATIENTS_DIR
from backend.models.patient import Patient

BENCHMARK_PATH = EVALUATION_DIR / "benchmark.json"


class BenchmarkItem(BaseModel):
    item_id: str
    patient_id: str
    question: str
    gold_evidence: list[str]  # chunk_ids, e.g. "pmid_28770321_chunk_0"
    expected_claims: list[str] = []
    notes: str = ""


def load_benchmark() -> list[BenchmarkItem]:
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Benchmark file not found: {BENCHMARK_PATH}")
    data = json.loads(BENCHMARK_PATH.read_text())
    return [BenchmarkItem(**item) for item in data]


def load_patient(patient_id: str) -> Patient:
    path = PATIENTS_DIR / f"{patient_id}.json"
    return Patient(**json.loads(path.read_text()))
