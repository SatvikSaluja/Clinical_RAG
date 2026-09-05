"""Structured synthetic patient-context representation (spec section 3).

All patient data used by this system is synthetic and for research/demo
purposes only - see the disclaimer surfaced in the API and UI.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LabResult(BaseModel):
    test: str
    value: float
    unit: str
    reference_range: str
    date: str
    status: Literal["normal", "high", "low", "critical"]


class Medication(BaseModel):
    name: str
    dose: str
    frequency: str
    start_date: Optional[str] = None
    indication: Optional[str] = None


class HistoryEvent(BaseModel):
    date: str
    event: str
    category: Literal[
        "diagnosis", "procedure", "hospitalization", "symptom",
        "family_history", "abnormal_lab", "other",
    ] = "other"


class Patient(BaseModel):
    patient_id: str
    name: str = Field(..., description="Synthetic name - not a real person")
    age: int
    sex: Literal["male", "female", "other"]
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    history: list[HistoryEvent] = Field(default_factory=list)
    summary: Optional[str] = Field(
        default=None,
        description="One-line synthetic clinical summary for display purposes",
    )
