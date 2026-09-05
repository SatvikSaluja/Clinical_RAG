from __future__ import annotations

from pydantic import BaseModel

from backend.models.result import PipelineConfig


class QueryRequest(BaseModel):
    patient_id: str
    question: str
    config: PipelineConfig | None = None


class PatientListItem(BaseModel):
    patient_id: str
    name: str
    age: int
    sex: str
    summary: str | None = None
