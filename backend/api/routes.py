from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend.api.schemas import PatientListItem, QueryRequest
from backend.config.settings import PATIENTS_DIR, RESULTS_DIR
from backend.models.patient import Patient
from backend.models.result import PipelineResult
from backend.pipeline import run_pipeline

router = APIRouter()


def _load_patient(patient_id: str) -> Patient:
    path = PATIENTS_DIR / f"{patient_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown patient_id '{patient_id}'")
    return Patient(**json.loads(path.read_text()))


@router.get("/patients", response_model=list[PatientListItem])
def list_patients():
    items = []
    for path in sorted(PATIENTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        items.append(
            PatientListItem(
                patient_id=data["patient_id"], name=data["name"], age=data["age"],
                sex=data["sex"], summary=data.get("summary"),
            )
        )
    return items


@router.get("/patients/{patient_id}", response_model=Patient)
def get_patient(patient_id: str):
    return _load_patient(patient_id)


@router.post("/query", response_model=PipelineResult)
def query(request: QueryRequest):
    patient = _load_patient(request.patient_id)
    try:
        return run_pipeline(patient, request.question, config=request.config)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/dashboard")
def dashboard():
    """Surface whatever evaluation result JSON files currently exist under
    experiments/results/ - returns an empty structure (never fabricated
    numbers) until `python -m backend.evaluation.baselines`/`.ablation` have
    actually been run.
    """
    results = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            results[path.stem] = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
    return {"available_result_files": list(results.keys()), "results": results}
