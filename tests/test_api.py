from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_patients_returns_synthetic_patients():
    resp = client.get("/api/patients")
    assert resp.status_code == 200
    patients = resp.json()
    assert len(patients) >= 1
    ids = {p["patient_id"] for p in patients}
    assert "patient_001" in ids


def test_get_unknown_patient_returns_404():
    resp = client.get("/api/patients/does_not_exist")
    assert resp.status_code == 404


def test_dashboard_endpoint_never_errors_even_with_no_results():
    resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    assert "available_result_files" in resp.json()
