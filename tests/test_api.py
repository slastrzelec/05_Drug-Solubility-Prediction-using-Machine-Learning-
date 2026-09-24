"""Tests for the FastAPI service (api.py). Run in-process via TestClient -
no server or Docker needed. See SPEC.md's step-5 addendum.
"""

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def test_health_returns_ok_and_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_smiles_returns_expected_fields():
    response = client.post("/predict", json={"smiles": ASPIRIN_SMILES})
    assert response.status_code == 200
    body = response.json()

    assert body["smiles"] == ASPIRIN_SMILES
    assert isinstance(body["log_solubility"], float)
    assert body["actual_solubility_mol_per_l"] == pytest.approx(
        10 ** body["log_solubility"]
    )
    assert body["category"] in {"🟢 High", "🟡 Medium", "🔴 Low"}
    assert body["interval_low"] < body["log_solubility"] < body["interval_high"]
    assert 0.0 <= body["max_train_similarity"] <= 1.0
    assert isinstance(body["in_applicability_domain"], bool)


def test_predict_invalid_smiles_returns_400():
    response = client.post("/predict", json={"smiles": "not a smiles!!"})
    assert response.status_code == 400
    assert "Invalid SMILES" in response.json()["detail"]


def test_predict_missing_field_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_empty_string_returns_422_via_min_length():
    # Pydantic's min_length=1 rejects "" before it ever reaches RDKit,
    # unlike predict_solubility() itself which treats "" as a valid
    # (empty) molecule - see tests/test_solubility.py.
    response = client.post("/predict", json={"smiles": ""})
    assert response.status_code == 422
