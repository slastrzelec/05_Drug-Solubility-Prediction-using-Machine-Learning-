"""FastAPI service for the Drug Solubility Predictor.

Independent of the Streamlit app (app.py): both load the same trained
artifacts directly and share the same prediction code in solubility.py,
but neither calls the other. See SPEC.md's step-5 addendum.

Run locally:    uvicorn api:app --reload
Docs:            http://localhost:8000/docs
"""

import json
import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rdkit import Chem

from solubility import predict_with_uncertainty
from uncertainty import load_training_fingerprints

_pipeline = None
_train_fingerprints = None
_interval_half_width = None


def _load_artifacts():
    """Idempotent: loads the trained artifacts once. Called both from the
    lifespan handler (normal startup) and lazily from each endpoint, so
    tests that hit the app without a real server lifecycle (e.g. a bare
    TestClient(app) with no context manager) still work."""
    global _pipeline, _train_fingerprints, _interval_half_width
    if _pipeline is not None:
        return
    if not os.path.exists("drug_solubility_pipeline.joblib"):
        raise RuntimeError("drug_solubility_pipeline.joblib not found")
    _pipeline = joblib.load("drug_solubility_pipeline.joblib")
    _train_fingerprints = load_training_fingerprints()
    with open("uncertainty_calibration.json", encoding="utf-8") as f:
        _interval_half_width = json.load(f)["q90_half_width"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="Drug Solubility Predictor API",
    description="Predicts aqueous solubility of drug molecules from SMILES notation.",
    version="2.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="SMILES notation of the molecule, e.g. CC(=O)Oc1ccccc1C(=O)O (aspirin)",
    )


class PredictResponse(BaseModel):
    smiles: str
    log_solubility: float
    actual_solubility_mol_per_l: float
    category: str
    interval_low: float
    interval_high: float
    max_train_similarity: float
    in_applicability_domain: bool


@app.get("/health")
def health():
    _load_artifacts()
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    _load_artifacts()

    # Reject a syntactically well-formed but chemically meaningless SMILES
    # with a proper 400 before touching the model, same validation RDKit
    # already does inside predict_with_uncertainty - checked here too so
    # the API gives a clear client error rather than a 500.
    if Chem.MolFromSmiles(request.smiles) is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES notation")

    from solubility import categorize_solubility

    result, error = predict_with_uncertainty(
        request.smiles, _pipeline, _train_fingerprints, _interval_half_width
    )
    if error:
        raise HTTPException(status_code=400, detail=error)

    category, _description, _color = categorize_solubility(result["log_solubility"])

    return PredictResponse(
        smiles=request.smiles,
        log_solubility=result["log_solubility"],
        actual_solubility_mol_per_l=result["actual_solubility"],
        category=category,
        interval_low=result["interval_low"],
        interval_high=result["interval_high"],
        max_train_similarity=result["max_train_similarity"],
        in_applicability_domain=result["in_domain"],
    )
