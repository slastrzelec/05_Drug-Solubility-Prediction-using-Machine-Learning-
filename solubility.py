"""Core prediction logic for the Drug Solubility Predictor.

Kept separate from app.py so it can be unit tested without importing
Streamlit (app.py runs its UI code at import time, which makes testing
functions defined inline in it impractical).
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def predict_solubility(smiles, model, scaler):
    """Predict aqueous solubility for a given SMILES string.

    Returns a (result, error) tuple: on success, result is a dict with
    'log_solubility', 'actual_solubility' and 'mol' and error is None;
    on failure, result is None and error is a human-readable message.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES notation"

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = np.array(fp).reshape(1, -1)

        fp_scaled = scaler.transform(fp_array)
        log_solubility = model.predict(fp_scaled)[0]
        actual_solubility = 10 ** log_solubility

        return {
            "log_solubility": log_solubility,
            "actual_solubility": actual_solubility,
            "mol": mol,
        }, None
    except Exception as e:
        return None, str(e)


def categorize_solubility(log_sol):
    """Classify a log(solubility) value as High / Medium / Low.

    Returns (label, description, color_hex).
    """
    if log_sol > -1:
        return "🟢 High", "High solubility", "#00ff41"
    elif log_sol > -3:
        return "🟡 Medium", "Moderate solubility", "#ffaa00"
    else:
        return "🔴 Low", "Low solubility", "#ff0000"
