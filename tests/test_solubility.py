"""Unit tests for the core prediction logic in solubility.py.

These test the pure functions used by the Streamlit app (app.py imports
them) without needing Streamlit itself or the trained model files.
"""

import numpy as np
import pytest

from solubility import categorize_solubility, predict_solubility

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


class FakeScaler:
    """Stand-in for the fitted StandardScaler: passes features through
    unscaled so tests don't depend on the trained scaler.joblib."""

    def transform(self, X):
        return X


class FakeModel:
    """Stand-in for the trained RandomForestRegressor: returns a fixed
    prediction so tests are deterministic and don't need the ~2MB
    trained model file."""

    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        return np.array([self.prediction])


# --- predict_solubility ---------------------------------------------------

def test_predict_solubility_valid_smiles_returns_result():
    result, error = predict_solubility(ASPIRIN_SMILES, FakeModel(-2.55), FakeScaler())

    assert error is None
    assert result["log_solubility"] == pytest.approx(-2.55)
    assert result["actual_solubility"] == pytest.approx(10 ** -2.55)
    assert result["mol"] is not None


def test_predict_solubility_invalid_smiles_returns_error():
    result, error = predict_solubility("not a smiles!!", FakeModel(0.0), FakeScaler())

    assert result is None
    assert error == "Invalid SMILES notation"


def test_predict_solubility_empty_string_parses_as_empty_molecule():
    # RDKit treats "" as a valid (empty) molecule rather than raising, so
    # this does NOT hit the "Invalid SMILES notation" branch - unlike a
    # genuinely malformed string such as "not a smiles!!".
    result, error = predict_solubility("", FakeModel(0.0), FakeScaler())

    assert error is None
    assert result["mol"] is not None
    assert result["mol"].GetNumAtoms() == 0


def test_predict_solubility_actual_solubility_is_power_of_ten_of_log():
    result, error = predict_solubility(ASPIRIN_SMILES, FakeModel(-3.0), FakeScaler())

    assert error is None
    assert result["actual_solubility"] == pytest.approx(1e-3)


# --- categorize_solubility -------------------------------------------------

@pytest.mark.parametrize(
    "log_sol,expected_label",
    [
        (1.58, "🟢 High"),      # top of the ESOL dataset range
        (0.0, "🟢 High"),
        (-0.5, "🟢 High"),
        (-1.5, "🟡 Medium"),
        (-2.99, "🟡 Medium"),
        (-3.0, "🔴 Low"),       # boundary: not > -3, so falls to Low
        (-3.01, "🔴 Low"),
        (-11.60, "🔴 Low"),     # bottom of the ESOL dataset range
    ],
)
def test_categorize_solubility_thresholds(log_sol, expected_label):
    label, _description, _color = categorize_solubility(log_sol)
    assert label == expected_label


def test_categorize_solubility_returns_label_description_and_color():
    label, description, color = categorize_solubility(0.0)
    assert label == "🟢 High"
    assert description == "High solubility"
    assert color.startswith("#")
