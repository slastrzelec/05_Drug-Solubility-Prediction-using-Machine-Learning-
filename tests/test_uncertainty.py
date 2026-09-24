"""Tests for the step-4 uncertainty additions: the applicability-domain
check and the prediction interval helper (uncertainty.py), plus the
combined predict_with_uncertainty() in solubility.py.
"""

import numpy as np
import pytest
from rdkit import Chem

from solubility import predict_with_uncertainty
from uncertainty import applicability_domain, compute_fingerprint, prediction_interval

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


class FakePipeline:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        return np.array([self.prediction])


# --- prediction_interval ----------------------------------------------------

def test_prediction_interval_is_symmetric_around_point_estimate():
    low, high = prediction_interval(-2.0, 0.5)
    assert low == pytest.approx(-2.5)
    assert high == pytest.approx(-1.5)


def test_prediction_interval_zero_width_collapses_to_point():
    low, high = prediction_interval(-2.0, 0.0)
    assert low == high == -2.0


# --- applicability_domain ----------------------------------------------------

def test_applicability_domain_identical_molecule_is_fully_in_domain():
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    train_fps = [compute_fingerprint(mol)]  # the training set contains this exact molecule
    max_sim, in_domain = applicability_domain(mol, train_fps)
    assert max_sim == pytest.approx(1.0)
    assert in_domain is True


def test_applicability_domain_empty_training_set_is_out_of_domain():
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    max_sim, in_domain = applicability_domain(mol, [])
    assert max_sim == 0.0
    assert in_domain is False


def test_applicability_domain_respects_custom_threshold():
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    other = Chem.MolFromSmiles("O")  # water: very dissimilar to aspirin
    train_fps = [compute_fingerprint(other)]
    max_sim, in_domain_default = applicability_domain(mol, train_fps)
    # with threshold 0.0 everything counts as in-domain
    _, in_domain_permissive = applicability_domain(mol, train_fps, threshold=0.0)
    assert in_domain_permissive is True
    assert max_sim < 0.4  # sanity: aspirin and water really are dissimilar


# --- predict_with_uncertainty ------------------------------------------------

def test_predict_with_uncertainty_adds_interval_and_domain_fields():
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    train_fps = [compute_fingerprint(mol)]

    result, error = predict_with_uncertainty(
        ASPIRIN_SMILES, FakePipeline(-2.0), train_fps, interval_half_width=0.5
    )

    assert error is None
    assert result["log_solubility"] == pytest.approx(-2.0)
    assert result["interval_low"] == pytest.approx(-2.5)
    assert result["interval_high"] == pytest.approx(-1.5)
    assert result["in_domain"] is True
    assert result["max_train_similarity"] == pytest.approx(1.0)


def test_predict_with_uncertainty_propagates_invalid_smiles_error():
    result, error = predict_with_uncertainty(
        "not a smiles!!", FakePipeline(0.0), [], interval_half_width=0.5
    )
    assert result is None
    assert error == "Invalid SMILES notation"
