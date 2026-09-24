"""Unit tests for the shared feature-building code in features.py."""

import numpy as np
import pytest
from rdkit import Chem

from features import DESCRIPTOR_NAMES, FINGERPRINT_BITS, compute_descriptors, featurize_mol, feature_names

ASPIRIN = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")


def test_featurize_mol_returns_expected_length():
    vector = featurize_mol(ASPIRIN)
    assert vector.shape == (FINGERPRINT_BITS + len(DESCRIPTOR_NAMES),)


def test_featurize_mol_fingerprint_part_is_binary():
    vector = featurize_mol(ASPIRIN)
    fingerprint_part = vector[:FINGERPRINT_BITS]
    assert set(np.unique(fingerprint_part)).issubset({0.0, 1.0})


def test_featurize_mol_is_deterministic():
    v1 = featurize_mol(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    v2 = featurize_mol(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    assert np.array_equal(v1, v2)


def test_featurize_mol_different_molecules_differ():
    aspirin = featurize_mol(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    water = featurize_mol(Chem.MolFromSmiles("O"))
    assert not np.array_equal(aspirin, water)


def test_compute_descriptors_molecular_weight_is_positive():
    descriptors = compute_descriptors(ASPIRIN)
    mol_wt_index = DESCRIPTOR_NAMES.index("MolWt")
    assert descriptors[mol_wt_index] == pytest.approx(180.16, abs=0.1)


def test_compute_descriptors_aromatic_ring_count():
    descriptors = compute_descriptors(ASPIRIN)
    ring_index = DESCRIPTOR_NAMES.index("NumAromaticRings")
    assert descriptors[ring_index] == 1


def test_feature_names_matches_vector_length():
    assert len(feature_names()) == FINGERPRINT_BITS + len(DESCRIPTOR_NAMES)
    assert feature_names()[-len(DESCRIPTOR_NAMES):] == DESCRIPTOR_NAMES
