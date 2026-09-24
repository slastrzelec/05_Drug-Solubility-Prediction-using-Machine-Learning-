"""Feature construction shared between training and the app.

Builds the combined feature vector used by the v2 model: a 2048-bit Morgan
fingerprint plus a small set of physicochemical descriptors, in a fixed
column order so training and inference always agree on feature layout.
"""

import numpy as np
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048

DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
]


def compute_descriptors(mol):
    """Compute the fixed set of physicochemical descriptors for a molecule,
    in DESCRIPTOR_NAMES order."""
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
    ]


def featurize_mol(mol):
    """Build the combined [fingerprint | descriptors] feature vector for an
    already-parsed RDKit molecule. Returns a 1D numpy array of length
    FINGERPRINT_BITS + len(DESCRIPTOR_NAMES)."""
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
    )
    fp_array = np.array(fp, dtype=float)
    descriptors = np.array(compute_descriptors(mol), dtype=float)
    return np.concatenate([fp_array, descriptors])


def feature_names():
    return [f"fp_{i}" for i in range(FINGERPRINT_BITS)] + list(DESCRIPTOR_NAMES)
