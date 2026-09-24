"""Inference-time additions on top of the trained pipeline: a prediction
interval (split conformal, calibrated on the step-3 test set) and an
applicability-domain check (Tanimoto similarity to the training set).

See SPEC.md, "Addendum: Step 4" for the rationale and its honest
limitations. Nothing here retrains or touches the fitted scaler/selector/
model in drug_solubility_pipeline.joblib.
"""

import pickle

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from features import FINGERPRINT_BITS, FINGERPRINT_RADIUS

AD_SIMILARITY_THRESHOLD = 0.4
TRAIN_FINGERPRINTS_PATH = "train_fingerprints.pkl"


def compute_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
    )


def build_training_fingerprints(smiles_list, path=TRAIN_FINGERPRINTS_PATH):
    """Compute and save Morgan fingerprints for a list of SMILES (meant to
    be called once, on the training split only, from train_v2.py)."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(compute_fingerprint(mol))
    with open(path, "wb") as f:
        pickle.dump(fps, f)
    return fps


def load_training_fingerprints(path=TRAIN_FINGERPRINTS_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def applicability_domain(mol, train_fingerprints, threshold=AD_SIMILARITY_THRESHOLD):
    """Return (max_similarity, in_domain) for a query molecule against the
    training set's fingerprints. in_domain is False when the query is
    structurally dissimilar to everything the model was trained on -
    exactly the regime where the AqSolDB external validation showed
    accuracy dropping (see README.md)."""
    query_fp = compute_fingerprint(mol)
    if not train_fingerprints:
        return 0.0, False
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, train_fingerprints)
    max_similarity = max(similarities)
    return max_similarity, max_similarity >= threshold


def prediction_interval(point_estimate, half_width):
    """Symmetric interval around a point estimate using a half-width
    calibrated elsewhere (see train_v2.py / uncertainty_calibration.json)."""
    return point_estimate - half_width, point_estimate + half_width
