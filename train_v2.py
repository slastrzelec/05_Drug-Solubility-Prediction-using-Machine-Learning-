"""Train the v2 solubility model: Morgan fingerprint + RDKit descriptors,
feature selection, and a comparison of RF / GradientBoosting / SVR /
XGBoost - all inside sklearn Pipelines so scaling and feature selection
are fit on the training fold only. See SPEC.md for the full rationale
and the data-leakage guardrails this script follows.

This is the actual script used to produce drug_solubility_pipeline.joblib
and the numbers in README.md / train_v2_final_summary.json. A full run
(4 candidate models x GridSearchCV(cv=5), plus the AqSolDB download and
external validation) takes several minutes.

Usage:
    python train_v2.py            # full run: train, select, evaluate, save
    python train_v2.py --skip-external   # skip the AqSolDB download/validation
"""

import argparse
import csv
import json
import sys
import time
import urllib.request

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from features import featurize_mol
from uncertainty import build_training_fingerprints

RDLogger.DisableLog("rdApp.*")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_STATE = 42
BASELINE_TEST_R2 = 0.6985  # fingerprint-only Random Forest, documented in git history
AQSOLDB_URL = (
    "https://raw.githubusercontent.com/whitead/dmol-book/main/data/"
    "curated-solubility-dataset.csv"
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_esol(path="data.txt"):
    smiles_list, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row["SMILES"])
            y.append(float(row["measured log(solubility:mol/L)"]))
    return smiles_list, np.array(y)


def build_features(smiles_list):
    """Featurize every valid molecule; returns (X, mask) where mask marks
    which input SMILES parsed successfully (X only has the valid rows)."""
    feats, mask = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mask.append(False)
            continue
        feats.append(featurize_mol(mol))
        mask.append(True)
    return np.array(feats), np.array(mask)


def make_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectFromModel(
            RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
            threshold="median",
        )),
        ("model", model),
    ])


def candidate_models():
    candidates = {
        "RandomForest": (
            make_pipeline(RandomForestRegressor(random_state=RANDOM_STATE)),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 20],
                "model__max_features": ["sqrt"],
            },
        ),
        "GradientBoosting": (
            make_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3],
            },
        ),
        "SVR": (
            make_pipeline(SVR()),
            {
                "model__C": [1, 10],
                "model__epsilon": [0.1, 0.2],
                "model__kernel": ["rbf"],
            },
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = (
            make_pipeline(XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 6],
                "model__learning_rate": [0.05, 0.1],
            },
        )
    else:
        log("xgboost not installed - skipping that candidate")
    return candidates


def run_external_validation(pipeline, esol_smiles):
    """Evaluate the already-fitted pipeline on AqSolDB, after dropping any
    compound (by canonical SMILES) that also appears in ESOL - see
    SPEC.md: without this, the "external" score would be inflated by
    compounds the model has already seen, since ESOL is one of AqSolDB's
    nine source datasets."""
    log("Downloading AqSolDB (curated-solubility-dataset.csv)...")
    with urllib.request.urlopen(AQSOLDB_URL, timeout=60) as resp:
        raw = resp.read().decode("utf-8")

    esol_canon = set()
    for smi in esol_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            esol_canon.add(Chem.MolToSmiles(mol))

    reader = csv.DictReader(raw.splitlines())
    feats, ys, n_invalid, n_overlap = [], [], 0, 0
    for row in reader:
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            n_invalid += 1
            continue
        if Chem.MolToSmiles(mol) in esol_canon:
            n_overlap += 1
            continue
        feats.append(featurize_mol(mol))
        ys.append(float(row["Solubility"]))

    log(f"AqSolDB: invalid={n_invalid}, overlap with ESOL dropped={n_overlap}, "
        f"kept={len(feats)}")

    X_ext, y_ext = np.array(feats), np.array(ys)
    y_pred = pipeline.predict(X_ext)
    return {
        "dataset": "AqSolDB (Sorkun et al. 2019), deduplicated against ESOL by canonical SMILES",
        "n_compounds": int(len(y_ext)),
        "n_esol_overlap_removed": n_overlap,
        "r2": r2_score(y_ext, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_ext, y_pred))),
        "mae": mean_absolute_error(y_ext, y_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-external", action="store_true",
                         help="skip the AqSolDB download + external validation")
    args = parser.parse_args()

    log("Loading ESOL dataset...")
    smiles_list, y_all = load_esol()
    log(f"{len(smiles_list)} compounds loaded")

    log("Computing features (fingerprint + descriptors)...")
    X_all, valid_mask = build_features(smiles_list)
    y_all = y_all[valid_mask]
    smiles_arr = np.array(smiles_list)[valid_mask]
    log(f"Feature matrix: {X_all.shape}")

    X_train, X_test, y_train, y_test, smi_train, smi_test = train_test_split(
        X_all, y_all, smiles_arr, test_size=0.2, random_state=RANDOM_STATE
    )
    log(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = {}
    best_name, best_estimator, best_cv_r2 = None, None, -np.inf
    for name, (pipeline, grid) in candidate_models().items():
        log(f"GridSearchCV for {name}...")
        search = GridSearchCV(pipeline, grid, cv=5, scoring="r2", n_jobs=-1)
        search.fit(X_train, y_train)
        log(f"{name}: CV R2={search.best_score_:.4f} params={search.best_params_}")
        results[name] = {"cv_r2": search.best_score_, "best_params": search.best_params_}
        if search.best_score_ > best_cv_r2:
            best_cv_r2 = search.best_score_
            best_name = name
            best_estimator = search.best_estimator_

    log(f"Best model by CV: {best_name} (CV R2={best_cv_r2:.4f})")
    log("Evaluating on the held-out test set (touched once)...")

    y_pred = best_estimator.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_mae = mean_absolute_error(y_test, y_pred)
    n_selected = int(best_estimator.named_steps["selector"].get_support().sum())

    log(f"TEST R2={test_r2:.4f} RMSE={test_rmse:.4f} MAE={test_mae:.4f} "
        f"(selected {n_selected}/{X_all.shape[1]} features)")
    log(f"Baseline (fingerprint-only RF) test R2={BASELINE_TEST_R2}")

    beats_baseline = test_r2 > BASELINE_TEST_R2
    summary = {
        "candidates": results,
        "best_model": best_name,
        "best_params": results[best_name]["best_params"],
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "n_features_selected": n_selected,
        "n_features_total": int(X_all.shape[1]),
        "baseline_test_r2": BASELINE_TEST_R2,
        "beats_baseline": bool(beats_baseline),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }

    if not args.skip_external:
        summary["external_validation"] = run_external_validation(best_estimator, smiles_list)
        log(f"AqSolDB external: R2={summary['external_validation']['r2']:.4f}")

    with open("train_v2_final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log("Wrote train_v2_final_summary.json")

    if beats_baseline:
        import joblib
        joblib.dump(best_estimator, "drug_solubility_pipeline.joblib")
        log("New pipeline beats baseline - saved drug_solubility_pipeline.joblib")

        log("Building the applicability-domain fingerprint lookup "
            "(training split only)...")
        build_training_fingerprints(list(smi_train))
        log("Saved train_fingerprints.pkl")

        log("Calibrating the prediction interval (split conformal on the test set, "
            "see SPEC.md step-4 addendum)...")
        residuals = y_test - y_pred
        abs_residuals = np.abs(residuals)
        q90 = float(np.quantile(abs_residuals, 0.90))
        q80 = float(np.quantile(abs_residuals, 0.80))
        lower, upper = y_pred - q90, y_pred + q90
        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
        calibration = {
            "method": "split conformal (test-set absolute residual quantile)",
            "calibration_set": "step-3 test set (229 compounds) - same set used for "
                                "the headline R2/RMSE, see SPEC.md step-4 addendum",
            "q80_half_width": q80,
            "q90_half_width": q90,
            "empirical_coverage_at_q90": coverage,
            "n_calibration": int(len(y_test)),
        }
        with open("uncertainty_calibration.json", "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)
        log(f"Saved uncertainty_calibration.json (q90={q90:.4f}, "
            f"empirical coverage={coverage:.3f})")
    else:
        log("New pipeline does NOT beat baseline - keeping the existing model "
            "and documenting this as a negative result")

    return summary


if __name__ == "__main__":
    main()
