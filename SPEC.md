# SPEC: Model performance improvement (step 3 of the project plan)

## Objective

Improve the solubility model beyond the current baseline (Random Forest on
2048-bit Morgan fingerprints only: test R² = 0.6985, RMSE = 1.1459) without
introducing data leakage, and get an honest read of how well it generalizes
outside the ESOL dataset it was trained on.

## Baseline (unchanged, kept as the reference to beat)

- Model: `RandomForestRegressor`, existing hyperparameters
- Features: Morgan fingerprints (radius=2, 2048 bits) only
- Split: existing 80/20 train/test, `random_state=42` (unchanged — the point
  is to compare fairly against the number already documented in the README)

## Changes

1. **Add physicochemical descriptors** alongside the fingerprint: molecular
   weight, LogP (Crippen), TPSA, H-bond donor/acceptor counts, rotatable
   bond count, aromatic ring count. ~10 extra features, computed with RDKit,
   concatenated to the 2048 fingerprint bits, then scaled together.
2. **Feature selection** down from 2048+ to a few hundred, to reduce the
   features-vs-samples ratio (2048 features on 915 training rows invites
   overfitting). Selection is fit **inside the training fold only**
   (`SelectFromModel` on a Random Forest, wrapped in the same
   `sklearn.pipeline.Pipeline` as the scaler and the final model) — it never
   sees the test set, so the selected feature set cannot leak test
   information into training.
3. **Try XGBoost** (already installed, `xgboost==3.2.0`) alongside the
   existing Random Forest / Gradient Boosting / SVR comparison, tuned with
   the same `GridSearchCV(cv=5)` approach already used in the notebook.
4. **External validation on AqSolDB** (Sorkun et al.), a separate, larger
   aqueous solubility dataset. Used *only* as a final generalization check
   on the already-chosen model — never for training, tuning, or model
   selection. Before use: any compound (by canonical SMILES) that also
   appears in the ESOL training or test split is dropped from the AqSolDB
   evaluation set, since ESOL is known to overlap with AqSolDB — without
   this dedup step the "external" R² would be inflated by compounds the
   model has already seen.

## Data-leakage guardrails (explicit, since this is the exact failure mode from a past bad experience)

- All preprocessing that learns from data — `StandardScaler`,
  `SelectFromModel` — lives inside a single `sklearn.Pipeline` fit only on
  the training fold, both during `GridSearchCV` and for the final fit. No
  manual `fit_transform` on the full dataset before splitting.
- The existing 229-row ESOL test set is touched exactly once, for the
  final chosen model. It is not used to pick between fingerprint-only vs.
  fingerprint+descriptors, RF vs. XGBoost, or any hyperparameter — all of
  that is decided by 5-fold CV on the training set only.
- AqSolDB is deduplicated against ESOL (train **and** test) before scoring,
  as above.
- The trained artifacts committed to the repo (`.joblib` files) are
  retrained from scratch inside this pipeline — the current model/scaler
  files are not reused or fine-tuned, to avoid carrying over any
  fingerprint-only assumptions into the new feature space.

## Deliverables

- Updated `drug_solubility.ipynb` section (or a new `notebook_v2` section)
  documenting the new feature set, model comparison, and the deduplicated
  AqSolDB external-validation score
- Retrained `drug_solubility_model.joblib` + `scaler.joblib` (+ the fitted
  feature selector, saved alongside them) if the new pipeline beats the
  baseline; if it doesn't, the baseline stays and the negative result is
  documented honestly in the README rather than hidden
- `app.py` updated to build the same combined feature vector at inference
  time
- `tests/` gains coverage for descriptor computation and for the
  leakage guardrail itself (asserts the fitted selector's feature mask
  doesn't change if test-set rows are shuffled/excluded before transform)
- README "Results" section updated with the new numbers, including the
  external AqSolDB score reported alongside the ESOL test score (not
  blended into one number)

## Out of scope for this step

Uncertainty intervals / applicability-domain check (step 4) and
FastAPI+Docker productionization (step 5) — separate, later steps.

## Risk if the new approach doesn't beat the baseline

Documented as a negative result, not discarded silently — an honest
"we tried X, it didn't help, here's why" is itself a legitimate portfolio
signal for a recruiter, and beats the current line's implicit claim that
2048 raw bits with no descriptors is the last word.

---

## Addendum: Step 4 - prediction uncertainty + applicability domain

Two independent, inference-time additions on top of the step 3 pipeline (no
retraining, no change to the fitted scaler/selector/model):

1. **Prediction interval** via split conformal prediction: the already
   held-out test set's residuals (computed once in step 3, reused here only
   to calibrate an interval width - not for any further model selection)
   give an empirical 90% absolute-residual quantile. Every new prediction
   gets `[pred - q90, pred + q90]`. This is a real but honest limitation -
   with a proper 3-way split the calibration set would be separate from the
   test set used for the headline R²/RMSE; here they're the same 229 rows
   because the dataset is small. Documented as such rather than overstated
   as a rigorous conformal guarantee.
2. **Applicability domain check**: Tanimoto similarity between the query
   molecule's Morgan fingerprint and its nearest neighbor in the 915-row
   training set. Below a similarity threshold (0.4, a common medicinal-
   chemistry convention for "structurally similar"), the prediction is
   flagged as extrapolation - consistent with the AqSolDB external-validation
   finding that accuracy drops outside the training distribution.

No new data enters training; only `data.txt`'s existing train split is
reused to build a fingerprint lookup table. No leakage risk beyond what
step 3 already accepted.
