# Drug Solubility Prediction using Machine Learning

A machine learning project that predicts the aqueous solubility of drug molecules from their chemical structure, using Morgan fingerprints and a Random Forest regressor, wrapped in an interactive Streamlit app.

## 🎯 Project Overview

**Problem:** given a drug's chemical structure (SMILES notation), predict its solubility in water — log(mol/L).

**Approach:**
1. Convert chemical structures to numerical features (Morgan fingerprints)
2. Train and compare multiple ML models
3. Optimize hyperparameters with `GridSearchCV`
4. Ship the best model (Random Forest, test R² ≈ 0.70) behind a Streamlit UI

This property matters in pharmaceutical development because it directly affects a drug's bioavailability and dosing.

## 📊 Dataset

- **Source:** ESOL dataset (Delaney, 2004) — 1,144 drug compounds
- **Features:** 2,048 Morgan fingerprint bits per molecule
- **Target:** log(solubility : mol/L), range -11.60 to 1.58
- **Split:** 80/20 (915 training / 229 test samples), `random_state=42`

## 🔬 Methodology

### Feature engineering

Morgan fingerprints (radius=2, 2048 bits) encode each molecule's connectivity and local atomic environment as a binary vector — the standard cheminformatics descriptor for this kind of task.

```python
from rdkit.Chem import AllChem
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
```

Features were standardized with `StandardScaler` before training.

### Model comparison

| Model | Test R² | Test RMSE | Test MAE |
|---|---|---|---|
| Linear Regression | -0.14 | 2.23 | 1.12 |
| Ridge Regression | 0.34 | 1.70 | 1.25 |
| **Random Forest** | **0.70** | **1.14** | **0.86** |
| Gradient Boosting | 0.62 | 1.28 | 0.98 |
| SVR | 0.63 | 1.27 | 0.96 |

Linear models underfit — solubility doesn't depend linearly on individual fingerprint bits — so the non-linear ensemble methods (Random Forest, Gradient Boosting, SVR) clearly outperform them. Random Forest was selected as the best trade-off of accuracy, training speed, and interpretability (feature importances).

### Hyperparameter tuning

`GridSearchCV`, 5-fold CV, 270 parameter combinations (1,350 fits total). Best parameters:

```
n_estimators: 100
max_depth: None
max_features: sqrt
min_samples_split: 2
min_samples_leaf: 1
```

Best CV R²: 0.657 — close to the tuned model's test performance, and identical to the untuned defaults, meaning the original configuration was already close to optimal.

## 📈 Results

**Final model (Random Forest), test set:**

- R² = 0.70 (explains ~70% of solubility variance)
- RMSE = 1.14 log(mol/L), MAE = 0.86 log(mol/L)
- Train R² = 0.94 → train/test gap of ~0.24, i.e. mild but controlled overfitting

**5-fold cross-validation:** mean R² = 0.658 ± 0.017 — stable across folds, no single fold underperforms.

**Performance by solubility range** (test set):

| Range | log(sol) | R² | MAE |
|---|---|---|---|
| High solubility | > -1 | 0.78 | 0.52 |
| Medium solubility | -1 to -3 | 0.72 | 0.89 |
| Low solubility | < -3 | 0.61 | 1.24 |

The model is most accurate for well-dissolved compounds and least accurate for poorly soluble ones — the harder, more sparsely represented range in the dataset.

**Feature importance:** only 128 of the 2,048 fingerprint bits (6.2%) account for 80% of the model's predictive power. The single most important feature, `fp_1380`, accounts for 11.6% of total importance on its own and appears in 234 molecules — likely encoding a polar substructure associated with hydrogen bonding.

| Rank | Feature | Importance |
|---|---|---|
| 1 | fp_1380 | 11.6% |
| 2 | fp_1143 | 6.0% |
| 3 | fp_1683 | 4.2% |
| 4 | fp_561 | 3.2% |
| 5 | fp_1087 | 2.8% |

**Known weak spots:** the worst individual errors (|residual| > 2.5) are organosilanes and heavily halogenated compounds — categories that are rare in the training data (~3% of the test set).

## 🧪 Example predictions

| Drug | SMILES | Predicted log(sol) | Category |
|---|---|---|---|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | -2.55 | 🟡 Medium |
| Ibuprofen | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` | -3.15 | 🔴 Low |
| Paracetamol | `CC(=O)Nc1ccc(O)cc1` | -1.21 | 🟢 High |
| Caffeine | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` | -1.47 | 🟢 High |

Categories: 🟢 High (log(sol) > -1), 🟡 Medium (-1 to -3), 🔴 Low (< -3).

## 🚀 Streamlit app

An interactive app (`app.py`) built on top of the trained model:

- **Predict tab** — paste a SMILES string, get an instant prediction, molecule drawing, and solubility category with interpretation
- **Database examples** — precomputed predictions for common drugs (aspirin, ibuprofen, paracetamol, caffeine, naproxen, diclofenac) as a quick reference
- **About the model** — performance metrics, model comparison chart, technical details
- **How to use** — SMILES notation primer and where to find SMILES for a given drug (PubChem, DrugBank, ChemSpider)

Deployed on Streamlit Community Cloud; `packages.txt` installs the system libraries (`libxrender1`, `libxext6`, `libsm6`, ...) that RDKit's `Chem.Draw` module needs on that platform.

## 🛠️ Installation & usage

```bash
pip install -r requirements.txt
streamlit run app.py
```

Batch / scripted predictions:

```python
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

model = joblib.load('drug_solubility_model.joblib')
scaler = joblib.load('scaler.joblib')

smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
mol = Chem.MolFromSmiles(smiles)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
fp_scaled = scaler.transform(np.array(fp).reshape(1, -1))
prediction = model.predict(fp_scaled)[0]

print(f"Predicted log(solubility): {prediction:.2f}")
print(f"Actual solubility: {10 ** prediction:.2e} mol/L")
```

**SMILES troubleshooting:** the notation must be a single line with no spaces and valid atom/bond syntax (e.g. `CC(=O)Oc1ccccc1C(=O)O`, not `CC(=O) O c1ccccc1 C(=O)O`). Verify against [PubChem](https://pubchem.ncbi.nlm.nih.gov/) if a prediction fails.

## 📁 Project structure

```
05_drug_solub/
├── app.py                       # Streamlit application
├── drug_solubility.ipynb        # Full analysis notebook (EDA, training, evaluation)
├── drug_solubility_model.joblib # Trained Random Forest model
├── scaler.joblib                # Fitted StandardScaler
├── project_summary.json         # Machine-readable results summary
├── data.txt                     # ESOL dataset
├── requirements.txt
├── packages.txt                 # System deps for RDKit on Streamlit Cloud
└── README.md
```

## 🎓 Skills demonstrated

Cheminformatics (SMILES, Morgan fingerprints, molecular descriptors) · machine learning (model selection, hyperparameter tuning, cross-validation, feature importance) · Python (pandas, numpy, scikit-learn, RDKit, Streamlit) · deployment (Streamlit Community Cloud).

## 📊 Possible next steps

- Additional descriptors (MACCS keys, RDKit physicochemical descriptors) alongside the fingerprint
- Gradient-boosted alternatives (XGBoost/LightGBM) and feature selection on the 2,048-bit fingerprint
- External validation on a second dataset (e.g. AqSolDB)
- Prediction uncertainty intervals and an applicability-domain check
- Automated tests + CI

## 📝 References

- Delaney, J. S. (2004). ESOL: Estimating aqueous solubility directly from molecular structure. *Journal of Chemical Information and Computer Sciences*, 44(3), 1000–1005.
- Morgan, H. L. (1965). The generation of a unique machine description for chemical structures.
- [RDKit documentation](https://www.rdkit.org/)

## 📄 License

MIT License.
