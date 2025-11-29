Drug Solubility Prediction using Machine Learning

A machine learning project that predicts aqueous solubility of drug molecules using Morgan fingerprints and Random Forest regression.
🎯 Project Overview

This project combines cheminformatics with machine learning to predict how well drugs dissolve in water (aqueous solubility). This property is crucial in pharmaceutical development as it affects drug bioavailability and efficacy.

Problem: Given a drug's chemical structure (SMILES notation), predict its solubility in water (log(mol/L))

Approach:

    Convert chemical structures to numerical features (Morgan fingerprints)
    Train and compare multiple ML models
    Optimize hyperparameters using GridSearchCV
    Achieve predictive model with R² = 0.70 on test set

📊 Dataset

    Source: ESOL Dataset (1,144 drug compounds)
    Features: 2,048 Morgan fingerprint bits (molecular descriptors)
    Target: log(solubility:mol/L) [range: -11.60 to 1.58]
    Train/Test Split: 80/20 (915 training, 229 test samples)

🔬 Methodology
1. Feature Engineering

    Morgan Fingerprints (radius=2, 2048 bits)
    Each bit represents a specific molecular substructure
    Captures chemical connectivity and topology

2. Models Tested

Model	Test R²	Test RMSE	MAE
Linear Regression	-0.1372	2.2253	1.1214
Ridge Regression	0.3362	1.7001	1.2505
Random Forest	0.6985	1.1459	0.8599
Gradient Boosting	0.6235	1.2805	0.9774
SVR	0.6283	1.2723	0.9588

Best Model: Random Forest ✓
3. Hyperparameter Optimization

    Method: GridSearchCV with 5-fold cross-validation
    Combinations tested: 270
    Best parameters:
        n_estimators: 100
        max_depth: None
        max_features: sqrt
        min_samples_split: 2
        min_samples_leaf: 1

4. Performance Metrics (Test Set)

    R² Score: 0.6985 (explains ~70% of solubility variance)
    RMSE: 1.1459 log(mol/L)
    MAE: 0.8599 log(mol/L)
    Overfitting: Minimal (Train R² - Test R² = 0.24)

📈 Key Insights
Feature Importance

Only 128 out of 2,048 features (6.2%) are needed to explain 80% of the model's predictions. This suggests:

    Most important fragments are highly specific molecular substructures
    Feature #1380 is the most predictive (11.6% importance)
    Many fingerprint bits are irrelevant for solubility

Top 10 Important Features

fp_1380: 11.60%  ← Most critical fragment
fp_1143: 6.04%
fp_1683: 4.24%
fp_561:  3.24%
...

Model Insights

    Random Forest significantly outperforms linear models
    Complex non-linear relationships between structure and solubility
    Solubility is not linearly dependent on individual features
    Slight overfitting is acceptable and controlled

🧪 Example Predictions

Drug	SMILES	Predicted log(sol)	Category
Aspirin	CC(=O)Oc1ccccc1C(=O)O	-2.55	Medium
Ibuprofen	CC(C)Cc1ccc(cc1)C(C)C(=O)O	-3.15	Low
Paracetamol	CC(=O)Nc1ccc(O)cc1	-1.21	High
Caffeine	CN1C=NC2=C1C(=O)N(C(=O)N2C)C	-1.47	High

Solubility Categories:

    🟢 High: log(sol) > -1 (very soluble)
    🟡 Medium: -1 > log(sol) > -3 (moderately soluble)
    🔴 Low: log(sol) < -3 (poorly soluble)

📁 Project Structure

drug-solubility-prediction/
├── README.md
├── notebook.ipynb                    # Complete analysis notebook
├── drug_solubility_model.pkl         # Trained Random Forest model
├── scaler.pkl                        # Fitted StandardScaler
├── project_summary.json              # Results summary
└── data/
    └── data.txt                      # ESOL dataset

🛠️ Installation & Usage
Requirements
bash

pip install pandas numpy scikit-learn rdkit matplotlib seaborn

Quick Start
python

import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Load saved model and scaler
with open('drug_solubility_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict solubility for a new drug
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
mol = Chem.MolFromSmiles(smiles)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
fp_scaled = scaler.transform(np.array(fp).reshape(1, -1))
prediction = model.predict(fp_scaled)[0]

print(f"Predicted log(solubility): {prediction:.2f}")
print(f"Actual solubility: {10**prediction:.2e} mol/L")

🔍 Technical Details
Why Morgan Fingerprints?

    Standard in cheminformatics
    Captures molecular connectivity patterns
    Interpretable bit positions
    Computationally efficient

Why Random Forest?

    Non-linear relationships in chemistry
    Handles high-dimensional data well
    Provides feature importance scores
    Robust to outliers
    Good balance between bias and variance

📚 Learning Resources Used

    RDKit: Molecular descriptor calculation
    scikit-learn: Machine learning models and evaluation
    Cheminformatics: Understanding SMILES, fingerprints, molecular properties

🎓 Skills Demonstrated

✓ Chemistry: Understanding drug properties, molecular structures, SMILES notation ✓ Machine Learning: Model selection, hyperparameter tuning, cross-validation ✓ Data Science: Feature engineering, exploratory analysis, evaluation metrics ✓ Python: pandas, numpy, scikit-learn, RDKit ✓ Project Management: GitHub, documentation, reproducibility
📊 Future Improvements

    Add more molecular descriptors (MACCS, RDKit descriptors)
    Test deep learning approaches (neural networks, graph neural networks)
    Implement SHAP values for model interpretability
    Deploy as web app (Flask/Streamlit)
    Add uncertainty quantification
    Test on external validation sets

📝 References

    Delaney, J. S. (2004). ESOL: Estimating aqueous solubility directly from molecular structure. Journal of Chemical Information and Computer Sciences, 44(3), 1000-1005.
    RDKit Documentation: https://www.rdkit.org/
    Morgan, H. L. (1965). The generation of a unique machine description for chemical structures.

👨‍💻 Author

Created as a portfolio project combining chemistry with machine learning.
📄 License

This project is open source and available under the MIT License.
