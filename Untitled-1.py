# Uruchom to LOKALNIE i wrzuć nowy plik na GitHub

import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import AllChem

# Wczytaj dane
df = pd.read_csv('data.txt')

# Feature engineering
fingerprints = []
for smiles in df['SMILES']:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprints.append(np.array(fp))

X = np.array(fingerprints)
y = df['measured log(solubility:mol/L)'].values

# Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Save with joblib (bardziej stabilny)
joblib.dump(model, 'drug_solubility_model.joblib', compress=3)
joblib.dump(scaler, 'scaler.joblib', compress=3)

print("✅ Model saved!")
print(f"Test R²: {model.score(X_test_scaled, y_test):.4f}")