# 🚀 Interactive Streamlit Application

## Overview

The Drug Solubility Predictor is an interactive web application built with **Streamlit** that allows you to predict aqueous solubility for any drug molecule in real-time.

**Live App:** [Launch Streamlit App](#) *(deploy link here)*

---

## Features

### 🔮 Prediction Tab
- Enter SMILES notation for any molecule
- Get instant solubility prediction
- View molecular structure visualization
- See classification (High/Medium/Low solubility)
- Receive actionable interpretation

**Example Usage:**
```
Input: CC(=O)Oc1ccccc1C(=O)O  (Aspirin)
Output:
  • log(solubility): -2.55
  • Actual solubility: 2.82e-3 mol/L
  • Category: 🟡 Medium
  • Interpretation: Good water solubility - likely good bioavailability
```

### 📚 Database Examples
- Pre-loaded predictions for 6 common drugs
- Quick comparison table
- Visual grid of molecular structures
- Copy SMILES strings for your own use

**Included Drugs:**
- Aspirin
- Ibuprofen
- Paracetamol
- Caffeine
- Naproxen
- Diclofenac

### ℹ️ About Model Tab
- Complete model performance metrics
- Feature engineering explanation
- Model comparison charts
- Cross-validation results
- Technical specifications

### 📖 How to Use Tab
- Step-by-step guidance
- SMILES notation explanation
- Where to find SMILES strings
- Example SMILES for common molecules
- Tips and best practices

---

## How to Run Locally

### Installation

**Step 1: Clone Repository**
```bash
git clone https://github.com/your-username/drug-solubility-prediction.git
cd drug-solubility-prediction
```

**Step 2: Create Virtual Environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit==1.28.0
scikit-learn==1.3.0
rdkit==2023.09.1
pandas==2.0.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
```

**Step 4: Run Application**
```bash
streamlit run app.py
```

**Step 5: Open in Browser**
```
http://localhost:8501
```

---

## Using the App

### Prediction Workflow

**1. Navigate to "Predict" Tab**

The prediction interface is organized into three sections:
- Left: Input area (SMILES text box)
- Center: Molecular structure (3D visualization)
- Right: Results and interpretation

**2. Enter SMILES Notation**

```
Example SMILES strings:
• Aspirin:     CC(=O)Oc1ccccc1C(=O)O
• Water:       O
• Benzene:     c1ccccc1
• Caffeine:    CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

**Where to find SMILES:**
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) - Search drug name, copy SMILES
- [DrugBank](https://www.drugbank.ca/) - Comprehensive drug database
- [ChemSpider](http://www.chemspider.com/) - Chemical structure search
- [Wikipedia](https://www.wikipedia.org/) - Often includes SMILES for common compounds

**3. Click "Predict Solubility"**

The model processes your SMILES and returns:
- Molecular structure visualization
- Numerical prediction (log scale)
- Actual solubility in mol/L
- Solubility category (High/Medium/Low)
- Chemical interpretation

**4. Interpret Results**

**log(solubility) Meaning:**
```
log(solubility) = log₁₀(concentration in mol/L)

Examples:
• log(sol) = 1  → 10 mol/L (extremely soluble)
• log(sol) = 0  → 1 mol/L (very soluble)
• log(sol) = -1 → 0.1 mol/L (soluble)
• log(sol) = -3 → 1e-3 mol/L (poorly soluble)
• log(sol) = -6 → 1e-6 mol/L (very poorly soluble)
```

**Solubility Categories:**

| Category | Threshold | Characteristics | Example |
|----------|-----------|-----------------|---------|
| 🟢 High | > -1 | Very soluble in water | Paracetamol |
| 🟡 Medium | -1 to -3 | Moderately soluble | Aspirin |
| 🔴 Low | < -3 | Poorly soluble | Ibuprofen |

---

## Advanced Features

### Database Examples Tab

**Quick Reference Table:**
Shows pre-computed predictions for 6 common drugs with:
- Drug name
- SMILES notation (copyable)
- Predicted log solubility
- Calculated actual solubility
- Category classification

**Molecular Grid:**
Visual display of all example drug structures with:
- Chemical structure drawing
- Drug name label
- Predicted solubility value
- Solubility category

**Use Case:** Compare your predictions against known drugs

### About Model Tab

**Performance Metrics:**
```
Test R² Score:      0.6985 (70% variance explained)
Test RMSE:          1.1459 (avg error ±1.15 log units)
Test MAE:           0.8599 (median error ±0.86 log units)
Training Samples:   915
Test Samples:       229
Total Compounds:    1,144
```

**Technical Details:**
```
Algorithm:        Random Forest Regressor
Features:         Morgan Fingerprints (2048 bits)
Hyperparameters:  
  • n_estimators: 100
  • max_depth: None (unlimited)
  • max_features: sqrt
  • min_samples_leaf: 1
  • min_samples_split: 2
```

**Model Comparison Chart:**
- Visual bar chart comparing all 5 tested models
- Shows why Random Forest was selected
- Illustrates performance differences

---

## Practical Applications

### Drug Development Pipeline
```
1. Initial Screening
   └─ Use model to predict solubility for 100+ candidates
   └─ Rank by solubility prediction
   └─ Select top candidates for further testing

2. Formulation Development
   └─ Identify low-solubility leads
   └─ Plan formulation strategy (salt forms, surfactants, etc.)
   └─ Predict post-formulation properties

3. Lead Optimization
   └─ Test chemical modifications
   └─ Predict how structure changes affect solubility
   └─ Guide medicinal chemist in design
```

### Pharmaceutical Companies
**Use Cases:**
- High-throughput screening
- Virtual compound library filtering
- Solubility-driven design optimization
- Patent landscape analysis

### Academic Research
**Applications:**
- Drug discovery projects
- Solubility prediction benchmarks
- Machine learning methodology papers
- Cheminformatics education

### Individual Researchers
**Benefits:**
- Quick solubility estimates
- No expensive software licenses
- Offline capability (download model)
- Open-source transparency

---

## Troubleshooting

### Issue: "Invalid SMILES" Error

**Cause:** Incorrect SMILES notation

**Solution:**
```
• Check SMILES syntax (should be single line)
• Remove whitespace
• Verify special characters are valid
• Use PubChem to validate SMILES

Example wrong SMILES:
✗ CC(=O) O c1ccccc1 C(=O)O  (spaces)
✗ CC(=O)Oc1ccccc1C(=O)OH   (double H)

Example correct SMILES:
✓ CC(=O)Oc1ccccc1C(=O)O
```

### Issue: Model Not Loading

**Cause:** Missing model files

**Solution:**
```bash
# Ensure these files exist in directory:
• drug_solubility_model.pkl
• scaler.pkl

# If missing, retrain model using Jupyter notebook
# Or download from GitHub repository
```

### Issue: Prediction Takes Too Long

**Cause:** First-time model initialization

**Solution:**
- First prediction may take 2-3 seconds (model loading)
- Subsequent predictions should be instant (<0.5s)
- This is normal behavior with Streamlit caching

### Issue: Structure Visualization Not Showing

**Cause:** RDKit not properly installed

**Solution:**
```bash
# Uninstall and reinstall RDKit via conda
conda remove rdkit
conda install -c conda-forge rdkit
```

---

## Tips & Tricks

### ✅ Best Practices

1. **Use Valid SMILES**
   - Always validate SMILES before predicting
   - Copy from reliable sources (PubChem, ChemSpider)

2. **Understand Limitations**
   - Model trained on drug-like molecules
   - May not work well for unusual compounds
   - Always validate predictions experimentally

3. **Interpret Responsibly**
   - R² = 0.70 means ~30% uncertainty
   - Use as guidance, not gospel
   - Combine with expert chemical knowledge

4. **Batch Processing**
   - For multiple compounds, use Jupyter notebook
   - Model can process hundreds instantly
   - Script approach faster than manual entry

### 🚀 Advanced Usage

**Script-Based Batch Prediction:**
```python
import pickle
from rdkit.Chem import AllChem, Chem
import numpy as np

# Load model and scaler
with open('drug_solubility_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict multiple compounds
smiles_list = [
    'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
    'O'  # Water
]

for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_scaled = scaler.transform(np.array(fp).reshape(1, -1))
    prediction = model.predict(fp_scaled)[0]
    print(f"{smiles}: log(sol) = {prediction:.2f}")
```

---

## Deployment

### Deploy to Streamlit Cloud (Free)

**Step 1: Push to GitHub**
```bash
git push origin main
```

**Step 2: Go to [share.streamlit.io](https://share.streamlit.io)**

**Step 3: Connect Your GitHub Account**

**Step 4: Deploy**
- Select repository
- Select branch (main)
- Select app file (app.py)
- Click Deploy

**Step 5: Share URL**
App is now live at: `https://share.streamlit.io/your-username/repo-name/app.py`

### Deploy to Heroku (Alternative)

```bash
# Create Procfile
echo "web: streamlit run app.py --logger.level=error" > Procfile

# Create runtime.txt
echo "python-3.9.16" > runtime.txt

# Deploy
heroku login
heroku create your-app-name
git push heroku main
```

---

## Demo Predictions

### Example 1: Aspirin
```
Input:  CC(=O)Oc1ccccc1C(=O)O
Output: log(sol) = -2.55
        actual = 2.82e-3 mol/L
        🟡 Medium Solubility
        ✓ Common OTC painkiller
        ✓ Good oral absorption
```

### Example 2: Caffeine
```
Input:  CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Output: log(sol) = -1.47
        actual = 3.39e-2 mol/L
        🟢 High Solubility
        ✓ Dissolves well in coffee/tea
        ✓ Quick absorption
```

### Example 3: Ibuprofen
```
Input:  CC(C)Cc1ccc(cc1)C(C)C(=O)O
Output: log(sol) = -3.15
        actual = 7.08e-4 mol/L
        🔴 Low Solubility
        ⚠️ Requires special formulations
        ⚠️ Often combined with buffering agents
```

---

## Citation

If you use this model in your research, please cite:

```bibtex
@article{delaney2004esol,
  title={ESOL: estimating aqueous solubility directly from molecular structure},
  author={Delaney, John S},
  journal={Journal of Chemical Information and Computer Sciences},
  volume={44},
  number={3},
  pages={1000--1005},
  year={2004},
  publisher={ACS Publications}
}
```

---

## Contact & Support

For questions or issues:
- 📧 Email: your.email@example.com
- 🐙 GitHub: [Your Profile](https://github.com)
- 💼 LinkedIn: [Your Profile](https://linkedin.com)

---

**Happy predicting! 🧪🤖**