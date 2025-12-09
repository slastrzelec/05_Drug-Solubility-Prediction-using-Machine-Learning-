import streamlit as st
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Crippen, Lipinski
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Drug Solubility Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler
@st.cache_resource
def load_model():
    import os
    try:
        if os.path.exists('drug_solubility_model.joblib'):
            model = joblib.load('drug_solubility_model.joblib')
            scaler = joblib.load('scaler.joblib')
            return model, scaler
        else:
            st.error("⚠️ Model files not found!")
            return None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None, None

# Prediction function
def predict_solubility(smiles, model, scaler):
    """Predict solubility for a given SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES notation"
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = np.array(fp).reshape(1, -1)
        
        fp_scaled = scaler.transform(fp_array)
        log_solubility = model.predict(fp_scaled)[0]
        actual_solubility = 10 ** log_solubility
        
        return {
            'log_solubility': log_solubility,
            'actual_solubility': actual_solubility,
            'mol': mol
        }, None
    except Exception as e:
        return None, str(e)

# Categorize solubility
def categorize_solubility(log_sol):
    if log_sol > -1:
        return "🟢 High", "High solubility", "#00ff41"
    elif log_sol > -3:
        return "🟡 Medium", "Moderate solubility", "#ffaa00"
    else:
        return "🔴 Low", "Low solubility", "#ff0000"

# Wizualizacja struktury molekularnej
def display_molecule_structure(smiles):
    """Display molecular structure based on SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("❌ Invalid SMILES notation!")
            return False
        
        # Draw molecule - smaller size
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img, caption=f"Structure: {smiles}")
        
        # Display molecule information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Molecular Weight", f"{Descriptors.MolWt(mol):.2f} g/mol")
        with col2:
            st.metric("H-bond Donors/Acceptors", Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol))
        with col3:
            st.metric("LogP (Lipophilicity)", f"{Crippen.MolLogP(mol):.2f}")
        
        return True
    except Exception as e:
        st.error(f"❌ Error drawing structure: {str(e)}")
        return False

st.title("💊 Drug Solubility Predictor")
st.markdown("Predict aqueous solubility of drug molecules using Machine Learning")
st.markdown("---")

# Load model
model, scaler = load_model()

if model is None or scaler is None:
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📚 Database Examples", "ℹ️ About Model", "📖 How to Use"])

# TAB 1: Prediction
with tab1:
    st.header("Make a Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_input(
            "Enter SMILES notation:",
            placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O (Aspirin)",
            value="CC(=O)Oc1ccccc1C(=O)O"
        )
    
    with col2:
        predict_button = st.button("🔍 Predict Solubility", use_container_width=True)
    
    if predict_button and smiles_input:
        result, error = predict_solubility(smiles_input, model, scaler)
        
        if error:
            st.error(f"❌ Error: {error}")
        else:
            st.markdown("---")
            
            # Two columns: Structure on left, Results on right
            col_struct, col_results = st.columns([1, 1])
            
            with col_struct:
                st.subheader("📐 Molecular Structure")
                display_molecule_structure(smiles_input)
            
            with col_results:
                st.subheader("🔬 Prediction Results")
                
                log_sol = result['log_solubility']
                actual_sol = result['actual_solubility']
                category, desc, color = categorize_solubility(log_sol)
                
                st.metric("log(solubility)", f"{log_sol:.3f}")
                st.metric("Actual solubility", f"{actual_sol:.2e} mol/L")
                
                st.markdown(f"<h3 style='color:{color}'>{category}</h3>", unsafe_allow_html=True)
                st.info(f"**Classification**: {desc}")
                
                st.markdown("""
                **Log Solubility Ranges:**
                - 🟢 **High** (> -1): Highly soluble
                - 🟡 **Medium** (-1 to -3): Moderately soluble
                - 🔴 **Low** (< -3): Poorly soluble
                """)
                
                if log_sol > -1:
                    st.success("✓ Good water solubility")
                elif log_sol > -3:
                    st.warning("⚠️ Moderate solubility")
                else:
                    st.error("✗ Poor solubility")
    else:
        # Show structure immediately after entering SMILES (without prediction)
        if smiles_input:
            st.subheader("📐 Molecular Structure")
            display_molecule_structure(smiles_input)

# TAB 2: Database Examples
with tab2:
    st.header("Example Drugs from Database")
    
    examples = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Naproxen": "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
        "Diclofenac": "O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl"
    }
    
    predictions_data = []
    for drug_name, smiles in examples.items():
        result, _ = predict_solubility(smiles, model, scaler)
        if result:
            category, _, _ = categorize_solubility(result['log_solubility'])
            predictions_data.append({
                "Drug": drug_name,
                "log(solubility)": f"{result['log_solubility']:.3f}",
                "Actual Solubility (mol/L)": f"{result['actual_solubility']:.2e}",
                "Category": category
            })
    
    df_examples = pd.DataFrame(predictions_data)
    st.dataframe(df_examples, use_container_width=True, hide_index=True)
    
    st.subheader("Molecular Structures")
    try:
        mols = [Chem.MolFromSmiles(smiles) for smiles in examples.values()]
        img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(250, 250), 
                                   legends=list(examples.keys()), returnPNG=False)
        st.image(img, use_container_width=True)
    except:
        st.info("Structure visualization unavailable")

# TAB 3: About Model
with tab3:
    st.header("About the Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Performance")
        metrics = {
            "Test R² Score": "0.6985",
            "Test RMSE": "1.1459",
            "Test MAE": "0.8599",
            "Training Samples": "915",
            "Test Samples": "229",
            "Total Compounds": "1,144"
        }
        for metric, value in metrics.items():
            st.metric(metric, value)
    
    with col2:
        st.subheader("🔬 Technical Details")
        st.markdown("""
        **Model Architecture:**
        - Algorithm: Random Forest Regressor
        - n_estimators: 100
        - max_depth: None
        
        **Feature Engineering:**
        - Morgan Fingerprints (radius=2)
        - 2048 binary features
        """)

# TAB 4: How to Use
with tab4:
    st.header("How to Use This App")
    
    st.markdown("""
    ### Step 1: Get SMILES Notation
    SMILES is a notation for chemical structures.
    
    **Where to find SMILES:**
    - [PubChem](https://pubchem.ncbi.nlm.nih.gov/)
    - [DrugBank](https://www.drugbank.ca/)
    - [ChemSpider](http://www.chemspider.com/)
    
    ### Step 2: Enter SMILES
    Copy-paste the SMILES string and the structure will display automatically!
    
    ### Step 3: Click "Predict Solubility"
    Get instant predictions about solubility
    
    ### Example SMILES
    - Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
    - Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧪 Drug Solubility Prediction | Built with Streamlit, RDKit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)