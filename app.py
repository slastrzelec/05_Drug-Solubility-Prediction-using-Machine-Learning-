import streamlit as st
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Try to import Draw, fallback if it fails
try:
    from rdkit.Chem import Draw
    HAS_DRAW = True
except ImportError:
    HAS_DRAW = False
    st.warning("⚠️ Molecular structure visualization not available on this platform")

import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Drug Solubility Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        with open('drug_solubility_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'drug_solubility_model.pkl' and 'scaler.pkl' are in the directory.")
        return None, None

# Prediction function
def predict_solubility(smiles, model, scaler):
    """Predict solubility for a given SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES notation"
        
        # Generate fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = np.array(fp).reshape(1, -1)
        
        # Scale and predict
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

# Header
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
            # Display results
            col1, col2, col3 = st.columns(3)
            
            # Molecule visualization
            with col1:
                st.subheader("Structure")
                if HAS_DRAW:
                    try:
                        img = Draw.MolToImage(result['mol'], size=(300, 300))
                        st.image(img, use_container_width=True)
                    except:
                        st.info("📝 SMILES: " + smiles_input)
                else:
                    st.info("📝 SMILES:\n" + smiles_input)
            
            # Prediction results
            with col2:
                st.subheader("Prediction Results")
                log_sol = result['log_solubility']
                actual_sol = result['actual_solubility']
                category, desc, color = categorize_solubility(log_sol)
                
                st.metric("log(solubility)", f"{log_sol:.3f}")
                st.metric("Actual solubility", f"{actual_sol:.2e} mol/L")
                st.markdown(f"<h3 style='color:{color}'>{category}</h3>", unsafe_allow_html=True)
                st.info(f"**Classification**: {desc}")
            
            # Interpretation
            with col3:
                st.subheader("Interpretation")
                st.markdown("""
                **Log Solubility Ranges:**
                - 🟢 **High** (> -1): Highly soluble in water
                - 🟡 **Medium** (-1 to -3): Moderately soluble
                - 🔴 **Low** (< -3): Poorly soluble in water
                
                **Formula**: 
                Actual Solubility = 10^(log_solubility)
                """)
                
                if log_sol > -1:
                    st.success("✓ Good water solubility - likely good bioavailability")
                elif log_sol > -3:
                    st.warning("⚠️ Moderate solubility - may need formulation help")
                else:
                    st.error("✗ Poor solubility - needs special formulation or prodrug approach")

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
    
    # Create prediction for each example
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
    
    # Visualize molecules
    if HAS_DRAW:
        st.subheader("Molecular Structures")
        try:
            mols = [Chem.MolFromSmiles(smiles) for smiles in examples.values()]
            img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(250, 250), 
                                       legends=list(examples.keys()), returnPNG=False)
            st.image(img, use_container_width=True)
        except:
            st.info("📝 Structure visualization unavailable, but predictions are available above!")
    else:
        st.info("📝 Molecular structure visualization not available, but predictions are ready above!")

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
        - max_features: sqrt
        
        **Feature Engineering:**
        - Morgan Fingerprints (radius=2)
        - 2048 binary features
        - Only 128 features (6.2%) needed for 80% explanation
        
        **Dataset:**
        - ESOL Dataset (1,144 compounds)
        - Solubility range: -11.60 to 1.58 log(mol/L)
        """)
    
    st.subheader("📈 Model Comparison")
    comparison_data = {
        "Model": ["Linear Regression", "Ridge", "Random Forest", "Gradient Boosting", "SVR"],
        "Test R²": [-0.1372, 0.3362, 0.6985, 0.6235, 0.6283],
        "Test RMSE": [2.2253, 1.7001, 1.1459, 1.2805, 1.2723]
    }
    df_comparison = pd.DataFrame(comparison_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_comparison))
    width = 0.35
    ax.bar(x - width/2, df_comparison['Test R²'], width, label='Test R²', color='skyblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Test R² Score')
    ax.set_title('Model Comparison: Test R² Score')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# TAB 4: How to Use
with tab4:
    st.header("How to Use This App")
    
    st.markdown("""
    ### Step 1: Get SMILES Notation
    SMILES (Simplified Molecular Input Line Entry System) is a notation for chemical structures.
    
    **Where to find SMILES:**
    - [PubChem](https://pubchem.ncbi.nlm.nih.gov/) - Search for drug name
    - [DrugBank](https://www.drugbank.ca/) - Drug database
    - [ChemSpider](http://www.chemspider.com/) - Chemical structure search
    
    ### Step 2: Enter SMILES in Prediction Tab
    Copy-paste the SMILES string into the text field and click "Predict Solubility"
    
    ### Step 3: Interpret Results
    - **log(solubility)**: Logarithmic solubility value
    - **Actual solubility**: Converted to mol/L units
    - **Category**: Classification (High/Medium/Low)
    
    ### Step 4: Use for Decision Making
    - High solubility: Good for oral bioavailability
    - Medium solubility: May need formulation optimization
    - Low solubility: Consider alternative approaches
    
    ### Example SMILES Strings
    - Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
    - Water: `O`
    - Benzene: `c1ccccc1`
    - Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
    
    ### ⚠️ Important Notes
    - Model trained on diverse drug compounds
    - Predictions are estimates (R² = 0.70)
    - Always validate predictions experimentally
    - SMILES must be chemically valid
    """)
    
    st.info("💡 **Tip**: Use the 'Database Examples' tab to see predictions for known drugs!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧪 Drug Solubility Prediction | Built with Streamlit, RDKit & Machine Learning</p>
    <p><small>Model: Random Forest | Features: Morgan Fingerprints (2048 bits) | Dataset: 1,144 compounds</small></p>
</div>
""", unsafe_allow_html=True)