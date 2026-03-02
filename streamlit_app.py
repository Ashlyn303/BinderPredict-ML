import streamlit as st
import torch
import torch.nn as nn
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Root-level imports ensure joblib compatibility for legacy unpickling
from data_loader import DeviationFeatureEncoder, build_reference_forms
from esm_feature_extractor import ESMFeatureExtractor

import py3Dmol
from stmol import showmol

# =============================================================================
# APP CONFIGURATION & MODELS
# =============================================================================

st.set_page_config(
    page_title="mBG17 Peptide Predictor",
    page_icon="🧬",
    layout="wide"
)

# Constants & Paths
REFERENCE_SEQ = "-SLQEDLEALE"
WT_SEQUENCE = "DDFSKQLQQS"  # SARS-CoV-2 N-protein WT Epitope
SCFV_SEQUENCE = "MAEVKLEESGGGLVQPGGSMKFSCVASGFTFSDYWMNWVRQSPDKGLEWVAEIRLKSNNYATHYAASVKGRFTISRDDSKSSVYLQMNNLRAEDSGIYYCTRSAMDYWGQGTSVTVSSGGGGSGGGGSGGGGSDIVMSQSPSSLAVSVGEKITMSCKSSQSLLYTSDQKNYLAWFQQKPGQSPKLLIFWASTRDSGVPDRFTGSGSGTDFTLTISSVKAEDLAVYYCQQFYNYPRTFGGGTKLEIDD"
SEQ_LENGTH = 11

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_DIR = os.path.join(SCRIPT_DIR, 'pytorch_results')
ESM_DIR = os.path.join(SCRIPT_DIR, 'esm_results')
HYBRID_DIR = os.path.join(SCRIPT_DIR, 'hybrid_results')
WT_PDB_PATH = os.path.join(SCRIPT_DIR, 'AF2_WildTypeEpitope', 'AF2_predicted_WT_DDFSKQLQQS_w_mBG17.pdb')
TOP_10K_PATH = os.path.join(PYTORCH_DIR, 'top_10k_predictions_pytorch.csv')
SHAP_PATH = os.path.join(SCRIPT_DIR, 'assets', 'docs', 'shap_heatmap_2d.png')

# Model specific constants
SEQ_LENGTH = 11  # Standard length including gaps

# Model Class (Must match training)
class PeptideNet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.128):
        super(PeptideNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

class ESMToPLDDT(nn.Module):
    def __init__(self, embedding_dim=320, dropout_rate=0.2):
        super(ESMToPLDDT, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

class HybridPeptideNet(nn.Module):
    def __init__(self, manual_dim, esm_dim, dropout_rate=0.2):
        super(HybridPeptideNet, self).__init__()
        input_dim = manual_dim + esm_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)

@st.cache_resource
def get_esm_extractor():
    return ESMFeatureExtractor()

@st.cache_resource
def load_resources(model_type='manual'):
    """Loads model and preprocessing joblibs."""
    try:
        # Encoder and Selector are shared by Manual and Hybrid
        encoder = joblib.load(os.path.join(PYTORCH_DIR, 'encoder.joblib'))
        selector = joblib.load(os.path.join(PYTORCH_DIR, 'selector.joblib'))
        
        if model_type == 'manual':
            scaler = joblib.load(os.path.join(PYTORCH_DIR, 'scaler.joblib'))
            input_dim = len(selector.get_support(indices=True))
            model = PeptideNet(input_dim)
            model.load_state_dict(torch.load(os.path.join(PYTORCH_DIR, 'peptide_predictor_pytorch.pt'), map_location='cpu'))
        
        elif model_type == 'esm':
            # ESM only model
            model = ESMToPLDDT(embedding_dim=320)
            model.load_state_dict(torch.load(os.path.join(ESM_DIR, 'esm_plddt_model.pt'), map_location='cpu'))
            scaler = None # ESM embeddings are usually self-normalized or raw
            
        elif model_type == 'hybrid':
            # Hybrid model (Manual + ESM)
            scaler = joblib.load(os.path.join(HYBRID_DIR, 'hybrid_scaler.joblib'))
            manual_dim = len(selector.get_support(indices=True))
            model = HybridPeptideNet(manual_dim, 320)
            model.load_state_dict(torch.load(os.path.join(HYBRID_DIR, 'hybrid_plddt_model.pt'), map_location='cpu'))
        
        model.eval()
        return model, encoder, selector, scaler
    except Exception as e:
        st.error(f"Error loading {model_type} model resources: {e}")
        return None, None, None, None

def calculate_properties(sequence):
    """Calculate physicochemical properties of the peptide sequence."""
    # Remove gaps for property calculation
    clean_seq = sequence.replace('-', '')
    if not clean_seq:
        return None
    
    analysis = ProteinAnalysis(clean_seq)
    
    properties = {
        'MW (Da)': analysis.molecular_weight(),
        'GRAVY (Hydrophobicity)': analysis.gravy(),
        'Isoelectric Point (pI)': analysis.isoelectric_point(),
        'Net Charge (pH 7.2)': analysis.charge_at_pH(7.2),
        'Aromaticity': analysis.aromaticity()
    }
    return properties

def show_pdb(pdb_path, height=500, width=800, highlight_res=None, spin=False):
    """Renders a PDB file in 3D using py3Dmol with advanced interaction."""
    if not os.path.exists(pdb_path):
        st.warning("Structure file not found.")
        return

    with open(pdb_path, 'r') as f:
        pdb_data = f.read()

    view = py3Dmol.view(height=height, width=width)
    view.addModel(pdb_data, 'pdb')
    
    # 1. Base Styles
    # scFv (Chain A) - Cartoon backbone
    view.setStyle({'chain': 'A'}, {'cartoon': {'color': '#4c72b0', 'opacity': 0.7}}) 
    # Epitope (Chain B) - Cartoon backbone + Transparent sticks
    view.setStyle({'chain': 'B'}, {'cartoon': {'color': '#c44e52', 'opacity': 0.9}})
    view.addStyle({'chain': 'B'}, {'stick': {'radius': 0.1, 'colorscheme': 'redCarbon', 'opacity': 0.5}})
    
    # 2. Advanced Click Interaction (JavaScript-based logic in viewer)
    # Enable clicking on any atom to show a label AND sticks for that residue
    view.setClickable({}, True, """
        function(atom, viewer, event, container) {
            if(!atom.clicked) {
                // Add Label
                atom.label = viewer.addLabel(atom.resn + ":" + atom.resi + " (" + atom.chain + ")", 
                    {position: atom, backgroundColor: 'white', fontColor: 'black', backgroundOpacity: 0.8});
                
                // Add Sidechain Sticks for this residue
                atom.sticks = viewer.addStyle({chain: atom.chain, resi: atom.resi}, {stick: {radius: 0.2, colorscheme: 'CPK'}});
                
                atom.clicked = true;
            } else {
                viewer.removeLabel(atom.label);
                // To remove style we usually need to re-render or use specific API
                // For simplicity, let's keep it togglable via labels
                delete atom.label;
                delete atom.clicked;
                viewer.render();
            }
            viewer.render();
        }
    """)

    # 3. Dynamic Selection Logic (Highlight + Proximity Sidechains)
    if highlight_res is not None:
        target_res = str(highlight_res)
        
        # A. Highlight Selected Epitope Residue (Strong sticks)
        view.addStyle({'chain': 'B', 'resi': target_res}, 
                      {'stick': {'radius': 0.4, 'colorscheme': 'redCarbon'}})
        
        # B. Proximity Sidechains (RCSB style)
        # Show scFv sidechains (Chain A) within 5.0A of the selected epitope residue
        view.addStyle({
            'chain': 'A', 
            'byres': True, 
            'within': {'distance': 5.0, 'sel': {'chain': 'B', 'resi': target_res}}
        }, {'stick': {'radius': 0.15, 'colorscheme': 'blueCarbon'}})
        
        # Zoom to the interaction site
        view.zoomTo({'chain': 'B', 'resi': target_res})
    else:
        # Initial View Framing
        view.zoomTo()
        view.translate(-180, 20)
        view.zoom(0.9)

    if spin:
        view.spin(True)
    else:
        view.spin(False)
        
    showmol(view, height=height, width=width)

# Residue Biophysical Properties (WT: DDFSKQLQQS)
WT_RESIDUE_INFO = {
    1: {"name": "D (Aspartic Acid)", "property": "Negatively Charged, Acidic", "role": "Interacts with scFv polar groups."},
    2: {"name": "D (Aspartic Acid)", "property": "Negatively Charged, Acidic", "role": "Solvent exposed or ionic bonding."},
    3: {"name": "F (Phenylalanine)", "property": "Hydrophobic, Aromatic", "role": "Anchors into hydrophobic pocket of scFv."},
    4: {"name": "S (Serine)", "property": "Polar, Uncharged", "role": "Hydrogen bonding partner."},
    5: {"name": "K (Lysine)", "property": "Positively Charged, Basic", "role": "Long side chain, potential electrostatic interaction."},
    6: {"name": "Q (Glutamine)", "property": "Polar, Uncharged", "role": "Hydrogen bond donor/acceptor."},
    7: {"name": "L (Leucine)", "property": "Hydrophobic", "role": "Steric fit in scFv cavity."},
    8: {"name": "Q (Glutamine)", "property": "Polar, Uncharged", "role": "Flexible surface interaction."},
    9: {"name": "Q (Glutamine)", "property": "Polar, Uncharged", "role": "End of helical segment."},
    10: {"name": "S (Serine)", "property": "Polar, Uncharged", "role": "C-terminal interaction site."}
}

def extract_model_features(seq, current_type, encoder, selector, scaler, esm_extractor):
    """Generates the appropriate feature vector for the selected model type."""
    try:
        if current_type == 'manual':
            feat = encoder.encode_features([seq])
            feat_sel = selector.transform(feat)
            return scaler.transform(feat_sel)
        
        elif current_type == 'esm':
            # ESM expects cleaned sequences (no gaps)
            clean_seq = seq.replace('-', '')
            return esm_extractor.get_embeddings([clean_seq])
        
        elif current_type == 'hybrid':
            # Manual part
            feat_manual = encoder.encode_features([seq])
            feat_manual_sel = selector.transform(feat_manual)
            # ESM part
            clean_seq = seq.replace('-', '')
            feat_esm = esm_extractor.get_embeddings([clean_seq])
            # Combine and scale
            feat_combined = np.concatenate([feat_manual_sel, feat_esm], axis=1)
            return scaler.transform(feat_combined)
    except Exception as e:
        st.error(f"Feature extraction error for {current_type}: {e}")
        return None

# =============================================================================
# UI COMPONENTS
# =============================================================================

def main():
    st.title("🧬 mBG17 scFv Binder Predictor")
    
    # Sidebar: Information
    with st.sidebar:
        st.header("Project Context")
        st.info("""
        **mBG17 scFv** is a high-affinity binder engineered to recognize the SARS-CoV-2 nucleocapsid (N) protein.
        The model evaluates how small peptide variations (10-aa) affect the predicted stability and confidence.
        """)
        st.markdown("---")
        st.write("**Champion Model:** Phase 3 Hybrid (Multi-Feature)")
        st.write("**Live Predictor:** Manual (One-Hot Deviations)")
        st.write("**Metric:** pLDDT (Confidence)")
        
        st.warning("""
        **Scientific Disclaimer:**
        This model was trained on **Synthetic Designer Sequences** (generated via RFDiffusion & ProteinMPNN) plus targeted point mutations. It was NOT trained on Wild-Type (WT) viral data. Use the WT comparison for biological context, but be aware of the model's synthetic training bias.
        """)

    # Model Selection UI
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Model Configuration")
    
    model_choice = st.sidebar.radio(
        "Active Predictor:",
        ["Manual Deviation", "Biological (ESM)", "Hybrid Champion"],
        help="Manual: Fast deviation features. ESM: LLM embeddings. Hybrid: The best of both."
    )
    
    model_map = {
        "Manual Deviation": "manual",
        "Biological (ESM)": "esm",
        "Hybrid Champion": "hybrid"
    }
    current_type = model_map[model_choice]
    
    model, encoder, selector, scaler = load_resources(current_type)
    esm_extractor = get_esm_extractor() if current_type in ['esm', 'hybrid'] else None

    if model is None:
        st.warning(f"Please ensure that **{current_type}** training has been completed.")
        return

    # Model Context
    with st.sidebar.expander("Model Details"):
        if current_type == 'manual':
            st.write("3-layer MLP using custom amino-acid deviation features from mBG17 reference.")
        elif current_type == 'esm':
            st.write("Uses evolutionary embeddings from ESM-2 (8M) to predict stability without needing a reference.")
        else:
            st.write("Champion model combining site-specific deviations with deep evolutionary context.")

    # Create Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Live Predictor", 
        "� Peptide Explorer",
        "📥 Batch Predictor",
        "�📊 Model Analysis", 
        "📚 Background"
    ])

    with tab1:
        st.subheader("Interactive Prediction")
        st.markdown(f"""
        Enter a peptide sequence and get an instant pLDDT score. 
        *Current Model: **{model_choice}***
        """)
        
        input_seq_raw = st.text_input("Enter 10-amino acid sequence (or 11 including gaps):", value="YSVEPSTRA-L")
        
        if st.button("Predict pLDDT"):
            # Process input
            input_seq = input_seq_raw.strip().upper()
            target_len = len(encoder.reference_sequence)
            
            if len(input_seq) == 10 and target_len == 11:
                input_seq = "-" + input_seq
                st.info(f"Interpreting 10-aa input as aligned sequence: `{input_seq}`")
            
            if len(input_seq) != target_len:
                st.warning(f"Warning: Sequence length ({len(input_seq)}) does not match expected ({target_len}). Adjusting...")
                input_seq = input_seq.ljust(target_len, '-')[:target_len]
            
            try:
                features_scaled = extract_model_features(input_seq, current_type, encoder, selector, scaler, esm_extractor)
                
                if features_scaled is not None:
                    with torch.no_grad():
                        pred = model(torch.FloatTensor(features_scaled)).item()
                
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.metric(label="Predicted pLDDT", value=f"{pred:.2f}")
                        if pred > 80: st.success("High Confidence Binder")
                        elif pred > 60: st.info("Moderate Confidence")
                        else: st.warning("Low Confidence / Unstable")

                    with col2:
                        ref_seq = encoder.reference_sequence
                        ref_features_scaled = extract_model_features(ref_seq, current_type, encoder, selector, scaler, esm_extractor)
                        
                        if ref_features_scaled is not None:
                            with torch.no_grad():
                                ref_pred = model(torch.FloatTensor(ref_features_scaled)).item()
                            
                            diff = pred - ref_pred
                            st.write(f"Reference Sequence: `{ref_seq}`")
                            st.write(f"Reference pLDDT: **{ref_pred:.2f}**")
                            st.write(f"Difference: **{diff:+.2f}**")
                    
                # Physicochemical Analysis
                st.markdown("### 🧬 Physicochemical Properties")
                input_props = calculate_properties(input_seq)
                ref_props = calculate_properties(ref_seq)
                
                if input_props and ref_props:
                    prop_df = pd.DataFrame({
                        'Property': list(input_props.keys()),
                        'Input': [f"{v:.2f}" if isinstance(v, float) else v for v in input_props.values()],
                        'mBG17 Ref': [f"{v:.2f}" if isinstance(v, float) else v for v in ref_props.values()],
                    })
                    st.table(prop_df)
                
                st.markdown("---")
                st.subheader("Biological Mutation Analysis (Target: WT Epitope)")
                st.write(f"Comparing your design against the SARS-CoV-2 N-protein WT sequence: `{WT_SEQUENCE}`")
                
                # Compare clean sequences (10-aa vs 10-aa)
                input_clean = input_seq.replace('-', '')
                
                mutations = []
                mut_data = []
                
                # Match positions (assuming 10-aa input map to 10-aa WT)
                for i, (aa, wt_aa) in enumerate(zip(input_clean, WT_SEQUENCE)):
                    if aa != wt_aa:
                        mutations.append(f"Pos {i+1}: {wt_aa} → {aa}")
                        mut_data.append({'Position': i+1, 'Change': f"{wt_aa}→{aa}"})
                
                if mutations:
                    st.info(", ".join(mutations))
                    fig, ax = plt.subplots(figsize=(8, 2))
                    positions = list(range(len(WT_SEQUENCE)))
                    colors = ['#4c72b0' if WT_SEQUENCE[i] == input_clean[i] else '#c44e52' for i in range(len(WT_SEQUENCE))]
                    ax.bar(positions, [1]*len(WT_SEQUENCE), color=colors)
                    ax.set_xticks(positions)
                    ax.set_xticklabels(list(input_clean))
                    ax.set_yticks([])
                    ax.set_title("Mutations from WT Native Epitope (Red = Mutation)")
                    st.pyplot(fig)
                else:
                    st.write("Sequence matches Wild-Type Epitope exactly.")

            except Exception as e:
                st.error(f"Prediction Error: {e}")

    with tab2:
        st.subheader("🛒 Peptide Explorer (Top 10,000 Prescreened)")
        st.markdown("""
        Browse the top results discovery pipeline. These sequences were generated via a 3-body deviation model
        and screened for the highest predicted pLDDT.
        """)
        
        if os.path.exists(TOP_10K_PATH):
            df_10k = pd.read_csv(TOP_10K_PATH)
            
            # Filters
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                min_plddt = st.slider("Minimum Predicted pLDDT", 
                                     float(df_10k['predicted_pLDDT'].min()), 
                                     float(df_10k['predicted_pLDDT'].max()), 
                                     85.0)
            with col_f2:
                search_motif = st.text_input("Search for Sequence Motif (e.g. 'DDF')").upper()
            
            filtered_df = df_10k[df_10k['predicted_pLDDT'] >= min_plddt]
            if search_motif:
                filtered_df = filtered_df[filtered_df['sequence'].str.contains(search_motif)]
            
            st.write(f"Showing {len(filtered_df)} results:")
            st.dataframe(filtered_df, use_container_width=True)
            
            st.download_button(
                label="Download Filtered CSV",
                data=filtered_df.to_csv(index=False),
                file_name="filtered_top_peptides.csv",
                mime="text/csv"
            )
        else:
            st.info("Top 10K results file not found. Run the discovery script first.")

    with tab3:
        st.subheader("📥 Batch Predictor")
        st.markdown("""
        Upload a CSV file containing a column named **'sequence'** to predict pLDDT for multiple peptides at once.
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            if 'sequence' in batch_df.columns:
                if st.button("Run Batch Prediction"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, row in batch_df.iterrows():
                        seq = str(row['sequence']).strip().upper()
                        # Auto-align 10-aa
                        if len(seq) == 10: seq = "-" + seq
                        
                        # Pad/Crop
                        target_len = len(encoder.reference_sequence)
                        seq = seq.ljust(target_len, '-')[:target_len]
                        
                        try:
                            feat_sc = extract_model_features(seq, current_type, encoder, selector, scaler, esm_extractor)
                            if feat_sc is not None:
                                with torch.no_grad():
                                    p = model(torch.FloatTensor(feat_sc)).item()
                                results.append(p)
                            else:
                                results.append(np.nan)
                        except:
                            results.append(np.nan)
                        
                        progress_bar.progress((i + 1) / len(batch_df))
                    
                    batch_df['predicted_pLDDT'] = results
                    st.success(f"Successfully predicted {len(batch_df)} sequences!")
                    st.dataframe(batch_df)
                    
                    st.download_button(
                        label="Download Predictions",
                        data=batch_df.to_csv(index=False),
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.error("CSV must contain a 'sequence' column.")

    with tab4:
        st.subheader("Final Model Benchmarks")
        st.markdown("""
        How do our models compare? We evaluated three distinct approaches:
        1. **Manual (One-Hot)**: Custom engineered deviations from reference.
        2. **Biological (ESM-2)**: Large Language Model embeddings from evolutionary history.
        3. **Hybrid (Champion)**: The combination of both, providing the highest accuracy.
        """)

        # 1. Comparison Plots
        comp_dir = os.path.join(SCRIPT_DIR, 'model_comparison')
        if os.path.exists(comp_dir):
            col_bench, col_corr = st.columns(2)
            with col_bench:
                st.image(os.path.join(comp_dir, 'performance_benchmarks.png'), caption="MSE & R² Comparison")
            with col_corr:
                st.image(os.path.join(comp_dir, 'correlation_comparison.png'), caption="Prediction Correlations (Actual vs Predicted)")
        else:
            st.info("Run `final_model_comparison.py` to generate benchmarking charts.")

        st.markdown("---")
        st.subheader("📊 Detailed Model Diagnostics")
        
        diag_tabs = st.tabs(["Manual Model", "Biological (ESM) Model", "Hybrid Model"])
        
        model_diag_config = [
            ("Manual", PYTORCH_DIR, diag_tabs[0]),
            ("ESM", ESM_DIR, diag_tabs[1]),
            ("Hybrid", HYBRID_DIR, diag_tabs[2])
        ]
        
        for name, base_dir, d_tab in model_diag_config:
            with d_tab:
                diag_dir = os.path.join(base_dir, 'diagnostics')
                if os.path.exists(diag_dir):
                    st.markdown(f"**{name} Model Performance breakdown:**")
                    col_act, col_res = st.columns(2)
                    with col_act:
                        st.image(os.path.join(diag_dir, 'actual_vs_predicted.png'), caption=f"{name}: Regression Plot")
                    with col_res:
                        st.image(os.path.join(diag_dir, 'residual_analysis.png'), caption=f"{name}: Residual Analysis")
                    
                    col_dist, col_err = st.columns(2)
                    with col_dist:
                        st.image(os.path.join(diag_dir, 'residual_distribution.png'), caption=f"{name}: Error Density")
                    with col_err:
                        # Fallback for manual's specific legacy plot name
                        err_path = os.path.join(diag_dir, 'error_profile.png')
                        if not os.path.exists(err_path):
                            err_path = os.path.join(diag_dir, 'error_by_length.png')
                        
                        if os.path.exists(err_path):
                            st.image(err_path, caption=f"{name}: Error Profile")
                else:
                    st.info(f"Diagnostics for {name} model not yet generated.")
        
        # 2. SHAP Heatmap
        st.markdown("---")
        st.subheader("Feature Importance: Manual Model (SHAP)")
        shap_path = SHAP_PATH
        if os.path.exists(shap_path):
            st.image(shap_path, caption="2D Heatmap of Amino Acid positions and their impact on pLDDT.")

    with tab5:
        st.subheader("Scientific Context: mBG17 scFv")
        
        # 1. Scientific Narrative (Full Width)
        st.markdown("""
        ### Overview
        The **mBG17 scFv** is a high-affinity engineered antibody fragment designed for diagnostic and therapeutic detection of SARS-CoV-2.
        
        ### Key Scientific Background
        *   **Engineered Target**: Specifically targets the **SARS-CoV-2 nucleocapsid (N) protein** (Terry et al., 2020).
        *   **Wild-type Epitope**: The native binding site is sequence `DDFSKQLQQS` (residues 401-410 on the N-protein).
        *   **Optimization Opportunity**: The complementarity-determining regions (CDRs) of this scFv were optimized against a static epitope, leaving room for further peptide refinement.
        *   **Validation Benchmark**: The well-characterized `DDFSKQLQQS` sequence serves as our critical reference point for all pLDDT predictions.
        *   **Unique Database Status**: mBG17 scFv is absent from current large-scale protein databases, making it an ideal "unbiased" test case for novel AI prediction.
        """)
        
        st.info("💡 **Goal**: This model aims to 'reverse-engineer' the binding landscape by discovering which peptide mutations maximize the pLDDT confidence scores.")
        st.markdown("---")

        # 2. Interactive Structural Hub (Full Width)
        st.subheader("🏛️ Interactive WT Structure & Interaction Hub")
        
        col_3d, col_seq = st.columns([1.2, 1])
        
        with col_3d:
            # Control Widget for 3D View
            col_ctrl1, col_ctrl2 = st.columns([1,1])
            selected_idx = st.session_state.get('wt_selected_residue', None)
            
            with col_ctrl1:
                default_spin = True if selected_idx is None else False
                auto_spin = st.checkbox("Auto-Spin Structure", value=default_spin)
            
            with col_ctrl2:
                if st.button("Center / Reset View"):
                    st.session_state.wt_selected_residue = None
                    st.rerun()

            # Render 3D View
            show_pdb(WT_PDB_PATH, spin=auto_spin, highlight_res=st.session_state.get('wt_selected_residue'))
            st.write("*Interaction: scFv (Blue) binds the Viral Epitope (Red sticks). Click atoms for labels.*")

        with col_seq:
            # Sequence Explorers
            st.markdown("#### 🧬 Viral Epitope Explorer")
            st.write("Click a residue to highlight the binding pocket.")
            
            res_cols = st.columns(len(WT_SEQUENCE))
            for i, char in enumerate(WT_SEQUENCE):
                res_num = i + 1
                if res_cols[i].button(char, key=f"res_btn_{res_num}", 
                                       type="primary" if selected_idx == res_num else "secondary"):
                    st.session_state.wt_selected_residue = res_num
                    st.rerun()

            # Show Info for Selected Residue
            if selected_idx:
                info = WT_RESIDUE_INFO[selected_idx]
                st.success(f"**Residue {selected_idx}: {info['name']}**")
                st.write(f"🧬 **Property:** {info['property']}")
                st.write(f"🎯 **Functional Role:** {info['role']}")
                if st.button("Clear Selection"):
                    st.session_state.wt_selected_residue = None
                    st.rerun()
            else:
                st.info("Select a residue above or click the 3D structure to explore.")

            st.markdown("#### 🛡️ mBG17 scFv Sequence")
            with st.expander("Show Full Binder Sequence (245 residues)"):
                scfv_chunk_size = 50
                for i in range(0, len(SCFV_SEQUENCE), scfv_chunk_size):
                    chunk = SCFV_SEQUENCE[i:i+scfv_chunk_size]
                    st.code(f"{i+1:3d}: {chunk}")

        st.markdown("---")
        st.markdown("### Binding Schematic & Legend")
        st.success("Target: SARS-CoV-2 N-Protein | Epitope: Peptide (10-aa) | Binder: mBG17 scFv")
        st.write("*Figure: The scFv interacts with the peptide epitope via the engineered CDR loops. Use the interactive tools above to physically verify these interactions.*")

if __name__ == "__main__":
    main()
