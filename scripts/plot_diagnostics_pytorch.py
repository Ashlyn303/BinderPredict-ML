#%%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler
from src.data_loader import ProteinDataLoader, SequenceAnalyzer, DeviationFeatureEncoder, build_reference_forms

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_RESULTS_DIR = os.path.join(SCRIPT_DIR, 'pytorch_results')
DIAGNOSTIC_DIR = os.path.join(PYTORCH_RESULTS_DIR, 'diagnostics')
os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)

CSV_FILES = [
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/AF2_IPSAE/BB10-1/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/AF2_IPSAE/BB10-2/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/AF2_IPSAE/BB10-3/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/AF2_IPSAE/BB10-4/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/AF2_IPSAE/BB10-5/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/mBG17_BB10_FineNet/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/mBG17_BB10_FineNet2/all_files/usalign_alignment_pLDDT_0_combined_data.csv'
]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =============================================================================
# MODEL ARCHITECTURE (Must match training)
# =============================================================================

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

# =============================================================================
# DIAGNOSTIC ENGINE
# =============================================================================

def run_diagnostics():
    print(f"Running Professional Model Diagnostics on {DEVICE}...")

    # 1. Load Preprocessing State
    try:
        encoder = joblib.load(os.path.join(PYTORCH_RESULTS_DIR, 'encoder.joblib'))
        selector = joblib.load(os.path.join(PYTORCH_RESULTS_DIR, 'selector.joblib'))
        scaler = joblib.load(os.path.join(PYTORCH_RESULTS_DIR, 'scaler.joblib'))
    except FileNotFoundError:
        print("Error: Could not find joblib files. Please re-run train_pytorch_plddt.py first.")
        return

    # 2. Load and Prepare Original Data
    loader = ProteinDataLoader(CSV_FILES)
    loader.load_data()
    sequences, plddt_values, _ = loader.process_sequences()
    
    # We need to know original sequence lengths for diagnostics
    orig_lengths = [len(s.replace('-','')) for s in sequences]

    # 3. Encode, Select, and Scale
    X_full = encoder.encode_features(sequences)
    X_selected = selector.transform(X_full)
    X_scaled = scaler.transform(X_selected)

    # 4. Load Model
    model = PeptideNet(X_selected.shape[1]).to(DEVICE)
    model_path = os.path.join(PYTORCH_RESULTS_DIR, 'peptide_predictor_pytorch.pt')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 5. Generate Full Predictions
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_scaled).to(DEVICE)).cpu().numpy().flatten()

    # 6. Calculate Metrics
    results_df = pd.DataFrame({
        'sequence': sequences,
        'length': orig_lengths,
        'actual': plddt_values,
        'predicted': preds,
        'residual': plddt_values - preds,
        'abs_error': np.abs(plddt_values - preds)
    })

    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================

    sns.set_theme(style="whitegrid", palette="muted")

    # A. Actual vs Predicted (Reliability Check)
    plt.figure(figsize=(10, 8))
    sns.regplot(data=results_df, x='actual', y='predicted', 
                scatter_kws={'alpha':0.4, 's':10}, line_kws={'color':'red', 'ls':'--'})
    plt.plot([results_df['actual'].min(), results_df['actual'].max()], 
             [results_df['actual'].min(), results_df['actual'].max()], 
             'k', lw=1) # Identity line
    plt.title("Reliability Analysis: Actual vs. Predicted pLDDT")
    plt.savefig(os.path.join(DIAGNOSTIC_DIR, 'actual_vs_predicted.png'), dpi=300)
    plt.close()

    # B. Residual Plot (Bias Check)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=results_df, x='predicted', y='residual', alpha=0.4, s=10)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residual Analysis: Detection of Systematic Bias")
    plt.xlabel("Predicted pLDDT")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.savefig(os.path.join(DIAGNOSTIC_DIR, 'residual_analysis.png'), dpi=300)
    plt.close()

    # C. Error by Sequence Length (Condition Check)
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=results_df, x='length', y='abs_error')
    plt.title("Prediction Reliability by Sequence Length")
    plt.xlabel("Peptide Length")
    plt.ylabel("Absolute Error (pLDDT Units)")
    plt.savefig(os.path.join(DIAGNOSTIC_DIR, 'error_by_length.png'), dpi=300)
    plt.close()

    # D. Residual Distribution (Gaussian Check)
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['residual'], kde=True, bins=50)
    plt.title("Distribution of Errors (Residuals)")
    plt.savefig(os.path.join(DIAGNOSTIC_DIR, 'residual_distribution.png'), dpi=300)
    plt.close()

    # =========================================================================
    # BLIND SPOT IDENTIFICATION
    # =========================================================================

    print("\n" + "="*40)
    print("BIOLOGICAL BLIND SPOTS (Largest Failures)")
    print("="*40)
    top_fails = results_df.sort_values(by='abs_error', ascending=False).head(5)
    print(top_fails[['sequence', 'length', 'actual', 'predicted', 'residual']])
    print("="*40)

    print(f"\nDiagnostic plots saved to: {DIAGNOSTIC_DIR}")

if __name__ == "__main__":
    run_diagnostics()
