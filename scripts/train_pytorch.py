#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import re
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from src.data_loader import ProteinDataLoader, SequenceAnalyzer, DeviationFeatureEncoder, build_reference_forms
import joblib

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_FILES = [
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/AF2_IPSAE/BB10-1/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/AF2_IPSAE/BB10-2/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/AF2_IPSAE/BB10-3/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/AF2_IPSAE/BB10-4/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/AF2_IPSAE/BB10-5/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/mBG17_BB10_FineNet/all_files/usalign_alignment_pLDDT_0_combined_data.csv',
    '/Users/szu-hsuan/Dropbox/CSU/TagTeam/mBG17/mBG17_BB10/mBG17_BB10_FineNet2/all_files/usalign_alignment_pLDDT_0_combined_data.csv'
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'pytorch_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

REFERENCE_SEQ_INPUT = "-SLQEDLEALE"
CHOSEN_K = 1200  # Optimized via Optuna
BATCH_SIZE = 64
EPOCHS = 100     # Higher, but will use early stopping logic if needed
LEARNING_RATE = 0.000186 # Optimized via Optuna
WEIGHT_DECAY = 8.11e-06 # Optimized via Optuna
DROPOUT_RATE = 0.128   # Optimized via Optuna
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =============================================================================
# MODEL ARCHITECTURE
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
# MAIN PIPELINE
# =============================================================================

def train_pytorch_model():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    loader = ProteinDataLoader(CSV_FILES)
    loader.load_data()
    sequences, plddt_values, _ = loader.process_sequences()

    # 2. Analyze Sequences
    analyzer = SequenceAnalyzer(sequences, plddt_values)
    sequences = analyzer.analyze_sequence_lengths()
    valid_positions = analyzer.analyze_position_content(n_top_positions=10)
    sequences, plddt_values = analyzer.validate_sequences()

    # 3. Feature Engineering
    ref_clean, reference_seq_padded = build_reference_forms(REFERENCE_SEQ_INPUT, analyzer.sequence_length)
    encoder = DeviationFeatureEncoder(
        sequences=sequences,
        valid_positions=valid_positions,
        reference_sequence=reference_seq_padded,
        pair_mode='all',
        triplet_mode='none'
    )
    X = encoder.encode_features()
    
    # 4. Feature Selection
    selector = SelectKBest(score_func=f_regression, k=min(CHOSEN_K, X.shape[1]))
    X_selected = selector.fit_transform(X, plddt_values)
    selected_feature_names = [encoder.feature_names[i] for i in selector.get_support(indices=True)]

    # 5. Split and Prepare Data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, plddt_values, test_size=0.2, random_state=42)
    
    # --- NEW: Professional Scaling ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X_selected) # For full prediction later

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 6. Training
    model = PeptideNet(X_selected.shape[1], DROPOUT_RATE).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    test_loss += criterion(model(batch_X), batch_y).item()
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")

    # 7. Predictions and Top 10K
    model.eval()
    
    with torch.no_grad():
        test_preds = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy().flatten()
    
    test_results = pd.DataFrame({
        'actual': y_test,
        'predicted_manual': test_preds
    })
    test_results.to_csv(os.path.join(OUTPUT_DIR, 'manual_test_predictions.csv'), index=False)

    # Predict on ALL data from dataset
    print("Predicting on all dataset samples...")
    full_dataset = TensorDataset(torch.FloatTensor(X_full_scaled))
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE)
    
    all_full_preds = []
    with torch.no_grad():
        for (batch_X,) in full_loader:
            batch_X = batch_X.to(DEVICE)
            all_full_preds.append(model(batch_X).cpu().numpy())
    all_full_preds = np.concatenate(all_full_preds).flatten()

    # Create results dataframe for dataset (Aggregating duplicates)
    dataset_results = pd.DataFrame({
        'sequence': sequences,
        'actual_pLDDT': plddt_values,
        'predicted_pLDDT': all_full_preds
    })
    
    # Take mean for duplicate sequences in dataset
    dataset_results = dataset_results.groupby('sequence').agg({
        'actual_pLDDT': 'mean',
        'predicted_pLDDT': 'mean'
    }).reset_index()
    dataset_results['type'] = 'dataset'
    
    # --- NEW: Predict on ALL single and double mutants of the reference ---
    print("Generating and predicting on all single/double mutants of reference...")
    synthetic_mutants = generate_mutants_for_ranking(encoder, reference_seq_padded, valid_positions)
    X_synthetic = encoder.encode_features(synthetic_mutants)
    X_syn_selected = selector.transform(X_synthetic)
    X_syn_scaled = scaler.transform(X_syn_selected)
    
    with torch.no_grad():
        syn_preds = model(torch.FloatTensor(X_syn_scaled).to(DEVICE)).cpu().numpy().flatten()
        
    synthetic_results = pd.DataFrame({
        'sequence': synthetic_mutants,
        'type': 'synthetic',
        'actual_pLDDT': np.nan,
        'predicted_pLDDT': syn_preds
    })
    
    # Combine and pick Top 10,000 UNIQUE sequences
    combined_results = pd.concat([dataset_results, synthetic_results], ignore_index=True)
    top_10k = combined_results.sort_values(by='predicted_pLDDT', ascending=False).head(10000)
    top_10k.to_csv(os.path.join(OUTPUT_DIR, 'top_10k_predictions_pytorch.csv'), index=False)
    print(f"Top 10K unique inclusive predictions saved to {OUTPUT_DIR}/top_10k_predictions_pytorch.csv")

    # 8. SHAP Explanation
    print("Calculating SHAP values...")
    background = torch.FloatTensor(X_train[:100]).to(DEVICE)
    explainer = shap.DeepExplainer(model, background)
    test_samples = torch.FloatTensor(X_test[:200]).to(DEVICE)
    shap_values = explainer.shap_values(test_samples)

    if isinstance(shap_values, list): shap_values = shap_values[0]

    # Standard Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:200], feature_names=selected_feature_names, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_pytorch.png'), bbox_inches='tight')
    plt.close()

    # --- NEW: 2D SHAP Heatmap (Position vs Amino Acid) ---
    plot_shap_heatmap(shap_values, selected_feature_names, OUTPUT_DIR)

    # Save model and preprocessing state
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'peptide_predictor_pytorch.pt'))
    joblib.dump(encoder, os.path.join(OUTPUT_DIR, 'encoder.joblib'))
    joblib.dump(selector, os.path.join(OUTPUT_DIR, 'selector.joblib'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    
    print(f"Model and results saved in {OUTPUT_DIR}")

def plot_shap_heatmap(shap_values, feature_names, output_dir):
    """
    Creates a 2D Heatmap of SHAP values: Amino Acid vs Position.
    Only considers 1-body features (e.g., 'pos1_A').
    """
    print("Generating 2D SHAP Heatmap...")
    
    # 1. Aggregate mean absolute SHAP values per feature
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    
    # 2. Parse feature names into (position, aa)
    # We only care about 1-body features like 'pos1_A'
    data = []
    positions = set()
    aas = set()
    
    for i, name in enumerate(feature_names):
        # Match 'pos1_A' but not 'pos1pos2_AA'
        match = re.match(r'^pos(\d+)_([A-Z\-])$', name)
        if match:
            pos = int(match.group(1))
            aa = match.group(2)
            val = np.mean(shap_values[:, i]) # Use signed mean to see direction (higher/lower)
            data.append((pos, aa, val))
            positions.add(pos)
            aas.add(aa)
            
    if not data:
        print("Warning: No 1-body features found for heatmap.")
        return

    # 3. Create Pivot Table
    pos_list = sorted(list(positions))
    aa_list = sorted(list(aas))
    heatmap_df = pd.DataFrame(0.0, index=aa_list, columns=pos_list, dtype=float)
    
    for pos, aa, val in data:
        heatmap_df.at[aa, pos] = float(val)
        
    # Ensure all data is float and handle any remaining NaNs just in case
    heatmap_df = heatmap_df.astype(float).fillna(0.0)
        
    # 4. Plot
    import seaborn as sns
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df, cmap='RdBu_r', center=0, annot=True, fmt=".3f", cbar_kws={'label': 'Mean SHAP Value (Effect on pLDDT)'})
    plt.title("Position-Specific Amino Acid Effects on pLDDT (SHAP)")
    plt.xlabel("Position")
    plt.ylabel("Amino Acid")
    plt.savefig(os.path.join(output_dir, 'shap_heatmap_2d.png'), bbox_inches='tight')
    plt.close()
    print(f"2D SHAP Heatmap saved as {output_dir}/shap_heatmap_2d.png")

def generate_mutants_for_ranking(encoder, ref_padded, valid_positions):
    """
    Generates all single and double mutants for the valid positions.
    """
    mutants = []
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY" # Standard amino acids
    
    ref_chars = list(ref_padded)
    
    # 1. Single Mutants
    for pos in valid_positions:
        original_aa = ref_chars[pos]
        for aa in aa_alphabet:
            if aa != original_aa:
                mutant = ref_chars.copy()
                mutant[pos] = aa
                mutants.append("".join(mutant))
                
    # 2. Double Mutants 
    for i, pos1 in enumerate(valid_positions):
        for j, pos2 in enumerate(valid_positions):
            if i >= j: continue # Avoid double counting
            
            orig1, orig2 = ref_chars[pos1], ref_chars[pos2]
            for aa1 in aa_alphabet:
                for aa2 in aa_alphabet:
                    if aa1 != orig1 or aa2 != orig2:
                        mutant = ref_chars.copy()
                        mutant[pos1] = aa1
                        mutant[pos2] = aa2
                        mutants.append("".join(mutant))
                        
    return list(set(mutants)) # Ensure unique

if __name__ == "__main__":
    train_pytorch_model()
