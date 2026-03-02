#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from src.data_loader import ProteinDataLoader, SequenceAnalyzer, DeviationFeatureEncoder, build_reference_forms
import itertools

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

REFERENCE_SEQ_INPUT = "-SLQEDLEALE"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- GRID SEARCH PARAMETERS ---
PARAM_GRID = {
    'chosen_k': [500, 800, 1500],
    'batch_size': [32, 64],
    'learning_rate': [0.001, 0.0005],
    'weight_decay': [0, 1e-4]
}
EPOCHS = 50 

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class PeptideNet(nn.Module):
    def __init__(self, input_dim):
        super(PeptideNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # Increased dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# =============================================================================
# TUNING LOOP
# =============================================================================

def tune_model():
    print(f"Starting Grid Search on {DEVICE}...")
    
    # 1. Load and Prepare Data ONCE
    loader = ProteinDataLoader(CSV_FILES)
    loader.load_data()
    sequences, plddt_values, _ = loader.process_sequences()
    analyzer = SequenceAnalyzer(sequences, plddt_values)
    sequences = analyzer.analyze_sequence_lengths()
    valid_positions = analyzer.analyze_position_content(n_top_positions=10)
    sequences, plddt_values = analyzer.validate_sequences()
    ref_clean, reference_seq_padded = build_reference_forms(REFERENCE_SEQ_INPUT, analyzer.sequence_length)
    
    encoder = DeviationFeatureEncoder(
        sequences=sequences,
        valid_positions=valid_positions,
        reference_sequence=reference_seq_padded,
        pair_mode='all'
    )
    X_full = encoder.encode_features()

    results = []

    # 2. Iterate through Grid
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for i, params in enumerate(combinations):
        print(f"\n[Trial {i+1}/{len(combinations)}] Params: {params}")
        
        # Feature Selection
        selector = SelectKBest(score_func=f_regression, k=min(params['chosen_k'], X_full.shape[1]))
        X_selected = selector.fit_transform(X_full, plddt_values)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_selected, plddt_values, test_size=0.2, random_state=42)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1))
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

        # Model
        model = PeptideNet(X_selected.shape[1]).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation at end of epoch
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    val_loss += criterion(model(batch_X), batch_y).item()
            
            val_loss /= len(test_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        print(f"Final Best Val Loss (MSE): {best_val_loss:.4f}")
        
        res = params.copy()
        res['best_val_mse'] = best_val_loss
        results.append(res)

    # 3. Save and Report
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='best_val_mse')
    results_df.to_csv('tuning_results.csv', index=False)
    
    print("\n" + "="*30)
    print("TOP 5 PARAMETER COMBINATIONS:")
    print(results_df.head(5))
    print("="*30)
    print("Full results saved to tuning_results.csv")

if __name__ == "__main__":
    tune_model()
