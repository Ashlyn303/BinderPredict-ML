#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import optuna
import wandb
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from src.data_loader import ProteinDataLoader, SequenceAnalyzer, DeviationFeatureEncoder, build_reference_forms

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
N_TRIALS = 30 # Number of Bayesian trials
N_FOLDS = 5   # For Cross-Validation

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class PeptideNet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
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
# OBJECTIVE FUNCTION FOR OPTUNA
# =============================================================================

def objective(trial, X_full, plddt_values, groups):
    # --- Suggest Hyperparameters ---
    params = {
        'chosen_k': trial.suggest_int('chosen_k', 300, 1500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5)
    }

    # Initialize W&B for this trial
    wandb.init(
        project="peptide-plddt-optimization",
        config=params,
        name=f"trial_{trial.number}",
        reinit=True,
        group="optuna-groupv-cv"
    )

    # 1. Feature Selection (Done once per trial)
    selector = SelectKBest(score_func=f_regression, k=min(params['chosen_k'], X_full.shape[1]))
    X_selected = selector.fit_transform(X_full, plddt_values)

    # 2. Cross-Validation
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_mses = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_selected, plddt_values, groups)):
        # Split
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = plddt_values[train_idx], plddt_values[val_idx]

        # --- NEW: Professional Scaling ---
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Prepare DataLoaders
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).view(-1, 1))
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params['batch_size'])

        # Build Model
        model = PeptideNet(X_selected.shape[1], params['dropout_rate']).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        
        # Train for fixed epochs (could also use pruning here)
        best_fold_loss = float('inf')
        for epoch in range(15): # Short epochs for tuning
            model.train()
            for bX, by in train_loader:
                bX, by = bX.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bX), by)
                loss.backward()
                optimizer.step()
            
            # Eval
            model.eval()
            v_loss = 0
            with torch.no_grad():
                for bX, by in val_loader:
                    bX, by = bX.to(DEVICE), by.to(DEVICE)
                    v_loss += criterion(model(bX), by).item()
            v_loss /= len(val_loader)
            if v_loss < best_fold_loss: best_fold_loss = v_loss
        
        fold_mses.append(best_fold_loss)
        wandb.log({f"fold_{fold}_best_mse": best_fold_loss})

    mean_mse = np.mean(fold_mses)
    wandb.log({"mean_cv_mse": mean_mse})
    wandb.finish()
    
    return mean_mse

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_advanced_tuning():
    print(f"Preparing data for Advanced Tuning on {DEVICE}...")
    
    # Load Data
    loader = ProteinDataLoader(CSV_FILES)
    loader.load_data()
    sequences, plddt_values, groups = loader.process_sequences()
    
    # Analyze
    analyzer = SequenceAnalyzer(sequences, plddt_values)
    sequences = analyzer.analyze_sequence_lengths()
    valid_positions = analyzer.analyze_position_content(n_top_positions=10)
    sequences, plddt_values = analyzer.validate_sequences()
    ref_clean, reference_seq_padded = build_reference_forms(REFERENCE_SEQ_INPUT, analyzer.sequence_length)
    
    # Encode Once (Heavy task)
    encoder = DeviationFeatureEncoder(
        sequences=sequences,
        valid_positions=valid_positions,
        reference_sequence=reference_seq_padded,
        pair_mode='all'
    )
    X_full = encoder.encode_features()
    
    print(f"Data ready. Found {X_full.shape[1]} total features.")
    
    # Setup Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, X_full, plddt_values, groups), n_trials=N_TRIALS)
    
    print("\n" + "="*30)
    print("BEST HYPERPARAMETERS FOUND:")
    print(study.best_params)
    print(f"Best Mean CV MSE: {study.best_value:.4f}")
    print("="*30)
    
    # Save best params to a file
    import json
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print("Best parameters saved to best_hyperparameters.json")

if __name__ == "__main__":
    run_advanced_tuning()
