#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from src.data_loader import ProteinDataLoader, SequenceAnalyzer, DeviationFeatureEncoder, build_reference_forms
from src.esm_feature_extractor import ESMFeatureExtractor
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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'hybrid_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

REFERENCE_SEQ_INPUT = "-SLQEDLEALE"
CHOSEN_K_MANUAL = 1200 # Using our optimized K from Phase 1
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.0003
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =============================================================================
# HYBRID MODEL ARCHITECTURE
# =============================================================================

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

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def train_hybrid_model():
    print(f"Starting Hybrid Training on {DEVICE}...")

    # 1. Load Data
    loader = ProteinDataLoader(CSV_FILES)
    loader.load_data()
    sequences, plddt_values, _ = loader.process_sequences()

    # 2. Extract Manual Features (Deviation-based)
    analyzer = SequenceAnalyzer(sequences, plddt_values)
    analyzer.analyze_sequence_lengths()
    valid_positions = analyzer.analyze_position_content(n_top_positions=10)
    sequences, plddt_values = analyzer.validate_sequences()
    
    ref_clean, ref_padded = build_reference_forms(REFERENCE_SEQ_INPUT, analyzer.sequence_length)
    encoder = DeviationFeatureEncoder(sequences, valid_positions, ref_padded, pair_mode='all')
    X_manual = encoder.encode_features()
    
    selector = SelectKBest(score_func=f_regression, k=min(CHOSEN_K_MANUAL, X_manual.shape[1]))
    X_manual_selected = selector.fit_transform(X_manual, plddt_values)

    # 3. Extract Biological Features (ESM-2)
    print("Extracting ESM-2 Biological Embeddings...")
    clean_sequences = [s.replace('-', '') for s in sequences]
    esm_extractor = ESMFeatureExtractor(device=DEVICE)
    X_esm = esm_extractor.get_embeddings(clean_sequences, batch_size=64)

    # 4. Combine and Scale
    print(f"Concatenating features: Manual({X_manual_selected.shape[1]}) + ESM({X_esm.shape[1]})")
    X_combined = np.concatenate([X_manual_selected, X_esm], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_combined, plddt_values, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 5. Training Loop
    model = HybridPeptideNet(X_manual_selected.shape[1], X_esm.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print("Training Hybrid Model...")
    history = {'train_loss': [], 'test_loss': []}
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
        
        avg_train = train_loss/len(train_loader)
        history['train_loss'].append(avg_train)
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                test_loss += criterion(model(batch_X), batch_y).item()
        
        avg_test = test_loss/len(test_loader)
        history['test_loss'].append(avg_test)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train:.4f} | Test: {avg_test:.4f}")

    # 6. Save Results
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy().flatten()
    
    test_results = pd.DataFrame({
        'actual': y_test,
        'predicted_hybrid': test_preds
    })
    test_results.to_csv(os.path.join(OUTPUT_DIR, 'hybrid_test_predictions.csv'), index=False)

    plt.figure(figsize=(10,6))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.title("Hybrid Model Training (Manual + ESM-2)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_history.png'))
    plt.close()

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'hybrid_plddt_model.pt'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'hybrid_scaler.joblib'))
    
    print(f"Hybrid model and history saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    train_hybrid_model()
