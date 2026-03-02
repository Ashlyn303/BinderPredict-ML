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
from src.data_loader import ProteinDataLoader
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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'esm_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.0005
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =============================================================================
# MODEL ARCHITECTURE (Optimized for ESM Embeddings)
# =============================================================================

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

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def train_esm_model():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    loader = ProteinDataLoader(CSV_FILES)
    loader.load_data()
    # ESM doesn't need "standardized window" logic as much, 
    # but we'll use the already picking cleaned sequences
    sequences, plddt_values, _ = loader.process_sequences()
    
    # Clean sequences for ESM (no gaps)
    clean_sequences = [s.replace('-', '') for s in sequences]

    # 2. Extract ESM Embeddings (One-time heavy cost)
    print("Initializing ESM Feature Extractor...")
    extractor = ESMFeatureExtractor(device=DEVICE)
    print(f"Extracting embeddings for {len(clean_sequences)} sequences...")
    X_embeddings = extractor.get_embeddings(clean_sequences, batch_size=64)
    
    print(f"Feature Extraction Complete. Embedding dim: {X_embeddings.shape[1]}")

    # 3. Split and Prepare Data
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, plddt_values, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 4. Training
    model = ESMToPLDDT(embedding_dim=X_embeddings.shape[1], dropout_rate=0.2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print("Starting ESM-based training...")
    history = {'train_loss': [], 'test_loss': []}
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss/len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Test Eval
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                total_test_loss += criterion(model(batch_X), batch_y).item()
        
        avg_test_loss = total_test_loss/len(test_loader)
        history['test_loss'].append(avg_test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # 5. Plot History
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title("ESM-Based Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()

    # 6. Final Predictions for Comparison
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy().flatten()
    
    # Save a CSV with predictions for manual inspection
    test_results = pd.DataFrame({
        'sequence': [sequences[idx] for idx in range(len(test_preds))], # Map back to original indices somehow if needed
        'actual': y_test,
        'predicted_esm': test_preds
    })
    test_results.to_csv(os.path.join(OUTPUT_DIR, 'esm_test_predictions.csv'), index=False)

    # 7. Save Model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'esm_plddt_model.pt'))
    # No joblib needed for feature names since ESM is a dense embedding
    print(f"Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    train_esm_model()
