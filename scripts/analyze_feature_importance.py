#%%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import DeviationFeatureEncoder

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_RESULTS_DIR = os.path.join(SCRIPT_DIR, 'pytorch_results')
INSIGHTS_DIR = os.path.join(PYTORCH_RESULTS_DIR, 'scientific_insights')
os.makedirs(INSIGHTS_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

def analyze_importance():
    print("Extracting Biochemical Insights from Production Model...")

    # 1. Load Preprocessing State
    try:
        encoder = joblib.load(os.path.join(PYTORCH_RESULTS_DIR, 'encoder.joblib'))
        selector = joblib.load(os.path.join(PYTORCH_RESULTS_DIR, 'selector.joblib'))
    except FileNotFoundError:
        print("Error: Required joblib files not found.")
        return

    # 2. Get Selection Scores (f_regression)
    # These scores tell us which features have the strongest statistical 
    # linear relationship with pLDDT before the Neural Network even sees them.
    scores = selector.scores_
    feature_names = encoder.feature_names
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'statistical_score': scores
    })
    
    # Sort and take top 20
    top_20 = importance_df.sort_values(by='statistical_score', ascending=False).head(20)

    # 3. Visualization: Statistical Importance
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=top_20, x='statistical_score', y='feature', palette="viridis")
    plt.title("Top 20 Statistically Significant Biochemical Features (f_regression)")
    plt.xlabel("Statistical Significance Score")
    plt.ylabel("Feature (Position_AA or Pair)")
    plt.savefig(os.path.join(INSIGHTS_DIR, 'top_features_statistical.png'), bbox_inches='tight')
    plt.close()

    # 4. Neural Network Weights (First Layer Importance)
    # The first layer weights indicate which features the model 
    # prioritized during training.
    from train_pytorch_plddt import PeptideNet
    
    input_dim = len(selector.get_support(indices=True))
    model = PeptideNet(input_dim)
    model.load_state_dict(torch.load(os.path.join(PYTORCH_RESULTS_DIR, 'peptide_predictor_pytorch.pt'), map_location=DEVICE))
    model.eval()
    
    # Get weights of the first linear layer
    # Shape: (256, input_dim)
    weights = model.network[0].weight.data.cpu().numpy()
    # Mean absolute weight per input feature
    feature_weights = np.mean(np.abs(weights), axis=0)
    
    selected_names = [feature_names[i] for i in selector.get_support(indices=True)]
    
    weight_df = pd.DataFrame({
        'feature': selected_names,
        'model_weight': feature_weights
    })
    
    top_20_weights = weight_df.sort_values(by='model_weight', ascending=False).head(20)

    # 5. Visualization: Model Weight Importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_20_weights, x='model_weight', y='feature', palette="magma")
    plt.title("Top 20 Features prioritized by the Neural Network")
    plt.xlabel("Mean Absolute Weight (First Layer)")
    plt.ylabel("Feature")
    plt.savefig(os.path.join(INSIGHTS_DIR, 'top_features_model_weights.png'), bbox_inches='tight')
    plt.close()

    # 6. Summary Report
    print("\n" + "="*40)
    print("TOP BIOCHEMICAL DRIVERS")
    print("="*40)
    print(top_20_weights[['feature', 'model_weight']].to_string(index=False))
    print("="*40)
    
    # Save the full importance table
    weight_df.to_csv(os.path.join(INSIGHTS_DIR, 'all_feature_importances.csv'), index=False)
    print(f"\nScientific insight reports saved to: {INSIGHTS_DIR}")

if __name__ == "__main__":
    analyze_importance()
