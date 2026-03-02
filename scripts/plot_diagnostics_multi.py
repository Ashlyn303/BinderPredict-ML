import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_CONFIG = {
    'manual': {
        'path': os.path.join(SCRIPT_DIR, 'pytorch_results', 'manual_test_predictions.csv'),
        'pred_col': 'predicted_manual',
        'out_dir': os.path.join(SCRIPT_DIR, 'pytorch_results', 'diagnostics'),
        'color': 'blue'
    },
    'esm': {
        'path': os.path.join(SCRIPT_DIR, 'esm_results', 'esm_test_predictions.csv'),
        'pred_col': 'predicted_esm',
        'out_dir': os.path.join(SCRIPT_DIR, 'esm_results', 'diagnostics'),
        'color': 'green'
    },
    'hybrid': {
        'path': os.path.join(SCRIPT_DIR, 'hybrid_results', 'hybrid_test_predictions.csv'),
        'pred_col': 'predicted_hybrid',
        'out_dir': os.path.join(SCRIPT_DIR, 'hybrid_results', 'diagnostics'),
        'color': 'red'
    }
}

def generate_diagnostics(model_name, config):
    print(f"Generating diagnostics for {model_name}...")
    if not os.path.exists(config['path']):
        print(f"Skipping {model_name}: File not found.")
        return
    
    os.makedirs(config['out_dir'], exist_ok=True)
    df = pd.read_csv(config['path'])
    df['residual'] = df['actual'] - df[config['pred_col']]
    
    sns.set_theme(style="whitegrid")
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x='actual', y=config['pred_col'], color=config['color'], 
                scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'black', 'ls': '--'})
    plt.plot([df['actual'].min(), df['actual'].max()], [df['actual'].min(), df['actual'].max()], 'k', lw=1)
    plt.title(f"{model_name.upper()} Model: Actual vs Predicted", fontsize=14)
    plt.xlabel("Actual pLDDT")
    plt.ylabel("Predicted pLDDT")
    plt.savefig(os.path.join(config['out_dir'], 'actual_vs_predicted.png'), dpi=300)
    plt.close()
    
    # 2. Residual Analysis (Predicted vs Residual)
    plt.figure(figsize=(8, 6))
    plt.scatter(df[config['pred_col']], df['residual'], alpha=0.3, s=10, color=config['color'])
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.title(f"{model_name.upper()} Model: Residual Analysis", fontsize=14)
    plt.xlabel("Predicted pLDDT")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.savefig(os.path.join(config['out_dir'], 'residual_analysis.png'), dpi=300)
    plt.close()
    
    # 3. Residual Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['residual'], kde=True, color=config['color'])
    plt.title(f"{model_name.upper()} Model: Residual Distribution (Error Density)", fontsize=14)
    plt.xlabel("Residual (Error)")
    plt.savefig(os.path.join(config['out_dir'], 'residual_distribution.png'), dpi=300)
    plt.close()

    # 4. Error by pLDDT Magnitude (Binned Actual vs MSE)
    plt.figure(figsize=(8, 6))
    df['actual_bin'] = pd.cut(df['actual'], bins=10)
    binned_error = df.groupby('actual_bin')['residual'].apply(lambda x: np.sqrt(np.mean(x**2)))
    binned_error.plot(kind='bar', color=config['color'], alpha=0.7)
    plt.title(f"{model_name.upper()} Model: RMSE by Actual pLDDT Bin", fontsize=14)
    plt.xlabel("Actual pLDDT Range")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config['out_dir'], 'error_profile.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    for name, cfg in MODELS_CONFIG.items():
        generate_diagnostics(name, cfg)
    print("All diagnostic plots generated successfully.")
