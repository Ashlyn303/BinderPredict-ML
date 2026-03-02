#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANUAL_RESULTS = os.path.join(SCRIPT_DIR, 'pytorch_results', 'manual_test_predictions.csv')
ESM_RESULTS = os.path.join(SCRIPT_DIR, 'esm_results', 'esm_test_predictions.csv')
HYBRID_RESULTS = os.path.join(SCRIPT_DIR, 'hybrid_results', 'hybrid_test_predictions.csv')
COMPARISON_DIR = os.path.join(SCRIPT_DIR, 'model_comparison')
os.makedirs(COMPARISON_DIR, exist_ok=True)

def load_and_calculate(path, pred_column):
    """Loads results and calculates metrics."""
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return None, None
    df = pd.read_csv(path)
    mse = mean_squared_error(df['actual'], df[pred_column])
    r2 = r2_score(df['actual'], df[pred_column])
    return df, (mse, r2)

def generate_comparison():
    print("Generating Professional Model Comparison Suite...")

    # 1. Load Data
    manual_df, manual_metrics = load_and_calculate(MANUAL_RESULTS, 'predicted_manual')
    esm_df, esm_metrics = load_and_calculate(ESM_RESULTS, 'predicted_esm')
    hybrid_df, hybrid_metrics = load_and_calculate(HYBRID_RESULTS, 'predicted_hybrid')

    # Prep for bar chart
    results = []
    if manual_metrics: results.append(('Manual (One-Hot)', manual_metrics[0], manual_metrics[1]))
    if esm_metrics: results.append(('Biological (ESM-2)', esm_metrics[0], esm_metrics[1]))
    if hybrid_metrics: results.append(('Hybrid (Manual+ESM)', hybrid_metrics[0], hybrid_metrics[1]))

    if not results:
        print("Error: No result files found to compare. Please run the training scripts first.")
        return

    res_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R2'])

    # 2. Main Performance Chart (Bar Chart)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.set_theme(style="whitegrid")

    # MSE plot
    sns.barplot(data=res_df, x='Model', y='MSE', palette='muted', ax=ax1)
    ax1.set_title("Mean Squared Error (Lower is Better)", fontsize=14)
    for i, v in enumerate(res_df['MSE']):
        ax1.text(i, v + 2, f"{v:.2f}", ha='center', weight='bold')

    # R2 Plot
    sns.barplot(data=res_df, x='Model', y='R2', palette='pastel', ax=ax2)
    ax2.set_title("R² Score (Higher is Better)", fontsize=14)
    ax2.set_ylim(0, 1)
    for i, v in enumerate(res_df['R2']):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha='center', weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'performance_benchmarks.png'), dpi=300)
    plt.close()

    # 3. Side-by-Side Correlation Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    plot_data = [
        (manual_df, 'predicted_manual', 'Manual Model', axes[0], 'blue'),
        (esm_df, 'predicted_esm', 'Biological (ESM-2) Model', axes[1], 'green'),
        (hybrid_df, 'predicted_hybrid', 'Hybrid Model', axes[2], 'red')
    ]

    for df, col, title, ax, color in plot_data:
        if df is not None:
            sns.regplot(data=df, x='actual', y=col, ax=ax, scatter_kws={'alpha':0.2, 's':10}, line_kws={'color':color, 'ls':'--'})
            ax.plot([df['actual'].min(), df['actual'].max()], [df['actual'].min(), df['actual'].max()], 'k', lw=1)
            ax.set_title(title)
            ax.set_xlabel("Actual pLDDT")
            ax.set_ylabel("Predicted pLDDT")
            
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'correlation_comparison.png'), dpi=300)
    plt.close()

    print(f"Professional comparison maps saved to: {COMPARISON_DIR}")
    print("\nSummary Metrics:")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    generate_comparison()
