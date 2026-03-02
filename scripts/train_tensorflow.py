import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from data_loader import ProteinDataLoader, SequenceAnalyzer, DeviationFeatureEncoder, build_reference_forms

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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'tensorflow_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

REFERENCE_SEQ_INPUT = "-SLQEDLEALE"
CHOSEN_K = 1500
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def train_tensorflow_model():
    # 1. Load Data
    loader = ProteinDataLoader(CSV_FILES)
    loader.load_data()
    sequences, plddt_values = loader.process_sequences()

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

    # 6. Build Model
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_selected.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
                  loss='mse', metrics=['mae'])

    # 7. Training
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # 8. Predictions and Top 10K
    print("Predicting on all data for top 10K selection...")
    all_preds = model.predict(X_selected, batch_size=BATCH_SIZE).flatten()

    results_df = pd.DataFrame({
        'sequence': sequences,
        'actual_pLDDT': plddt_values,
        'predicted_pLDDT': all_preds
    })
    
    top_10k = results_df.sort_values(by='predicted_pLDDT', ascending=False).head(10000)
    top_10k.to_csv(os.path.join(OUTPUT_DIR, 'top_10k_predictions_tensorflow.csv'), index=False)
    print(f"Top 10K predictions saved to {OUTPUT_DIR}/top_10k_predictions_tensorflow.csv")

    # 9. SHAP Explanation
    print("Calculating SHAP values...")
    # KernelExplainer is slow but DeepExplainer for TF2 is often buggy with complex graphs
    background = X_train[:100]
    explainer = shap.KernelExplainer(model.predict, background)
    test_samples = X_test[:100]
    shap_values = explainer.shap_values(test_samples)

    if isinstance(shap_values, list): shap_values = shap_values[0]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:100], feature_names=selected_feature_names, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_tensorflow.png'), bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {OUTPUT_DIR}/shap_summary_tensorflow.png")

    # Save model
    model.save(os.path.join(OUTPUT_DIR, 'peptide_predictor_tensorflow.h5'))
    print("Model saved.")

if __name__ == "__main__":
    train_tensorflow_model()
