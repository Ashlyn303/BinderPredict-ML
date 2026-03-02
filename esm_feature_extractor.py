#%%
import torch
from transformers import AutoTokenizer, EsmModel
import numpy as np
from tqdm import tqdm

class ESMFeatureExtractor:
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", device=None):
        """
        Initializes the ESM-2 feature extractor.
        Default is the 8M parameter model (the smallest and fastest).
        """
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading ESM-2 model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(self.device).eval()
        
    def get_embeddings(self, sequences, batch_size=32):
        """
        Converts a list of sequences into dense embeddings.
        Returns a numpy array of shape (n_sequences, embedding_dim).
        We use the mean-pooling over the sequence length.
        """
        all_embeddings = []
        
        # Process in batches to manage memory
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting ESM Embeddings"):
            batch_seqs = list(sequences[i:i + batch_size])
            
            # Clean sequences (ESM doesn't like gaps '-' at the start/end usually, 
            # but we'll remove them for the "meaning" of the sequence)
            clean_seqs = [s.replace('-', '') for s in batch_seqs]
            
            inputs = self.tokenizer(clean_seqs, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use the 'last_hidden_state'
            # Shape: (batch_size, seq_len, 320)
            last_hidden_states = outputs.last_hidden_state
            
            # Global Average Pooling (Mean over the sequence dimension, ignoring padding)
            # mask out the padding tokens
            attention_mask = inputs.attention_mask.unsqueeze(-1)
            masked_hidden_states = last_hidden_states * attention_mask
            
            # Sum and divide by actual length
            sum_embeddings = torch.sum(masked_hidden_states, dim=1)
            lengths = torch.sum(attention_mask, dim=1)
            mean_embeddings = sum_embeddings / lengths
            
            all_embeddings.append(mean_embeddings.cpu().numpy())
            
        return np.concatenate(all_embeddings, axis=0)

if __name__ == "__main__":
    # Quick Test
    extractor = ESMFeatureExtractor()
    test_seqs = ["SLQEDLEALE", "DERQESLNKL", "SELSELSSQR"]
    embs = extractor.get_embeddings(test_seqs)
    print(f"Embedding shape: {embs.shape}")
    print("Test successful!")
