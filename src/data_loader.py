import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from sklearn.feature_selection import SelectKBest, f_regression

# =============================================================================
# CONFIGURATION
# =============================================================================

AA_LIST = list('ACDEFGHIKLMNPQRSTVWY-')
MIN_AA_THRESHOLD = 50

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_plddt_value(plddt_str):
    if pd.isna(plddt_str):
        return None
    plddt_str = str(plddt_str).strip()
    try:
        return float(plddt_str)
    except ValueError:
        pass
    if plddt_str.startswith('[') and plddt_str.endswith(']'):
        try:
            values_str = plddt_str[1:-1]
            values = [float(x.strip()) for x in values_str.split(',') if x.strip()]
            return np.mean(values) if values else None
        except (ValueError, IndexError):
            pass
    numbers = re.findall(r'-?\d+\.?\d*', plddt_str)
    if numbers:
        try:
            values = [float(x) for x in numbers]
            return np.mean(values)
        except ValueError:
            pass
    return None

VALID_AA_PATTERN = re.compile(r"[ACDEFGHIKLMNPQRSTVWY\-]+")

def _is_valid_seq(s):
    if pd.isna(s):
        return False
    s = str(s).strip()
    if not s:
        return False
    return bool(VALID_AA_PATTERN.fullmatch(s))

def process_aligned_sequence(sequence):
    if pd.isna(sequence):
        return None
    sequence = str(sequence).strip()
    return sequence if sequence else None

def pick_final_query_seq(row):
    cand = row.get("Final_Query_Peptide")
    if _is_valid_seq(cand):
        return cand
    orient = row.get("Final_Orientation", "")
    if orient == "Original":
        cand = row.get("Query_Peptide_Alignment")
    elif orient == "Flipped":
        cand = row.get("Query_Peptide_Flipped")
    if _is_valid_seq(cand):
        return cand
    cand = row.get("Query_Peptide_Alignment")
    if _is_valid_seq(cand):
        return cand
    cand = row.get("sequence") or row.get("Original seq.") or row.get("Original Seq.")
    if _is_valid_seq(cand):
        return cand
    return None

def row_passes_ss_filter(row, window_size=10, min_frac=0.30):
    ss = row.get("secondary_structure")
    if pd.isna(ss):
        return False
    ss = str(ss).strip()
    if not ss:
        return False
    window = ss[:window_size]
    denom = max(1, min(len(ss), window_size))
    h_count = sum(1 for ch in window if ch == 'H')
    return (h_count / denom) >= min_frac

def tm_score_to_weight(tm):
    if tm is None or pd.isna(tm):
        return 1.0
    try:
        tm = float(tm)
    except Exception:
        return 1.0
    tm = max(0.0, min(1.0, tm))
    return tm

def classify_feature_name(fname: str):
    if '_' not in fname:
        return None
    left, right = fname.split('_', 1)
    npos = left.count('pos')
    if npos in (1, 2, 3):
        return npos
    L = len(right)
    if L in (1, 2, 3):
        return L
    return None

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

class ProteinDataLoader:
    def __init__(self, csv_files):
        self.csv_files = csv_files
        self.data = None
        self.sequences = None
        self.plddt_values = None
        self.sample_weights = None

    def load_data(self, use_ss_filter=False):
        combined_data = []
        for i, file_path in enumerate(self.csv_files):
            try:
                df = pd.read_csv(file_path)
                plddt_col = self._find_plddt_column(df)
                if plddt_col is None: continue
                
                df['pLDDT_column_name'] = plddt_col
                df['__file_idx__'] = i # Save file index for GroupKFold
                new_required_any = ["Final_Query_Peptide", "Query_Peptide_Alignment", "Query_Peptide_Flipped", "sequence"]
                has_new = any(col in df.columns for col in new_required_any)
                df['__schema__'] = 'new' if has_new else 'legacy'
                combined_data.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if not combined_data:
            raise ValueError("No valid CSV files were loaded")
        self.data = pd.concat(combined_data, ignore_index=True)
        return self.data

    def _find_plddt_column(self, df):
        if "pLDDT" in df.columns: return "pLDDT"
        for col in df.columns:
            if 'plddt' in col.lower(): return col
        return None

    def process_sequences(self, use_ss_filter=False, use_tm_weighting=False):
        sequences = []
        plddt_values = []
        sample_weights = []
        groups = [] # Track file index for GroupKFold

        for idx, row in self.data.iterrows():
            plddt_col = row['pLDDT_column_name']
            plddt_val = parse_plddt_value(row[plddt_col])
            if plddt_val is None: continue

            if use_ss_filter and not row_passes_ss_filter(row):
                continue

            aligned_seq = None
            if row.get('__schema__') == 'new':
                aligned_seq = pick_final_query_seq(row)
            else:
                alignment_str = row.get('Final_Structural_Alignment')
                if pd.notna(alignment_str):
                    m = re.search(r"\('.*?', '.*?', '(.*?)'\)", str(alignment_str))
                    if m: aligned_seq = m.group(1)
                if not aligned_seq:
                    aligned_seq = row.get('Original seq.') or row.get('Original Seq.')

            aligned_seq = process_aligned_sequence(aligned_seq)
            if not aligned_seq: continue

            sequences.append(aligned_seq)
            plddt_values.append(plddt_val)

            if use_tm_weighting:
                w = tm_score_to_weight(row.get("tm_score") or row.get("TM_score") or row.get("TMScore"))
            else:
                w = 1.0
            sample_weights.append(w)
            groups.append(row.get('__file_idx__', 0))

        self.sequences = np.array(sequences)
        self.plddt_values = np.array(plddt_values)
        self.sample_weights = np.array(sample_weights)
        self.groups = np.array(groups)
        return self.sequences, self.plddt_values, self.groups

class SequenceAnalyzer:
    def __init__(self, sequences, plddt_values):
        self.sequences = sequences
        self.plddt_values = plddt_values
        self.sequence_length = None
        self.valid_positions = None
        self.position_aa_counts = {}
        self.position_gap_counts = {}

    def analyze_sequence_lengths(self):
        seq_lengths = [len(seq) for seq in self.sequences]
        max_length = max(seq_lengths)
        padded_sequences = []
        for seq in self.sequences:
            if len(seq) < max_length:
                padded_sequences.append(seq + '-' * (max_length - len(seq)))
            else:
                padded_sequences.append(seq)
        self.sequences = np.array(padded_sequences)
        self.sequence_length = max_length
        return self.sequences

    def analyze_position_content(self, n_top_positions=10):
        total_sequences = len(self.sequences)
        positions_to_exclude = []
        for pos in range(self.sequence_length):
            aa_counter = Counter()
            gap_count = 0
            for seq in self.sequences:
                char = seq[pos]
                if char == '-': gap_count += 1
                else: aa_counter[char] += 1
            self.position_aa_counts[pos] = aa_counter
            self.position_gap_counts[pos] = gap_count
            
            total_aa_count = sum(aa_counter.values())
            gap_percentage = (gap_count / total_sequences) * 100
            if total_aa_count < MIN_AA_THRESHOLD or gap_percentage > 90:
                positions_to_exclude.append(pos)

        # Implementation of consecutive window selection
        position_scores = {}
        for pos in range(self.sequence_length):
            if pos in positions_to_exclude:
                position_scores[pos] = {'combined': -np.inf}
                continue

            aa_counts = self.position_aa_counts[pos]
            gap_count = self.position_gap_counts[pos]
            total_aa_count = sum(aa_counts.values())
            
            # Entropy
            aa_probs = np.array([count / total_aa_count for count in aa_counts.values()])
            entropy = -np.sum(aa_probs * np.log2(aa_probs + 1e-10))
            non_gap_pct = (total_sequences - gap_count) / total_sequences

            # F-statistic (simplified/estimated if needed, but let's try to keep it)
            f_statistic = 0.0
            unique_aas = list(aa_counts.keys())
            if len(unique_aas) > 1:
                X_pos = np.zeros((len(self.sequences), len(unique_aas)))
                aa_to_idx = {aa: i for i, aa in enumerate(unique_aas)}
                for seq_idx, seq in enumerate(self.sequences):
                    ch = seq[pos]
                    if ch != '-' and ch in aa_to_idx:
                        X_pos[seq_idx, aa_to_idx[ch]] = 1
                try:
                    f_scores, _ = f_regression(X_pos, self.plddt_values)
                    f_statistic = float(np.max(f_scores))
                except: f_statistic = 0.0

            position_scores[pos] = {
                'combined': 0.60 * f_statistic + 25.0 * entropy + 15.0 * non_gap_pct,
                'non_gap_pct': non_gap_pct,
                'entropy': entropy
            }

        best = None
        for start in range(0, self.sequence_length - n_top_positions + 1):
            window = list(range(start, start + n_top_positions))
            if any(position_scores[p]['combined'] == -np.inf for p in window): continue
            
            combined_sum = sum(position_scores[p]['combined'] for p in window)
            avg_non_gap = np.mean([position_scores[p]['non_gap_pct'] for p in window])
            avg_entropy = np.mean([position_scores[p]['entropy'] for p in window])
            
            candidate = (combined_sum, avg_non_gap, avg_entropy, window)
            if best is None or candidate > best:
                best = candidate
        
        if best:
            self.valid_positions = best[3]
        else:
            self.valid_positions = [p for p in range(self.sequence_length) if p not in positions_to_exclude][:n_top_positions]
        
        return self.valid_positions

    def validate_sequences(self):
        valid_aa_set = set(list('ACDEFGHIKLMNPQRSTVWY-'))
        valid_seq_indices = [i for i, seq in enumerate(self.sequences) if all(aa in valid_aa_set for aa in seq)]
        self.sequences = self.sequences[valid_seq_indices]
        self.plddt_values = self.plddt_values[valid_seq_indices]
        return self.sequences, self.plddt_values

class DeviationFeatureEncoder:
    def __init__(self, sequences, valid_positions, reference_sequence, drop_aa='-', 
                 pair_mode='adjacent', pair_vocab='observed', pair_topk=200,
                 triplet_mode='none', triplet_vocab='observed', triplet_topk=200):
        self.sequences = sequences
        self.valid_positions = list(valid_positions)
        self.reference_sequence = reference_sequence
        self.drop_aa = drop_aa
        self.pair_mode = pair_mode
        self.pair_vocab = pair_vocab
        self.pair_topk = pair_topk
        self.triplet_mode = triplet_mode
        self.triplet_vocab = triplet_vocab
        self.triplet_topk = triplet_topk

        # Alphabet
        aa_set = set()
        for seq in sequences:
            for pos in self.valid_positions:
                if pos < len(seq) and seq[pos] != self.drop_aa:
                    aa_set.add(seq[pos])
        self.amino_acids = sorted(aa_set)

        # Pairs/Triples
        vp = self.valid_positions
        if pair_mode == 'adjacent': self.pairs = [(vp[i], vp[i+1]) for i in range(len(vp)-1)]
        elif pair_mode == 'all': self.pairs = [(vp[i], vp[j]) for i in range(len(vp)) for j in range(i+1, len(vp))]
        else: self.pairs = []

        if triplet_mode == 'adjacent': self.triples = [(vp[i], vp[i+1], vp[i+2]) for i in range(len(vp)-2)]
        elif triplet_mode == 'all':
            self.triples = [(vp[i], vp[j], vp[k]) for i in range(len(vp)) for j in range(i+1, len(vp)) for k in range(j+1, len(vp))]
        else: self.triples = []

        self.ref_pos_aa = {pos: (reference_sequence[pos] if pos < len(reference_sequence) else self.drop_aa) for pos in self.valid_positions}
        
        def safe_cat(a, b): return None if (a == self.drop_aa or b == self.drop_aa) else (a + b)
        def safe_cat3(a, b, c): return None if (a == self.drop_aa or b == self.drop_aa or c == self.drop_aa) else (a + b + c)

        self.ref_pair_aa = {(i, j): safe_cat(self.ref_pos_aa[i], self.ref_pos_aa[j]) for (i, j) in self.pairs}
        self.ref_triplet_aa = {(i, j, k): safe_cat3(self.ref_pos_aa[i], self.ref_pos_aa[j], self.ref_pos_aa[k]) for (i, j, k) in self.triples}

        # Vocabularies
        self.pair_to_vocab = {}
        for (i, j) in self.pairs:
            cnt = Counter()
            for seq in sequences:
                if i < len(seq) and j < len(seq):
                    ai, aj = seq[i], seq[j]
                    if ai != self.drop_aa and aj != self.drop_aa: cnt[ai+aj] += 1
            vocab = set(cnt.keys()) if self.pair_vocab == 'observed' else set([p for p, _ in cnt.most_common(self.pair_topk)])
            ref_dp = self.ref_pair_aa[(i, j)]
            if ref_dp: vocab.add(ref_dp)
            self.pair_to_vocab[(i, j)] = sorted(vocab)

        self.triple_to_vocab = {}
        for (i, j, k) in self.triples:
            cnt = Counter()
            for seq in sequences:
                if i < len(seq) and j < len(seq) and k < len(seq):
                    ai, aj, ak = seq[i], seq[j], seq[k]
                    if ai != self.drop_aa and aj != self.drop_aa and ak != self.drop_aa: cnt[ai+aj+ak] += 1
            vocab = set(cnt.keys()) if self.triplet_vocab == 'observed' else set([t for t, _ in cnt.most_common(self.triplet_topk)])
            ref_tri = self.ref_triplet_aa[(i, j, k)]
            if ref_tri: vocab.add(ref_tri)
            self.triple_to_vocab[(i, j, k)] = sorted(vocab)

        # Feature Names
        self.feature_names = []
        for pos in self.valid_positions:
            for aa in self.amino_acids: self.feature_names.append(f"pos{pos+1}_{aa}")
        for (i, j), vocab in self.pair_to_vocab.items():
            for dp in vocab: self.feature_names.append(f"pos{i+1}pos{j+1}_{dp}")
        for (i, j, k), vocab in self.triple_to_vocab.items():
            for tri in vocab: self.feature_names.append(f"pos{i+1}pos{j+1}pos{k+1}_{tri}")

    def encode_features(self, sequences=None):
        target_seqs = sequences if sequences is not None else self.sequences
        X = [self._encode_one(seq) for seq in target_seqs]
        return np.array(X, dtype=float)

    def _encode_one(self, seq):
        row = []
        for pos in self.valid_positions:
            seq_aa = seq[pos] if pos < len(seq) else self.drop_aa
            ref_aa = self.ref_pos_aa[pos]
            for aa in self.amino_acids:
                row.append((1.0 if seq_aa == aa else 0.0) - (1.0 if ref_aa == aa else 0.0))
        for (i, j), vocab in self.pair_to_vocab.items():
            ai, aj = (seq[i] if i < len(seq) else self.drop_aa), (seq[j] if j < len(seq) else self.drop_aa)
            seq_dp = None if (ai == self.drop_aa or aj == self.drop_aa) else (ai+aj)
            ref_dp = self.ref_pair_aa[(i, j)]
            for dp in vocab:
                row.append((1.0 if seq_dp == dp else 0.0) - (1.0 if ref_dp == dp else 0.0))
        for (i, j, k), vocab in self.triple_to_vocab.items():
            ai, aj, ak = (seq[i] if i < len(seq) else self.drop_aa), (seq[j] if j < len(seq) else self.drop_aa), (seq[k] if k < len(seq) else self.drop_aa)
            seq_tri = None if (ai == self.drop_aa or aj == self.drop_aa or ak == self.drop_aa) else (ai+aj+ak)
            ref_tri = self.ref_triplet_aa[(i, j, k)]
            for tri in vocab:
                row.append((1.0 if seq_tri == tri else 0.0) - (1.0 if ref_tri == tri else 0.0))
        return np.array(row, dtype=float)

def build_reference_forms(ref_input: str, target_len: int):
    ref_clean = ref_input.replace('-', '')
    if len(ref_input) < target_len:
        reference_padded = ref_input + '-' * (target_len - len(ref_input))
    else:
        reference_padded = ref_input[:target_len]
    return ref_clean, reference_padded
