# -*- coding: utf-8 -*-
import torch
import os
import pickle  
from transformers import T5Tokenizer, T5EncoderModel
from conf import T5MODEL_FOLD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[FeatureExtract] : {device}")

CACHE_STAT_PATH = "/home/www/PepLM-GNN/webserver/static/download/cached_train_stats.pkl"

SMALL_DATA_PATH = "/home/www/PepLM-GNN/webserver/static/download/Test167_negative.txt"

# --------------------------
# 2. Load the T5 model
# --------------------------
if not os.path.exists(T5MODEL_FOLD):
    raise FileNotFoundError(f"The T5 model path does not exist: {T5MODEL_FOLD}")

tokenizer = T5Tokenizer.from_pretrained(T5MODEL_FOLD, do_lower_case=False, local_files_only=True)
t5_model = T5EncoderModel.from_pretrained(T5MODEL_FOLD, local_files_only=True).to(device)
t5_model.eval()
print(f"[FeatureExtract] T5 model loading complete: {next(t5_model.parameters()).device}")

# --------------------------
# 3. Core Feature Function
# --------------------------
def extract_t5_residue_features(sequence, target_device=device):
    clean_seq = sequence.strip()
    if not clean_seq:
        raise ValueError(f"Invalid Sequence: {sequence}")
    sequence_with_space = " ".join(list(clean_seq))
    inputs = tokenizer(sequence_with_space, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(next(t5_model.parameters()).device)
    with torch.no_grad():
        outputs = t5_model(** inputs)
    residue_feat = outputs.last_hidden_state[0][:len(clean_seq)]
    return residue_feat.to(target_device)

def aggregate_residue_features(residue_features):
    if residue_features.numel() == 0:
        raise ValueError("Residue feature is empty")
    return torch.mean(residue_features, dim=0)

def standardize_feature(seq_feature, mean, std):
    if mean.device != seq_feature.device:
        mean = mean.to(seq_feature.device)
    if std.device != seq_feature.device:
        std = std.to(seq_feature.device)
    return (seq_feature - mean) / (std + 1e-8)

# --------------------------
# 4. Caching Logic (Prioritize loading from cache; compute and save if no cache exists)
# --------------------------
def load_or_calculate_stats():
    """
    Core Logic:
    1. Check if cache file exists → If present, load directly
    2. If absent, load small batch of data → Compute statistics → Save to cache file
    """
   
    if os.path.exists(CACHE_STAT_PATH):
        try:
            with open(CACHE_STAT_PATH, 'rb') as f:
                stats = pickle.load(f)
            
            required_keys = ['pep_mean', 'pep_std', 'prot_mean', 'prot_std']
            if all(key in stats for key in required_keys):
                
                pep_mean = stats['pep_mean'].to(device)
                pep_std = stats['pep_std'].to(device)
                prot_mean = stats['prot_mean'].to(device)
                prot_std = stats['prot_std'].to(device)
                return pep_mean, pep_std, prot_mean, prot_std
            else:
                print(f"[FeatureExtract] Cache file is incomplete; recalculating.")
        except Exception as e:
            print(f"[FeatureExtract] Cache file corruption")
    
    
    
    # Check if small batch data exists (required only on first run)
    if not os.path.exists(SMALL_DATA_PATH):
        raise FileNotFoundError(f"The initial calculation requires a small batch of data, but the path does not exist:{SMALL_DATA_PATH}")
    
    
    small_pairs = []
    with open(SMALL_DATA_PATH, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or "\t" not in line:
                continue
            pep, prot = line.split("\t", 1)
            small_pairs.append((pep.strip(), prot.strip()))
            if i >= 99:  
                break
    if len(small_pairs) == 0:
        raise ValueError("No valid sequence pairs in small-batch data")
    
   
    pep_features = []
    prot_features = []
    for pep_seq, prot_seq in small_pairs:
        pep_residue = extract_t5_residue_features(pep_seq, target_device=torch.device("cpu"))  
        pep_feat = aggregate_residue_features(pep_residue)
        pep_features.append(pep_feat)
        
        prot_residue = extract_t5_residue_features(prot_seq, target_device=torch.device("cpu"))
        prot_feat = aggregate_residue_features(prot_residue)
        prot_features.append(prot_feat)
    
    pep_mean = torch.mean(torch.stack(pep_features), dim=0)
    pep_std = torch.std(torch.stack(pep_features), dim=0)
    prot_mean = torch.mean(torch.stack(prot_features), dim=0)
    prot_std = torch.std(torch.stack(prot_features), dim=0)
    
    
    stats_to_save = {
        'pep_mean': pep_mean,
        'pep_std': pep_std,
        'prot_mean': prot_mean,
        'prot_std': prot_std
    }
   
    os.makedirs(os.path.dirname(CACHE_STAT_PATH), exist_ok=True)
    with open(CACHE_STAT_PATH, 'wb') as f:
        pickle.dump(stats_to_save, f)
    
    
    
    return pep_mean.to(device), pep_std.to(device), prot_mean.to(device), prot_std.to(device)

# --------------------------
# 5. Execute cache logic
# --------------------------
pep_mean, pep_std, prot_mean, prot_std = load_or_calculate_stats()

# --------------------------
# 6. Feature Extraction Entry Point
# --------------------------
def extract_sequence_feature(sequence, is_peptide=True, target_device=device):
    residue_feat = extract_t5_residue_features(sequence, target_device=target_device)
    seq_feat = aggregate_residue_features(residue_feat)
    if is_peptide:
        return standardize_feature(seq_feat, pep_mean, pep_std).unsqueeze(0)
    else:
        return standardize_feature(seq_feat, prot_mean, prot_std).unsqueeze(0)

def parallel_feature_extract(pro_seq_list, pep_seq_list, uip, target_device=device):
    if not pro_seq_list or not pep_seq_list:
        raise ValueError("The sequence list cannot be empty.")
    pro_feat = extract_sequence_feature(pro_seq_list[0].strip(), is_peptide=False, target_device=target_device)
    pep_feat = extract_sequence_feature(pep_seq_list[0].strip(), is_peptide=True, target_device=target_device)
    return (pro_feat, None, None, None, None, None), (pep_feat, None, None, None, None, None)