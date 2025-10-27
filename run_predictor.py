# -*- coding: utf-8 -*-
import os
import torch
import pickle
import argparse
from conf import MODEL_FOLD, USER_FOLD
from FeatureExtract import parallel_feature_extract
from model import load_ppi_model
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def build_graph(pep_feat, pro_feat):
   
    pep_feat = pep_feat.to(device)
    pro_feat = pro_feat.to(device)
    
    x = torch.cat([pep_feat, pro_feat], dim=0)  
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)  
    batch = torch.tensor([0, 0], dtype=torch.long, device=device) 
    return Data(x=x, edge_index=edge_index, batch=batch)


def predict_interaction(pep_feat, pro_feat, model):
   
    graph = build_graph(pep_feat, pro_feat)
    with torch.no_grad():  
        logits = model(graph)
        prob = torch.sigmoid(logits).item()  
    label = 1 if prob >= 0.5 else 0  
    return prob, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-uip", required=True, help="User Directory (Stores FASTA files and results)")
    parser.add_argument("-do_shap", default="0", help="SHAP(0=no,1=yes)")
    args = parser.parse_args()
    uip = args.uip  # User directory path (e.g., /home/www/PepLM-GNN/webserver/temp/user/xxx)

    # 1. Read the FASTA sequence entered by the user
    pep_seq_list = []
    with open(os.path.join(uip, 'Peptide_Seq.fasta'), 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):  
                pep_seq_list.append(line)
                break  
    
    pro_seq_list = []
    with open(os.path.join(uip, 'Protein_Seq.fasta'), 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                pro_seq_list.append(line)
                break  

    # 2. Extract T5 features
   
    protein_features, peptide_features = parallel_feature_extract(
        pro_seq_list=pro_seq_list,
        pep_seq_list=pep_seq_list,
        uip=uip,
        target_device=device  
    )
    
   
    pep_feat = peptide_features[0]  # (1, 1024)
    pro_feat = protein_features[0]  # (1, 1024)
    
    # 3. Load the model and make predictions.
    
    model = load_ppi_model(
        model_path=os.path.join(MODEL_FOLD, 'best_model.pth'),
        device=device  
    )
    model.eval()  
    
    prob, label = predict_interaction(pep_feat, pro_feat, model)
    

    # 4. save
    result = {
        "peptide": pep_seq_list[0] if pep_seq_list else "",  
        "protein": pro_seq_list[0] if pro_seq_list else "",
        "interaction_probability": round(prob, 4),  
        "interaction_result": "Interact" if label == 1 else "No Interaction"  
    }

    
    result_path = os.path.join(uip, 'result.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
   