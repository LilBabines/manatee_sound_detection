# %%
! pip install pandas

# %%

import pandas as pd
import numpy as np
import os
import glob
import re
from pathlib import Path
#localisation = pd.read_csv("C:/Users/dubus/Documents/1_Son synthèse/training/training_juillet24/training_set_all_deep.csv") 
#print(localisation)


prediction_dir = Path(r"runs/predict/") #ADJUST THIS PATH TO YOUR PREDICTION FOLDER
prediction_files = sorted(str(p) for p in prediction_dir.rglob("*predictions*.csv"))
labels = pd.read_csv("test_set_annotation/test_set_all_deep.csv") #ADJUST THIS PATH TO YOUR test_set_annotation FOLDER

labels = labels[labels['subsample'] == 'yes']

print("Nombre de lignes après filtrage :", len(labels))
for p in prediction_files:
    print(p) 


# %%
def compare_csv(labels,predictions,output_path):
    #replace "_REC_" and "_REC " with "_"
    labels['Fichier'] = labels['Fichier'].str.replace("_REC_", "_").str.replace("_REC ", "_")
    predictions['filename'] = predictions['filename'].str.replace("00033565_20250216T150000+0100_REC [-03.94513+011.34217]_loc1200-1251.09065759637s.wav", "00033565_20250216T150000+0100_REC_[-03.94513+011.34217]_loc1200-1260s.wav")
    predictions['filename'] = predictions['filename'].str.replace("_REC_", "_").str.replace("_REC ", "_")
    labels['Fichier'] = labels['Fichier'].str.replace("_Rec_", "_").str.replace("_Rec ", "_")
    predictions['filename'] = predictions['filename'].str.replace("_Rec_", "_").str.replace("_Rec ", "_")
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    cpt = 0

    # Ajout des colonnes pour pred_1 et pred_2 dans labels
    labels['start_time_pred_1'] = np.nan
    labels['id_pred_1'] = np.nan
    labels['probs_manatee_pred_1'] = np.nan
    labels['probs_no_manatee_pred_1'] = np.nan
    labels['start_time_pred_2'] = np.nan
    labels['id_pred_2'] = np.nan
    labels['probs_manatee_pred_2'] = np.nan
    labels['probs_no_manatee_pred_2'] = np.nan

    for i,row in labels.iterrows():
        file = row['Fichier']+".wav"
        loc = row['Localisation']
        label = 1 if row['Manual_ID']=='manatee' else 0
        sub_preds = predictions[predictions['filename'] == file]
        if len(sub_preds) == 0:
            print(f"No predictions for {file}")
            cpt += 1
            continue
        starts = sub_preds['start_time'].values
        # print(loc)
        # print(starts)
        # print()
        # print(min(starts-max(loc-5,0)))
        squared_diffs = (starts+5 - (loc+0.1)) ** 2
        min_index = np.argsort(squared_diffs)[:2]
        best_start_1 = starts[min_index[0]]
        best_start_2 = starts[min_index[1]]
        pred_1_row = sub_preds[sub_preds['start_time'] == best_start_1].iloc[0]
        pred_2_row = sub_preds[sub_preds['start_time'] == best_start_2].iloc[0]
        if best_start_1 <= loc+0.1 <= best_start_1 + 10 :
            labels.at[i, 'start_time_pred_1'] = pred_1_row['start_time']
            labels.at[i, 'id_pred_1'] = pred_1_row['id_preds']
            labels.at[i, 'probs_manatee_pred_1'] = pred_1_row['probs_manatee']
            labels.at[i, 'probs_no_manatee_pred_1'] = pred_1_row['probs_no_manatee']
        if best_start_2 <= loc+0.1 <= best_start_2 + 10 : 
            labels.at[i, 'start_time_pred_2'] = pred_2_row['start_time']
            labels.at[i, 'id_pred_2'] = pred_2_row['id_preds']
            labels.at[i, 'probs_manatee_pred_2'] = pred_2_row['probs_manatee']
            labels.at[i, 'probs_no_manatee_pred_2'] = pred_2_row['probs_no_manatee']

    
    labels.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

# %%
for pred_file in prediction_files:
    print(f"\n=== Processing {Path(pred_file).name} ===")
    predictions = pd.read_csv(pred_file)
    
    # Récupère le sous-dossier et le nom du fichier
    pred_path = Path(pred_file)
    subfolder = pred_path.parent.name  # nom du sous-dossier
    pred_name = pred_path.stem  # nom sans extension
    
    # Crée le chemin de sortie avec le sous-dossier
    output_path = f"{prediction_dir}/{pred_name}_results.csv"
    
    compare_csv(labels.copy(), predictions, output_path)

