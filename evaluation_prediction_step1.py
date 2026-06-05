# Install
!pip install pandas

# Imports
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# PATHS TO ADJUST
# =========================

prediction_file = Path(r"runs/predict/predictions.csv")  # ADJUST THIS PATH TO YOUR PREDICTION FILE
labels_file = Path(r"test_set_annotation/test_set_all_deep.csv") 

# Create output path from the input prediction file
output_file = prediction_file.with_name(prediction_file.stem + "_results.csv")

# =========================
# READ INPUT FILES
# =========================

labels = pd.read_csv(labels_file)
predictions = pd.read_csv(prediction_file)

labels = labels[labels["subsample"] == "yes"]

print("Number of rows after filtering:", len(labels))
print("Prediction file:", prediction_file)
print("Output file:", output_file)


# =========================
# CREATE FUNCTION
# =========================

def compare_csv(labels, predictions, output_path):
    # Replace "_REC_" and "_REC " with "_"
    labels["Fichier"] = labels["Fichier"].str.replace("_REC_", "_").str.replace("_REC ", "_")
    predictions["filename"] = predictions["filename"].str.replace(
        "00033565_20250216T150000+0100_REC [-03.94513+011.34217]_loc1200-1251.09065759637s.wav",
        "00033565_20250216T150000+0100_REC_[-03.94513+011.34217]_loc1200-1260s.wav"
    )
    predictions["filename"] = predictions["filename"].str.replace("_REC_", "_").str.replace("_REC ", "_")
    labels["Fichier"] = labels["Fichier"].str.replace("_Rec_", "_").str.replace("_Rec ", "_")
    predictions["filename"] = predictions["filename"].str.replace("_Rec_", "_").str.replace("_Rec ", "_")

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    cpt = 0

    # Add columns for pred_1 and pred_2 in labels
    labels["start_time_pred_1"] = np.nan
    labels["id_pred_1"] = np.nan
    labels["probs_manatee_pred_1"] = np.nan
    labels["probs_no_manatee_pred_1"] = np.nan
    labels["start_time_pred_2"] = np.nan
    labels["id_pred_2"] = np.nan
    labels["probs_manatee_pred_2"] = np.nan
    labels["probs_no_manatee_pred_2"] = np.nan

    for i, row in labels.iterrows():
        file = row["Fichier"] + ".wav"
        loc = row["Localisation"]
        label = 1 if row["Manual_ID"] == "manatee" else 0
        sub_preds = predictions[predictions["filename"] == file]

        if len(sub_preds) == 0:
            print(f"No predictions for {file}")
            cpt += 1
            continue

        starts = sub_preds["start_time"].values

        squared_diffs = (starts + 5 - (loc + 0.1)) ** 2
        min_index = np.argsort(squared_diffs)[:2]

        best_start_1 = starts[min_index[0]]
        best_start_2 = starts[min_index[1]]

        pred_1_row = sub_preds[sub_preds["start_time"] == best_start_1].iloc[0]
        pred_2_row = sub_preds[sub_preds["start_time"] == best_start_2].iloc[0]

        if best_start_1 <= loc + 0.1 <= best_start_1 + 10:
            labels.at[i, "start_time_pred_1"] = pred_1_row["start_time"]
            labels.at[i, "id_pred_1"] = pred_1_row["id_preds"]
            labels.at[i, "probs_manatee_pred_1"] = pred_1_row["probs_manatee"]
            labels.at[i, "probs_no_manatee_pred_1"] = pred_1_row["probs_no_manatee"]

        if best_start_2 <= loc + 0.1 <= best_start_2 + 10:
            labels.at[i, "start_time_pred_2"] = pred_2_row["start_time"]
            labels.at[i, "id_pred_2"] = pred_2_row["id_preds"]
            labels.at[i, "probs_manatee_pred_2"] = pred_2_row["probs_manatee"]
            labels.at[i, "probs_no_manatee_pred_2"] = pred_2_row["probs_no_manatee"]

    labels.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")


# =========================
# RUN ON A SINGLE FILE
# =========================

compare_csv(labels.copy(), predictions, output_file)
