"""
export_scaler_json.py
=====================
Exports the fitted IQR scaler parameters to scaler.json
for inclusion in the supplementary code repository.

This addresses Review Point 7.3:
  "The IQR scaler is fitted on combined benign training data.
   Without the Q1, Q2, Q3 values per feature, the preprocessing
   cannot be reproduced."

Loads from:
  artefacts_v2/iqr_scaler.npz
  artefacts_v2/selected_features.csv

Saves to:
  scaler.json  (upload to GitHub repository)

Run from project root:
    python export_scaler_json.py
"""

import os
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = r"C:/MLProject/zero_day_project"
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "artefacts_v2")

def run():
    #    Load scaler                                                        
    scaler_path = os.path.join(ARTEFACT_DIR, "iqr_scaler.npz")
    print(f"Loading scaler from: {scaler_path}")
    d = np.load(scaler_path, allow_pickle=True)
    print(f"  Keys: {list(d.keys())}")

    q1  = d["q1"].tolist()
    q2  = d["q2"].tolist()
    q3  = d["q3"].tolist()
    iqr = d["iqr"].tolist()

    #    Load feature names                                                 
    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    features  = pd.read_csv(feat_path, header=None)[0].tolist()
    features  = [f for f in features
                 if not str(f).strip().lstrip("-").isdigit()]
    print(f"  Features: {len(features)}")

    #    Build JSON                                                         
    scaler_json = {
        "description": (
            "IQR scaler fitted on combined benign traffic from "
            "CIC-IDS2017, CSE-CIC-IDS2018, and UNSW-NB15 "
            "(600,000 samples per dataset, 1,800,000 total). "
            "Transformation: z = (x - Q2) / (Q3 - Q1), "
            "clipped to [-10, 10]."
        ),
        "n_features":   len(features),
        "feature_names": features,
        "clip_range":   [-10, 10],
        "training_data": {
            "datasets": [
                "CIC-IDS2017",
                "CSE-CIC-IDS2018",
                "UNSW-NB15"
            ],
            "samples_per_dataset": 600000,
            "total_samples": 1800000,
            "split": "80% train / 20% val",
            "label": "benign only"
        },
        "parameters": {}
    }

    for i, feat in enumerate(features):
        scaler_json["parameters"][feat] = {
            "Q1":  round(float(q1[i]),  8),
            "Q2":  round(float(q2[i]),  8),
            "Q3":  round(float(q3[i]),  8),
            "IQR": round(float(iqr[i]), 8),
        }

    #    Save to project root                                               
    out_path = os.path.join(PROJECT_ROOT, "scaler.json")
    with open(out_path, "w") as f:
        json.dump(scaler_json, f, indent=2)
    print(f"\nSaved -> {out_path}")

    #    Print summary                                                      
    print("\n" + "=" * 65)
    print("SCALER PARAMETERS SUMMARY")
    print("=" * 65)
    print(f"  {'Feature':<35} {'Q1':>10} {'Q2':>10} "
          f"{'Q3':>10} {'IQR':>10}")
    print("  " + "-" * 67)
    for feat in features:
        p = scaler_json["parameters"][feat]
        print(f"  {feat:<35} {p['Q1']:>10.4f} {p['Q2']:>10.4f} "
              f"{p['Q3']:>10.4f} {p['IQR']:>10.4f}")

    print("\n" + "=" * 65)
    print("LATEX TEXT for Section 5 (IQR normalisation bullet):")
    print("=" * 65)
    print("""
The fitted scaler parameters ($Q_1$, $Q_2$, $Q_3$ per 
feature) are provided in the supplementary repository 
as \\texttt{scaler.json} to ensure full reproducibility 
of the preprocessing pipeline.
""")

    print("Upload scaler.json to your GitHub repository.")
    print("Done.")


if __name__ == "__main__":
    run()
