import os
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, f1_score

from models.autoencoder import Autoencoder

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"

def load(name):
    return np.load(os.path.join(ARTEFACT_DIR, name))

print("Loading AE...")
ae = Autoencoder().load(os.path.join(ARTEFACT_DIR, "ae_model.pt"))

d17  = load("test_cic17.npz")
d18  = load("test_cic18.npz")
dsw  = load("test_unsw.npz")
dbot = load("test_botiot.npz")

datasets = [
    ("CIC-IDS2017", np.clip(d17["X"], -10, 10), d17["y"]),
    ("CIC-IDS2018", np.clip(d18["X"], -10, 10), d18["y"]),
    ("UNSW-NB15",   np.clip(dsw["X"], -10, 10), dsw["y"]),
    ("BoT-IoT",     np.clip(dbot["X"], -10, 10), dbot["y"]),
]

print()
print("=" * 70)
print("AE ONLY (saved ae_model.pt)")
print("=" * 70)

for name, X, y in datasets:

    scores = ae.anomaly_scores(X)

    benign = scores[y == 0]

    if len(benign) > 0:
        tau = float(np.percentile(benign, 95))
    else:
        tau = float(np.percentile(scores, 95))

    y_pred = (scores > tau).astype(int)

    rec = recall_score(y, y_pred, zero_division=0)
    f1  = f1_score(y, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y, scores)
        auc_str = f"{auc:.4f}"
    except:
        auc_str = "—"

    print(
        f"{name:<18} "
        f"AUC={auc_str:<8} "
        f"Recall={rec:.4f} "
        f"F1={f1:.4f}"
    )