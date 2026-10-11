"""
train_autoencoder_v2.py
=======================
Trains AE on combined benign traffic (CIC2017 + CIC2018 + UNSW).
Loads from artefacts_v2/. Saves to artefacts_v2/.
v1 files untouched.

Run from project root:
    python train_autoencoder_v2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from models.autoencoder import Autoencoder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
AE_PATH      = os.path.join(ARTEFACT_DIR, "ae_model.pt")


def load(name):
    d = np.load(os.path.join(ARTEFACT_DIR, name))
    return d


def run():
    log.info("Loading v2 datasets ...")
    X_train = np.clip(load("train.npz")["X_train"], -10, 10)
    X_val   = np.clip(load("val.npz")["X_val"],     -10, 10)
    d17     = load("test_cic17.npz");  X_17 = np.clip(d17["X"], -10, 10); y_17 = d17["y"]
    d18     = load("test_cic18.npz");  X_18 = np.clip(d18["X"], -10, 10); y_18 = d18["y"]
    dsw     = load("test_unsw.npz");   X_sw = np.clip(dsw["X"], -10, 10); y_sw = dsw["y"]
    dbot    = load("test_botiot.npz"); X_bt = np.clip(dbot["X"],-10, 10); y_bt = dbot["y"]

    input_dim = X_train.shape[1]
    log.info("  Input dim: %d | X_train: %s | X_val: %s",
             input_dim, X_train.shape, X_val.shape)

    #    Train                                                              
    ae = Autoencoder(input_dim=input_dim)
    ae.fit(X_train, X_val)

    #    REN + LSAS                                                         
    log.info("Fitting REN score stats ...")
    ae.fit_score_stats(X_train)
    log.info("Fitting LSAS latent stats ...")
    ae.fit_latent_stats(X_train)

    #    Threshold                                                          
    ae.fit_threshold(X_val)

    #    Quick eval                                                         
    print("")
    print("=" * 55)
    print("AE v2 Evaluation")
    print("=" * 55)
    header = "  " + f"{'Dataset':<22}" + f"{'Acc':>6}" + f"{'Prec':>7}" + f"{'Rec':>7}" + f"{'F1':>7}" + f"{'AUC':>7}"
    print(header)
    print("  " + "-" * 55)

    for name, X, y in [("CIC2017", X_17, y_17), ("CIC2018", X_18, y_18),
                        ("UNSW",    X_sw, y_sw),  ("BoT-IoT", X_bt, y_bt)]:
        scores = ae.anomaly_scores(X)
        # Adaptive p95 threshold from benign (if any benign exist)
        benign = scores[y == 0]
        if len(benign) > 0:
            tau = float(np.percentile(benign, 95))
        else:
            tau = ae.threshold_   # BoT-IoT is all attack
        y_pred = (scores > tau).astype(int)
        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        try:    auc = roc_auc_score(y, scores)
        except: auc = 0.0
        print("  " + f"{name:<22}" + f"{acc:>6.4f}" + f"{prec:>7.4f}" +
              f"{rec:>7.4f}" + f"{f1:>7.4f}" + f"{auc:>7.4f}")

    print("  " + "-" * 55)

    #    Save                                                               
    ae.save(AE_PATH)
    log.info("AE v2 saved -> %s", AE_PATH)
    log.info("Next: python train_vae_v2.py")


if __name__ == "__main__":
    run()
