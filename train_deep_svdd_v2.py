"""
train_deep_svdd_v2.py
=====================
Trains Deep SVDD on combined benign traffic (CIC2017 + CIC2018 + UNSW).
Replaces Isolation Forest in the v2 ensemble.

Run from project root:
    python train_deep_svdd_v2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from models.deep_svdd import DeepSVDD

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTEFACT_DIR  = r"C:/MLProject/zero_day_project/artefacts_v2"
SVDD_PATH     = os.path.join(ARTEFACT_DIR, "svdd_model.pt")


def load(name):
    return np.load(os.path.join(ARTEFACT_DIR, name))


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
    svdd = DeepSVDD(
        input_dim    = input_dim,
        latent_dim   = 16,
        lr           = 1e-3,
        batch_size   = 256,
        max_epochs   = 100,
        patience     = 10,
        warmup_epochs= 10,
    )
    svdd.fit(X_train, X_val)

    #    REN stats                                                          
    log.info("Fitting REN score stats ...")
    svdd.fit_score_stats(X_train)

    #    Threshold                                                          
    svdd.fit_threshold(X_val)

    #    Evaluation                                                         
    print("")
    print("=" * 55)
    print("Deep SVDD v2 Evaluation")
    print("=" * 55)
    print("  " + f"{'Dataset':<22}" +
          f"{'Acc':>6}" + f"{'Prec':>7}" + f"{'Rec':>7}" +
          f"{'F1':>7}" + f"{'AUC':>7}")
    print("  " + "-" * 55)

    for name, X, y in [("CIC2017", X_17, y_17), ("CIC2018", X_18, y_18),
                        ("UNSW",    X_sw, y_sw),  ("BoT-IoT", X_bt, y_bt)]:
        scores = svdd.anomaly_scores(X)
        benign = scores[y == 0]
        tau    = float(np.percentile(benign, 95)) if len(benign) > 0 else svdd.threshold_
        y_pred = (scores > tau).astype(int)
        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        try:    auc = roc_auc_score(y, scores)
        except: auc = float("nan")
        auc_str = f"{auc:>7.4f}" if not np.isnan(auc) else "    nan"
        print("  " + f"{name:<22}" + f"{acc:>6.4f}" + f"{prec:>7.4f}" +
              f"{rec:>7.4f}" + f"{f1:>7.4f}" + auc_str)

    print("  " + "-" * 55)

    #    Save                                                               
    svdd.save(SVDD_PATH)
    log.info("Deep SVDD saved -> %s", SVDD_PATH)
    log.info("Next: python train_ensemble_v2.py")


if __name__ == "__main__":
    run()
