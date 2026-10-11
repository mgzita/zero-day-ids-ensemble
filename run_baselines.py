"""
run_baselines.py
================
Evaluates standard one-class anomaly detection baselines under the
same experimental protocol as the v2 ensemble.

Baselines:
  1. OC-SVM  (One-Class Support Vector Machine)
     - Standard kernel-based one-class detector
     - Trained on combined benign (subset — OC-SVM doesn't scale to 1.8M)
     - nu parameter controls the fraction of outliers expected

  2. HBOS  (Histogram-Based Outlier Score)
     - Fast, non-parametric univariate histogram anomaly detector
     - Trains a histogram per feature, scores = sum of log(1/bin_density)
     - Scales well to large datasets

  3. LOF   (Local Outlier Factor)
     - Density-based local anomaly detection
     - Compares local density of each sample to its neighbours
     - Included as a third standard baseline

All baselines:
  - Trained on same combined benign data as v2 (subsampled for OC-SVM/LOF)
  - Evaluated on same 4 test sets
  - Threshold: p95 of benign calibration scores (same as v2)

Run from project root:
    python run_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
RNG = np.random.default_rng(42)

# OC-SVM and LOF cannot scale to 1.8M — use this many training samples
OCSVM_TRAIN_SIZE = 10_000
LOF_TRAIN_SIZE   = 10_000


# =============================================================================
# HBOS (Histogram-Based Outlier Score)
# Fast, scales to large datasets
# =============================================================================

class HBOS:
    """
    Histogram-Based Outlier Score.
    Trains one histogram per feature. Anomaly score = sum of
    -log(density) across features. Higher = more anomalous.
    """
    def __init__(self, n_bins=50):
        self.n_bins = n_bins
        self.histograms_ = []

    def fit(self, X: np.ndarray) -> "HBOS":
        self.histograms_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            counts, edges = np.histogram(col, bins=self.n_bins, density=True)
            counts = np.maximum(counts, 1e-10)  # avoid log(0)
            self.histograms_.append((counts, edges))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(X), dtype=np.float32)
        for j, (counts, edges) in enumerate(self.histograms_):
            col    = X[:, j]
            idx    = np.searchsorted(edges[1:-1], col)  # bin index
            idx    = np.clip(idx, 0, len(counts) - 1)
            scores += -np.log(counts[idx])
        return scores


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def evaluate(scores, y, name, tau):
    y_pred = (scores > tau).astype(int)
    acc  = accuracy_score(y,  y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y,   y_pred, zero_division=0)
    f1   = f1_score(y,       y_pred, zero_division=0)
    try:    auc = roc_auc_score(y, scores)
    except: auc = float("nan")
    return dict(name=name, acc=acc, prec=prec, rec=rec, f1=f1, auc=auc)


def print_row(r, col=22):
    auc_s = f"{r['auc']:>7.4f}" if not np.isnan(r['auc']) else "    nan"
    print("  " + f"{r['name']:<{col}}" +
          f"{r['acc']:>6.4f}" + f"{r['prec']:>7.4f}" +
          f"{r['rec']:>7.4f}" + f"{r['f1']:>7.4f}" + auc_s)


# =============================================================================
# MAIN
# =============================================================================

def run():
    #    Load preprocessed data                                             
    log.info("Loading v2 preprocessed data ...")
    X_train = np.clip(np.load(os.path.join(ARTEFACT_DIR,"train.npz"))["X_train"],
                      -10, 10)
    X_val   = np.clip(np.load(os.path.join(ARTEFACT_DIR,"val.npz"))["X_val"],
                      -10, 10)
    d17  = np.load(os.path.join(ARTEFACT_DIR,"test_cic17.npz"))
    d18  = np.load(os.path.join(ARTEFACT_DIR,"test_cic18.npz"))
    dsw  = np.load(os.path.join(ARTEFACT_DIR,"test_unsw.npz"))
    dbot = np.load(os.path.join(ARTEFACT_DIR,"test_botiot.npz"))

    X_17=np.clip(d17["X"],-10,10);  y_17=d17["y"]
    X_18=np.clip(d18["X"],-10,10);  y_18=d18["y"]
    X_sw=np.clip(dsw["X"],-10,10);  y_sw=dsw["y"]
    X_bt=np.clip(dbot["X"],-10,10); y_bt=dbot["y"]

    log.info("  X_train: %s", X_train.shape)

    datasets = [
        ("CIC-IDS2017",     X_17, y_17),
        ("CSE-CIC-IDS2018", X_18, y_18),
        ("UNSW-NB15",       X_sw, y_sw),
        ("BoT-IoT",         X_bt, y_bt),
    ]

    header  = ("  " + f"{'Dataset':<22}" +
               f"{'Acc':>6}" + f"{'Prec':>7}" + f"{'Rec':>7}" +
               f"{'F1':>7}" + f"{'AUC':>7}")
    divider = "  " + "-" * 57

    all_results = []

    # =========================================================================
    # 1. HBOS
    # =========================================================================
    print("")
    print("=" * 62)
    print("Baseline 1: HBOS (Histogram-Based Outlier Score)")
    print("Training on full combined benign (1,800,000 rows) ...")
    print("=" * 62)

    hbos = HBOS(n_bins=50)
    hbos.fit(X_train)
    log.info("HBOS fitted.")

    print(header); print(divider)
    for ds_name, X, y in datasets:
        scores  = hbos.score_samples(X)
        benign  = scores[y==0]
        if len(benign) > 0:
            n_cal = max(1000, int(len(benign)*0.10))
            idx   = RNG.choice(len(benign), size=n_cal, replace=False)
            tau   = float(np.percentile(scores[y==0][idx], 95))
        else:
            tau = float(np.percentile(hbos.score_samples(X_val[:50_000]), 95))
        r = evaluate(scores, y, ds_name, tau)
        r["baseline"] = "HBOS"
        all_results.append(r)
        print_row(r)
    print(divider)

    # =========================================================================
    # 2. OC-SVM
    # =========================================================================
    print("")
    print("=" * 62)
    print("Baseline 2: OC-SVM (One-Class SVM, RBF kernel)")
    print(f"Training on {OCSVM_TRAIN_SIZE:,} benign samples (subset) ...")
    print("=" * 62)

    # OC-SVM cannot handle 1.8M samples — subsample
    idx_ocsvm = RNG.choice(len(X_train), size=OCSVM_TRAIN_SIZE, replace=False)
    X_ocsvm   = X_train[idx_ocsvm]

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X_ocsvm)
    log.info("OC-SVM fitted.")

    print(header); print(divider)
    for ds_name, X, y in datasets:
        # decision_function: higher = more normal, so negate for anomaly score
        scores  = -ocsvm.decision_function(X).astype(np.float32)
        benign  = scores[y==0]
        if len(benign) > 0:
            n_cal = max(1000, int(len(benign)*0.10))
            idx   = RNG.choice(len(benign), size=n_cal, replace=False)
            tau   = float(np.percentile(benign[idx], 95))
        else:
            tau = float(np.percentile(-ocsvm.decision_function(X_val[:10_000]), 95))
        r = evaluate(scores, y, ds_name, tau)
        r["baseline"] = "OC-SVM"
        all_results.append(r)
        print_row(r)
    print(divider)

    # =========================================================================
    # 3. LOF
    # =========================================================================
    print("")
    print("=" * 62)
    print("Baseline 3: LOF (Local Outlier Factor)")
    print(f"Training on {LOF_TRAIN_SIZE:,} benign samples (subset) ...")
    print("=" * 62)

    idx_lof = RNG.choice(len(X_train), size=LOF_TRAIN_SIZE, replace=False)
    X_lof   = X_train[idx_lof]

    # novelty=True allows predict on new data
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    lof.fit(X_lof)
    log.info("LOF fitted.")

    print(header); print(divider)
    for ds_name, X, y in datasets:
        scores  = -lof.decision_function(X).astype(np.float32)
        benign  = scores[y==0]
        if len(benign) > 0:
            n_cal = max(1000, int(len(benign)*0.10))
            idx   = RNG.choice(len(benign), size=n_cal, replace=False)
            tau   = float(np.percentile(benign[idx], 95))
        else:
            tau = float(np.percentile(-lof.decision_function(X_val[:5_000]), 95))
        r = evaluate(scores, y, ds_name, tau)
        r["baseline"] = "LOF"
        all_results.append(r)
        print_row(r)
    print(divider)

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print("")
    print("=" * 72)
    print("BASELINE COMPARISON — AUC Summary")
    print("v2 Ensemble results included for reference")
    print("=" * 72)

    # v2 ensemble AUC from previous results
    v2_auc = {
        "CIC-IDS2017":     0.9370,
        "CSE-CIC-IDS2018": 0.9564,
        "UNSW-NB15":       0.8657,
        "BoT-IoT":         float("nan"),
    }

    print("  " + f"{'Method':<20}" +
          f"{'CIC2017':>10}" + f"{'CIC2018':>10}" +
          f"{'UNSW':>10}" + f"{'BoT-IoT':>12}")
    print("  " + "-" * 64)

    df = pd.DataFrame(all_results)
    for baseline in ["HBOS", "OC-SVM", "LOF"]:
        sub = df[df["baseline"]==baseline]
        row = "  " + f"{baseline:<20}"
        for ds_name, _, _ in datasets:
            r = sub[sub["name"]==ds_name]
            if len(r) > 0:
                auc = r.iloc[0]["auc"]
                row += f"{auc:>10.4f}" if not np.isnan(auc) else "       nan"
            else:
                row += "       N/A"
        print(row)

    # v2 ensemble row
    row = "  " + f"{'v2 Ensemble (ours)':<20}"
    for ds_name, _, _ in datasets:
        auc = v2_auc.get(ds_name, float("nan"))
        row += f"{auc:>10.4f}" if not np.isnan(auc) else "       nan"
    print(row)
    print("  " + "-" * 64)

    # Save
    csv_path = os.path.join(ARTEFACT_DIR, "baseline_results.csv")
    df.to_csv(csv_path, index=False)
    log.info("Results saved -> %s", csv_path)
    print(f"\nResults saved -> {csv_path}")
    print("Done.")


if __name__ == "__main__":
    run()
