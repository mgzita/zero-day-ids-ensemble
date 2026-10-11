"""
train_ensemble.py
=================
Builds and evaluates the 4-signal ensemble detector:
    AE reconstruction (REN) + VAE reconstruction (REN)
    + VAE latent distance (LSAS) + Isolation Forest

Weights informed by per-dataset AUC analysis:
    0.30 * AE_recon + 0.35 * VAE_recon + 0.25 * VAE_latent + 0.10 * IF

Run from project root:
    python train_ensemble.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              confusion_matrix, roc_auc_score)

from preprocessing.pipeline import PreprocessingPipeline
from models.autoencoder      import Autoencoder
from models.vae              import VAE
from models.isolation_forest import IsolationForestModel
from models.ensemble         import EnsembleDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)

ARTEFACT_DIR  = r"C:/MLProject/zero_day_project/artefacts"
AE_PATH       = os.path.join(ARTEFACT_DIR, "ae_model.pt")
VAE_PATH      = os.path.join(ARTEFACT_DIR, "vae_model.pt")
IF_PATH       = os.path.join(ARTEFACT_DIR, "if_model.pkl")
ENSEMBLE_PATH = os.path.join(ARTEFACT_DIR, "ensemble.pkl")


def run():
    #    Load datasets                                                      
    print("")
    print("=" * 60)
    print("Loading preprocessed datasets ...")
    print("=" * 60)

    data    = PreprocessingPipeline.load_splits(ARTEFACT_DIR)
    X_train = np.clip(data["X_train"], -10, 10)
    X_val   = np.clip(data["X_val"],   -10, 10)
    X_test  = np.clip(data["X_test"],  -10, 10)
    y_test  = data["y_test"]
    X_cic18 = np.clip(data["X_cic18"], -10, 10)
    y_cic18 = data["y_cic18"]
    X_unsw  = np.clip(data["X_unsw"],  -10, 10)
    y_unsw  = data["y_unsw"]

    print("  X_train : " + str(X_train.shape))
    print("  X_val   : " + str(X_val.shape))
    print("  X_test  : " + str(X_test.shape))
    print("  X_cic18 : " + str(X_cic18.shape))
    print("  X_unsw  : " + str(X_unsw.shape))

    #    Load individual models                                             
    print("")
    print("=" * 60)
    print("Loading trained models ...")
    print("=" * 60)

    ae      = Autoencoder().load(AE_PATH)
    vae     = VAE().load(VAE_PATH)
    iforest = IsolationForestModel().load(IF_PATH)

    # Verify LSAS stats are loaded
    if vae.latent_centroid_ is None:
        print("  WARNING: VAE latent stats missing — run refit_score_stats.py first")
    else:
        print("  VAE LSAS centroid loaded  ")
    print("  AE, VAE, IF loaded")

    #    Build ensemble                                                     
    print("")
    print("=" * 60)
    print("Building 4-signal ensemble ...")
    print("  Signals: AE_recon(0.30) + VAE_recon(0.35) + VAE_latent(0.25) + IF(0.10)")
    print("=" * 60)

    ensemble = EnsembleDetector(ae, vae, iforest, n_sigma=3.0)
    ensemble.fit_normaliser(X_val)
    ensemble.fit_threshold(X_val)

    #    Per-dataset adaptive thresholds                                    
    print("")
    print("=" * 60)
    print("Fitting per-dataset adaptive thresholds (10% benign samples) ...")
    print("=" * 60)

    datasets = [
        ("CIC-IDS2017 (test)", X_test,  y_test),
        ("CSE-CIC-IDS2018",    X_cic18, y_cic18),
        ("UNSW-NB15",          X_unsw,  y_unsw),
    ]

    dataset_thresholds = {}
    for name, X, y in datasets:
        X_benign  = X[y == 0]
        n_cal     = max(1000, int(len(X_benign) * 0.10))
        rng       = np.random.default_rng(42)
        idx       = rng.choice(len(X_benign), size=n_cal, replace=False)
        X_cal     = X_benign[idx]
        scores    = ensemble.ensemble_scores(X_cal)
        mu        = float(scores.mean())
        sigma     = float(scores.std())
        tau_sigma = mu + 3.0 * sigma
        tau_p95   = float(np.percentile(scores, 95))
        tau       = min(tau_sigma, tau_p95)
        dataset_thresholds[name] = tau
        print("  " + name + ": n_cal=" + str(n_cal) +
              "  mu=" + f"{mu:.4f}" +
              "  sigma=" + f"{sigma:.4f}" +
              "  tau_sigma=" + f"{tau_sigma:.4f}" +
              "  tau_p95=" + f"{tau_p95:.4f}" +
              "  chosen=" + f"{tau:.4f}")

    #    Full evaluation                                                    
    print("")
    print("=" * 60)
    print("FULL EVALUATION (adaptive threshold)")
    print("=" * 60)
    print("")
    print("  " + f"{'Dataset':<25}" +
          f"{'Acc':>6}" + f"{'Prec':>7}" +
          f"{'Rec':>7}" + f"{'F1':>7}" + f"{'AUC':>7}")
    print("  " + "-" * 55)

    results = {}
    for name, X, y in datasets:
        tau    = dataset_thresholds[name]
        scores = ensemble.ensemble_scores(X)
        y_pred = (scores > tau).astype(int)
        m      = _compute_metrics(y, y_pred, scores)
        results[name] = m
        print("  " + f"{name:<25}" +
              f"{m['accuracy']:>6.4f}" +
              f"{m['precision']:>7.4f}" +
              f"{m['recall']:>7.4f}" +
              f"{m['f1']:>7.4f}" +
              f"{m['auc']:>7.4f}")

    print("  " + "-" * 55)

    #    Confusion matrices                                                 
    print("")
    print("=" * 60)
    print("CONFUSION MATRICES")
    print("=" * 60)

    for name, X, y in datasets:
        tau    = dataset_thresholds[name]
        scores = ensemble.ensemble_scores(X)
        y_pred = (scores > tau).astype(int)
        cm     = confusion_matrix(y, y_pred)
        n_b    = int((y == 0).sum())
        n_a    = int((y == 1).sum())
        fpr    = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0
        fnr    = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0]+cm[1,1]) > 0 else 0
        print("")
        print("  [" + name + "]")
        print("  Benign=" + f"{n_b:,}" + "  Attack=" + f"{n_a:,}")
        print("  TN=" + f"{cm[0,0]:,}" + "  FP=" + f"{cm[0,1]:,}")
        print("  FN=" + f"{cm[1,0]:,}" + "  TP=" + f"{cm[1,1]:,}")
        print("  FPR=" + f"{fpr:.4f}" + "  FNR=" + f"{fnr:.4f}")

    #    Individual model comparison (CIC2017)                              
    print("")
    print("=" * 60)
    print("INDIVIDUAL MODEL COMPARISON (CIC2017 test)")
    print("=" * 60)

    cic17_tau  = dataset_thresholds["CIC-IDS2017 (test)"]
    ens_scores = ensemble.ensemble_scores(X_test)

    # AE recon only (p95 adaptive)
    ae_scores  = ae.anomaly_scores(X_test)
    ae_tau     = float(np.percentile(ae_scores[y_test == 0], 95))
    vae_scores = vae.anomaly_scores(X_test)
    vae_tau    = float(np.percentile(vae_scores[y_test == 0], 95))
    if_scores  = iforest.anomaly_scores(X_test)
    if_tau     = float(np.percentile(if_scores[y_test == 0], 95))

    models_cmp = [
        ("AE (recon)",     (ae_scores  > ae_tau).astype(int)),
        ("VAE (recon)",    (vae_scores > vae_tau).astype(int)),
        ("IF",             (if_scores  > if_tau).astype(int)),
        ("Ensemble (4-sig)",(ens_scores > cic17_tau).astype(int)),
    ]

    print("")
    print("  " + f"{'Model':<18}" +
          f"{'Acc':>6}" + f"{'Prec':>7}" +
          f"{'Rec':>7}" + f"{'F1':>7}")
    print("  " + "-" * 45)

    for mname, y_pred in models_cmp:
        acc  = accuracy_score(y_test,  y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test,    y_pred, zero_division=0)
        f1   = f1_score(y_test,        y_pred, zero_division=0)
        print("  " + f"{mname:<18}" +
              f"{acc:>6.4f}" + f"{prec:>7.4f}" +
              f"{rec:>7.4f}" + f"{f1:>7.4f}")

    #    Score diagnostics                                                  
    print("")
    print("=" * 60)
    print("SCORE DIAGNOSTICS (benign vs attack separation)")
    print("=" * 60)

    for name, X, y in datasets:
        scores   = ensemble.ensemble_scores(X)
        b_scores = scores[y == 0]
        a_scores = scores[y == 1]
        b_mu  = float(b_scores.mean())
        b_std = float(b_scores.std())
        b_p99 = float(np.percentile(b_scores, 99))
        a_mu  = float(a_scores.mean()) if len(a_scores) > 0 else 0.0
        a_p1  = float(np.percentile(a_scores, 1)) if len(a_scores) > 0 else 0.0
        sep   = a_mu - b_mu
        tau   = dataset_thresholds[name]
        try:
            auc = roc_auc_score(y, scores)
        except Exception:
            auc = 0.0
        print("")
        print("  [" + name + "]")
        print("  Benign : mu=" + f"{b_mu:.4f}" +
              "  std=" + f"{b_std:.4f}" + "  p99=" + f"{b_p99:.4f}")
        print("  Attack : mu=" + f"{a_mu:.4f}" + "  p1=" + f"{a_p1:.4f}")
        print("  Tau=" + f"{tau:.4f}" +
              "  Sep=" + f"{sep:.4f}" +
              "  AUC=" + f"{auc:.4f}")

    #    Save                                                               
    ensemble.save(ENSEMBLE_PATH)
    print("")
    print("  Ensemble saved -> " + ENSEMBLE_PATH)
    print("All done.")


def _compute_metrics(y_true, y_pred, scores):
    acc  = accuracy_score(y_true,  y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true,    y_pred, zero_division=0)
    f1   = f1_score(y_true,        y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = 0.0
    return dict(accuracy=acc, precision=prec,
                recall=rec, f1=f1, auc=auc)


if __name__ == "__main__":
    run()
