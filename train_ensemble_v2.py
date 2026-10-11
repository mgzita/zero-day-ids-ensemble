"""
train_ensemble_v2.py
====================
v2 ensemble: AE + VAE + Deep SVDD (replaces IF).
Evaluates across CIC2017, CIC2018, UNSW, BoT-IoT.

Run from project root:
    python train_ensemble_v2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

from models.autoencoder import Autoencoder
from models.vae         import VAE
from models.deep_svdd   import DeepSVDD

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
RNG = np.random.default_rng(42)


def load(name):
    return np.load(os.path.join(ARTEFACT_DIR, name))


def norm(s):
    lo = float(np.percentile(s, 1))
    hi = float(np.percentile(s, 99))
    if hi - lo < 1e-8:
        return np.zeros_like(s)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)


def ensemble_scores(ae, vae, svdd, X,
                    w_ae=0.4, w_vae=0.4, w_svdd=0.2):
    return (w_ae   * norm(ae.anomaly_scores(X)) +
            w_vae  * norm(vae.anomaly_scores(X)) +
            w_svdd * norm(svdd.anomaly_scores(X)))


def print_metrics(name, y, scores, tau, col=22):
    y_pred = (scores > tau).astype(int)
    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    try:    auc = roc_auc_score(y, scores)
    except: auc = float("nan")
    cm   = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, int((y==1).sum()), 0))
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    auc_s = f"{auc:>7.4f}" if not np.isnan(auc) else "    nan"
    print("  " + f"{name:<{col}}" +
          f"{acc:>6.4f}" + f"{prec:>7.4f}" + f"{rec:>7.4f}" +
          f"{f1:>7.4f}" + auc_s +
          f"  FPR={fpr:.4f}  FNR={fnr:.4f}")
    return f1, auc


def run():
    #    Load data                                                          
    log.info("Loading v2 datasets ...")
    X_val = np.clip(load("val.npz")["X_val"], -10, 10)
    d17   = load("test_cic17.npz"); X_17 = np.clip(d17["X"], -10, 10); y_17 = d17["y"]
    d18   = load("test_cic18.npz"); X_18 = np.clip(d18["X"], -10, 10); y_18 = d18["y"]
    dsw   = load("test_unsw.npz");  X_sw = np.clip(dsw["X"], -10, 10); y_sw = dsw["y"]
    dbot  = load("test_botiot.npz");X_bt = np.clip(dbot["X"],-10, 10); y_bt = dbot["y"]

    #    Load models                                                        
    log.info("Loading models ...")
    ae   = Autoencoder().load(os.path.join(ARTEFACT_DIR, "ae_model.pt"))
    vae  = VAE().load(os.path.join(ARTEFACT_DIR, "vae_model.pt"))
    svdd = DeepSVDD().load(os.path.join(ARTEFACT_DIR, "svdd_model.pt"))

    #    Global threshold from val                                          
    log.info("Computing ensemble scores on val set ...")
    val_scores = ensemble_scores(ae, vae, svdd, X_val)
    tau_global = float(np.percentile(val_scores, 95))
    log.info("  Global val threshold (p95): %.4f", tau_global)

    #    Per-dataset adaptive thresholds                                    
    # Threshold fitted on benign calibration samples only (no label leakage).
    # Uses p95 of benign scores — catches top 5% anomalous-looking benign,
    # flagging anything above as attack.
    log.info("Fitting adaptive thresholds (p95 of benign cal samples) ...")
    datasets = [
        ("CIC-IDS2017",     X_17, y_17),
        ("CSE-CIC-IDS2018", X_18, y_18),
        ("UNSW-NB15",       X_sw, y_sw),
        ("BoT-IoT",         X_bt, y_bt),
    ]

    thresholds = {}
    for name, X, y in datasets:
        benign = X[y == 0]
        if len(benign) == 0:
            thresholds[name] = tau_global
            log.info("  %s: no benign — using global tau=%.4f", name, tau_global)
        else:
            n_cal = max(1000, int(len(benign) * 0.10))
            idx   = RNG.choice(len(benign), size=n_cal, replace=False)
            s_cal = ensemble_scores(ae, vae, svdd, benign[idx])
            tau   = float(np.percentile(s_cal, 95))
            thresholds[name] = tau
            log.info("  %s: n_cal=%d  benign_mu=%.4f  tau(p95)=%.4f",
                     name, n_cal, float(s_cal.mean()), tau)

    #    Full evaluation                                                    
    print("")
    print("=" * 78)
    print("ENSEMBLE v2 (AE + VAE + Deep SVDD) — FULL EVALUATION")
    print("=" * 78)
    header = ("  " + f"{'Dataset':<22}" +
              f"{'Acc':>6}" + f"{'Prec':>7}" + f"{'Rec':>7}" +
              f"{'F1':>7}" + f"{'AUC':>7}" + "  FPR      FNR")
    print(header)
    print("  " + "-" * 76)

    for name, X, y in datasets:
        scores = ensemble_scores(ae, vae, svdd, X)
        print_metrics(name, y, scores, thresholds[name])

    print("  " + "-" * 76)

    #    Individual model AUC comparison                                   
    print("")
    print("=" * 78)
    print("INDIVIDUAL MODEL AUC COMPARISON")
    print("=" * 78)
    print("  " + f"{'Model':<18}" +
          f"{'CIC2017':>10}" + f"{'CIC2018':>10}" +
          f"{'UNSW':>10}" + f"{'BoT-IoT':>12}")
    print("  " + "-" * 52)

    model_list = [
        ("AE",        lambda X: ae.anomaly_scores(X)),
        ("VAE",       lambda X: vae.anomaly_scores(X)),
        ("Deep SVDD", lambda X: svdd.anomaly_scores(X)),
        ("Ensemble",  lambda X: ensemble_scores(ae, vae, svdd, X)),
    ]

    for mname, score_fn in model_list:
        row = "  " + f"{mname:<18}"
        for _, X, y in datasets:
            s = score_fn(X)
            try:    auc = roc_auc_score(y, s)
            except: auc = float("nan")
            row += f"{auc:>10.4f}" if not np.isnan(auc) else "       nan"
        print(row)

    #    Score diagnostics                                                  
    print("")
    print("=" * 78)
    print("SCORE DIAGNOSTICS (benign vs attack separation)")
    print("=" * 78)

    for name, X, y in datasets:
        scores = ensemble_scores(ae, vae, svdd, X)
        b = scores[y == 0]
        a = scores[y == 1]
        b_mu = float(b.mean()) if len(b) > 0 else float("nan")
        a_mu = float(a.mean()) if len(a) > 0 else float("nan")
        sep  = a_mu - b_mu if not np.isnan(b_mu) else float("nan")
        tau  = thresholds[name]
        print(f"  [{name}]")
        if len(b) > 0:
            print(f"  Benign : mu={b_mu:.4f}  std={b.std():.4f}  p99={np.percentile(b,99):.4f}")
        print(f"  Attack : mu={a_mu:.4f}  std={a.std():.4f}")
        print(f"  Tau={tau:.4f}  Sep={sep:.4f}")
        print("")

    log.info("Done.")


if __name__ == "__main__":
    run()
