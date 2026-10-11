"""
run_dedup_seed42_save.py
========================
Retrains the full ensemble on deduplicated data for seed 42 only,
saves models, and computes full confusion matrix metrics.

Run from project root:
    python run_dedup_seed42_save.py

Expected time: ~4 hours
Models saved to: artefacts_v2/dedup/seed42/
Results saved to: artefacts_v2/dedup/seed42_confusion.csv
"""
import sys, os, gc, logging
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from models.autoencoder import Autoencoder
from models.vae import VAE
from models.deep_svdd import DeepSVDD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
DEDUP_DIR    = os.path.join(ARTEFACT_DIR, "dedup")
SEED42_DIR   = os.path.join(DEDUP_DIR, "seed42")
OUT_PATH     = os.path.join(DEDUP_DIR, "seed42_confusion.csv")
SEED         = 42
os.makedirs(SEED42_DIR, exist_ok=True)


def norm_minmax(s):
    lo = float(np.percentile(s, 1))
    hi = float(np.percentile(s, 99))
    if hi - lo < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def get_ae_scores(ae, X):
    ae.model_.eval()
    device = next(ae.model_.parameters()).device
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 10000):
            batch = torch.FloatTensor(X[i:i+10000]).to(device)
            recon = ae.model_(batch)
            scores.append(((batch - recon)**2).mean(dim=1).cpu().numpy())
    return np.concatenate(scores)


def get_vae_scores(vae, X):
    vae.model_.eval()
    device = next(vae.model_.parameters()).device
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 10000):
            batch = torch.FloatTensor(X[i:i+10000]).to(device)
            x_recon, mu, log_var = vae.model_(batch)
            recon = ((batch - x_recon)**2).mean(dim=1)
            kl = -0.5*(1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
            scores.append((recon + kl).cpu().numpy())
    return np.concatenate(scores)


def get_svdd_scores(svdd, X):
    svdd.model_.eval()
    device = next(svdd.model_.parameters()).device
    centre = svdd.centre_.to(device)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 10000):
            batch = torch.FloatTensor(X[i:i+10000]).to(device)
            z = svdd.model_(batch)
            scores.append(((z - centre)**2).sum(dim=1).cpu().numpy())
    return np.concatenate(scores)


def ensemble_scores(ae, vae, svdd, X):
    return (0.4 * norm_minmax(get_ae_scores(ae, X)) +
            0.4 * norm_minmax(get_vae_scores(vae, X)) +
            0.2 * norm_minmax(get_svdd_scores(svdd, X)))


def run():
    log.info("Loading deduplicated data...")
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    d_val   = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_train = d_train["X_train"]
    X_val   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    log.info("Train: %s  Val: %s", X_train.shape, X_val.shape)

    # Check if models already saved
    ae_path   = os.path.join(SEED42_DIR, "ae.pt")
    vae_path  = os.path.join(SEED42_DIR, "vae.pt")
    svdd_path = os.path.join(SEED42_DIR, "svdd.pt")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # AE
    ae = Autoencoder(input_dim=18)
    if os.path.exists(ae_path):
        log.info("Loading AE from checkpoint...")
        ae.load(ae_path)
    else:
        log.info("Training AE (seed 42)...")
        ae.fit(X_train, X_val)
        ae.save(ae_path)
        log.info("AE saved -> %s", ae_path)

    # VAE
    vae = VAE(input_dim=18)
    if os.path.exists(vae_path):
        log.info("Loading VAE from checkpoint...")
        vae.load(vae_path)
    else:
        log.info("Training VAE (seed 42)...")
        vae.fit(X_train, X_val)
        vae.save(vae_path)
        log.info("VAE saved -> %s", vae_path)

    # SVDD
    svdd = DeepSVDD(input_dim=18)
    if os.path.exists(svdd_path):
        log.info("Loading SVDD from checkpoint...")
        svdd.load(svdd_path)
    else:
        log.info("Training SVDD (seed 42)...")
        svdd.fit(X_train, X_val)
        svdd.save(svdd_path)
        log.info("SVDD saved -> %s", svdd_path)

    # Compute threshold
    val_scores = ensemble_scores(ae, vae, svdd, X_val)
    tau = float(np.percentile(val_scores, 95))
    log.info("Val tau (p95) = %.4f", tau)

    # Evaluate with full confusion matrix
    test_sets = [
        ("CIC-IDS2017",     "test_cic17_dedup.npz"),
        ("CSE-CIC-IDS2018", "test_cic18_dedup.npz"),
        ("UNSW-NB15",       "test_unsw_dedup.npz"),
        ("BoT-IoT",         "test_botiot_dedup.npz"),
    ]

    results = []
    log.info("\nCONFUSION MATRIX RESULTS (seed 42, dedup):")
    log.info("=" * 60)

    for name, fname in test_sets:
        path = os.path.join(DEDUP_DIR, fname)
        if not os.path.exists(path):
            log.warning("Not found: %s", path)
            continue

        d = np.load(path)
        X = np.clip(d["X"], -10, 10).astype(np.float32)
        y = d["y"].astype(int)

        scores = ensemble_scores(ae, vae, svdd, X)
        y_pred = (scores > tau).astype(int)

        # Confusion matrix
        TP = int(((y==1) & (y_pred==1)).sum())
        TN = int(((y==0) & (y_pred==0)).sum())
        FP = int(((y==0) & (y_pred==1)).sum())
        FN = int(((y==1) & (y_pred==0)).sum())
        total = TP + TN + FP + FN

        if len(np.unique(y)) < 2:
            # BoT-IoT attack only
            DR = TP / (TP + FN) if (TP + FN) > 0 else 0
            log.info("%s:", name)
            log.info("  TP=%d FN=%d DR=%.4f tau=%.4f",
                     TP, FN, DR, tau)
            results.append({
                "dataset": name, "TP": TP, "TN": TN,
                "FP": FP, "FN": FN, "tau": round(tau, 4),
                "DR": round(DR, 4)
            })
        else:
            acc  = (TP + TN) / total
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0
            rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            fnr  = FN / (TP + FN) if (TP + FN) > 0 else 0
            fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0
            auc  = roc_auc_score(y, scores)

            log.info("%s:", name)
            log.info("  TP=%d TN=%d FP=%d FN=%d",
                     TP, TN, FP, FN)
            log.info("  Acc=%.3f Prec=%.3f Rec=%.3f F1=%.3f FNR=%.3f FPR=%.3f AUC=%.4f",
                     acc, prec, rec, f1, fnr, fpr, auc)

            results.append({
                "dataset": name, "TP": TP, "TN": TN,
                "FP": FP, "FN": FN, "tau": round(tau, 4),
                "Accuracy": round(acc, 3),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1": round(f1, 3),
                "FNR": round(fnr, 3),
                "FPR": round(fpr, 3),
                "AUC": round(auc, 4),
            })

    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False)
    log.info("\nSaved -> %s", OUT_PATH)
    log.info("Models saved in: %s", SEED42_DIR)
    log.info("Done.")


if __name__ == "__main__":
    run()
