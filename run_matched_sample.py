"""
run_matched_sample.py
=====================
Compares ensemble vs OC-SVM/LOF on matched 10,000 sample
training set to assess whether gap is due to architecture
or training data size.

Run from project root:
    python run_matched_sample.py

Expected time: ~1-2 hours
Results saved to: artefacts_v2/matched_sample_results.csv
"""
import sys, os, logging
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
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
OUT_PATH     = os.path.join(ARTEFACT_DIR, "matched_sample_results.csv")
SEED         = 42
N_MATCHED    = 10000  # same as OC-SVM/LOF


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
    log.info("Loading data...")
    d_train = np.load(os.path.join(ARTEFACT_DIR, "train.npz"))
    d_val   = np.load(os.path.join(ARTEFACT_DIR, "val.npz"))

    X_train_full = d_train["X_train"]
    X_val_full   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)

    # Matched sample — 10,000 from combined training set
    np.random.seed(SEED)
    idx = np.random.choice(len(X_train_full), N_MATCHED, replace=False)
    X_matched = X_train_full[idx]
    X_val_matched = X_val_full[:2000]

    log.info("Matched training set: %s", X_matched.shape)

    test_sets = [
        ("CIC-IDS2017",     "test_cic17.npz"),
        ("CSE-CIC-IDS2018", "test_cic18.npz"),
        ("UNSW-NB15",       "test_unsw.npz"),
    ]

    results = []

    # -------------------------------------------------------------------------
    # 1. Ensemble on matched 10,000 samples
    # -------------------------------------------------------------------------
    log.info("=" * 60)
    log.info("ENSEMBLE on %d samples (matched)", N_MATCHED)
    log.info("=" * 60)

    torch.manual_seed(SEED)
    ae   = Autoencoder(input_dim=18)
    vae  = VAE(input_dim=18)
    svdd = DeepSVDD(input_dim=18)

    log.info("Training AE...")
    ae.fit(X_matched, X_val_matched)
    log.info("Training VAE...")
    vae.fit(X_matched, X_val_matched)
    log.info("Training SVDD...")
    svdd.fit(X_matched, X_val_matched)

    for name, fname in test_sets:
        d = np.load(os.path.join(ARTEFACT_DIR, fname))
        X = np.clip(d["X"], -10, 10).astype(np.float32)
        y = d["y"].astype(int)
        scores = ensemble_scores(ae, vae, svdd, X)
        auc = roc_auc_score(y, scores)
        results.append({"method": f"Ensemble (n={N_MATCHED})",
                        "dataset": name, "AUC": round(auc, 4)})
        log.info("  %-20s AUC = %.4f", name, auc)

    # -------------------------------------------------------------------------
    # 2. OC-SVM on 10,000 samples
    # -------------------------------------------------------------------------
    log.info("=" * 60)
    log.info("OC-SVM on %d samples", N_MATCHED)
    log.info("=" * 60)

    ocsvm = OneClassSVM(kernel="rbf", nu=0.05)
    ocsvm.fit(X_matched)

    for name, fname in test_sets:
        d = np.load(os.path.join(ARTEFACT_DIR, fname))
        X = np.clip(d["X"], -10, 10).astype(np.float32)
        y = d["y"].astype(int)
        scores = -ocsvm.score_samples(X)
        auc = roc_auc_score(y, scores)
        results.append({"method": "OC-SVM", "dataset": name, "AUC": round(auc, 4)})
        log.info("  %-20s AUC = %.4f", name, auc)

    # -------------------------------------------------------------------------
    # 3. LOF on 10,000 samples
    # -------------------------------------------------------------------------
    log.info("=" * 60)
    log.info("LOF on %d samples", N_MATCHED)
    log.info("=" * 60)

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(X_matched)

    for name, fname in test_sets:
        d = np.load(os.path.join(ARTEFACT_DIR, fname))
        X = np.clip(d["X"], -10, 10).astype(np.float32)
        y = d["y"].astype(int)
        scores = -lof.score_samples(X)
        auc = roc_auc_score(y, scores)
        results.append({"method": "LOF", "dataset": name, "AUC": round(auc, 4)})
        log.info("  %-20s AUC = %.4f", name, auc)

    # -------------------------------------------------------------------------
    # 4. Ensemble on FULL training set (reference)
    # -------------------------------------------------------------------------
    log.info("=" * 60)
    log.info("ENSEMBLE on FULL %d samples (reference)", len(X_train_full))
    log.info("=" * 60)

    torch.manual_seed(SEED)
    ae_full   = Autoencoder(input_dim=18)
    vae_full  = VAE(input_dim=18)
    svdd_full = DeepSVDD(input_dim=18)

    log.info("Training AE (full)...")
    ae_full.fit(X_train_full, X_val_full)
    log.info("Training VAE (full)...")
    vae_full.fit(X_train_full, X_val_full)
    log.info("Training SVDD (full)...")
    svdd_full.fit(X_train_full, X_val_full)

    for name, fname in test_sets:
        d = np.load(os.path.join(ARTEFACT_DIR, fname))
        X = np.clip(d["X"], -10, 10).astype(np.float32)
        y = d["y"].astype(int)
        scores = ensemble_scores(ae_full, vae_full, svdd_full, X)
        auc = roc_auc_score(y, scores)
        results.append({"method": "Ensemble (full)",
                        "dataset": name, "AUC": round(auc, 4)})
        log.info("  %-20s AUC = %.4f", name, auc)

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False)
    log.info("\nSaved -> %s", OUT_PATH)

    # Summary table
    log.info("\n" + "=" * 60)
    log.info("MATCHED SAMPLE COMPARISON SUMMARY")
    log.info("=" * 60)
    pivot = df.pivot(index="method", columns="dataset", values="AUC")
    log.info("\n%s", pivot.to_string())
    log.info("\nDone.")


if __name__ == "__main__":
    run()
