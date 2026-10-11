"""
run_single_env_ensemble.py
==========================
Trains the full AE+VAE+SVDD ensemble on a SINGLE
environment (CIC-IDS2017 only) using deduplicated data.

Compares against:
  - AE-Single (already run)
  - Full ensemble 3-environment (main dedup results)

This separates architecture effect from diversity effect
for the FULL ensemble (not just AE).

Run from project root:
    python run_single_env_ensemble.py

Results saved to: artefacts_v2/dedup/single_env_ensemble_results.csv
Expected time: ~4 hours (seed 42 only for speed)
"""
import sys, os, gc, logging
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import roc_auc_score, f1_score, recall_score
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
OUT_PATH     = os.path.join(DEDUP_DIR, "single_env_ensemble_results.csv")
SEEDS        = [42, 123, 456, 789, 999]


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


def load_existing():
    if os.path.exists(OUT_PATH):
        df = pd.read_csv(OUT_PATH)
        done = df["seed"].tolist()
        return df.to_dict("records"), done
    return [], []


def run():
    log.info("Loading deduplicated data...")
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    d_val   = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))

    X_train_full = d_train["X_train"]
    X_val_full   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)

    log.info("Full train: %s", X_train_full.shape)

    # Use first third = CIC-IDS2017 portion
    n      = len(X_train_full)
    n_val  = len(X_val_full)
    third  = n // 3
    third_val = n_val // 3

    X_train_single = X_train_full[:third]
    X_val_single   = X_val_full[:third_val]

    log.info("CIC17-only train: %s  val: %s",
             X_train_single.shape, X_val_single.shape)

    # Load test sets
    test_sets = []
    for name, fname in [
        ("CIC-IDS2017",     "test_cic17_dedup.npz"),
        ("CSE-CIC-IDS2018", "test_cic18_dedup.npz"),
        ("UNSW-NB15",       "test_unsw_dedup.npz"),
        ("BoT-IoT",         "test_botiot_dedup.npz"),
    ]:
        path = os.path.join(DEDUP_DIR, fname)
        if os.path.exists(path):
            d = np.load(path)
            X = np.clip(d["X"], -10, 10).astype(np.float32)
            y = d["y"].astype(int)
            test_sets.append((name, X, y))

    results, done_seeds = load_existing()

    for seed in SEEDS:
        if seed in done_seeds:
            log.info("Seed %d already done — skipping", seed)
            continue

        log.info("=" * 60)
        log.info("SEED %d — Single-env ensemble (CIC17 only)", seed)
        log.info("=" * 60)

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Train full ensemble on CIC17 only
        ae = Autoencoder(input_dim=18)
        log.info("Training AE...")
        ae.fit(X_train_single, X_val_single)

        vae = VAE(input_dim=18)
        log.info("Training VAE...")
        vae.fit(X_train_single, X_val_single)

        svdd = DeepSVDD(input_dim=18)
        log.info("Training SVDD...")
        svdd.fit(X_train_single, X_val_single)

        # Threshold from single-env val set
        val_scores = ensemble_scores(ae, vae, svdd, X_val_single)
        tau = float(np.percentile(val_scores, 95))
        log.info("Val tau (p95) = %.4f", tau)

        seed_result = {"seed": seed, "training": "CIC17-only"}

        for name, X, y in test_sets:
            scores = ensemble_scores(ae, vae, svdd, X)

            if len(np.unique(y)) < 2:
                dr = float((scores > tau).mean())
                seed_result[f"{name}_DR"] = round(dr, 4)
                log.info("  %-20s DR  = %.4f", name, dr)
            else:
                auc = roc_auc_score(y, scores)
                rec = recall_score(y, (scores > tau).astype(int), zero_division=0)
                f1  = f1_score(y, (scores > tau).astype(int), zero_division=0)
                seed_result[f"{name}_AUC"]    = round(auc, 4)
                seed_result[f"{name}_Recall"] = round(rec, 4)
                seed_result[f"{name}_F1"]     = round(f1, 4)
                log.info("  %-20s AUC=%.4f Rec=%.4f F1=%.4f",
                         name, auc, rec, f1)

        results.append(seed_result)
        pd.DataFrame(results).to_csv(OUT_PATH, index=False)
        log.info("Saved -> %s", OUT_PATH)
        gc.collect()

    # Summary
    df = pd.DataFrame(results)
    log.info("\n" + "=" * 60)
    log.info("SINGLE-ENV ENSEMBLE SUMMARY (mean+/-std, %d seeds)", len(df))
    log.info("=" * 60)

    # Compare against 3-env ensemble
    three_env = {
        "CIC-IDS2017_AUC":     0.830,
        "CSE-CIC-IDS2018_AUC": 0.986,
        "UNSW-NB15_AUC":       0.740,
        "BoT-IoT_DR":          0.580,
    }

    for col in [c for c in df.columns if c not in ["seed", "training"]]:
        mean_val = df[col].mean()
        std_val  = df[col].std()
        ref = three_env.get(col)
        diff = f"  (vs 3-env: {mean_val-ref:+.3f})" if ref else ""
        log.info("  %-28s %.4f +/- %.4f%s", col, mean_val, std_val, diff)

    log.info("\nDiversity effect (single→multi env, full ensemble):")
    for col, ref in three_env.items():
        if col in df.columns:
            diff = ref - df[col].mean()
            log.info("  %-28s %+.3f", col, diff)

    log.info("Done.")


if __name__ == "__main__":
    run()
