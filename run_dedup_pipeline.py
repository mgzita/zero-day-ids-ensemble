"""
run_dedup_pipeline.py
=====================
Step 1: Deduplicates all npz files (train, val, test sets)
Step 2: Retrains full 5-seed pipeline on deduplicated data
Step 3: Re-evaluates all metrics and saves results

Run from project root:
    python run_dedup_pipeline.py

Results saved to: artefacts_v2/dedup/
Expected time: ~25 hours (dedup ~30 min + retrain ~20 hours + reeval ~5 hours)
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
OUT_PATH     = os.path.join(DEDUP_DIR, "dedup_results.csv")
SEEDS        = [42, 123, 456, 789, 999]
os.makedirs(DEDUP_DIR, exist_ok=True)


# =============================================================================
# STEP 1: DEDUPLICATION
# =============================================================================
def deduplicate_npz(in_path, out_path, x_key, y_key=None):
    """Load npz, remove exact duplicate rows, save deduplicated version."""
    d = np.load(in_path)
    X = d[x_key]
    y = d[y_key].astype(int) if y_key and y_key in d else None

    n_before = len(X)
    df = pd.DataFrame(X)

    if y is not None:
        df['__label__'] = y
        df_dedup = df.drop_duplicates()
        y_dedup = df_dedup['__label__'].values.astype(int)
        X_dedup = df_dedup.drop(columns=['__label__']).values.astype(np.float32)
    else:
        df_dedup = df.drop_duplicates()
        X_dedup = df_dedup.values.astype(np.float32)
        y_dedup = None

    n_after  = len(X_dedup)
    n_removed = n_before - n_after

    log.info("  %s: %d → %d rows (%d removed, %.2f%%)",
             os.path.basename(in_path), n_before, n_after,
             n_removed, n_removed/n_before*100)

    if y_dedup is not None:
        np.savez_compressed(out_path, **{x_key: X_dedup, y_key: y_dedup})
    else:
        np.savez_compressed(out_path, **{x_key: X_dedup})

    return n_before, n_after, n_removed


def run_deduplication():
    log.info("=" * 60)
    log.info("STEP 1: DEDUPLICATION")
    log.info("=" * 60)

    datasets = [
        ("train.npz",        "train_dedup.npz",   "X_train", None),
        ("val.npz",          "val_dedup.npz",     "X_val",   None),
        ("test_cic17.npz",   "test_cic17_dedup.npz",  "X", "y"),
        ("test_cic18.npz",   "test_cic18_dedup.npz",  "X", "y"),
        ("test_unsw.npz",    "test_unsw_dedup.npz",   "X", "y"),
        ("test_botiot.npz",  "test_botiot_dedup.npz", "X", "y"),
    ]

    summary = []
    for in_file, out_file, x_key, y_key in datasets:
        in_path  = os.path.join(ARTEFACT_DIR, in_file)
        out_path = os.path.join(DEDUP_DIR, out_file)

        if os.path.exists(out_path):
            log.info("  %s already deduplicated — skipping", in_file)
            d = np.load(out_path)
            summary.append((in_file, "skipped", len(d[x_key])))
            continue

        if not os.path.exists(in_path):
            log.warning("  %s not found — skipping", in_file)
            continue

        n_before, n_after, n_removed = deduplicate_npz(
            in_path, out_path, x_key, y_key)
        summary.append((in_file, n_before, n_after, n_removed))

    log.info("\nDEDUPLICATION SUMMARY:")
    for row in summary:
        if len(row) == 3:
            log.info("  %-30s %s (n=%d)", row[0], row[1], row[2])
        else:
            log.info("  %-30s %d → %d (-%d, %.1f%%)",
                     row[0], row[1], row[2], row[3],
                     row[3]/row[1]*100)


# =============================================================================
# STEP 2: RETRAIN ON DEDUPLICATED DATA
# =============================================================================
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


def load_existing_results():
    if os.path.exists(OUT_PATH):
        df = pd.read_csv(OUT_PATH)
        done = df["seed"].tolist()
        log.info("Found existing dedup results for seeds: %s", done)
        return df.to_dict("records"), done
    return [], []


def run_retraining():
    log.info("=" * 60)
    log.info("STEP 2: RETRAINING ON DEDUPLICATED DATA")
    log.info("=" * 60)

    # Load deduplicated training data
    log.info("Loading deduplicated train/val data...")
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    d_val   = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_train = d_train["X_train"]
    X_val   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    log.info("Train: %s  Val: %s", X_train.shape, X_val.shape)

    # Test sets
    test_sets = [
        ("CIC-IDS2017",     "test_cic17_dedup.npz"),
        ("CSE-CIC-IDS2018", "test_cic18_dedup.npz"),
        ("UNSW-NB15",       "test_unsw_dedup.npz"),
        ("BoT-IoT",         "test_botiot_dedup.npz"),
    ]

    results, done_seeds = load_existing_results()

    for seed in SEEDS:
        if seed in done_seeds:
            log.info("Seed %d already done — skipping", seed)
            continue

        log.info("=" * 60)
        log.info("SEED %d", seed)
        log.info("=" * 60)

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Train models
        ae = Autoencoder(input_dim=18)
        log.info("Training AE...")
        ae.fit(X_train, X_val)

        vae = VAE(input_dim=18)
        log.info("Training VAE...")
        vae.fit(X_train, X_val)

        svdd = DeepSVDD(input_dim=18)
        log.info("Training Deep SVDD...")
        svdd.fit(X_train, X_val)

        # Compute threshold from val
        val_scores = ensemble_scores(ae, vae, svdd, X_val)
        tau = float(np.percentile(val_scores, 95))
        log.info("Val tau (p95) = %.4f", tau)

        seed_result = {"seed": seed}

        for name, fname in test_sets:
            path = os.path.join(DEDUP_DIR, fname)
            if not os.path.exists(path):
                log.warning("Not found: %s", path)
                continue

            d = np.load(path)
            X = np.clip(d["X"], -10, 10).astype(np.float32)
            y = d["y"].astype(int)

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
    log.info("DEDUP PIPELINE SUMMARY (mean +/- std, %d seeds)", len(SEEDS))
    log.info("=" * 60)

    original = {
        "CIC-IDS2017_AUC":     0.912,
        "CSE-CIC-IDS2018_AUC": 0.959,
        "UNSW-NB15_AUC":       0.884,
        "BoT-IoT_DR":          0.827,
    }

    for col in [c for c in df.columns if c != "seed"]:
        mean_val = df[col].mean()
        std_val  = df[col].std()
        ref = original.get(col)
        diff = f"  (diff vs original: {mean_val-ref:+.3f})" if ref else ""
        log.info("  %-28s %.4f +/- %.4f%s", col, mean_val, std_val, diff)

    log.info("Done.")


# =============================================================================
# MAIN
# =============================================================================
def run():
    run_deduplication()
    run_retraining()


if __name__ == "__main__":
    run()
