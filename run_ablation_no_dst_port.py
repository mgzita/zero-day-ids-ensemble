"""
run_ablation_no_dst_port.py
============================
Reruns the full 5-seed ensemble pipeline with
destination_port removed from the feature set.

Models are saved after each seed so the script
can be restarted without retraining completed seeds.

Run from project root:
    python run_ablation_no_dst_port.py

Results saved to: artefacts_v2/ablation_no_dst_port_results.csv
Models saved to:  artefacts_v2/ablation_dst_port/seed_{N}/
"""
import sys, os, gc, logging
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import roc_auc_score
from models.autoencoder import Autoencoder
from models.vae import VAE
from models.deep_svdd import DeepSVDD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTEFACT_DIR   = r"C:/MLProject/zero_day_project/artefacts_v2"
CHECKPOINT_DIR = os.path.join(ARTEFACT_DIR, "ablation_dst_port")
OUT_PATH       = os.path.join(ARTEFACT_DIR, "ablation_no_dst_port_results.csv")
SEEDS          = [42, 123, 456, 789, 999]
REMOVE_FEATURE = "destination_port"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_features():
    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    all_features = pd.read_csv(feat_path, header=None)[0].tolist()
    all_features = [f for f in all_features
                    if str(f).strip() and not str(f).strip().lstrip("-").isdigit()]
    features = [f for f in all_features if f != REMOVE_FEATURE]
    feat_idx = [all_features.index(f) for f in features]
    log.info("Removed '%s' — %d features remaining", REMOVE_FEATURE, len(features))
    return features, feat_idx


def norm_minmax(s):
    """Min-max normalize using 1st/99th percentiles."""
    lo = float(np.percentile(s, 1))
    hi = float(np.percentile(s, 99))
    if hi - lo < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def get_ae_scores(ae, X):
    """Compute AE reconstruction scores directly."""
    device = ae.device_ if hasattr(ae, 'device_') else torch.device('cpu')
    ae.model_.eval()
    scores = []
    batch_size = 10000
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X_t[i:i+batch_size].to(device)
            recon = ae.model_(batch)
            err = ((batch - recon) ** 2).mean(dim=1)
            scores.append(err.cpu().numpy())
    return np.concatenate(scores)


def get_vae_scores(vae, X):
    """Compute VAE ELBO scores directly using posterior mean."""
    device = vae.device_ if hasattr(vae, 'device_') else torch.device('cpu')
    vae.model_.eval()
    scores = []
    batch_size = 10000
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X_t[i:i+batch_size].to(device)
            x_recon, mu, log_var = vae.model_(batch)
            recon_loss = ((batch - x_recon) ** 2).mean(dim=1)
            kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
            elbo = recon_loss + kl_loss
            scores.append(elbo.cpu().numpy())
    return np.concatenate(scores)


def get_svdd_scores(svdd, X):
    """Compute Deep SVDD distance scores directly."""
    device = svdd.device if hasattr(svdd, 'device') else torch.device('cpu')
    svdd.model_.eval()
    centre = svdd.centre_.to(device)
    scores = []
    batch_size = 10000
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X_t[i:i+batch_size].to(device)
            z = svdd.model_(batch)
            dist = ((z - centre) ** 2).sum(dim=1)
            scores.append(dist.cpu().numpy())
    return np.concatenate(scores)


def ensemble_scores(ae, vae, svdd, X):
    s_ae   = get_ae_scores(ae, X)
    s_vae  = get_vae_scores(vae, X)
    s_svdd = get_svdd_scores(svdd, X)
    return (0.4 * norm_minmax(s_ae) +
            0.4 * norm_minmax(s_vae) +
            0.2 * norm_minmax(s_svdd))


def load_existing_results():
    if os.path.exists(OUT_PATH):
        df = pd.read_csv(OUT_PATH)
        done_seeds = df["seed"].tolist()
        log.info("Found existing results for seeds: %s", done_seeds)
        return df.to_dict("records"), done_seeds
    return [], []


def run():
    features, feat_idx = get_features()
    input_dim = len(features)

    log.info("Loading train/val data...")
    d_train = np.load(os.path.join(ARTEFACT_DIR, "train.npz"))
    d_val   = np.load(os.path.join(ARTEFACT_DIR, "val.npz"))
    X_train = d_train["X_train"][:, feat_idx]
    X_val   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)[:, feat_idx]
    log.info("Train: %s  Val: %s", X_train.shape, X_val.shape)

    test_sets = [
        ("CIC-IDS2017",     os.path.join(ARTEFACT_DIR, "test_cic17.npz")),
        ("CSE-CIC-IDS2018", os.path.join(ARTEFACT_DIR, "test_cic18.npz")),
        ("UNSW-NB15",       os.path.join(ARTEFACT_DIR, "test_unsw.npz")),
        ("BoT-IoT",         os.path.join(ARTEFACT_DIR, "test_botiot.npz")),
    ]

    results, done_seeds = load_existing_results()

    for seed in SEEDS:
        if seed in done_seeds:
            log.info("Seed %d already completed — skipping", seed)
            continue

        log.info("=" * 60)
        log.info("SEED %d", seed)
        log.info("=" * 60)

        seed_dir = os.path.join(CHECKPOINT_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        ae_path   = os.path.join(seed_dir, "ae.pt")
        vae_path  = os.path.join(seed_dir, "vae.pt")
        svdd_path = os.path.join(seed_dir, "svdd.pt")

        np.random.seed(seed)
        torch.manual_seed(seed)

        # AE
        # AE - load from checkpoint or retrain
        ae = Autoencoder(input_dim=input_dim)
        if os.path.exists(ae_path):
            log.info("Loading AE from checkpoint...")
            ae.load(ae_path)
        else:
            log.info("Training AE...")
            ae.fit(X_train, X_val)
            ae.save(ae_path)
            log.info("AE saved -> %s", ae_path)

        # VAE - always retrain fresh (saves avoid dimension mismatch)
        vae = VAE(input_dim=input_dim)
        log.info("Training VAE...")
        vae.fit(X_train, X_val)

        # SVDD - load from checkpoint or retrain
        svdd = DeepSVDD(input_dim=input_dim)
        if os.path.exists(svdd_path):
            log.info("Loading SVDD from checkpoint...")
            svdd.load(svdd_path)
        else:
            log.info("Training Deep SVDD...")
            svdd.fit(X_train, X_val)
            svdd.save(svdd_path)
            log.info("SVDD saved -> %s", svdd_path)

        # Validate model dimensions before evaluation
        log.info("Validating model dimensions...")
        test_batch = torch.FloatTensor(X_val[:10])
        try:
            with torch.no_grad():
                ae_out = ae.model_(test_batch.to(ae.device_) if hasattr(ae,'device_') else test_batch)
                assert ae_out.shape == test_batch.shape, f"AE dim mismatch: {ae_out.shape} vs {test_batch.shape}"
            log.info("  AE output shape: %s ✅", ae_out.shape)
        except Exception as e:
            log.error("  AE validation failed: %s", e)
            raise

        try:
            with torch.no_grad():
                x_recon, mu, log_var = vae.model_(test_batch.to(vae.device_) if hasattr(vae,'device_') else test_batch)
                assert x_recon.shape == test_batch.shape, f"VAE dim mismatch: {x_recon.shape} vs {test_batch.shape}"
            log.info("  VAE output shape: %s ✅", x_recon.shape)
        except Exception as e:
            log.error("  VAE validation failed: %s", e)
            raise

        log.info("All model dimensions validated ✅")

        # Evaluate
        log.info("Evaluating...")
        seed_result = {"seed": seed}

        # Compute val scores for BoT-IoT threshold
        val_scores = ensemble_scores(ae, vae, svdd, X_val)
        tau = float(np.percentile(val_scores, 95))
        log.info("  Val tau (p95) = %.4f", tau)

        for name, path in test_sets:
            if not os.path.exists(path):
                log.warning("Not found: %s", path)
                continue

            d = np.load(path)
            X = np.clip(d["X"], -10, 10).astype(np.float32)[:, feat_idx]
            y = d["y"].astype(int)

            scores = ensemble_scores(ae, vae, svdd, X)

            if len(np.unique(y)) < 2:
                dr = float((scores > tau).mean())
                seed_result[f"{name}_DR"] = round(dr, 4)
                log.info("  %-20s DR  = %.4f", name, dr)
            else:
                auc = roc_auc_score(y, scores)
                seed_result[f"{name}_AUC"] = round(auc, 4)
                log.info("  %-20s AUC = %.4f", name, auc)

        results.append(seed_result)
        pd.DataFrame(results).to_csv(OUT_PATH, index=False)
        log.info("Saved -> %s", OUT_PATH)
        gc.collect()

    # Summary
    df = pd.DataFrame(results)
    log.info("\n" + "=" * 60)
    log.info("SUMMARY — WITHOUT destination_port")
    log.info("=" * 60)

    main = {
        "CIC-IDS2017_AUC":     0.912,
        "CSE-CIC-IDS2018_AUC": 0.959,
        "UNSW-NB15_AUC":       0.884,
        "BoT-IoT_DR":          0.827,
    }

    for col in [c for c in df.columns if c != "seed"]:
        mean_val = df[col].mean()
        std_val  = df[col].std()
        ref      = main.get(col)
        diff_str = f"  (diff vs main: {mean_val - ref:+.3f})" if ref else ""
        log.info("  %-25s %.4f +/- %.4f%s", col, mean_val, std_val, diff_str)

    log.info("Done.")


if __name__ == "__main__":
    run()
