"""
run_ablation_imputation.py
===========================
Compares three imputation strategies for the two
missing BoT-IoT features (bwd_iat_std and
bwd_packet_length_std):
  1. Zero-fill (current approach)
  2. Source-median imputation (median from training data)
  3. Feature removal (drop both features from all datasets)

Run from project root:
    python run_ablation_imputation.py

Results saved to: artefacts_v2/ablation_imputation_results.csv
Expected time: ~12 hours (3 strategies × ~4 hours)
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

ARTEFACT_DIR  = r"C:/MLProject/zero_day_project/artefacts_v2"
OUT_PATH      = os.path.join(ARTEFACT_DIR, "ablation_imputation_results.csv")
SEEDS         = [42, 123, 456, 789, 999]

# The two missing BoT-IoT features
MISSING_FEATURES = ["bwd_iat_std", "bwd_packet_length_std"]
BOTIOT_PATH = r"C:/MLProject/zero_day_project/artefacts_v2/test_botiot.npz"


def get_features():
    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    all_features = pd.read_csv(feat_path, header=None)[0].tolist()
    all_features = [f for f in all_features
                    if str(f).strip() and not str(f).strip().lstrip("-").isdigit()]
    return all_features


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
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        for i in range(0, len(X), 10000):
            batch = X_t[i:i+10000].to(device)
            recon = ae.model_(batch)
            err = ((batch - recon) ** 2).mean(dim=1)
            scores.append(err.cpu().numpy())
    return np.concatenate(scores)


def get_vae_scores(vae, X):
    vae.model_.eval()
    device = next(vae.model_.parameters()).device
    scores = []
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        for i in range(0, len(X), 10000):
            batch = X_t[i:i+10000].to(device)
            x_recon, mu, log_var = vae.model_(batch)
            recon_loss = ((batch - x_recon) ** 2).mean(dim=1)
            kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
            scores.append((recon_loss + kl_loss).cpu().numpy())
    return np.concatenate(scores)


def get_svdd_scores(svdd, X):
    svdd.model_.eval()
    device = next(svdd.model_.parameters()).device
    centre = svdd.centre_.to(device)
    scores = []
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        for i in range(0, len(X), 10000):
            batch = X_t[i:i+10000].to(device)
            z = svdd.model_(batch)
            dist = ((z - centre) ** 2).sum(dim=1)
            scores.append(dist.cpu().numpy())
    return np.concatenate(scores)


def ensemble_scores(ae, vae, svdd, X):
    return (0.4 * norm_minmax(get_ae_scores(ae, X)) +
            0.4 * norm_minmax(get_vae_scores(vae, X)) +
            0.2 * norm_minmax(get_svdd_scores(svdd, X)))


def train_models(X_train, X_val, input_dim, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    ae = Autoencoder(input_dim=input_dim)
    log.info("  Training AE...")
    ae.fit(X_train, X_val)

    vae = VAE(input_dim=input_dim)
    log.info("  Training VAE...")
    vae.fit(X_train, X_val)

    svdd = DeepSVDD(input_dim=input_dim)
    log.info("  Training SVDD...")
    svdd.fit(X_train, X_val)

    return ae, vae, svdd


def evaluate(ae, vae, svdd, X_val, X_botiot, feat_idx_botiot=None):
    """Evaluate on BoT-IoT only (main focus of imputation study)."""
    val_scores = ensemble_scores(ae, vae, svdd, X_val)
    tau = float(np.percentile(val_scores, 95))

    if feat_idx_botiot is not None:
        X_bot = X_botiot[:, feat_idx_botiot]
    else:
        X_bot = X_botiot

    bot_scores = ensemble_scores(ae, vae, svdd, X_bot)
    dr = float((bot_scores > tau).mean())
    return dr, tau


def load_existing():
    if os.path.exists(OUT_PATH):
        df = pd.read_csv(OUT_PATH)
        done = list(zip(df["strategy"].tolist(), df["seed"].tolist()))
        return df.to_dict("records"), done
    return [], []


def run():
    features = get_features()
    missing_idx = [features.index(f) for f in MISSING_FEATURES if f in features]
    log.info("All features: %d", len(features))
    log.info("Missing feature indices: %s -> %s", MISSING_FEATURES, missing_idx)

    # Load training data
    log.info("Loading train/val data...")
    d_train = np.load(os.path.join(ARTEFACT_DIR, "train.npz"))
    d_val   = np.load(os.path.join(ARTEFACT_DIR, "val.npz"))
    X_train_full = d_train["X_train"]
    X_val_full   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)

    # Load BoT-IoT test data
    d_bot = np.load(BOTIOT_PATH)
    X_botiot_full = np.clip(d_bot["X"], -10, 10).astype(np.float32)

    # Compute source medians for imputation (from training data)
    log.info("Computing source medians for imputation...")
    source_medians = {}
    for idx, feat in zip(missing_idx, MISSING_FEATURES):
        median_val = float(np.median(X_train_full[:, idx]))
        source_medians[feat] = median_val
        log.info("  %s median = %.4f", feat, median_val)

    results, done = load_existing()

    strategies = {
        "zero_fill": {
            "desc": "Zero-fill missing features (current approach)",
            "input_dim": len(features),
            "transform_botiot": lambda X: X,  # already zero-filled in npz
        },
        "median_fill": {
            "desc": "Source-median imputation for missing features",
            "input_dim": len(features),
            "transform_botiot": lambda X: _median_fill(X, missing_idx,
                                                        list(source_medians.values())),
        },
        "feature_removal": {
            "desc": "Remove missing features from all datasets",
            "input_dim": len(features) - len(missing_idx),
            "feat_idx_keep": [i for i in range(len(features)) if i not in missing_idx],
        },
    }

    for strategy_name, strategy in strategies.items():
        input_dim = strategy["input_dim"]

        if strategy_name == "feature_removal":
            keep_idx = strategy["feat_idx_keep"]
            X_train = X_train_full[:, keep_idx]
            X_val   = X_val_full[:, keep_idx]
            X_bot   = X_botiot_full[:, keep_idx]
        else:
            X_train = X_train_full
            X_val   = X_val_full
            if strategy_name == "median_fill":
                X_bot = strategy["transform_botiot"](X_botiot_full.copy())
            else:
                X_bot = X_botiot_full

        log.info("=" * 60)
        log.info("STRATEGY: %s (%s)", strategy_name, strategy["desc"])
        log.info("Input dim: %d", input_dim)
        log.info("=" * 60)

        seed_drs = []
        for seed in SEEDS:
            if (strategy_name, seed) in done:
                log.info("Seed %d already done — skipping", seed)
                existing = [r for r in results
                            if r["strategy"] == strategy_name and r["seed"] == seed]
                if existing:
                    seed_drs.append(existing[0]["BoT_IoT_DR"])
                continue

            log.info("  Seed %d", seed)
            ae, vae, svdd = train_models(X_train, X_val, input_dim, seed)
            dr, tau = evaluate(ae, vae, svdd, X_val, X_bot)

            log.info("  BoT-IoT DR = %.4f (tau=%.4f)", dr, tau)
            seed_drs.append(dr)

            result = {
                "strategy": strategy_name,
                "seed": seed,
                "BoT_IoT_DR": round(dr, 4),
                "tau": round(tau, 4),
            }
            results.append(result)
            pd.DataFrame(results).to_csv(OUT_PATH, index=False)
            log.info("  Saved -> %s", OUT_PATH)
            done.append((strategy_name, seed))
            gc.collect()

        if seed_drs:
            log.info("  %s mean DR = %.4f +/- %.4f",
                     strategy_name, np.mean(seed_drs), np.std(seed_drs))

    # Final summary
    df = pd.DataFrame(results)
    log.info("\n" + "=" * 60)
    log.info("IMPUTATION COMPARISON SUMMARY")
    log.info("=" * 60)
    log.info("Main result (zero-fill): DR = 0.827 +/- 0.072")
    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat]["BoT_IoT_DR"]
        log.info("  %-20s DR = %.4f +/- %.4f", strat, sub.mean(), sub.std())
    log.info("Done.")


def _median_fill(X, missing_idx, median_vals):
    """Replace zero-filled features with source medians."""
    X = X.copy()
    for idx, val in zip(missing_idx, median_vals):
        X[:, idx] = val
    return X


if __name__ == "__main__":
    run()
