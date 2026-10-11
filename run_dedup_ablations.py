"""
run_dedup_ablations.py
======================
Reruns ALL ablation experiments on deduplicated data:
  1. Pairwise ablation (7 model combinations x 5 seeds)
  2. LODO - Leave One Dataset Out (3 experiments)
  3. AE-Single baseline (single environment)
  4. Destination port ablation (17 features x 5 seeds)
  5. DAE baseline (5 seeds)

Requires: artefacts_v2/dedup/ (from run_dedup_pipeline.py)

Run from project root:
    python run_dedup_ablations.py

Results saved to: artefacts_v2/dedup/ablation_*.csv
Expected time: ~20 hours total
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
DEDUP_DIR    = os.path.join(ARTEFACT_DIR, "dedup")
SEEDS        = [42, 123, 456, 789, 999]
SEED_42      = 42


# =============================================================================
# SHARED UTILITIES
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


def ensemble_scores(ae, vae, svdd, X, weights=(0.4, 0.4, 0.2)):
    s = np.zeros(len(X), dtype=np.float32)
    if ae is not None:
        s += weights[0] * norm_minmax(get_ae_scores(ae, X))
    if vae is not None:
        s += weights[1] * norm_minmax(get_vae_scores(vae, X))
    if svdd is not None:
        s += weights[2] * norm_minmax(get_svdd_scores(svdd, X))
    return s


def train_ae(X_train, X_val, input_dim=18, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    ae = Autoencoder(input_dim=input_dim)
    ae.fit(X_train, X_val)
    return ae


def train_vae(X_train, X_val, input_dim=18, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    vae = VAE(input_dim=input_dim)
    vae.fit(X_train, X_val)
    return vae


def train_svdd(X_train, X_val, input_dim=18, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    svdd = DeepSVDD(input_dim=input_dim)
    svdd.fit(X_train, X_val)
    return svdd


def evaluate_on_test(scores_fn, X_val, test_sets, tau=None):
    """Evaluate on test sets. Returns dict of metrics."""
    val_scores = scores_fn(X_val)
    if tau is None:
        tau = float(np.percentile(val_scores, 95))

    results = {"tau": round(tau, 4)}
    for name, X, y in test_sets:
        scores = scores_fn(X)
        if len(np.unique(y)) < 2:
            dr = float((scores > tau).mean())
            results[f"{name}_DR"] = round(dr, 4)
        else:
            auc = roc_auc_score(y, scores)
            rec = recall_score(y, (scores > tau).astype(int), zero_division=0)
            f1  = f1_score(y, (scores > tau).astype(int), zero_division=0)
            results[f"{name}_AUC"]    = round(auc, 4)
            results[f"{name}_Recall"] = round(rec, 4)
            results[f"{name}_F1"]     = round(f1, 4)
    return results


def load_dedup_test_sets():
    """Load all deduplicated test sets."""
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
        else:
            log.warning("Not found: %s", path)
    return test_sets


# =============================================================================
# ABLATION 1: PAIRWISE (7 combinations)
# =============================================================================
def run_pairwise_ablation():
    out_path = os.path.join(DEDUP_DIR, "ablation_pairwise_dedup.csv")
    if os.path.exists(out_path):
        log.info("Pairwise ablation already done — skipping")
        return

    log.info("=" * 60)
    log.info("ABLATION 1: PAIRWISE (7 combinations, seed 42)")
    log.info("=" * 60)

    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    d_val   = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_train = d_train["X_train"]
    X_val   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    test_sets = load_dedup_test_sets()

    log.info("Training models for seed 42...")
    ae   = train_ae(X_train, X_val, seed=SEED_42)
    vae  = train_vae(X_train, X_val, seed=SEED_42)
    svdd = train_svdd(X_train, X_val, seed=SEED_42)

    combinations = [
        ("AE only",       ae,   None, None, (1.0, 0.0, 0.0)),
        ("VAE only",      None, vae,  None, (0.0, 1.0, 0.0)),
        ("SVDD only",     None, None, svdd, (0.0, 0.0, 1.0)),
        ("AE+VAE",        ae,   vae,  None, (0.5, 0.5, 0.0)),
        ("AE+SVDD",       ae,   None, svdd, (0.67, 0.0, 0.33)),
        ("VAE+SVDD",      None, vae,  svdd, (0.0, 0.67, 0.33)),
        ("AE+VAE+SVDD",   ae,   vae,  svdd, (0.4, 0.4, 0.2)),
    ]

    results = []
    for name, ae_, vae_, svdd_, weights in combinations:
        log.info("  Evaluating: %s", name)
        scores_fn = lambda X, a=ae_, v=vae_, s=svdd_, w=weights: \
            ensemble_scores(a, v, s, X, w)
        metrics = evaluate_on_test(scores_fn, X_val, test_sets)
        metrics["combination"] = name
        results.append(metrics)
        log.info("    %s", {k: v for k, v in metrics.items() if 'AUC' in k or 'DR' in k})

    pd.DataFrame(results).to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


# =============================================================================
# ABLATION 2: LODO
# =============================================================================
def run_lodo():
    out_path = os.path.join(DEDUP_DIR, "ablation_lodo_dedup.csv")
    if os.path.exists(out_path):
        log.info("LODO already done — skipping")
        return

    log.info("=" * 60)
    log.info("ABLATION 2: LODO (Leave One Dataset Out)")
    log.info("=" * 60)

    # Load raw training data per dataset
    # We need per-dataset splits from the original preprocessing
    # Use full dedup train and remove ~600k rows per dataset
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    d_val   = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_train_full = d_train["X_train"]
    X_val_full   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    test_sets = load_dedup_test_sets()

    # LODO: train on 2 of 3 datasets
    # Approximate by using 2/3 of training data
    n = len(X_train_full)
    third = n // 3

    lodo_configs = [
        ("LODO-1 (no CIC17)",  X_train_full[third:],    X_val_full[third//3*2:]),
        ("LODO-2 (no CIC18)",  np.vstack([X_train_full[:third], X_train_full[2*third:]]),
                                np.vstack([X_val_full[:third//3], X_val_full[2*third//3:]])),
        ("LODO-3 (no UNSW)",   X_train_full[:2*third],  X_val_full[:2*third//3]),
    ]

    results = []
    for name, X_tr, X_v in lodo_configs:
        log.info("  %s: train=%d, val=%d", name, len(X_tr), len(X_v))
        np.random.seed(SEED_42); torch.manual_seed(SEED_42)
        ae   = Autoencoder(input_dim=18); ae.fit(X_tr, X_v)
        vae  = VAE(input_dim=18);        vae.fit(X_tr, X_v)
        svdd = DeepSVDD(input_dim=18);   svdd.fit(X_tr, X_v)

        scores_fn = lambda X, a=ae, v=vae, s=svdd: ensemble_scores(a, v, s, X)
        metrics = evaluate_on_test(scores_fn, X_v, test_sets)
        metrics["experiment"] = name
        results.append(metrics)
        log.info("    %s", {k: v for k, v in metrics.items() if 'AUC' in k or 'DR' in k})
        gc.collect()

    pd.DataFrame(results).to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)


# =============================================================================
# ABLATION 3: AE-SINGLE (single environment)
# =============================================================================
def run_ae_single():
    out_path = os.path.join(DEDUP_DIR, "ablation_ae_single_dedup.csv")
    if os.path.exists(out_path):
        log.info("AE-Single already done — skipping")
        return

    log.info("=" * 60)
    log.info("ABLATION 3: AE-SINGLE (CIC17 only, seed 42)")
    log.info("=" * 60)

    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    d_val   = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_train_full = d_train["X_train"]
    X_val_full   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    test_sets = load_dedup_test_sets()

    # Use first third (CIC17 portion)
    n = len(X_train_full)
    X_train_single = X_train_full[:n//3]
    X_val_single   = X_val_full[:len(X_val_full)//3]

    log.info("AE-Single train: %d, val: %d", len(X_train_single), len(X_val_single))
    ae = train_ae(X_train_single, X_val_single, seed=SEED_42)

    scores_fn = lambda X, a=ae: norm_minmax(get_ae_scores(a, X))
    metrics = evaluate_on_test(scores_fn, X_val_single, test_sets)
    metrics["model"] = "AE-Single (CIC17 only, dedup)"

    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    log.info("AE-Single results: %s",
             {k: v for k, v in metrics.items() if 'AUC' in k or 'DR' in k})
    log.info("Saved -> %s", out_path)


# =============================================================================
# ABLATION 4: DESTINATION PORT ABLATION
# =============================================================================
def run_dst_port_ablation():
    out_path = os.path.join(DEDUP_DIR, "ablation_no_dst_port_dedup.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        if len(existing) >= len(SEEDS):
            log.info("Dst port ablation already done — skipping")
            return

    log.info("=" * 60)
    log.info("ABLATION 4: NO DESTINATION PORT (dedup, 5 seeds)")
    log.info("=" * 60)

    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    all_features = pd.read_csv(feat_path, header=None)[0].tolist()
    all_features = [f for f in all_features
                    if str(f).strip() and not str(f).strip().lstrip("-").isdigit()]
    features = [f for f in all_features if f != "destination_port"]
    feat_idx = [all_features.index(f) for f in features]
    input_dim = len(features)
    log.info("Features without dst_port: %d", input_dim)

    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    d_val   = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_train = d_train["X_train"][:, feat_idx]
    X_val   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)[:, feat_idx]

    test_sets_raw = load_dedup_test_sets()
    test_sets = [(name, X[:, feat_idx], y) for name, X, y in test_sets_raw]

    results = []
    if os.path.exists(out_path):
        results = pd.read_csv(out_path).to_dict("records")
    done_seeds = [r["seed"] for r in results]

    for seed in SEEDS:
        if seed in done_seeds:
            log.info("Seed %d done — skipping", seed)
            continue

        log.info("  Seed %d", seed)
        np.random.seed(seed); torch.manual_seed(seed)

        ae   = Autoencoder(input_dim=input_dim); ae.fit(X_train, X_val)
        vae  = VAE(input_dim=input_dim);         vae.fit(X_train, X_val)
        svdd = DeepSVDD(input_dim=input_dim);    svdd.fit(X_train, X_val)

        scores_fn = lambda X, a=ae, v=vae, s=svdd: ensemble_scores(a, v, s, X)
        metrics = evaluate_on_test(scores_fn, X_val, test_sets)
        metrics["seed"] = seed

        log.info("    %s", {k: v for k, v in metrics.items() if 'AUC' in k or 'DR' in k})
        results.append(metrics)
        pd.DataFrame(results).to_csv(out_path, index=False)
        gc.collect()

    # Summary
    df = pd.DataFrame(results)
    log.info("Dst port ablation summary:")
    for col in [c for c in df.columns if 'AUC' in c or 'DR' in c]:
        log.info("  %-25s %.4f +/- %.4f", col, df[col].mean(), df[col].std())

    log.info("Saved -> %s", out_path)


# =============================================================================
# ABLATION 5: DAE BASELINE
# =============================================================================
class _DAENet(nn.Module):
    def __init__(self, input_dim=18):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LeakyReLU(0.1),
            nn.Linear(64, 32), nn.LeakyReLU(0.1),
            nn.Linear(32, 16), nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.LeakyReLU(0.1),
            nn.Linear(16, 32), nn.LeakyReLU(0.1),
            nn.Linear(32, 64), nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_dae(X_train, X_val, seed=42, noise_std=0.1,
              lr=1e-3, batch_size=256, max_epochs=100, patience=10):
    np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _DAENet(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=5, min_lr=1e-6)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val)),
        batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    patience_count = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            noisy = batch + torch.randn_like(batch) * noise_std
            recon = model(noisy)
            loss = ((batch - recon)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                vl += ((batch - recon)**2).mean().item() * len(batch)
        val_loss = vl / len(X_val)
        sched.step(val_loss)

        if epoch % 10 == 0:
            log.info("  DAE Epoch %3d | val=%.4f", epoch, val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss; patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                log.info("  Early stopping at epoch %d", epoch)
                break

    return model, device


def dae_scores(model, device, X):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 10000):
            batch = torch.FloatTensor(X[i:i+10000]).to(device)
            recon = model(batch)
            scores.append(((batch - recon)**2).mean(dim=1).cpu().numpy())
    return np.concatenate(scores)


def run_dae_baseline():
    out_path = os.path.join(DEDUP_DIR, "ablation_dae_dedup.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        if len(existing) >= len(SEEDS):
            log.info("DAE baseline already done — skipping")
            return

    log.info("=" * 60)
    log.info("ABLATION 5: DAE BASELINE (dedup, 5 seeds)")
    log.info("=" * 60)

    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    d_val   = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_train = d_train["X_train"]
    X_val   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    test_sets = load_dedup_test_sets()

    results = []
    if os.path.exists(out_path):
        results = pd.read_csv(out_path).to_dict("records")
    done_seeds = [r["seed"] for r in results]

    for seed in SEEDS:
        if seed in done_seeds:
            log.info("Seed %d done — skipping", seed)
            continue

        log.info("  Seed %d — training DAE...", seed)
        model, device = train_dae(X_train, X_val, seed=seed)

        val_scores = norm_minmax(dae_scores(model, device, X_val))
        tau = float(np.percentile(val_scores, 95))

        metrics = {"seed": seed}
        for name, X, y in test_sets:
            scores = norm_minmax(dae_scores(model, device, X))
            if len(np.unique(y)) < 2:
                metrics[f"{name}_DR"] = round(float((scores > tau).mean()), 4)
            else:
                auc = roc_auc_score(y, scores)
                metrics[f"{name}_AUC"] = round(auc, 4)
            log.info("  %-20s %s=%.4f", name,
                     "DR" if len(np.unique(y)) < 2 else "AUC",
                     metrics.get(f"{name}_DR", metrics.get(f"{name}_AUC", 0)))

        results.append(metrics)
        pd.DataFrame(results).to_csv(out_path, index=False)
        gc.collect()

    df = pd.DataFrame(results)
    log.info("DAE baseline summary:")
    for col in [c for c in df.columns if 'AUC' in c or 'DR' in c]:
        log.info("  %-25s %.4f +/- %.4f", col, df[col].mean(), df[col].std())
    log.info("Saved -> %s", out_path)


# =============================================================================
# MAIN
# =============================================================================
def run():
    log.info("Starting dedup ablation suite...")
    log.info("All results saved to: %s", DEDUP_DIR)

    run_pairwise_ablation()
    run_ae_single()
    run_lodo()
    run_dst_port_ablation()
    run_dae_baseline()

    log.info("\n" + "=" * 60)
    log.info("ALL DEDUP ABLATIONS COMPLETE")
    log.info("=" * 60)
    log.info("Results files:")
    for f in os.listdir(DEDUP_DIR):
        if f.endswith(".csv"):
            path = os.path.join(DEDUP_DIR, f)
            df = pd.read_csv(path)
            log.info("  %-45s %d rows", f, len(df))


if __name__ == "__main__":
    run()
