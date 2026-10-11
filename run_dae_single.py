"""
run_dae_single.py
=================
Trains a Denoising Autoencoder (DAE) on CIC-IDS2017 ONLY
across 5 seeds, to compare against DAE-Multi (3 environments).

This is the controlled experiment proving:
  Same architecture (DAE) + 1 env  vs  Same architecture (DAE) + 3 envs
  → Diversity drives generalization, not architecture

Output:
  artefacts_v2/dedup/dae_single_results.csv

Run from project root:
    python run_dae_single.py

Expected time: ~8 hours (5 seeds x ~90 min each)
"""
import sys, os, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
DEDUP_DIR    = os.path.join(ARTEFACT_DIR, "dedup")
OUT_DIR      = DEDUP_DIR
SEEDS        = [42, 123, 456, 789, 999]
NOISE_STD    = 0.1   # Gaussian noise for DAE
EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 1e-3
PATIENCE     = 10
INPUT_DIM    = 18


# ── DAE Model ─────────────────────────────────────────────────────────────
class DAEModel(nn.Module):
    def __init__(self, input_dim=18):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, 8),         nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),         nn.ReLU(),
            nn.Linear(16, 32),        nn.ReLU(),
            nn.Linear(32, 64),        nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_dae(X_train, X_val, seed):
    """Train DAE with Gaussian noise corruption."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DAEModel(INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss(delta=1.0)

    X_train_t = torch.FloatTensor(X_train).to(device)
    X_val_t   = torch.FloatTensor(X_val).to(device)

    best_val_loss = float('inf')
    patience_count = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        # Shuffle
        idx = torch.randperm(len(X_train_t))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(X_train_t), BATCH_SIZE):
            batch = X_train_t[idx[i:i+BATCH_SIZE]]
            # Add Gaussian noise
            noisy = batch + torch.randn_like(batch) * NOISE_STD
            noisy = torch.clamp(noisy, -10, 10)

            optimizer.zero_grad()
            recon = model(noisy)
            loss = criterion(recon, batch)  # Reconstruct CLEAN from NOISY
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            noisy_val = X_val_t + torch.randn_like(X_val_t) * NOISE_STD
            noisy_val = torch.clamp(noisy_val, -10, 10)
            val_loss = criterion(model(noisy_val), X_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log.info("  Early stopping at epoch %d", epoch+1)
                break

    model.load_state_dict(best_state)
    return model


def get_scores(model, X, device):
    """Compute reconstruction error as anomaly score."""
    model.eval()
    X_t = torch.FloatTensor(X).to(device)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_t), 10000):
            batch = X_t[i:i+10000]
            recon = model(batch)
            mse = ((batch - recon)**2).mean(dim=1)
            scores.append(mse.cpu().numpy())
    return np.concatenate(scores)


def norm_scores(scores, p1, p99):
    """Min-max normalize using percentile bounds."""
    eps = 1e-8
    return np.clip((scores - p1) / (p99 - p1 + eps), 0, 1)


def evaluate(model, X_test, y_test, p1, p99, device):
    """Compute ROC-AUC on test set."""
    scores = get_scores(model, X_test, device)
    scores_norm = norm_scores(scores, p1, p99)
    if len(np.unique(y_test)) < 2:
        return None
    return roc_auc_score(y_test, scores_norm)


def run():
    log.info("Loading deduplicated data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Device: %s", device)

    # Load full training data
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(d_train["X_train"], -10, 10).astype(np.float32)
    n_total = len(X_train_full)
    n_per_env = n_total // 3

    # CIC17 portion only (first third)
    X_train_single = X_train_full[:n_per_env]
    log.info("CIC17-only training set: %d flows", len(X_train_single))

    # Validation (CIC17 portion only)
    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    n_val_per_env = len(X_val_full) // 3
    X_val_single = X_val_full[:n_val_per_env]
    log.info("CIC17-only validation set: %d flows", len(X_val_single))

    # Load test sets
    test_sets = {}
    for name, fname in [
        ("CIC17", "test_cic17_dedup.npz"),
        ("CIC18", "test_cic18_dedup.npz"),
        ("UNSW",  "test_unsw_dedup.npz"),
        ("BoT",   "test_botiot_dedup.npz"),
    ]:
        path = os.path.join(DEDUP_DIR, fname)
        if os.path.exists(path):
            d = np.load(path)
            test_sets[name] = (
                np.clip(d["X"], -10, 10).astype(np.float32),
                d["y"].astype(int)
            )
            log.info("Loaded %s: %d rows", name, len(test_sets[name][0]))

    results = []

    for seed in SEEDS:
        log.info("="*50)
        log.info("SEED %d", seed)
        log.info("="*50)
        t0 = time.time()

        # Train DAE-Single
        log.info("Training DAE-Single (CIC17 only, seed %d)...", seed)
        model = train_dae(X_train_single, X_val_single, seed)

        # Compute normalization parameters from training scores
        train_scores = get_scores(model, X_train_single, device)
        p1  = float(np.percentile(train_scores, 1))
        p99 = float(np.percentile(train_scores, 99))

        # Compute p95 threshold from validation
        val_scores = get_scores(model, X_val_single, device)
        val_norm = norm_scores(val_scores, p1, p99)
        tau = float(np.percentile(val_norm, 95))
        log.info("  tau=%.4f", tau)

        row = {'seed': seed, 'tau': tau}

        # Evaluate on all test sets
        for name, (X_test, y_test) in test_sets.items():
            if name == "BoT":
                # BoT is attack-only — compute DR
                scores = get_scores(model, X_test, device)
                scores_norm = norm_scores(scores, p1, p99)
                dr = float((scores_norm > tau).mean())
                row['BoT-IoT_DR'] = dr
                log.info("  BoT DR=%.4f", dr)
            else:
                auc = evaluate(model, X_test, y_test, p1, p99, device)
                col = {
                    'CIC17': 'CIC-IDS2017_AUC',
                    'CIC18': 'CSE-CIC-IDS2018_AUC',
                    'UNSW':  'UNSW-NB15_AUC',
                }[name]
                row[col] = auc
                log.info("  %s AUC=%.4f", name, auc)

        elapsed = time.time() - t0
        log.info("Seed %d done in %.1f min", seed, elapsed/60)
        results.append(row)

    # Save results
    df = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, "dae_single_results.csv")
    df.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)

    # Print summary
    print("\nDAE-SINGLE RESULTS SUMMARY:")
    print(df.to_string(index=False))
    print()

    cols = {
        'CIC17': 'CIC-IDS2017_AUC',
        'CIC18': 'CSE-CIC-IDS2018_AUC',
        'UNSW':  'UNSW-NB15_AUC',
        'BoT':   'BoT-IoT_DR',
    }

    print("MEAN ± STD:")
    for name, col in cols.items():
        if col in df.columns:
            m = df[col].mean()
            s = df[col].std()
            print(f"  {name}: {m:.4f} ± {s:.4f}")

    # Compare with DAE-Multi
    dae_multi = {
        'CIC17': 0.9249,
        'CIC18': 0.9646,
        'UNSW':  0.9432,
        'BoT':   0.6789,
    }
    print("\nDAE-SINGLE vs DAE-MULTI:")
    print(f"{'Dataset':<10} {'Single':>8} {'Multi':>8} {'Gain':>8}")
    print("-"*40)
    for name, col in cols.items():
        if col in df.columns:
            single = df[col].mean()
            multi  = dae_multi[name]
            gain   = multi - single
            print(f"{name:<10} {single:>8.4f} {multi:>8.4f} {gain:>+8.4f}")


if __name__ == "__main__":
    run()
