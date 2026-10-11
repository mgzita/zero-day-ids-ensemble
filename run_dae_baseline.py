"""
run_dae_baseline.py
====================
Trains a Denoising Autoencoder (DAE) as a modern deep baseline.

DAE adds Gaussian noise to the input during training and learns
to reconstruct the clean input. At inference, anomaly score is
the reconstruction error on the clean input.

Reference:
    Vincent et al. (2008) "Extracting and Composing Robust Features
    with Denoising Autoencoders." ICML 2008.

Same multi-environment training protocol as main ensemble:
    - Trained on 1.8M benign flows from 3 environments
    - Evaluated on all 5 datasets
    - 5 random seeds for stability

Run from project root:
    python run_dae_baseline.py

Results saved to: artefacts_v2/dae_baseline_results.csv
Expected time: ~8 hours (5 seeds × ~1.5 hours)
"""
import sys, os, gc, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import roc_auc_score, f1_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
OUT_PATH     = os.path.join(ARTEFACT_DIR, "dae_baseline_results.csv")
SEEDS        = [42, 123, 456, 789, 999]

# DAE hyperparameters (same as AE where possible)
INPUT_DIM   = 18
HIDDEN_DIMS = [64, 32, 16, 8]
LR          = 1e-3
BATCH_SIZE  = 256
MAX_EPOCHS  = 100
PATIENCE    = 10
NOISE_STD   = 0.1   # Gaussian noise standard deviation


# =============================================================================
# DAE NETWORK
# =============================================================================
class _DAENet(nn.Module):
    """
    Denoising Autoencoder network.
    Same architecture as AE but trained with noisy input.
    18 → 64 → 32 → 16 → 8 → 16 → 32 → 64 → 18
    """
    def __init__(self, input_dim: int = 18):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LeakyReLU(0.1),
            nn.Linear(64, 32),        nn.LeakyReLU(0.1),
            nn.Linear(32, 16),        nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),          nn.LeakyReLU(0.1),
            nn.Linear(16, 32),         nn.LeakyReLU(0.1),
            nn.Linear(32, 64),         nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# =============================================================================
# DAE TRAINER
# =============================================================================
class DenoisingAutoencoder:
    def __init__(self, input_dim=18, noise_std=NOISE_STD,
                 lr=LR, batch_size=BATCH_SIZE,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE):
        self.input_dim  = input_dim
        self.noise_std  = noise_std
        self.lr         = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience   = patience
        self.device     = torch.device("cuda" if torch.cuda.is_available()
                                        else "cpu")
        self.model_     = None

    def _make_loader(self, X, shuffle=True):
        ds = TensorDataset(torch.FloatTensor(X))
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=shuffle, num_workers=0)

    def fit(self, X_train, X_val=None):
        self.model_ = _DAENet(self.input_dim).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)

        train_loader = self._make_loader(X_train, shuffle=True)
        val_loader   = self._make_loader(X_val, shuffle=False) \
                       if X_val is not None else None

        best_val = float("inf")
        patience_count = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                # Add Gaussian noise to input
                noisy = batch + torch.randn_like(batch) * self.noise_std
                recon = self.model_(noisy)
                # Reconstruct CLEAN input from NOISY input
                loss = ((batch - recon) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(batch)
            train_loss /= len(X_train)

            val_loss = None
            if val_loader:
                self.model_.eval()
                vl = 0.0
                with torch.no_grad():
                    for (batch,) in val_loader:
                        batch = batch.to(self.device)
                        # Val: reconstruct clean from clean (no noise at inference)
                        recon = self.model_(batch)
                        vl += ((batch - recon) ** 2).mean().item() * len(batch)
                val_loss = vl / len(X_val)
                sched.step(val_loss)

                if epoch % 10 == 0 or epoch == 1:
                    log.info("  Epoch %3d/%d | train=%.4f | val=%.4f | lr=%.2e",
                             epoch, self.max_epochs, train_loss, val_loss,
                             opt.param_groups[0]["lr"])

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        log.info("  Early stopping at epoch %d", epoch)
                        break

        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Reconstruction error on CLEAN input (no noise at inference)."""
        self.model_.eval()
        scores = []
        X_t = torch.FloatTensor(X)
        with torch.no_grad():
            for i in range(0, len(X), 10000):
                batch = X_t[i:i+10000].to(self.device)
                recon = self.model_(batch)
                err = ((batch - recon) ** 2).mean(dim=1)
                scores.append(err.cpu().numpy())
        return np.concatenate(scores)


# =============================================================================
# MAIN
# =============================================================================
def norm_minmax(s):
    lo = float(np.percentile(s, 1))
    hi = float(np.percentile(s, 99))
    if hi - lo < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_existing():
    if os.path.exists(OUT_PATH):
        df = pd.read_csv(OUT_PATH)
        done = df["seed"].tolist()
        log.info("Found existing results for seeds: %s", done)
        return df.to_dict("records"), done
    return [], []


def run():
    log.info("Loading data...")
    d_train = np.load(os.path.join(ARTEFACT_DIR, "train.npz"))
    d_val   = np.load(os.path.join(ARTEFACT_DIR, "val.npz"))
    X_train = d_train["X_train"]
    X_val   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    log.info("Train: %s  Val: %s", X_train.shape, X_val.shape)

    test_sets = [
        ("CIC-IDS2017",     "test_cic17.npz"),
        ("CSE-CIC-IDS2018", "test_cic18.npz"),
        ("UNSW-NB15",       "test_unsw.npz"),
        ("BoT-IoT",         "test_botiot.npz"),
    ]

    results, done_seeds = load_existing()

    for seed in SEEDS:
        if seed in done_seeds:
            log.info("Seed %d already done — skipping", seed)
            continue

        log.info("=" * 60)
        log.info("SEED %d", seed)
        log.info("=" * 60)

        np.random.seed(seed)
        torch.manual_seed(seed)

        dae = DenoisingAutoencoder(input_dim=INPUT_DIM, noise_std=NOISE_STD)
        log.info("Training DAE (noise_std=%.2f)...", NOISE_STD)
        dae.fit(X_train, X_val)

        # Get val scores for threshold
        val_scores = norm_minmax(dae.anomaly_scores(X_val))
        tau = float(np.percentile(val_scores, 95))
        log.info("Val tau (p95) = %.4f", tau)

        seed_result = {"seed": seed, "noise_std": NOISE_STD}

        for name, fname in test_sets:
            path = os.path.join(ARTEFACT_DIR, fname)
            if not os.path.exists(path):
                log.warning("Not found: %s", path)
                continue

            d = np.load(path)
            X = np.clip(d["X"], -10, 10).astype(np.float32)
            y = d["y"].astype(int)

            scores = norm_minmax(dae.anomaly_scores(X))

            if len(np.unique(y)) < 2:
                dr = float((scores > tau).mean())
                seed_result[f"{name}_DR"] = round(dr, 4)
                log.info("  %-20s DR  = %.4f", name, dr)
            else:
                auc = roc_auc_score(y, scores)
                rec = recall_score(y, (scores > tau).astype(int), zero_division=0)
                f1  = f1_score(y, (scores > tau).astype(int), zero_division=0)
                seed_result[f"{name}_AUC"] = round(auc, 4)
                seed_result[f"{name}_Recall"] = round(rec, 4)
                seed_result[f"{name}_F1"] = round(f1, 4)
                log.info("  %-20s AUC=%.4f Rec=%.4f F1=%.4f",
                         name, auc, rec, f1)

        results.append(seed_result)
        pd.DataFrame(results).to_csv(OUT_PATH, index=False)
        log.info("Saved -> %s", OUT_PATH)
        gc.collect()

    # Summary
    df = pd.DataFrame(results)
    log.info("\n" + "=" * 60)
    log.info("DAE BASELINE SUMMARY (mean +/- std, %d seeds)", len(SEEDS))
    log.info("=" * 60)

    main_auc = {
        "CIC-IDS2017_AUC":     0.912,
        "CSE-CIC-IDS2018_AUC": 0.959,
        "UNSW-NB15_AUC":       0.884,
        "BoT-IoT_DR":          0.827,
    }

    for col in [c for c in df.columns if c not in ["seed", "noise_std"]]:
        mean_val = df[col].mean()
        std_val  = df[col].std()
        ref = main_auc.get(col)
        diff = f"  (diff vs ensemble: {mean_val-ref:+.3f})" if ref else ""
        log.info("  %-25s %.4f +/- %.4f%s", col, mean_val, std_val, diff)

    log.info("Done.")


if __name__ == "__main__":
    run()
