"""
models/deep_svdd.py
===================
Deep Support Vector Data Description (Deep SVDD).

Learns a neural network mapping that encloses normal training data
within a minimum-volume hypersphere in latent space.

Anomaly score = distance from the learned hypersphere centre.
Attacks lie far from the centre → high score.

Reference:
    Ruff et al. (2018) "Deep One-Class Classification"
    ICML 2018. https://arxiv.org/abs/1906.02629

Architecture:
    input_dim → 64 → 32 → 16 → latent_dim
    No bias terms (standard SVDD practice — prevents hypersphere collapse)
    LeakyReLU activations

Training:
    Loss = mean(||phi(x) - c||^2)  where c = centre of hypersphere
    Centre c is fixed after warm-up (mean of initial encodings)
    Minimising this loss pushes all normal data toward c

Key difference from AE:
    AE minimises reconstruction error (input space)
    SVDD minimises distance from centre (latent space)
    SVDD is therefore more directly optimised for anomaly detection
"""

import logging
import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


# =============================================================================
# NETWORK
# =============================================================================

class _SVDDNet(nn.Module):
    """
    Encoder-only network for Deep SVDD.
    No bias terms to prevent hypersphere collapse.
    """
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64, bias=False), nn.LeakyReLU(0.1),
            nn.Linear(64, 32,        bias=False), nn.LeakyReLU(0.1),
            nn.Linear(32, 16,        bias=False), nn.LeakyReLU(0.1),
            nn.Linear(16, latent_dim,bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# DEEP SVDD WRAPPER
# =============================================================================

class DeepSVDD:
    """
    Deep SVDD one-class anomaly detector.

    Same interface as Autoencoder and VAE:
        fit(X_train, X_val)
        anomaly_scores(X) → np.ndarray
        fit_threshold(X_val)
        predict(X) → np.ndarray
        save(path) / load(path)
    """

    def __init__(self,
                 input_dim:   int   = 32,
                 latent_dim:  int   = 16,
                 lr:          float = 1e-3,
                 batch_size:  int   = 256,
                 max_epochs:  int   = 100,
                 patience:    int   = 10,
                 warmup_epochs: int = 10,
                 n_sigma:     float = 3.0):

        self.input_dim     = input_dim
        self.latent_dim    = latent_dim
        self.lr            = lr
        self.batch_size    = batch_size
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.warmup_epochs = warmup_epochs
        self.n_sigma       = n_sigma

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_    = None
        self.centre_   = None   # hypersphere centre c (latent_dim,)
        self.radius_   = None   # soft boundary radius
        self.threshold_= None
        self.score_mu_ = None
        self.score_sigma_ = None
        self._fitted   = False

        log.info("DeepSVDD initialised | input_dim=%d | latent_dim=%d | device=%s",
                 input_dim, latent_dim, self.device)

    # =========================================================================
    # FITTING
    # =========================================================================

    def fit(self, X_train: np.ndarray,
            X_val:   Optional[np.ndarray] = None) -> "DeepSVDD":

        self.model_ = _SVDDNet(self.input_dim, self.latent_dim).to(self.device)
        optimiser   = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=5, min_lr=1e-6)

        train_loader = self._make_loader(X_train, shuffle=True)
        val_loader   = self._make_loader(X_val,   shuffle=False) if X_val is not None else None

        log.info("Training DeepSVDD | train=%d | val=%s | device=%s",
                 len(X_train),
                 len(X_val) if X_val is not None else "None",
                 self.device)

        # ── Warm-up: initialise centre from mean of initial encodings ──────
        log.info("  Warm-up: computing hypersphere centre (%d epochs) ...",
                 self.warmup_epochs)
        self.model_.train()
        with torch.no_grad():
            z_sum = torch.zeros(self.latent_dim).to(self.device)
            n     = 0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                z     = self.model_(batch)
                z_sum += z.sum(dim=0)
                n     += len(batch)
        centre = z_sum / n
        # Avoid centre collapse: if any dim is near zero, push it slightly
        centre[(centre.abs() < 0.01)] = 0.01
        self.centre_ = centre.detach()
        log.info("  Centre initialised | norm=%.4f", self.centre_.norm().item())

        # ── Main training ─────────────────────────────────────────────────
        best_val_loss = float("inf")
        patience_count = 0
        smooth = None

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                z     = self.model_(batch)
                # Loss = mean squared distance from centre
                dist  = torch.sum((z - self.centre_) ** 2, dim=1)
                loss  = dist.mean()
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimiser.step()
                train_loss += loss.item() * len(batch)

            train_loss /= len(X_train)

            # Validation
            val_loss = train_loss
            if val_loader is not None:
                self.model_.eval()
                vl = 0.0
                with torch.no_grad():
                    for (batch,) in val_loader:
                        batch = batch.to(self.device)
                        z     = self.model_(batch)
                        dist  = torch.sum((z - self.centre_) ** 2, dim=1)
                        vl   += dist.mean().item() * len(batch)
                val_loss = vl / len(X_val)

            smooth = val_loss if smooth is None else 0.9 * smooth + 0.1 * val_loss
            scheduler.step(val_loss)
            lr_now = optimiser.param_groups[0]["lr"]

            if epoch % 10 == 0 or epoch <= 5:
                log.info("Epoch %3d/%d | train=%.4f | val=%.4f | smooth=%.4f | lr=%.2e",
                         epoch, self.max_epochs, train_loss, val_loss, smooth, lr_now)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_count = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    log.info("Early stopping at epoch %d | best_val=%.6f",
                             epoch, best_val_loss)
                    break

        self.model_.load_state_dict(best_state)
        self._fitted = True
        log.info("DeepSVDD training complete. Best val_loss=%.6f", best_val_loss)
        return self

    def fit_score_stats(self, X_train: np.ndarray) -> "DeepSVDD":
        """Fit mu/sigma of anomaly scores on benign training data (REN)."""
        scores = self._raw_distances(X_train)
        self.score_mu_    = float(scores.mean())
        self.score_sigma_ = float(scores.std()) + 1e-8
        log.info("SVDD score stats: mu=%.6f  sigma=%.6f",
                 self.score_mu_, self.score_sigma_)
        return self

    def fit_threshold(self, X_val: np.ndarray,
                      percentile: float = 99.0) -> float:
        """Set threshold as µ + n_sigma*σ on val distances."""
        scores    = self._raw_distances(X_val)
        mu        = float(scores.mean())
        sigma     = float(scores.std())
        tau_sigma = mu + self.n_sigma * sigma
        tau_pct   = float(np.percentile(scores, percentile))
        self.threshold_ = min(tau_sigma, tau_pct * 3.0)
        log.info("SVDD errors: µ=%.4f  σ=%.4f  p%.0f=%.4f",
                 mu, sigma, percentile, tau_pct)
        log.info("SVDD threshold τ: sigma-based=%.4f | p99-cap=%.4f | chosen=%.4f",
                 tau_sigma, tau_pct * 3.0, self.threshold_)
        return self.threshold_

    # =========================================================================
    # INFERENCE
    # =========================================================================

    def _raw_distances(self, X: np.ndarray) -> np.ndarray:
        """Euclidean distance from centre in latent space."""
        self.model_.eval()
        loader = self._make_loader(X, shuffle=False)
        dists  = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                z     = self.model_(batch)
                dist  = torch.sqrt(
                    torch.sum((z - self.centre_) ** 2, dim=1)
                )
                dists.append(dist.cpu().numpy())
        return np.concatenate(dists, axis=0).astype(np.float32)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        REN z-score normalised anomaly scores.
        Falls back to raw distances if score stats not fitted.
        """
        raw = self._raw_distances(X)
        if self.score_mu_ is not None:
            return (raw - self.score_mu_) / self.score_sigma_
        return raw

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call fit_threshold() first.")
        return (self.anomaly_scores(X) > self.threshold_).astype(int)

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state":  self.model_.state_dict(),
            "centre":       self.centre_.cpu().numpy(),
            "threshold":    self.threshold_,
            "input_dim":    self.input_dim,
            "latent_dim":   self.latent_dim,
            "score_mu":     self.score_mu_,
            "score_sigma":  self.score_sigma_,
        }, path)
        log.info("DeepSVDD saved -> '%s'", path)

    def load(self, path: str) -> "DeepSVDD":
        cp = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim    = cp["input_dim"]
        self.latent_dim   = cp["latent_dim"]
        self.model_       = _SVDDNet(self.input_dim, self.latent_dim).to(self.device)
        self.model_.load_state_dict(cp["model_state"])
        self.centre_      = torch.tensor(cp["centre"]).to(self.device)
        self.threshold_   = cp["threshold"]
        self.score_mu_    = cp.get("score_mu",    None)
        self.score_sigma_ = cp.get("score_sigma", None)
        self._fitted      = True
        log.info("DeepSVDD loaded <- '%s' | threshold=%.6f", path, self.threshold_)
        return self

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _make_loader(self, X: np.ndarray, shuffle: bool) -> DataLoader:
        t = torch.tensor(X, dtype=torch.float32)
        return DataLoader(TensorDataset(t),
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          num_workers=0,
                          pin_memory=False)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
