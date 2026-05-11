"""
models/autoencoder.py
=====================
Autoencoder (AE) for zero-day attack detection.

Architecture  (paper section 4.3)
----------------------------------
Encoder : input -> 64 -> 32 -> 16 -> 8   (bottleneck)
Decoder : 8 -> 16 -> 32 -> 64 -> input

Activation  : ReLU on all hidden layers
Output layer: linear (no activation) -- regression reconstruction
Loss        : Huber loss (delta=1.0)  -- robust to outliers vs MSE
Optimiser   : Adam (lr=1e-3)

Training
--------
- Trained on BENIGN traffic only (X_train)
- Validated on BENIGN traffic only (X_val)
- Early stopping on val loss (patience=10)

Anomaly detection
-----------------
Reconstruction error = mean Huber loss per sample.
Score normalisation (REN): score = (error - mu_train) / sigma_train
Converts raw error into a z-score (dataset-agnostic).
Threshold τ = µ + 3σ computed on normalised X_val scores.
"""

import logging
import os
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class _AENet(nn.Module):
    """
    Symmetric autoencoder:
        Encoder: input_dim -> 64 -> 32 -> 16 -> 8
        Decoder: 8 -> 16 -> 32 -> 64 -> input_dim
    """

    def __init__(self, input_dim: int):
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
            nn.Linear(64, input_dim), # linear output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent vector z for input x."""
        return self.encoder(x)


# =============================================================================
# AUTOENCODER WRAPPER
# =============================================================================

class Autoencoder:
    """
    Autoencoder wrapper with fit / predict / save / load.

    Usage
    -----
    ae = Autoencoder(input_dim=32)
    ae.fit(X_train, X_val)
    scores = ae.reconstruction_error(X_test)
    ae.fit_threshold(X_val)
    y_pred = ae.predict(X_test)
    ae.save("artefacts/ae.pt")
    """

    def __init__(self,
                 input_dim:   int   = 32,
                 lr:          float = 1e-3,
                 batch_size:  int   = 256,
                 max_epochs:  int   = 100,
                 patience:    int   = 10,
                 huber_delta: float = 1.0,
                 device:      str   = None):

        self.input_dim   = input_dim
        self.lr          = lr
        self.batch_size  = batch_size
        self.max_epochs  = max_epochs
        self.patience    = patience
        self.huber_delta = huber_delta
        self.device      = device or ("cuda" if torch.cuda.is_available()
                                      else "cpu")

        self.model_       = _AENet(input_dim).to(self.device)
        self.threshold_   = None
        self.score_mu_    = None
        self.score_sigma_ = None
        self._fitted      = False
        self.grad_clip   = 1.0    # max gradient norm -- prevents loss spikes
        self.smooth_beta = 0.9    # exponential smoothing for val loss tracking

        log.info("Autoencoder initialised | input_dim=%d | device=%s",
                 input_dim, self.device)

    # =========================================================================
    # TRAINING
    # =========================================================================

    def fit(self,
            X_train: np.ndarray,
            X_val:   np.ndarray) -> "Autoencoder":
        """
        Train on benign X_train, validate on benign X_val.
        Early stopping on validation loss.

        Parameters
        ----------
        X_train : scaled benign training data  (n_train x input_dim)
        X_val   : scaled benign validation data (n_val   x input_dim)

        Returns self for chaining.
        """
        log.info("Training Autoencoder | train=%d | val=%d | device=%s",
                 len(X_train), len(X_val), self.device)

        train_loader = self._make_loader(X_train, shuffle=True)
        val_loader   = self._make_loader(X_val,   shuffle=False)

        optimiser  = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                         optimiser, mode="min", factor=0.5,
                         patience=5, min_lr=1e-6)
        loss_fn    = nn.HuberLoss(delta=self.huber_delta)

        best_val_loss  = float("inf")
        smoothed_val   = None   # exponentially smoothed val loss
        patience_count = 0
        best_weights   = None

        for epoch in range(1, self.max_epochs + 1):
            # ── Train ────────────────────────────────────────────────────
            self.model_.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimiser.zero_grad()
                recon = self.model_(batch)
                loss  = loss_fn(recon, batch)
                loss.backward()
                # Gradient clipping -- prevents loss spikes
                nn.utils.clip_grad_norm_(self.model_.parameters(),
                                         self.grad_clip)
                optimiser.step()
                train_loss += loss.item() * len(batch)
            train_loss /= len(X_train)

            # ── Validate ─────────────────────────────────────────────────
            val_loss = self._eval_loss(val_loader, loss_fn)

            # Exponential smoothing to reduce spike sensitivity
            if smoothed_val is None:
                smoothed_val = val_loss
            else:
                smoothed_val = (self.smooth_beta * smoothed_val
                                + (1 - self.smooth_beta) * val_loss)

            # Step LR scheduler on raw val loss
            scheduler.step(val_loss)
            current_lr = optimiser.param_groups[0]["lr"]

            log.info(
                "Epoch %3d/%d | train=%.4f | val=%.4f | smooth=%.4f | lr=%.2e",
                epoch, self.max_epochs, train_loss, val_loss,
                smoothed_val, current_lr
            )

            # ── Early stopping on SMOOTHED val loss ──────────────────────
            if smoothed_val < best_val_loss:
                best_val_loss  = smoothed_val
                patience_count = 0
                best_weights   = {k: v.cpu().clone()
                                  for k, v in self.model_.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    log.info("Early stopping at epoch %d (patience=%d).",
                             epoch, self.patience)
                    break

        # Restore best weights
        if best_weights:
            self.model_.load_state_dict(
                {k: v.to(self.device) for k, v in best_weights.items()}
            )

        self._fitted = True
        log.info("AE training complete. Best val_loss=%.6f", best_val_loss)
        return self

    # =========================================================================
    # THRESHOLD
    # =========================================================================

    def fit_threshold(self, X_val: np.ndarray,
                      n_sigma:     float = 3.0,
                      percentile:  float = 99.0) -> float:
        """
        Compute anomaly threshold on benign validation reconstruction errors.

        Uses the HIGHER of two methods for robustness:
          1. µ + n_sigma * σ  (paper method)
          2. percentile-based (e.g. 99th percentile of benign errors)

        When σ is inflated by outliers, the percentile method acts as a
        safety cap preventing τ from becoming unrealistically large.

        Parameters
        ----------
        X_val      : benign validation data
        n_sigma    : standard deviations above mean (default 3)
        percentile : fallback percentile cap (default 99.0)

        Returns threshold value.
        """
        self._check_fitted()
        errors = self.reconstruction_error(X_val)
        mu     = float(errors.mean())
        sigma  = float(errors.std())

        tau_sigma = mu + n_sigma * sigma
        tau_pct   = float(np.percentile(errors, percentile))

        # Use percentile if sigma-based threshold is unrealistically large
        # (happens when a few outlier benign samples have extreme errors)
        self.threshold_ = min(tau_sigma, tau_pct * 3.0)

        log.info("AE errors: µ=%.4f  σ=%.4f  p99=%.4f", mu, sigma, tau_pct)
        log.info("AE threshold τ: sigma-based=%.4f | p99-cap=%.4f | chosen=%.4f",
                 tau_sigma, tau_pct * 3.0, self.threshold_)
        return self.threshold_

    # =========================================================================
    # SCORE NORMALISATION (REN)
    # =========================================================================

    def fit_score_stats(self, X_train: np.ndarray) -> 'Autoencoder':
        # Compute mu and sigma of reconstruction errors on benign training data.
        # Used to convert raw errors into z-scores (REN technique).
        # Must be called after fit() and before anomaly_scores().
        self._check_fitted()
        errors            = self.reconstruction_error(X_train)
        self.score_mu_    = float(errors.mean())
        self.score_sigma_ = float(errors.std()) + 1e-8
        log.info('AE score stats: mu=%.6f  sigma=%.6f',
                 self.score_mu_, self.score_sigma_)
        return self

    def fit_latent_stats(self, X_train: np.ndarray) -> 'Autoencoder':
        # Encode benign training data and compute centroid + inv covariance
        # for Mahalanobis distance scoring (LSAS technique).
        self._check_fitted()
        Z = self.encode(X_train)
        self.latent_centroid_ = Z.mean(axis=0)
        cov = np.cov(Z.T)
        # Regularise covariance to ensure invertibility
        cov += np.eye(cov.shape[0]) * 1e-6
        self.latent_inv_cov_ = np.linalg.inv(cov)
        log.info("AE latent stats fitted: centroid shape=%s", Z.shape)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        # Return latent vectors z for input X. Shape: (n_samples, latent_dim)
        self._check_fitted()
        self.model_.eval()
        loader = self._make_loader(X, shuffle=False)
        zs = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                z = self.model_.encode(batch)
                zs.append(z.cpu().numpy())
        return np.concatenate(zs, axis=0)

    def latent_scores(self, X: np.ndarray) -> np.ndarray:
        # Mahalanobis distance from benign centroid in latent space.
        # More domain-invariant than reconstruction error.
        if self.latent_centroid_ is None:
            raise RuntimeError("Call fit_latent_stats() first.")
        Z = self.encode(X)
        diff = Z - self.latent_centroid_
        # Vectorised Mahalanobis: sqrt(diff @ inv_cov @ diff.T) per sample
        scores = np.sqrt(np.einsum('ij,jk,ik->i',
                                   diff, self.latent_inv_cov_, diff))
        return scores.astype(np.float32)

    # =========================================================================
    # INFERENCE
    # =========================================================================

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample mean Huber reconstruction error.

        Returns np.ndarray of shape (n_samples,)
        """
        self._check_fitted()
        self.model_.eval()
        loader = self._make_loader(X, shuffle=False)
        errors = []

        loss_fn = nn.HuberLoss(delta=self.huber_delta, reduction="none")

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = self.model_(batch)
                # Per-sample mean error across features
                err   = loss_fn(recon, batch).mean(dim=1)
                errors.append(err.cpu().numpy())

        return np.concatenate(errors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels using fitted threshold.
        Returns np.ndarray: 0 = benign, 1 = attack.
        """
        if self.threshold_ is None:
            raise RuntimeError(
                "Threshold not set. Call fit_threshold() first."
            )
        scores = self.reconstruction_error(X)
        return (scores > self.threshold_).astype(int)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        # Return REN z-score anomaly scores for ensemble fusion.
        # Falls back to raw error if fit_score_stats() not called.
        errors = self.reconstruction_error(X)
        if self.score_mu_ is not None:
            return (errors - self.score_mu_) / self.score_sigma_
        return errors

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save(self, path: str) -> None:
        """Save model weights + threshold to disk."""
        self._check_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state":      self.model_.state_dict(),
            "threshold":        self.threshold_,
            "input_dim":        self.input_dim,
            "huber_delta":      self.huber_delta,
            "score_mu":         self.score_mu_,
            "score_sigma":      self.score_sigma_,
            "latent_centroid":  self.latent_centroid_,
            "latent_inv_cov":   self.latent_inv_cov_,
        }, path)
        log.info("AE saved -> '%s'", path)

    def load(self, path: str) -> "Autoencoder":
        """Load model weights + threshold from disk."""
        checkpoint        = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim    = checkpoint["input_dim"]
        self.huber_delta  = checkpoint["huber_delta"]
        self.model_       = _AENet(self.input_dim).to(self.device)
        self.model_.load_state_dict(checkpoint["model_state"])
        self.threshold_   = checkpoint["threshold"]
        self.score_mu_        = checkpoint.get("score_mu",         None)
        self.score_sigma_     = checkpoint.get("score_sigma",      None)
        self.latent_centroid_ = checkpoint.get("latent_centroid",  None)
        self.latent_inv_cov_  = checkpoint.get("latent_inv_cov",   None)
        self._fitted          = True
        log.info("AE loaded <- '%s' | threshold=%.6f", path, self.threshold_)
        return self

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _make_loader(self, X: np.ndarray,
                     shuffle: bool) -> DataLoader:
        tensor  = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=shuffle)

    def _eval_loss(self, loader: DataLoader,
                   loss_fn: nn.Module) -> float:
        self.model_.eval()
        total_loss = 0.0
        n          = 0
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = self.model_(batch)
                loss  = loss_fn(recon, batch)
                total_loss += loss.item() * len(batch)
                n          += len(batch)
        return total_loss / n

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "Autoencoder not fitted. Call fit() first."
            )
