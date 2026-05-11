"""
models/vae.py
=============
Variational Autoencoder (VAE) for zero-day attack detection.

Architecture  (paper section 4.4)
----------------------------------
Encoder : input -> 64 -> 32 -> 16 -> mu, log_var  (bottleneck dim=8)
Decoder : z(8)  -> 16 -> 32 -> 64 -> input

The VAE differs from a plain AE in two key ways:
  1. The bottleneck is a probability distribution (mu, log_var) not a
     fixed vector. During training a sample z ~ N(mu, exp(log_var)) is
     drawn via the reparameterisation trick.
  2. The loss has two terms:
       - Reconstruction loss : Huber(input, reconstruction)
       - KL divergence       : -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
     KL divergence regularises the latent space, forcing the encoder to
     produce distributions close to N(0,1). This makes the VAE more
     robust to out-of-distribution inputs (i.e. attacks).

Anomaly detection
-----------------
Reconstruction error = Huber loss per sample (same as AE).
Threshold τ = µ + 3σ computed on benign validation errors.
"""

import logging
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class _VAENet(nn.Module):
    """
    Variational Autoencoder network.

    Encoder: input_dim -> 64 -> 32 -> 16 -> (mu, log_var) [dim=8 each]
    Decoder: 8 -> 16 -> 32 -> 64 -> input_dim
    """

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim

        # Shared encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
        )

        # Separate heads for mean and log-variance
        self.fc_mu      = nn.Linear(16, latent_dim)
        self.fc_log_var = nn.Linear(16, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, 64),         nn.ReLU(),
            nn.Linear(64, input_dim),  # linear output
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to (mu, log_var) of latent distribution."""
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterise(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick: z = mu + eps * std
        where eps ~ N(0, I).
        Allows gradients to flow through the sampling operation.
        At inference time (eval mode) we use mu directly (no noise).
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval time

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mu, log_var)."""
        mu, log_var = self.encode(x)
        z           = self.reparameterise(mu, log_var)
        recon       = self.decode(z)
        return recon, mu, log_var


# =============================================================================
# VAE LOSS
# =============================================================================

def vae_loss(recon: torch.Tensor,
             x:     torch.Tensor,
             mu:    torch.Tensor,
             log_var: torch.Tensor,
             huber_delta: float = 1.0,
             kl_weight:   float = 1.0) -> Tuple[torch.Tensor,
                                                 torch.Tensor,
                                                 torch.Tensor]:
    """
    VAE loss = reconstruction loss + KL divergence.

    Parameters
    ----------
    recon       : reconstructed input
    x           : original input
    mu          : latent mean
    log_var     : latent log-variance
    huber_delta : Huber loss delta
    kl_weight   : weight on KL term (beta-VAE style, default=1.0)

    Returns (total_loss, recon_loss, kl_loss) — all scalar tensors.
    """
    recon_loss = nn.HuberLoss(delta=huber_delta)(recon, x)

    # KL divergence: -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(
        1 + log_var - mu.pow(2) - log_var.exp()
    )

    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss


# =============================================================================
# VAE WRAPPER
# =============================================================================

class VAE:
    """
    VAE wrapper with fit / predict / save / load.

    Usage
    -----
    vae = VAE(input_dim=32)
    vae.fit(X_train, X_val)
    scores = vae.reconstruction_error(X_test)
    vae.fit_threshold(X_val)
    y_pred = vae.predict(X_test)
    vae.save("artefacts/vae_model.pt")
    """

    def __init__(self,
                 input_dim:   int   = 32,
                 latent_dim:  int   = 8,
                 lr:          float = 1e-3,
                 batch_size:  int   = 256,
                 max_epochs:  int   = 100,
                 patience:    int   = 10,
                 huber_delta: float = 1.0,
                 kl_weight:   float = 1.0,
                 device:      str   = None):

        self.input_dim   = input_dim
        self.latent_dim  = latent_dim
        self.lr          = lr
        self.batch_size  = batch_size
        self.max_epochs  = max_epochs
        self.patience    = patience
        self.huber_delta = huber_delta
        self.kl_weight   = kl_weight
        self.device      = device or ("cuda" if torch.cuda.is_available()
                                      else "cpu")
        self.grad_clip   = 1.0
        self.smooth_beta = 0.9

        self.model_     = _VAENet(input_dim, latent_dim).to(self.device)
        self.threshold_ = None
        self._fitted    = False

        log.info("VAE initialised | input_dim=%d | latent_dim=%d | device=%s",
                 input_dim, latent_dim, self.device)

    # =========================================================================
    # TRAINING
    # =========================================================================

    def fit(self, X_train: np.ndarray, X_val: np.ndarray) -> "VAE":
        """
        Train on benign X_train, validate on benign X_val.
        Early stopping on smoothed validation loss.
        """
        log.info("Training VAE | train=%d | val=%d | device=%s",
                 len(X_train), len(X_val), self.device)

        train_loader = self._make_loader(X_train, shuffle=True)
        val_loader   = self._make_loader(X_val,   shuffle=False)

        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimiser, mode="min", factor=0.5,
                        patience=5, min_lr=1e-6)

        best_val_loss  = float("inf")
        smoothed_val   = None
        patience_count = 0
        best_weights   = None

        # KL annealing: linearly ramp kl_weight from 0 -> kl_weight
        # over anneal_epochs. Prevents KL from dominating early training.
        anneal_epochs = 20

        for epoch in range(1, self.max_epochs + 1):

            # Linearly increase KL weight from 0 to self.kl_weight
            kl_w = self.kl_weight * min(1.0, epoch / anneal_epochs)

            # ── Train ────────────────────────────────────────────────────
            self.model_.train()
            train_total = train_recon = train_kl = 0.0

            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimiser.zero_grad()

                recon, mu, log_var = self.model_(batch)
                loss, rl, kl = vae_loss(recon, batch, mu, log_var,
                                        self.huber_delta, kl_w)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(),
                                         self.grad_clip)
                optimiser.step()

                n             = len(batch)
                train_total  += loss.item() * n
                train_recon  += rl.item()   * n
                train_kl     += kl.item()   * n

            n_train      = len(X_train)
            train_total /= n_train
            train_recon /= n_train
            train_kl    /= n_train

            # ── Validate ─────────────────────────────────────────────────
            val_total, val_recon, val_kl = self._eval_loss(val_loader)

            # Exponential smoothing
            if smoothed_val is None:
                smoothed_val = val_total
            else:
                smoothed_val = (self.smooth_beta * smoothed_val
                                + (1 - self.smooth_beta) * val_total)

            scheduler.step(val_total)
            current_lr = optimiser.param_groups[0]["lr"]

            log.info(
                "Epoch %3d/%d | train=%.4f (r=%.4f kl=%.4f) | "
                "val=%.4f (r=%.4f kl=%.4f) | smooth=%.4f | lr=%.2e | kl_w=%.2f",
                epoch, self.max_epochs,
                train_total, train_recon, train_kl,
                val_total,   val_recon,   val_kl,
                smoothed_val, current_lr, kl_w
            )

            # ── Early stopping on smoothed val ───────────────────────────
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

        if best_weights:
            self.model_.load_state_dict(
                {k: v.to(self.device) for k, v in best_weights.items()}
            )

        self._fitted = True
        log.info("VAE training complete. Best val_loss=%.6f", best_val_loss)
        return self

    # =========================================================================
    # THRESHOLD
    # =========================================================================

    def fit_threshold(self, X_val: np.ndarray,
                      n_sigma:    float = 3.0,
                      percentile: float = 99.0) -> float:
        """
        Compute anomaly threshold τ on benign validation reconstruction errors.
        Uses min(µ + n_sigma*σ, p99*3) to guard against inflated σ.
        """
        self._check_fitted()
        errors = self.reconstruction_error(X_val)
        mu     = float(errors.mean())
        sigma  = float(errors.std())

        tau_sigma = mu + n_sigma * sigma
        tau_pct   = float(np.percentile(errors, percentile))

        self.threshold_ = min(tau_sigma, tau_pct * 3.0)

        log.info("VAE errors: µ=%.4f  σ=%.4f  p99=%.4f", mu, sigma, tau_pct)
        log.info("VAE threshold τ: sigma-based=%.4f | p99-cap=%.4f | chosen=%.4f",
                 tau_sigma, tau_pct * 3.0, self.threshold_)
        return self.threshold_

    # =========================================================================
    # SCORE NORMALISATION (REN)
    # =========================================================================

    def fit_score_stats(self, X_train: np.ndarray) -> 'VAE':
        # Compute mu and sigma of reconstruction errors on benign training data.
        # Used to convert raw errors into z-scores (REN technique).
        self._check_fitted()
        errors            = self.reconstruction_error(X_train)
        self.score_mu_    = float(errors.mean())
        self.score_sigma_ = float(errors.std()) + 1e-8
        log.info('VAE score stats: mu=%.6f  sigma=%.6f',
                 self.score_mu_, self.score_sigma_)
        return self

    def fit_latent_stats(self, X_train: np.ndarray) -> 'VAE':
        # Encode benign training data and compute centroid + inv covariance
        # for Mahalanobis distance scoring (LSAS technique).
        # VAE latent space is regularised z~N(0,1) making this especially powerful.
        self._check_fitted()
        Z = self.encode(X_train)
        self.latent_centroid_ = Z.mean(axis=0)
        cov = np.cov(Z.T)
        cov += np.eye(cov.shape[0]) * 1e-6
        self.latent_inv_cov_ = np.linalg.inv(cov)
        log.info("VAE latent stats fitted: centroid shape=%s", Z.shape)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        # Return latent mu vectors for input X. Shape: (n_samples, latent_dim)
        # Uses mu directly (deterministic) for consistent distance scoring.
        self._check_fitted()
        self.model_.eval()
        loader = self._make_loader(X, shuffle=False)
        zs = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                mu, _ = self.model_.encode(batch)
                zs.append(mu.cpu().numpy())
        return np.concatenate(zs, axis=0)

    def latent_scores(self, X: np.ndarray) -> np.ndarray:
        # Mahalanobis distance from benign centroid in VAE latent space.
        # VAE KL regularisation makes this particularly domain-invariant.
        if self.latent_centroid_ is None:
            raise RuntimeError("Call fit_latent_stats() first.")
        Z = self.encode(X)
        diff = Z - self.latent_centroid_
        scores = np.sqrt(np.einsum('ij,jk,ik->i',
                                   diff, self.latent_inv_cov_, diff))
        return scores.astype(np.float32)

    # =========================================================================
    # INFERENCE
    # =========================================================================

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Per-sample mean Huber reconstruction error.
        Uses mu directly (no sampling) for deterministic inference.
        Returns np.ndarray of shape (n_samples,).
        """
        self._check_fitted()
        self.model_.eval()
        loader  = self._make_loader(X, shuffle=False)
        errors  = []
        loss_fn = nn.HuberLoss(delta=self.huber_delta, reduction="none")

        with torch.no_grad():
            for (batch,) in loader:
                batch             = batch.to(self.device)
                recon, mu, _      = self.model_(batch)
                err               = loss_fn(recon, batch).mean(dim=1)
                errors.append(err.cpu().numpy())

        return np.concatenate(errors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions: 0=benign, 1=attack."""
        if self.threshold_ is None:
            raise RuntimeError("Threshold not set. Call fit_threshold() first.")
        return (self.reconstruction_error(X) > self.threshold_).astype(int)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        # Return REN z-score anomaly scores for ensemble fusion.
        errors = self.reconstruction_error(X)
        if self.score_mu_ is not None:
            return (errors - self.score_mu_) / self.score_sigma_
        return errors

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save(self, path: str) -> None:
        self._check_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state":      self.model_.state_dict(),
            "threshold":        self.threshold_,
            "input_dim":        self.input_dim,
            "latent_dim":       self.latent_dim,
            "huber_delta":      self.huber_delta,
            "kl_weight":        self.kl_weight,
            "score_mu":         self.score_mu_,
            "score_sigma":      self.score_sigma_,
            "latent_centroid":  self.latent_centroid_,
            "latent_inv_cov":   self.latent_inv_cov_,
        }, path)
        log.info("VAE saved -> '%s'", path)

    def load(self, path: str) -> "VAE":
        checkpoint        = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim    = checkpoint["input_dim"]
        self.latent_dim   = checkpoint["latent_dim"]
        self.huber_delta  = checkpoint["huber_delta"]
        self.kl_weight    = checkpoint["kl_weight"]
        self.model_       = _VAENet(self.input_dim,
                                    self.latent_dim).to(self.device)
        self.model_.load_state_dict(checkpoint["model_state"])
        self.threshold_   = checkpoint["threshold"]
        self.score_mu_        = checkpoint.get("score_mu",        None)
        self.score_sigma_     = checkpoint.get("score_sigma",     None)
        self.latent_centroid_ = checkpoint.get("latent_centroid", None)
        self.latent_inv_cov_  = checkpoint.get("latent_inv_cov",  None)
        self._fitted          = True
        log.info("VAE loaded <- '%s' | threshold=%.6f", path, self.threshold_)
        return self

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _make_loader(self, X: np.ndarray, shuffle: bool) -> DataLoader:
        tensor  = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _eval_loss(self, loader: DataLoader
                   ) -> Tuple[float, float, float]:
        """Returns (total, recon, kl) losses averaged over loader."""
        self.model_.eval()
        total = recon = kl = 0.0
        n = 0
        with torch.no_grad():
            for (batch,) in loader:
                batch              = batch.to(self.device)
                rec, mu, log_var   = self.model_(batch)
                loss, rl, kl_loss  = vae_loss(rec, batch, mu, log_var,
                                              self.huber_delta, self.kl_weight)
                total += loss.item()    * len(batch)
                recon += rl.item()      * len(batch)
                kl    += kl_loss.item() * len(batch)
                n     += len(batch)
        return total / n, recon / n, kl / n

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("VAE not fitted. Call fit() first.")
