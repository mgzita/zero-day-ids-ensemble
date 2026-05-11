"""
models/ensemble.py
==================
Ensemble fusion of AE + VAE + Isolation Forest using 4 scoring signals:

    1. AE  reconstruction error  (REN z-score)
    2. VAE reconstruction error  (REN z-score)
    3. VAE latent distance       (LSAS Mahalanobis)
    4. Isolation Forest score

Fusion weights (informed by per-dataset AUC analysis):
-------------------------------------------------------
From LSAS diagnostic experiments:
  - AE recon:   AUC=0.908 CIC2017 | 0.416 CIC2018 | 0.199 UNSW
  - VAE recon:  AUC=0.883 CIC2017 | 0.407 CIC2018 | 0.771 UNSW
  - VAE latent: AUC=0.604 CIC2017 | 0.628 CIC2018 | 0.056 UNSW
  - IF:         AUC=0.778 CIC2017 | 0.559 CIC2018 | 0.087 UNSW

VAE latent (LSAS) is the only signal with positive cross-dataset AUC
on CIC2018. VAE recon is the only strong signal on UNSW.
AE recon is strongest on CIC2017.

Final weights balance all three datasets:
    0.30 * AE_recon + 0.35 * VAE_recon + 0.25 * VAE_latent + 0.10 * IF

Score normalisation
-------------------
All scores normalised to [0,1] using p1/p99 on benign validation data
before weighted fusion.
"""

import logging
import os
import pickle
from typing import Dict, Tuple

import numpy as np

log = logging.getLogger(__name__)


class EnsembleDetector:
    """
    4-signal weighted ensemble: AE_recon + VAE_recon + VAE_latent + IF.

    Usage
    -----
    ensemble = EnsembleDetector(ae, vae, iforest)
    ensemble.fit_normaliser(X_val)
    ensemble.fit_threshold(X_val)
    y_pred = ensemble.predict(X_test)
    ensemble.save("artefacts/ensemble.pkl")
    """

    # Weights informed by per-dataset AUC analysis
    WEIGHTS = {
        "ae_recon":   0.30,
        "vae_recon":  0.35,
        "vae_latent": 0.25,
        "if":         0.10,
    }

    def __init__(self, ae, vae, iforest, n_sigma: float = 3.0):
        self.ae      = ae
        self.vae     = vae
        self.iforest = iforest
        self.n_sigma = n_sigma

        self._norm_params: Dict[str, Tuple[float, float]] = {}
        self.threshold_  = None
        self._fitted     = False

    # =========================================================================
    # FITTING
    # =========================================================================

    def fit_normaliser(self, X_val: np.ndarray) -> "EnsembleDetector":
        """
        Fit p1/p99 normalisation params for each scoring signal
        on benign validation data.
        """
        log.info("Fitting score normalisers on %d val samples ...", len(X_val))

        raw = self._raw_scores(X_val)
        for name, scores in raw.items():
            s_min = float(np.percentile(scores, 1))
            s_max = float(np.percentile(scores, 99))
            if s_max - s_min < 1e-8:
                s_max = s_min + 1.0
            self._norm_params[name] = (s_min, s_max)
            log.info("  %s: p1=%.4f  p99=%.4f", name, s_min, s_max)
        return self

    def fit_threshold(self, X_val: np.ndarray,
                      percentile: float = 99.0) -> float:
        """
        Compute ensemble threshold on benign val scores.
        τ = min(µ + n_sigma*σ, p99*3)
        """
        if not self._norm_params:
            raise RuntimeError("Call fit_normaliser() before fit_threshold().")

        scores    = self.ensemble_scores(X_val)
        mu        = float(scores.mean())
        sigma     = float(scores.std())
        tau_sigma = mu + self.n_sigma * sigma
        tau_pct   = float(np.percentile(scores, percentile))

        self.threshold_ = min(tau_sigma, tau_pct * 3.0)
        self._fitted    = True

        log.info("Ensemble scores: µ=%.4f  σ=%.4f  p%.0f=%.4f",
                 mu, sigma, percentile, tau_pct)
        log.info("Ensemble threshold τ: sigma-based=%.4f | p99-cap=%.4f | chosen=%.4f",
                 tau_sigma, tau_pct * 3.0, self.threshold_)
        return self.threshold_

    # =========================================================================
    # INFERENCE
    # =========================================================================

    def ensemble_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute weighted ensemble anomaly scores.
        Returns np.ndarray of shape (n_samples,) in [0,1].
        Higher = more anomalous.
        """
        if not self._norm_params:
            raise RuntimeError("Call fit_normaliser() before ensemble_scores().")

        raw    = self._raw_scores(X)
        normed = {}
        for name, scores in raw.items():
            s_min, s_max = self._norm_params[name]
            normed[name] = np.clip(
                (scores - s_min) / (s_max - s_min), 0.0, 1.0
            )

        fused = (
            self.WEIGHTS["ae_recon"]   * normed["ae_recon"]   +
            self.WEIGHTS["vae_recon"]  * normed["vae_recon"]  +
            self.WEIGHTS["vae_latent"] * normed["vae_latent"] +
            self.WEIGHTS["if"]         * normed["if"]
        )
        return fused.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions: 0=benign, 1=attack."""
        if self.threshold_ is None:
            raise RuntimeError("Call fit_threshold() first.")
        return (self.ensemble_scores(X) > self.threshold_).astype(int)

    def predict_with_scores(self, X: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (y_pred, ensemble_scores)."""
        scores = self.ensemble_scores(X)
        y_pred = (scores > self.threshold_).astype(int)
        return y_pred, scores

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "norm_params": self._norm_params,
                "threshold":   self.threshold_,
                "n_sigma":     self.n_sigma,
                "weights":     self.WEIGHTS,
            }, f)
        log.info("Ensemble metadata saved -> '%s'", path)

    def load(self, path: str) -> "EnsembleDetector":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._norm_params = data["norm_params"]
        self.threshold_   = data["threshold"]
        self.n_sigma      = data["n_sigma"]
        self._fitted      = True
        log.info("Ensemble metadata loaded <- '%s' | threshold=%.6f",
                 path, self.threshold_)
        return self

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _raw_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get all 4 raw scoring signals."""
        scores = {
            "ae_recon":  self.ae.anomaly_scores(X),
            "vae_recon": self.vae.anomaly_scores(X),
            "if":        self.iforest.anomaly_scores(X),
        }
        # VAE latent (LSAS) — only if fitted
        if self.vae.latent_centroid_ is not None:
            scores["vae_latent"] = self.vae.latent_scores(X)
        else:
            log.warning("VAE latent stats not fitted — using zeros for vae_latent.")
            scores["vae_latent"] = np.zeros(len(X), dtype=np.float32)
        return scores
