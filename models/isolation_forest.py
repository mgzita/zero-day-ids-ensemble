"""
models/isolation_forest.py
==========================
Isolation Forest (IF) for zero-day attack detection.

How it works  (paper section 4.5)
----------------------------------
An Isolation Forest builds an ensemble of random decision trees.
Each tree isolates samples by randomly selecting a feature and a
split value. Anomalous samples (attacks) are isolated in fewer
splits because they are rare and different from the majority —
they end up closer to the root of the tree.

The anomaly score is based on the average path length to isolation:
- Short path  → easy to isolate → anomalous  → attack
- Long path   → hard to isolate → normal     → benign

Parameters  (paper section 4.5)
---------------------------------
n_estimators  : 100 trees
max_samples   : 256 (subsample per tree — keeps trees diverse)
contamination : "auto" (let sklearn decide based on score distribution)

Key difference from AE/VAE
---------------------------
IF is a non-parametric tree ensemble — it makes no assumptions about
data distribution and captures completely different anomaly patterns.
This is why the ensemble (AE + VAE + IF) outperforms any single model.

Anomaly detection
-----------------
sklearn IsolationForest outputs scores in [-1, 1] range internally.
We convert to a [0, 1] anomaly score where higher = more anomalous,
then apply threshold τ = µ + 3σ on validation scores (same as AE/VAE).
"""

import logging
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest as _SKLearnIF

log = logging.getLogger(__name__)


class IsolationForestModel:
    """
    Isolation Forest wrapper with fit / predict / save / load.

    Usage
    -----
    iforest = IsolationForestModel()
    iforest.fit(X_train)
    scores = iforest.anomaly_scores(X_test)
    iforest.fit_threshold(X_val)
    y_pred = iforest.predict(X_test)
    iforest.save("artefacts/if_model.pkl")
    """

    def __init__(self,
                 n_estimators:  int   = 100,
                 max_samples:   int   = 256,
                 n_jobs:        int   = -1,
                 random_state:  int   = 42):

        self.n_estimators = n_estimators
        self.max_samples  = max_samples
        self.n_jobs       = n_jobs
        self.random_state = random_state

        self.model_      = None
        self.threshold_  = None
        self._fitted     = False

        log.info(
            "IsolationForest initialised | "
            "n_estimators=%d | max_samples=%d",
            n_estimators, max_samples
        )

    # =========================================================================
    # TRAINING
    # =========================================================================

    def fit(self, X_train: np.ndarray) -> "IsolationForestModel":
        """
        Fit Isolation Forest on benign training data.

        Note: IF does not use a validation set during training —
        it is a non-parametric model with no gradient descent.
        Training is fast even on large datasets.

        Parameters
        ----------
        X_train : scaled benign training data (n_train x n_features)
        """
        log.info(
            "Fitting IsolationForest | train=%d | "
            "n_estimators=%d | max_samples=%d",
            len(X_train), self.n_estimators, self.max_samples
        )

        self.model_ = _SKLearnIF(
            n_estimators  = self.n_estimators,
            max_samples   = self.max_samples,
            contamination = "auto",
            n_jobs        = self.n_jobs,
            random_state  = self.random_state,
        )
        self.model_.fit(X_train)
        self._fitted = True

        log.info("IsolationForest fitted.")
        return self

    # =========================================================================
    # THRESHOLD
    # =========================================================================

    def fit_threshold(self, X_val: np.ndarray,
                      n_sigma:    float = 3.0,
                      percentile: float = 99.0) -> float:
        """
        Compute anomaly threshold τ on benign validation anomaly scores.
        Uses min(µ + n_sigma*σ, p99*3) to guard against inflated σ.

        Parameters
        ----------
        X_val      : benign validation data
        n_sigma    : standard deviations above mean (default 3)
        percentile : fallback percentile cap (default 99.0)
        """
        self._check_fitted()
        scores = self.anomaly_scores(X_val)
        mu     = float(scores.mean())
        sigma  = float(scores.std())

        tau_sigma = mu + n_sigma * sigma
        tau_pct   = float(np.percentile(scores, percentile))

        # For IF, scores are densely packed — use percentile directly
        # rather than µ+3σ which is too conservative for tree-based scores.
        self.threshold_ = tau_pct

        log.info("IF scores: µ=%.4f  σ=%.4f  p%.0f=%.4f",
                 mu, sigma, percentile, tau_pct)
        log.info(
            "IF threshold τ: sigma-based=%.4f | p%.0f=%.4f | chosen=%.4f",
            tau_sigma, percentile, tau_pct, self.threshold_
        )
        return self.threshold_

    # =========================================================================
    # INFERENCE
    # =========================================================================

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample anomaly scores in [0, 1] range.
        Higher score = more anomalous = more likely an attack.

        sklearn's decision_function returns negative outlier scores
        where more negative = more anomalous. We negate and normalise
        to [0, 1] for consistency with AE/VAE reconstruction errors.

        Returns np.ndarray of shape (n_samples,).
        """
        self._check_fitted()

        # Raw scores: more negative = more anomalous
        raw = self.model_.decision_function(X)

        # Negate so higher = more anomalous, then shift to [0, inf)
        scores = -raw + 0.5   # sklearn centres around 0, shift to ~[0, 1]
        return scores.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions: 0=benign, 1=attack."""
        if self.threshold_ is None:
            raise RuntimeError(
                "Threshold not set. Call fit_threshold() first."
            )
        return (self.anomaly_scores(X) > self.threshold_).astype(int)

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save(self, path: str) -> None:
        """Save model + threshold to disk using pickle."""
        self._check_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":      self.model_,
                "threshold":  self.threshold_,
                "params": {
                    "n_estimators": self.n_estimators,
                    "max_samples":  self.max_samples,
                    "random_state": self.random_state,
                }
            }, f)
        log.info("IsolationForest saved -> '%s'", path)

    def load(self, path: str) -> "IsolationForestModel":
        """Load model + threshold from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model_         = data["model"]
        self.threshold_     = data["threshold"]
        self.n_estimators   = data["params"]["n_estimators"]
        self.max_samples    = data["params"]["max_samples"]
        self.random_state   = data["params"]["random_state"]
        self._fitted        = True
        log.info(
            "IsolationForest loaded <- '%s' | threshold=%.6f",
            path, self.threshold_
        )
        return self

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "IsolationForest not fitted. Call fit() first."
            )
