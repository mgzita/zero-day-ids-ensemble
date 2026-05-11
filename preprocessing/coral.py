"""
preprocessing/coral.py
======================
CORAL (Correlation Alignment) domain adaptation.

Reference:
    Sun, Feng & Saenko (2016). "Return of Frustratingly Easy Domain Adaptation."
    AAAI 2016. https://arxiv.org/abs/1511.05547

Theory
------
CORAL aligns the second-order statistics (covariance) of source and target
feature distributions, so that a model trained on the source domain can
generalise to the target domain without retraining.

The alignment transform is:

    X_aligned = X_target @ A

where:

    A = Σ_target^(-1/2) @ Σ_source^(1/2)

After alignment:

    covariance(X_aligned) ≈ covariance(X_source)

Why this works for IDS cross-dataset generalisation:
- Different network environments produce different feature scales and
  correlations (Benign_CIC2017_mu=0.14 vs CIC2018_mu=0.40 vs UNSW_mu=0.54)
- CORAL maps the target dataset's feature geometry to match the source
- The AE/VAE then sees familiar-looking input, reducing score inversion

Usage
-----
    coral = CORAL()
    coral.fit(X_train_benign)        # fit source covariance on CIC2017 benign
    X_cic18_aligned = coral.transform(X_cic18_benign_cal, X_cic18)
    X_unsw_aligned  = coral.transform(X_unsw_benign_cal,  X_unsw)

    # Then evaluate as usual
    scores = ensemble.ensemble_scores(X_cic18_aligned)
"""

import logging
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)


class CORAL:
    """
    CORAL (Correlation Alignment) domain adaptation.

    Fits on source (training) domain benign data, then transforms
    target domain data to match the source covariance structure.
    """

    def __init__(self, reg: float = 1e-5):
        """
        Parameters
        ----------
        reg : float
            Regularisation added to diagonal of covariance matrices
            to ensure invertibility (default 1e-5).
        """
        self.reg             = reg
        self.source_cov_sqrt_: Optional[np.ndarray] = None
        self._fitted         = False

    def fit(self, X_source: np.ndarray) -> "CORAL":
        """
        Fit CORAL on source domain (CIC2017 benign training data).

        Computes Σ_source^(1/2) via eigendecomposition.

        Parameters
        ----------
        X_source : np.ndarray, shape (n_samples, n_features)
            Source domain benign samples (X_train).
        """
        log.info("CORAL: fitting on source data shape=%s", X_source.shape)

        cov_s = np.cov(X_source.T) + self.reg * np.eye(X_source.shape[1])

        # Symmetric square root via eigendecomposition:
        # Σ^(1/2) = V @ diag(λ^(1/2)) @ V^T
        eigvals, eigvecs = np.linalg.eigh(cov_s)
        eigvals = np.maximum(eigvals, 0)  # numerical safety
        self.source_cov_sqrt_ = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        self._fitted = True
        log.info("CORAL: source covariance sqrt fitted | dim=%d", X_source.shape[1])
        return self

    def transform(self,
                  X_target_cal: np.ndarray,
                  X_target:     np.ndarray) -> np.ndarray:
        """
        Align target domain data to source covariance.

        Parameters
        ----------
        X_target_cal : np.ndarray, shape (n_cal, n_features)
            Benign calibration samples from the target dataset.
            Used to estimate target covariance. In practice this is
            the same 10% benign calibration set used for thresholding.

        X_target : np.ndarray, shape (n_samples, n_features)
            Full target dataset to align (benign + attacks).

        Returns
        -------
        X_aligned : np.ndarray, shape (n_samples, n_features)
            Target data transformed to match source covariance.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        # Estimate target covariance from benign calibration samples
        cov_t = np.cov(X_target_cal.T) + self.reg * np.eye(X_target_cal.shape[1])

        # Symmetric inverse square root of target covariance:
        # Σ_target^(-1/2) = V @ diag(λ^(-1/2)) @ V^T
        eigvals, eigvecs = np.linalg.eigh(cov_t)
        eigvals = np.maximum(eigvals, 1e-10)  # avoid division by zero
        target_cov_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # CORAL transform: A = Σ_target^(-1/2) @ Σ_source^(1/2)
        A = target_cov_inv_sqrt @ self.source_cov_sqrt_

        # Centre target on its own mean before applying alignment
        mu_target = X_target_cal.mean(axis=0)
        X_centred = X_target - mu_target

        # Apply alignment
        X_aligned = X_centred @ A.T

        # Re-centre to source mean (optional but improves score stability)
        # We skip this to avoid any data leakage from training labels

        log.info("CORAL: aligned target shape=%s", X_aligned.shape)
        return X_aligned.astype(np.float32)

    def fit_transform(self, X_source: np.ndarray,
                      X_target:     np.ndarray) -> np.ndarray:
        """Fit on source and transform target (same-domain convenience)."""
        return self.fit(X_source).transform(X_source, X_target)
