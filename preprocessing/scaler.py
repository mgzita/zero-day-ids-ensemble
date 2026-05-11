"""
preprocessing/scaler.py
=======================
IQR-based robust scaler as defined in the paper (section 4.2 step 3):

    z = (x - Q2) / (Q3 - Q1)

Must be fitted on benign training data only, then reused for all splits
and external evaluation datasets without refitting.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


class IQRScaler:
    """
    Robust scaler using interquartile range normalisation.

    Usage
    -----
    scaler = IQRScaler()
    scaler.fit(X_train_benign)          # fit once on benign training data

    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    X_eval  = scaler.transform(X_external)

    scaler.save("artefacts/iqr_scaler.npz")   # persist
    scaler.load("artefacts/iqr_scaler.npz")   # reload later
    """

    def __init__(self):
        self.q1_  = None
        self.q2_  = None
        self.q3_  = None
        self.iqr_ = None

    # -------------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "IQRScaler":
        """
        Compute Q1, Q2 (median), Q3 and IQR from X.
        Columns with IQR == 0 are assigned IQR = 1 to avoid division by zero.
        """
        self.q1_  = np.percentile(X, 25, axis=0)
        self.q2_  = np.percentile(X, 50, axis=0)
        self.q3_  = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        self.iqr_[self.iqr_ == 0] = 1.0

        log.info("IQRScaler fitted: %d samples x %d features.", *X.shape)
        return self

    # -------------------------------------------------------------------------
    def transform(self, X: np.ndarray,
                  clip: float = 10.0) -> np.ndarray:
        """
        Apply z = (x - Q2) / IQR scaling, then clip to [-clip, +clip].
        Clipping prevents extreme values from blowing up reconstruction
        errors and the anomaly threshold σ computation.
        Default clip=10.0 preserves 99.9%+ of normal traffic variation.
        """
        self._check_fitted()
        z = (X - self.q2_) / self.iqr_
        return np.clip(z, -clip, clip)

    # -------------------------------------------------------------------------
    def fit_transform(self, X: np.ndarray,
                      clip: float = 10.0) -> np.ndarray:
        """Convenience: fit then transform in one call."""
        return self.fit(X).transform(X, clip=clip)

    # -------------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist scaler parameters to a .npz file."""
        self._check_fitted()
        np.savez(path, q1=self.q1_, q2=self.q2_, q3=self.q3_, iqr=self.iqr_)
        log.info("IQRScaler saved -> '%s'.", path)

    # -------------------------------------------------------------------------
    def load(self, path: str) -> "IQRScaler":
        """Load scaler parameters from a .npz file."""
        data      = np.load(path)
        self.q1_  = data["q1"]
        self.q2_  = data["q2"]
        self.q3_  = data["q3"]
        self.iqr_ = data["iqr"]
        log.info("IQRScaler loaded <- '%s'.", path)
        return self

    # -------------------------------------------------------------------------
    def _check_fitted(self):
        if self.q2_ is None:
            raise RuntimeError("IQRScaler is not fitted. Call fit() first.")
