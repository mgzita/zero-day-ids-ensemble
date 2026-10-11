"""
preprocessing/features.py
=========================
Feature alignment and selection across datasets.

Functions
---------
get_common_features   : intersection of numeric features across all datasets
align_features        : keep only the agreed feature set in a DataFrame
fit_correlation_selector : Spearman-based redundancy removal (paper section 4.2)
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config_v1 import CORRELATION_THRESHOLD

log = logging.getLogger(__name__)


# =============================================================================
# COMMON FEATURE ALIGNMENT
# =============================================================================

def get_common_features(*dataframes: pd.DataFrame) -> list:
    """
    Return a sorted list of numeric feature columns present in ALL
    supplied dataframes, excluding 'label'.

    Usage
    -----
    common = get_common_features(df_cic2017, df_cic2018, df_unsw)
    """
    feature_sets = [
        set(df.select_dtypes(include=[np.number]).columns) - {"label"}
        for df in dataframes
    ]
    common = sorted(set.intersection(*feature_sets))
    log.info(
        "Common features across %d datasets: %d features",
        len(dataframes), len(common)
    )
    return common


def align_features(df: pd.DataFrame,
                   feature_cols: list,
                   fill_missing: bool = False) -> pd.DataFrame:
    """
    Restrict df to feature_cols + 'label'.

    Parameters
    ----------
    df            : DataFrame to align
    feature_cols  : list of expected feature column names
    fill_missing  : if True, missing columns are filled with 0.0 instead
                    of raising an error. Use for cross-dataset eval where
                    some features may not exist (e.g. UNSW vs CIC2017).

    Raises KeyError if fill_missing=False and columns are missing.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        if fill_missing:
            log.info(
                "  align_features: filling %d missing columns with 0.0: %s",
                len(missing), missing
            )
            for col in missing:
                df[col] = 0.0
        else:
            raise KeyError(
                f"Dataset is missing {len(missing)} expected feature(s): "
                f"{missing[:5]} ..."
            )
    return df[feature_cols + ["label"]].copy()


# =============================================================================
# SPEARMAN CORRELATION SELECTOR  (paper section 4.2 step 2)
# =============================================================================

def fit_correlation_selector(df_benign: pd.DataFrame,
                              feature_cols: list,
                              threshold: float = CORRELATION_THRESHOLD
                              ) -> list:
    """
    Identify and remove one feature from each highly-correlated pair.
    Must be fitted on BENIGN training data only.

    Parameters
    ----------
    df_benign    : DataFrame containing only benign training rows
    feature_cols : list of candidate numeric feature names
    threshold    : Spearman correlation cutoff (default from config: 0.85)

    Returns
    -------
    selected : reduced list of feature names to keep
    """
    log.info(
        "Fitting Spearman selector on %d benign samples "
        "(threshold=%.2f) ...", len(df_benign), threshold
    )

    X        = df_benign[feature_cols].values.astype(np.float32)
    X_ranked = np.apply_along_axis(rankdata, 0, X)
    corr     = np.corrcoef(X_ranked.T)

    n        = len(feature_cols)
    drop_set = set()

    for i in range(n):
        if feature_cols[i] in drop_set:
            continue
        for j in range(i + 1, n):
            if feature_cols[j] in drop_set:
                continue
            if abs(corr[i, j]) > threshold:
                drop_set.add(feature_cols[j])   # drop the latter of the pair

    selected = [f for f in feature_cols if f not in drop_set]
    log.info(
        "Correlation selector: %d -> %d features (removed %d redundant)",
        n, len(selected), len(drop_set)
    )
    return selected
