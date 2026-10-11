"""
preprocessing/cleaner.py
========================
Cleans a raw loaded DataFrame.

Steps
-----
1. Drop non-numeric identifier columns (IPs, protocol strings, etc.)
2. Keep only numeric columns + label
3. Replace +/- inf with NaN
4. Drop columns with too many NaN values
5. Drop rows with remaining NaN values
6. Drop constant (zero-variance) columns
7. Log-transform heavy-tailed features (supervisor fix 5)

Log transformation
------------------
Network flow features like flow_duration, total_fwd_packets, and
packet length counts are heavily right-skewed with extreme outliers.
Applying log1p (log(1+x)) compresses the tail, making feature
distributions more comparable across datasets collected from different
network environments. This directly improves cross-dataset generalisation.

np.log1p is used (instead of np.log) to safely handle zero values:
    log1p(0) = 0
    log1p(x) = log(1+x) for x > 0
"""

import logging

import numpy as np
import pandas as pd

from .config_v1 import NON_NUMERIC_COLS, LOG_TRANSFORM_COLS

log = logging.getLogger(__name__)


def clean(df: pd.DataFrame, missing_thresh: float = 0.5) -> pd.DataFrame:
    """
    Clean a dataset DataFrame.

    Parameters
    ----------
    df             : raw loaded DataFrame (output of loader.load_dataset)
    missing_thresh : drop columns where NaN fraction exceeds this value

    Returns
    -------
    Cleaned pd.DataFrame with only numeric features + 'label'.
    """
    log.info("Cleaning: shape before = %s", df.shape)

    df = _drop_identifier_columns(df)
    df = _keep_numeric_and_label(df)
    df = _replace_infinities(df)
    df = _drop_high_nan_columns(df, missing_thresh)
    df = _drop_nan_rows(df)
    df = _drop_constant_columns(df)
    df = _log_transform(df)

    log.info("Cleaning: shape after  = %s", df.shape)
    return df


# =============================================================================
# PRIVATE STEPS
# =============================================================================

def _drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in NON_NUMERIC_COLS if c in df.columns]
    if to_drop:
        log.info("  Dropped identifier columns: %s", to_drop)
        df = df.drop(columns=to_drop)
    return df


def _keep_numeric_and_label(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce all non-label columns to numeric first.
    # After cross-dataset renaming + sampling, some numeric columns may have
    # been inferred as object dtype -- coerce forces them back to float/int.
    for col in df.columns:
        if col != "label":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if "label" not in numeric and "label" in df.columns:
        numeric.append("label")
    dropped = [c for c in df.columns if c not in numeric]
    if dropped:
        log.info("  Dropped %d non-numeric columns after coercion: %s",
                 len(dropped), dropped)
    return df[numeric]


def _replace_infinities(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicate columns first -- can arise from cross-dataset renaming
    df = df.loc[:, ~df.columns.duplicated()]
    feat_cols = [c for c in df.columns if c != "label"]
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    return df


def _drop_high_nan_columns(df: pd.DataFrame,
                            missing_thresh: float) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if c != "label"]
    nan_frac  = df[feat_cols].isnull().mean()
    to_drop   = nan_frac[nan_frac > missing_thresh].index.tolist()
    if to_drop:
        log.info("  Dropped %d high-NaN columns: %s", len(to_drop), to_drop)
        df = df.drop(columns=to_drop)
    return df


def _drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df     = df.dropna()
    dropped = before - len(df)
    if dropped:
        log.info("  Dropped %d rows containing NaN.", dropped)
    return df


def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols   = [c for c in df.columns if c != "label"]
    const_cols  = [c for c in feat_cols if df[c].nunique() <= 1]
    if const_cols:
        log.info("  Dropped %d constant columns: %s", len(const_cols), const_cols)
        df = df.drop(columns=const_cols)
    return df


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to heavy-tailed network flow features.
    Only transforms columns that exist in the dataframe.
    Negative values are clipped to 0 before transform (log1p requires x >= 0).
    """
    cols = [c for c in LOG_TRANSFORM_COLS if c in df.columns]
    if not cols:
        return df
    for col in cols:
        df[col] = np.log1p(df[col].clip(lower=0))
    log.info("  Log-transformed %d heavy-tailed features.", len(cols))
    return df
