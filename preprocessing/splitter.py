"""
preprocessing/splitter.py
=========================
Chronological 70 / 15 / 15 split (paper section 4.2 step 4).

Strategy
--------
1. Separate benign rows from the full dataset
2. Split benign rows chronologically:
       train = 70% of benign   (used to fit selector + scaler + models)
       val   = 15% of benign   (used for threshold tuning and early stopping)
       test  = remaining 15% of benign + ALL attack rows  (zero-day evaluation)

Returns DataFrames (not numpy arrays) so the pipeline can still fit the
Spearman correlation selector and IQR scaler on the training DataFrame
before converting to numpy.
"""

import logging
from typing import Tuple

import pandas as pd

from .config_v1 import TRAIN_RATIO, VAL_RATIO

log = logging.getLogger(__name__)


def split_dataset(
    df:           pd.DataFrame,
    train_ratio:  float = TRAIN_RATIO,
    val_ratio:    float = VAL_RATIO,
    label_col:    str   = "label",
    benign_label: int   = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split on benign rows; all attacks go into the test set.

    Parameters
    ----------
    df           : aligned DataFrame with label column (NOT yet scaled)
    train_ratio  : fraction of benign rows for training  (default 0.70)
    val_ratio    : fraction of benign rows for validation (default 0.15)
    label_col    : name of the label column
    benign_label : integer value representing benign class (default 0)

    Returns
    -------
    train_df, val_df, test_df   (DataFrames, NOT yet scaled)

    Split summary
    -------------
    train_df : 70% of benign rows only  -> fit selector, scaler, train models
    val_df   : 15% of benign rows only  -> threshold tuning, early stopping
    test_df  : 15% of benign + ALL attacks -> zero-day evaluation
    """
    df = df.copy().reset_index(drop=True)

    # ── Separate benign and attack rows ───────────────────────────────────
    benign_df = df[df[label_col] == benign_label].reset_index(drop=True)
    attack_df = df[df[label_col] != benign_label].reset_index(drop=True)

    log.info("Benign rows : %d", len(benign_df))
    log.info("Attack rows : %d", len(attack_df))

    # ── Chronological split on benign rows only ───────────────────────────
    n         = len(benign_df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train_df       = benign_df.iloc[:train_end].reset_index(drop=True)
    val_df         = benign_df.iloc[train_end:val_end].reset_index(drop=True)
    test_benign_df = benign_df.iloc[val_end:].reset_index(drop=True)

    # ── Test set = held-out benign + ALL attacks ──────────────────────────
    test_df = pd.concat([test_benign_df, attack_df], ignore_index=True)

    log.info("Train  : %d rows (benign only)", len(train_df))
    log.info("Val    : %d rows (benign only)", len(val_df))
    log.info("Test   : %d benign + %d attack = %d total",
             len(test_benign_df), len(attack_df), len(test_df))

    return train_df, val_df, test_df
