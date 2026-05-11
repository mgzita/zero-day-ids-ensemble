"""
preprocessing/loader.py
=======================
Loads raw CSV files for CIC-IDS2017, CIC-IDS2018, and UNSW-NB15.

Responsibilities
----------------
- Accepts a file path or a directory of CSV files
- Stratified sampling for large eval datasets (CIC2018 ~16M rows)
- Injects UNSW-NB15 column headers (raw files have none)
- Normalises all column names (strip, lowercase, underscores)
- Renames CIC2018 and UNSW columns to match CIC2017 naming convention
- Maps each dataset's label column to binary 0/1
"""

import os
import glob
import logging

import pandas as pd

from .config_v1 import UNSW_COLUMNS, CIC2018_RENAME, UNSW_RENAME, \
                   MAX_EVAL_ROWS, SAMPLED_DATASETS, UNSW_DATA_FILES

log = logging.getLogger(__name__)


# =============================================================================
# COLUMN NAME NORMALISER
# =============================================================================

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase, replace spaces/slashes with underscores."""
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[\s/]+", "_", regex=True)
                  .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


# =============================================================================
# LABEL MAPPERS  ->  0 = benign, 1 = attack
# =============================================================================

def _find_label_col(df: pd.DataFrame) -> str:
    for candidate in ["label", "attack_cat", "class", "attack"]:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"No label column found. Available: {list(df.columns)[:10]}"
    )

def _map_label_cic(df: pd.DataFrame) -> pd.DataFrame:
    col = _find_label_col(df)
    df["label"] = (
        df[col].astype(str).str.strip().str.upper() != "BENIGN"
    ).astype(int)
    return df

def _map_label_unsw(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["label"].astype(int)
    return df

_LABEL_MAPPERS = {
    "cic2017": _map_label_cic,
    "cic2018": _map_label_cic,
    "unsw":    _map_label_unsw,
}

SUPPORTED_DATASETS = list(_LABEL_MAPPERS.keys())


# =============================================================================
# CORE LOADER
# =============================================================================

def load_dataset(path: str, dataset_name: str) -> pd.DataFrame:
    """
    Load one or more CSV files into a single DataFrame.
    Applies stratified sampling for large datasets (e.g. CIC2018).

    Parameters
    ----------
    path         : path to a single CSV file or a directory of CSV files
    dataset_name : one of 'cic2017', 'cic2018', 'unsw'

    Returns
    -------
    pd.DataFrame with normalised + renamed columns and binary 'label' column.
    """
    if dataset_name not in _LABEL_MAPPERS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {SUPPORTED_DATASETS}"
        )

    # Load all files
    files  = _get_files(path, dataset_name == "unsw")
    frames = [_read_single(f, dataset_name == "unsw") for f in files]
    df     = pd.concat(frames, ignore_index=True)

    log.info("Raw shape: %s", df.shape)

    # Normalise + rename + label map
    df = _postprocess(df, dataset_name)

    # Stratified sample if this is a large eval dataset
    if dataset_name in SAMPLED_DATASETS and len(df) > MAX_EVAL_ROWS:
        df = _stratified_sample(df, MAX_EVAL_ROWS)

    label_dist = df["label"].value_counts().to_dict()
    n_numeric  = df.select_dtypes(include="number").shape[1]
    n_object   = df.select_dtypes(include="object").shape[1]
    log.info(
        "Loaded '%s': %d rows | benign=%d | attack=%d | numeric cols=%d | object cols=%d",
        dataset_name, len(df),
        label_dist.get(0, 0), label_dist.get(1, 0),
        n_numeric, n_object
    )
    return df


# =============================================================================
# STRATIFIED SAMPLING
# Samples MAX_EVAL_ROWS rows while preserving the benign/attack ratio.
# Example: 16M rows (83% benign, 17% attack) -> 500k rows (same ratio)
# =============================================================================

def _stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Sample n rows from df preserving the label distribution.

    Parameters
    ----------
    df : DataFrame with binary 'label' column
    n  : total number of rows to sample

    Returns
    -------
    Sampled DataFrame with same benign/attack ratio as original.
    """
    label_counts = df["label"].value_counts()
    total        = len(df)

    log.info("Stratified sampling: %d -> %d rows (ratio preserved) ...",
             total, n)

    sampled_frames = []
    for label_val, count in label_counts.items():
        # Calculate how many rows to take from this class
        proportion = count / total
        n_sample   = max(1, int(n * proportion))

        class_df = df[df["label"] == label_val]
        # If class has fewer rows than requested, take all of them
        n_sample = min(n_sample, len(class_df))

        sampled_frames.append(
            class_df.sample(n=n_sample, random_state=42)
        )
        log.info("  label=%d : %d -> %d rows (%.1f%%)",
                 label_val, count, n_sample, proportion * 100)

    result = pd.concat(sampled_frames, ignore_index=True)
    log.info("Stratified sample complete: %d rows", len(result))
    return result


# =============================================================================
# SHARED POST-PROCESSING
# =============================================================================

def _postprocess(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Normalise columns, apply rename map, map label."""
    df = normalise_columns(df)

    if dataset_name == "cic2018":
        df = _apply_rename(df, CIC2018_RENAME, "CIC2018")
    elif dataset_name == "unsw":
        df = _apply_rename(df, UNSW_RENAME, "UNSW")

    df = _LABEL_MAPPERS[dataset_name](df)
    return df


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _apply_rename(df: pd.DataFrame,
                  rename_map: dict,
                  dataset_label: str) -> pd.DataFrame:
    """Apply a column rename map. Only renames columns that exist."""
    applicable = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=applicable)
    log.info("  %s: renamed %d columns to match CIC2017 convention.",
             dataset_label, len(applicable))
    return df


def _get_files(path: str, is_unsw: bool) -> list:
    """Return sorted list of CSV files from a path or directory.
    For UNSW, only loads the 4 actual data files (excludes GT/features files).
    """
    if os.path.isdir(path):
        if is_unsw:
            # Only load the 4 known data files, ignore metadata CSVs
            files = [
                os.path.join(path, fname)
                for fname in UNSW_DATA_FILES
                if os.path.exists(os.path.join(path, fname))
            ]
            if not files:
                raise FileNotFoundError(
                    f"None of the expected UNSW data files found in: {path}\n"
                    f"Expected: {UNSW_DATA_FILES}"
                )
        else:
            files = sorted(glob.glob(os.path.join(path, "*.csv")))
            if not files:
                raise FileNotFoundError(f"No CSV files found in: {path}")
        log.info("Found %d file(s) in '%s'", len(files), path)
        return files
    return [path]


def _read_single(filepath: str, is_unsw: bool) -> pd.DataFrame:
    """Read a single CSV file. Falls back to latin-1 if UTF-8 fails."""
    log.info("  Reading %s", os.path.basename(filepath))
    for encoding in ("utf-8", "latin-1"):
        try:
            if is_unsw:
                return pd.read_csv(
                    filepath, header=None, names=UNSW_COLUMNS,
                    low_memory=False, encoding=encoding
                )
            return pd.read_csv(filepath, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            log.warning("  UTF-8 failed for %s, retrying with latin-1 ...",
                        os.path.basename(filepath))
    raise UnicodeDecodeError(
        f"Could not decode {filepath} with utf-8 or latin-1."
    )


# Expose for testing
def _rename_cic2018_columns(df): return _apply_rename(df, CIC2018_RENAME, "CIC2018")
def _rename_unsw_columns(df):    return _apply_rename(df, UNSW_RENAME,    "UNSW")
