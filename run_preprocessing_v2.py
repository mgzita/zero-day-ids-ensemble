"""
run_preprocessing_v2.py
=======================
Memory-safe v2 pipeline. Loads datasets in chunks, keeps only
benign rows in memory at any time to prevent crashes.

  TRAINING   : combined benign from CIC2017 + CIC2018 + UNSW (80% each)
  VALIDATION : held-out 20% benign from each dataset
  TEST       : BoT-IoT (fully unseen)

Outputs to artefacts_v2/. v1 files untouched.

Run from project root:
    python run_preprocessing_v2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing.scaler   import IQRScaler
from preprocessing_v2.config import (
    DATASET_PATHS, ARTEFACT_DIR,
    UNSW_COLUMNS, UNSW_DATA_FILES,
    BOTIOT_DATA_FILES, BOTIOT_RENAME,
    CIC2018_RENAME, UNSW_RENAME,
    NON_NUMERIC_COLS, LOG_TRANSFORM_COLS,
    TRAIN_RATIO, MAX_TRAIN_ROWS_PER_DS, MAX_VAL_ROWS_PER_DS,
    MAX_BOTIOT_ROWS, CORRELATION_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
os.makedirs(ARTEFACT_DIR, exist_ok=True)
RNG = np.random.default_rng(42)

CHUNK_SIZE = 100_000   # rows per chunk â€” safe for 8GB RAM


# =============================================================================
# SHARED CLEANING
# =============================================================================

def _clean(df: pd.DataFrame, rename: dict = None) -> pd.DataFrame:
    # Normalise column names
    df.columns = (df.columns.str.strip()
                             .str.lower()
                             .str.replace(r"[^a-z0-9_]", "_", regex=True))
    if rename:
        df.rename(columns=rename, inplace=True)

    # Drop non-numeric identifiers
    df.drop(columns=[c for c in NON_NUMERIC_COLS if c in df.columns],
            inplace=True, errors="ignore")

    # Convert to numeric
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if col != "label":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Log transform
    log_cols = [c for c in LOG_TRANSFORM_COLS if c in df.columns]
    if log_cols:
        df[log_cols] = np.log1p(df[log_cols].clip(lower=0))

    # Clean infs/NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    return df


def _make_label_cic(df: pd.DataFrame) -> pd.DataFrame:
    label_col = next((c for c in ["label", "flow_label"]
                      if c in df.columns), None)
    if label_col is None:
        df["label"] = 0
        return df
    df["label"] = (df[label_col].astype(str)
                                .str.strip().str.upper() != "BENIGN").astype(int)
    if label_col != "label":
        df.drop(columns=[label_col], inplace=True)
    return df


def _make_label_numeric(df: pd.DataFrame, col: str = "label") -> pd.DataFrame:
    df["label"] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


# =============================================================================
# CHUNK-BASED BENIGN LOADERS
# =============================================================================

def load_benign_cic(folder: str, rename: dict = None,
                    max_rows: int = None) -> pd.DataFrame:
    """Load benign rows only from a CIC dataset folder, chunk by chunk."""
    files   = sorted(Path(folder).glob("*.csv"))
    benign  = []
    total   = 0
    for f in files:
        log.info("  Reading %s ...", f.name)
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
            chunk = _clean(chunk, rename)
            chunk = _make_label_cic(chunk)
            chunk = chunk[chunk["label"] == 0].copy()
            benign.append(chunk)
            total += len(chunk)
            if max_rows and total >= max_rows:
                break
        if max_rows and total >= max_rows:
            break
        gc.collect()

    df = pd.concat(benign, ignore_index=True)
    if max_rows:
        df = df.iloc[:max_rows]
    log.info("  Benign rows loaded: %d", len(df))
    return df


def load_all_cic(folder: str, rename: dict = None,
                 max_rows: int = None) -> pd.DataFrame:
    """Load ALL rows (benign + attack) from a CIC folder for evaluation."""
    files  = sorted(Path(folder).glob("*.csv"))
    frames = []
    total  = 0
    for f in files:
        log.info("  Reading %s ...", f.name)
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
            chunk = _clean(chunk, rename)
            chunk = _make_label_cic(chunk)
            frames.append(chunk)
            total += len(chunk)
            if max_rows and total >= max_rows:
                break
        if max_rows and total >= max_rows:
            break
        gc.collect()

    df = pd.concat(frames, ignore_index=True)
    if max_rows:
        df = df.iloc[:max_rows]
    log.info("  Total rows: %d | benign=%d attack=%d",
             len(df), int((df["label"]==0).sum()), int((df["label"]==1).sum()))
    return df


def load_benign_unsw(max_rows: int = None) -> pd.DataFrame:
    folder = DATASET_PATHS["unsw"]
    benign = []
    total  = 0
    for fname in UNSW_DATA_FILES:
        fpath = os.path.join(folder, fname)
        log.info("  Reading %s ...", fname)
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE,
                                  header=None, names=UNSW_COLUMNS,
                                  low_memory=False):
            chunk = _clean(chunk, UNSW_RENAME)
            chunk = _make_label_numeric(chunk, "label")
            chunk = chunk[chunk["label"] == 0].copy()
            benign.append(chunk)
            total += len(chunk)
            if max_rows and total >= max_rows:
                break
        if max_rows and total >= max_rows:
            break
        gc.collect()

    df = pd.concat(benign, ignore_index=True)
    if max_rows:
        df = df.iloc[:max_rows]
    log.info("  UNSW benign rows: %d", len(df))
    return df


def load_all_unsw(max_rows: int = None) -> pd.DataFrame:
    folder = DATASET_PATHS["unsw"]
    frames = []
    total  = 0
    for fname in UNSW_DATA_FILES:
        fpath = os.path.join(folder, fname)
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE,
                                  header=None, names=UNSW_COLUMNS,
                                  low_memory=False):
            chunk = _clean(chunk, UNSW_RENAME)
            chunk = _make_label_numeric(chunk, "label")
            frames.append(chunk)
            total += len(chunk)
            if max_rows and total >= max_rows:
                break
        if max_rows and total >= max_rows:
            break
        gc.collect()

    df = pd.concat(frames, ignore_index=True)
    if max_rows:
        df = df.iloc[:max_rows]
    log.info("  UNSW total rows: %d | benign=%d attack=%d",
             len(df), int((df["label"]==0).sum()), int((df["label"]==1).sum()))
    return df


def load_botiot(max_rows: int = None) -> pd.DataFrame:
    folder = DATASET_PATHS["botiot"]
    frames = []
    total  = 0
    for fname in BOTIOT_DATA_FILES:
        fpath = os.path.join(folder, fname)
        log.info("  Reading %s ...", fname)
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
            chunk = _clean(chunk, BOTIOT_RENAME)
            chunk = _make_label_numeric(chunk, "label")
            frames.append(chunk)
            total += len(chunk)
            if max_rows and total >= max_rows:
                break
        if max_rows and total >= max_rows:
            break
        gc.collect()

    df = pd.concat(frames, ignore_index=True)
    if max_rows and max_rows < len(df):
        # Stratified sample only when we actually need to reduce size
        from sklearn.model_selection import train_test_split
        idx      = np.arange(len(df))
        keep, _  = train_test_split(idx, train_size=max_rows,
                                    stratify=df["label"].values, random_state=42)
        df = df.iloc[keep].reset_index(drop=True)
    log.info("  BoT-IoT rows: %d | benign=%d attack=%d",
             len(df), int((df["label"]==0).sum()), int((df["label"]==1).sum()))
    return df


# =============================================================================
# FEATURE SELECTION  (Spearman correlation)
# =============================================================================

def spearman_selection(X: np.ndarray, feature_names: list,
                        threshold: float) -> list:
    from scipy.stats import rankdata
    log.info("Spearman selection on %d samples x %d features ...",
             X.shape[0], X.shape[1])
    X_r  = np.apply_along_axis(rankdata, 0, X)
    corr = np.corrcoef(X_r.T)
    n    = len(feature_names)
    drop = set()
    for i in range(n):
        if feature_names[i] in drop:
            continue
        for j in range(i + 1, n):
            if feature_names[j] in drop:
                continue
            if abs(corr[i, j]) >= threshold:
                drop.add(feature_names[j])
    selected = [f for f in feature_names if f not in drop]
    log.info("  %d -> %d features (dropped %d)", n, len(selected), len(drop))
    return selected


# =============================================================================
# MAIN
# =============================================================================

def run():
    log.info("=" * 60)
    log.info("v2 Preprocessing â€” memory-safe chunk loading")
    log.info("=" * 60)

    max_benign = MAX_TRAIN_ROWS_PER_DS + MAX_VAL_ROWS_PER_DS

    #    Step 1: Load benign-only for training/val                          
    log.info("Step 1: Loading benign traffic for training ...")

    log.info("  CIC2017 benign ...")
    df17b = load_benign_cic(DATASET_PATHS["cic2017"], rename=None,
                            max_rows=max_benign)
    log.info("  CIC2018 benign ...")
    df18b = load_benign_cic(DATASET_PATHS["cic2018"], rename=CIC2018_RENAME,
                            max_rows=max_benign)
    log.info("  UNSW benign ...")
    dfswb = load_benign_unsw(max_rows=max_benign)

    #    Step 2: Common features                                            
    log.info("Step 2: Finding common features ...")
    def num_cols(df):
        return set(df.select_dtypes(include=[np.number]).columns) - {"label"}

    common = sorted(num_cols(df17b) & num_cols(df18b) & num_cols(dfswb))
    log.info("  Common features: %d", len(common))

    #    Step 3: Spearman feature selection                                 
    log.info("Step 3: Spearman correlation selection ...")
    # Sample up to 50k benign rows from CIC2017 for selection
    n_sel   = min(50_000, len(df17b))
    idx_sel = RNG.choice(len(df17b), size=n_sel, replace=False)
    X_sel   = df17b.iloc[idx_sel][common].values.astype(np.float32)
    selected = spearman_selection(X_sel, common, CORRELATION_THRESHOLD)
    n_feat   = len(selected)
    del X_sel; gc.collect()

    pd.Series(common).to_csv(
        os.path.join(ARTEFACT_DIR, "common_features.csv"), index=False)
    pd.Series(selected).to_csv(
        os.path.join(ARTEFACT_DIR, "selected_features.csv"), index=False)
    log.info("  Final features: %d", n_feat)

    #    Step 4: Build train/val arrays                                     
    log.info("Step 4: Building train/val splits ...")

    def extract_split(df):
        # Fill missing selected features with 0
        for f in selected:
            if f not in df.columns:
                df[f] = 0.0
        X   = df[selected].values.astype(np.float32)
        idx = np.arange(len(X))
        RNG.shuffle(idx)
        n_tr = int(len(idx) * TRAIN_RATIO)
        return X[idx[:n_tr]], X[idx[n_tr:]]

    tr17, va17 = extract_split(df17b)
    tr18, va18 = extract_split(df18b)
    trsw, vasw = extract_split(dfswb)
    del df17b, df18b, dfswb; gc.collect()

    X_train = np.concatenate([tr17, tr18, trsw], axis=0)
    X_val   = np.concatenate([va17, va18, vasw], axis=0)
    log.info("  X_train: %s  [CIC17=%d CIC18=%d UNSW=%d]",
             X_train.shape, len(tr17), len(tr18), len(trsw))
    log.info("  X_val  : %s  [CIC17=%d CIC18=%d UNSW=%d]",
             X_val.shape, len(va17), len(va18), len(vasw))
    del tr17,va17,tr18,va18,trsw,vasw; gc.collect()

    #    Step 5: Fit scaler                                                 
    log.info("Step 5: Fitting IQR scaler on %d training rows ...", len(X_train))
    scaler = IQRScaler()
    scaler.fit(X_train)
    scaler.save(os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"))

    def scale(X):
        return np.clip(scaler.transform(X), -10, 10).astype(np.float32)

    X_train = scale(X_train)
    X_val   = scale(X_val)

    np.savez_compressed(os.path.join(ARTEFACT_DIR, "train.npz"), X_train=X_train)
    np.savez_compressed(os.path.join(ARTEFACT_DIR, "val.npz"),   X_val=X_val)
    del X_train, X_val; gc.collect()

    #    Step 6: Full eval datasets (benign + attack)                       
    log.info("Step 6: Loading full eval datasets ...")

    log.info("  CIC2017 full (500k sample) ...")
    df17 = load_all_cic(DATASET_PATHS["cic2017"], max_rows=500_000)
    for f in selected:
        if f not in df17.columns: df17[f] = 0.0
    X17 = scale(df17[selected].values.astype(np.float32))
    y17 = df17["label"].values
    np.savez_compressed(os.path.join(ARTEFACT_DIR, "test_cic17.npz"), X=X17, y=y17)
    del df17, X17, y17; gc.collect()

    log.info("  CIC2018 full (500k sample) ...")
    df18 = load_all_cic(DATASET_PATHS["cic2018"], rename=CIC2018_RENAME,
                        max_rows=500_000)
    for f in selected:
        if f not in df18.columns: df18[f] = 0.0
    X18 = scale(df18[selected].values.astype(np.float32))
    y18 = df18["label"].values
    np.savez_compressed(os.path.join(ARTEFACT_DIR, "test_cic18.npz"), X=X18, y=y18)
    del df18, X18, y18; gc.collect()

    log.info("  UNSW full (500k sample) ...")
    dfsw = load_all_unsw(max_rows=500_000)
    for f in selected:
        if f not in dfsw.columns: dfsw[f] = 0.0
    Xsw = scale(dfsw[selected].values.astype(np.float32))
    ysw = dfsw["label"].values
    np.savez_compressed(os.path.join(ARTEFACT_DIR, "test_unsw.npz"), X=Xsw, y=ysw)
    del dfsw, Xsw, ysw; gc.collect()

    log.info("  BoT-IoT ...")
    dfbot = load_botiot(max_rows=MAX_BOTIOT_ROWS)
    missing = [f for f in selected if f not in dfbot.columns]
    log.info("  BoT-IoT missing features filled with 0: %d / %d",
             len(missing), n_feat)
    for f in missing:
        dfbot[f] = 0.0
    Xbot = scale(dfbot[selected].values.astype(np.float32))
    ybot = dfbot["label"].values
    np.savez_compressed(os.path.join(ARTEFACT_DIR, "test_botiot.npz"), X=Xbot, y=ybot)
    log.info("  BoT-IoT: %s | benign=%d attack=%d",
             Xbot.shape, int((ybot==0).sum()), int((ybot==1).sum()))
    del dfbot, Xbot, ybot; gc.collect()

    log.info("=" * 60)
    log.info("v2 Preprocessing complete. All files saved to %s", ARTEFACT_DIR)
    log.info("Next: python train_autoencoder_v2.py")
    log.info("=" * 60)


if __name__ == "__main__":
    run()

