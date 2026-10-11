"""
run_preprocessing.py
====================
Entry point for the full preprocessing pipeline.

Checkpoint behaviour
--------------------
Each dataset is saved to .npz immediately after processing.
If a run fails midway, completed datasets are NOT reprocessed on the next run.
Only missing .npz files are reprocessed on restart.

Run from the project root:
    python run_preprocessing.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np

from preprocessing_v2 import PreprocessingPipeline, DATASET_PATHS, ARTEFACT_DIR
from preprocessing.loader import load_dataset
from preprocessing.cleaner import clean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)

os.makedirs(ARTEFACT_DIR, exist_ok=True)

# =============================================================================
# Expected .npz checkpoint paths
# =============================================================================

NPZ = {
    "cic2017_train": os.path.join(ARTEFACT_DIR, "cic2017_train.npz"),
    "cic2017_val":   os.path.join(ARTEFACT_DIR, "cic2017_val.npz"),
    "cic2017_test":  os.path.join(ARTEFACT_DIR, "cic2017_test.npz"),
    "cic2018":       os.path.join(ARTEFACT_DIR, "cic2018.npz"),
    "unsw":          os.path.join(ARTEFACT_DIR, "unsw.npz"),
}

ARTEFACT_FILES = [
    os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"),
    os.path.join(ARTEFACT_DIR, "common_features.csv"),
    os.path.join(ARTEFACT_DIR, "selected_features.csv"),
]


def exists(key):   return os.path.exists(NPZ[key])
def artefacts_ok():return all(os.path.exists(p) for p in ARTEFACT_FILES)

def save_npz(key, X, y):
    np.savez_compressed(NPZ[key], X=X, y=y)
    logging.info("Checkpoint saved -> %s", NPZ[key])

def load_npz(key):
    d = np.load(NPZ[key])
    return d["X"], d["y"]


# =============================================================================
# MAIN
# =============================================================================

def run():

    #    All done                                                           
    if all(exists(k) for k in NPZ) and artefacts_ok():
        print("\nAll datasets already processed. Loading from .npz ...")
        data = PreprocessingPipeline.load_splits(ARTEFACT_DIR)
        _print_summary(data)
        return data

    pipeline = PreprocessingPipeline()

    # =========================================================================
    # STEP 1 — Load + cache eval datasets individually (with checkpointing)
    # Each dataset is loaded, cleaned, and cached in pipeline._eval_cache.
    # This happens BEFORE fit_transform so a crash on one dataset does not
    # force us to reload the others.
    # =========================================================================

    #    CIC-IDS2017                                                        
    _header("STEP 1a: Load + cache CIC-IDS2017")
    if exists("cic2017_train") and artefacts_ok():
        print("  Already processed. Loading artefacts only ...")
        pipeline.load(ARTEFACT_DIR)
        _cache_train_for_features(pipeline)
    else:
        _cache_train(pipeline)

    #    CIC-IDS2018                                                        
    _header("STEP 1b: Load + cache CIC-IDS2018")
    if exists("cic2018") and artefacts_ok():
        print("  Already processed. Caching headers for feature intersection ...")
        _cache_eval_for_features(pipeline, "cic2018")
    else:
        _cache_eval(pipeline, "cic2018")

    #    UNSW-NB15                                                          
    _header("STEP 1c: Load + cache UNSW-NB15")
    if exists("unsw") and artefacts_ok():
        print("  Already processed. Caching headers for feature intersection ...")
        _cache_eval_for_features(pipeline, "unsw")
    else:
        _cache_eval(pipeline, "unsw")

    # =========================================================================
    # STEP 2 — Fit + transform CIC-IDS2017
    # All datasets already cached -- fit_transform uses cache, no reloading
    # =========================================================================

    if exists("cic2017_train") and exists("cic2017_val") and \
       exists("cic2017_test") and artefacts_ok():
        _header("STEP 2: CIC2017 already processed — loading from .npz")
        X_train, y_train = load_npz("cic2017_train")
        X_val,   y_val   = load_npz("cic2017_val")
        X_test,  y_test  = load_npz("cic2017_test")
        print(f"  X_train : {X_train.shape} | X_val : {X_val.shape} | X_test : {X_test.shape}")
    else:
        _header("STEP 2: Fit + transform CIC-IDS2017")
        # eval_paths=None because CIC2018 + UNSW already cached in step 1
        # train data also already cached in step 1a
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.fit_transform(
            train_path   = DATASET_PATHS["cic2017"],
            dataset_name = "cic2017",
            eval_paths   = None,
        )
        print(f"\n  Features selected : {len(pipeline.selected_features_)}")
        print(f"  X_train : {X_train.shape}  (benign only)")
        print(f"  X_val   : {X_val.shape}    (benign only)")
        print(f"  X_test  : {X_test.shape}   (benign + attacks)")

        save_npz("cic2017_train", X_train, y_train)
        save_npz("cic2017_val",   X_val,   y_val)
        save_npz("cic2017_test",  X_test,  y_test)
        pipeline.save(ARTEFACT_DIR)
        print("    CIC2017 checkpoint saved.")

    # =========================================================================
    # STEP 3 — Transform CIC-IDS2018
    # =========================================================================

    if exists("cic2018"):
        _header("STEP 3: CIC2018 already processed — loading from .npz")
        X_cic18, y_cic18 = load_npz("cic2018")
        print(f"  X_cic18 : {X_cic18.shape}")
    else:
        _header("STEP 3: Transform CIC-IDS2018")
        X_cic18, y_cic18 = pipeline.transform("cic2018")
        print(f"  X_cic18 : {X_cic18.shape}")
        print(f"  Benign  : {(y_cic18==0).sum()} | Attack : {(y_cic18==1).sum()}")
        save_npz("cic2018", X_cic18, y_cic18)
        print("    CIC2018 checkpoint saved.")

    # =========================================================================
    # STEP 4 — Transform UNSW-NB15
    # =========================================================================

    if exists("unsw"):
        _header("STEP 4: UNSW already processed — loading from .npz")
        X_unsw, y_unsw = load_npz("unsw")
        print(f"  X_unsw  : {X_unsw.shape}")
    else:
        _header("STEP 4: Transform UNSW-NB15")
        X_unsw, y_unsw = pipeline.transform("unsw")
        print(f"  X_unsw  : {X_unsw.shape}")
        print(f"  Benign  : {(y_unsw==0).sum()} | Attack : {(y_unsw==1).sum()}")
        save_npz("unsw", X_unsw, y_unsw)
        print("    UNSW checkpoint saved.")

    #    Summary                                                            
    data = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "X_cic18": X_cic18, "y_cic18": y_cic18,
        "X_unsw":  X_unsw,  "y_unsw":  y_unsw,
        "pipeline": pipeline,
    }
    _print_summary(data)
    return data


# =============================================================================
# HELPERS
# =============================================================================

def _cache_train(pipeline):
    """Load, clean and cache CIC2017 training data into pipeline._train_cache."""
    logging.info("Loading training dataset 'cic2017' ...")
    df = load_dataset(DATASET_PATHS["cic2017"], "cic2017")
    df = clean(df)
    pipeline._train_cache = df
    logging.info("Cached 'cic2017': %d rows", len(df))


def _cache_train_for_features(pipeline):
    """
    Load just 1000 rows of CIC2017 to get column names for feature
    intersection. Used when CIC2017 .npz already exists.
    """
    import pandas as pd, glob
    from preprocessing.loader import normalise_columns
    path  = DATASET_PATHS["cic2017"]
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    df    = pd.read_csv(files[0], nrows=1000, low_memory=False)
    df    = normalise_columns(df)
    pipeline._train_cache = df
    logging.info("Cached 'cic2017' (header-only, %d rows).", len(df))


def _cache_eval(pipeline, name):
    """Load, clean and cache a full eval dataset into pipeline._eval_cache."""
    logging.info("Loading eval dataset '%s' ...", name)
    df = load_dataset(DATASET_PATHS[name], name)
    df = clean(df)
    pipeline._eval_cache[name] = df
    logging.info("Cached '%s': %d rows", name, len(df))


def _cache_eval_for_features(pipeline, name):
    """
    Load just enough of an eval dataset to compute common features.
    Used when the dataset's .npz already exists but we still need its
    column names for get_common_features() during fit_transform().
    Loads only 1000 rows -- very fast.
    """
    import pandas as pd
    from preprocessing.loader import normalise_columns, _apply_rename
    from preprocessing.config_v1 import UNSW_COLUMNS, CIC2018_RENAME, UNSW_RENAME
    import glob

    path = DATASET_PATHS[name]
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    first_file = files[0]

    if name == "unsw":
        from preprocessing.config_v1 import UNSW_DATA_FILES
        unsw_files = [os.path.join(path, f) for f in UNSW_DATA_FILES
                      if os.path.exists(os.path.join(path, f))]
        first_file = unsw_files[0]
        df = pd.read_csv(first_file, header=None, names=UNSW_COLUMNS,
                         nrows=1000, low_memory=False)
    else:
        df = pd.read_csv(first_file, nrows=1000, low_memory=False)

    df = normalise_columns(df)
    if name == "cic2018":
        from preprocessing.loader import _rename_cic2018_columns
        df = _rename_cic2018_columns(df)
    elif name == "unsw":
        from preprocessing.loader import _rename_unsw_columns
        df = _rename_unsw_columns(df)

    pipeline._eval_cache[name] = df
    logging.info("Cached '%s' (header-only, %d rows) for feature intersection.",
                 name, len(df))


def _header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _print_summary(data):
    print("\n" + "=" * 60)
    print("All datasets ready.")
    print("=" * 60)
    for key in ["X_train", "X_val", "X_test", "X_cic18", "X_unsw"]:
        if key in data:
            print(f"  {key:12s} : {data[key].shape}")


if __name__ == "__main__":
    run()
