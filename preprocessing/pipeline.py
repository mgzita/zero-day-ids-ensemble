"""
preprocessing/pipeline.py
=========================
Orchestrates all preprocessing modules in the correct order:

    1.  Load + clean CIC-IDS2017
    2.  Load + clean CIC-IDS2018 and UNSW -> find common features
    3.  Align CIC-IDS2017 to common features
    4.  Split into train_df / val_df / test_df   (DataFrames)
            train_df = 70% benign only
            val_df   = 15% benign only
            test_df  = 15% benign + ALL attacks
    5.  Fit Spearman correlation selector on train_df (benign only)
    6.  Reduce all DataFrames to selected features
    7.  Convert to numpy
    8.  Fit IQR scaler on combined benign from CIC2017 + CIC2018 + UNSW (supervisor fix 1)
    9.  Scale all splits
    10. Apply same feature selection + scaling to eval datasets
    11. Save everything to .npz files
"""

import logging
import os

import numpy as np
import pandas as pd

from .loader   import load_dataset
from .cleaner  import clean
from .features import get_common_features, align_features, fit_correlation_selector
from .scaler   import IQRScaler
from .splitter import split_dataset
from .config_v1   import ARTEFACT_DIR, CORRELATION_THRESHOLD, TRAIN_RATIO, VAL_RATIO

log = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for zero-day attack detection.

    First run  (processes raw CSVs)
    --------------------------------
    pipeline = PreprocessingPipeline()

    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.fit_transform(
        train_path   = DATASET_PATHS["cic2017"],
        dataset_name = "cic2017",
        eval_paths   = {
            "cic2018": DATASET_PATHS["cic2018"],
            "unsw":    DATASET_PATHS["unsw"],
        }
    )
    X_cic18, y_cic18 = pipeline.transform("cic2018")
    X_unsw,  y_unsw  = pipeline.transform("unsw")

    pipeline.save()
    pipeline.save_splits(X_train, y_train, X_val, y_val,
                         X_test,  y_test,  X_cic18, y_cic18,
                         X_unsw,  y_unsw)

    Subsequent runs  (loads from .npz, skips all preprocessing)
    ------------------------------------------------------------
    data = PreprocessingPipeline.load_splits(ARTEFACT_DIR)
    X_train = data["X_train"]
    ...
    """

    def __init__(self,
                 corr_threshold: float = CORRELATION_THRESHOLD,
                 train_ratio:    float = TRAIN_RATIO,
                 val_ratio:      float = VAL_RATIO,
                 artefact_dir:   str   = ARTEFACT_DIR):

        self.corr_threshold     = corr_threshold
        self.train_ratio        = train_ratio
        self.val_ratio          = val_ratio
        self.artefact_dir       = artefact_dir

        self.common_features_   = None
        self.selected_features_ = None
        self.scaler_            = IQRScaler()
        self._eval_cache        = {}   # eval datasets keyed by name
        self._train_cache       = None # CIC2017 DataFrame if pre-cached
        self._fitted            = False

    # =========================================================================
    # FIT + TRANSFORM  (CIC-IDS2017)
    # =========================================================================

    def fit_transform(self,
                      train_path:   str,
                      dataset_name: str,
                      eval_paths:   dict = None):
        """
        Full pipeline on CIC-IDS2017.

        Parameters
        ----------
        train_path   : path to CIC-IDS2017 folder or CSV file
        dataset_name : 'cic2017'
        eval_paths   : {'cic2018': path, 'unsw': path}
                       Loaded now to compute common features.
                       Cached internally for transform() calls.

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test  (scaled numpy arrays)
        """
        log.info("=" * 55)
        log.info("Fitting pipeline on '%s'", dataset_name)
        log.info("=" * 55)

        # 1. Load + clean CIC-IDS2017 (use pre-cached if available)
        if self._train_cache is not None:
            log.info("Using pre-cached training data (%d rows).",
                     len(self._train_cache))
            df = self._train_cache
        else:
            df = load_dataset(train_path, dataset_name)
            df = clean(df)

        # 2. Load + clean eval datasets; find common feature set
        if eval_paths:
            eval_dfs = self._load_and_cache_eval(eval_paths)
            self.common_features_ = get_common_features(df, *eval_dfs.values())
        else:
            self.common_features_ = sorted(
                set(df.select_dtypes(include=["number"]).columns) - {"label"}
            )
            log.info("No eval paths given -- using all %d training features.",
                     len(self.common_features_))

        # 3. Align CIC-IDS2017 to common features
        df = align_features(df, self.common_features_)

        # 4. Split into DataFrames FIRST
        #    train_df = 70% benign only
        #    val_df   = 15% benign only
        #    test_df  = 15% benign + ALL attacks
        log.info("Splitting dataset ...")
        train_df, val_df, test_df = split_dataset(
            df, self.train_ratio, self.val_ratio
        )

        # 5. Fit Spearman selector on benign training DataFrame only
        log.info("Fitting correlation selector on training split ...")
        self.selected_features_ = fit_correlation_selector(
            train_df, self.common_features_, self.corr_threshold
        )

        # 6. Reduce all DataFrames to selected features
        train_df = align_features(train_df, self.selected_features_)
        val_df   = align_features(val_df,   self.selected_features_)
        test_df  = align_features(test_df,  self.selected_features_)

        # 7. Convert to numpy
        X_train, y_train = _to_numpy(train_df, self.selected_features_)
        X_val,   y_val   = _to_numpy(val_df,   self.selected_features_)
        X_test,  y_test  = _to_numpy(test_df,  self.selected_features_)

        # 8. Fit IQR scaler on combined benign from all datasets
        #    (supervisor fix 1: global feature normalisation)
        #    Benign samples from CIC2018 and UNSW are aligned to selected
        #    features and combined with CIC2017 benign before fitting.
        log.info("Fitting IQR scaler on combined benign from all datasets ...")
        X_combined = self._build_combined_benign(X_train)
        self.scaler_.fit(X_combined)
        log.info("  Combined benign scaler fitted on %d rows.", len(X_combined))
        del X_combined  # free memory

        # 9. Scale all splits
        X_train = self.scaler_.transform(X_train)
        X_val   = self.scaler_.transform(X_val)
        X_test  = self.scaler_.transform(X_test)

        self._fitted = True
        log.info("Pipeline fitted. Final feature count: %d",
                 len(self.selected_features_))

        return X_train, X_val, X_test, y_train, y_val, y_test

    # =========================================================================
    # TRANSFORM EVALUATION DATASETS  (no refit)
    # =========================================================================

    def transform(self, dataset_name: str):
        """
        Apply fitted feature selection + scaling to a cached eval dataset.
        Must call fit_transform() first.

        Parameters
        ----------
        dataset_name : 'cic2018' or 'unsw'

        Returns X (np.ndarray), y (np.ndarray)
        """
        self._check_fitted()

        if dataset_name not in self._eval_cache:
            raise KeyError(
                f"'{dataset_name}' was not loaded during fit_transform(). "
                f"Pass its path in eval_paths."
            )

        df = self._eval_cache[dataset_name]
        # fill_missing=True: cross-dataset eval may lack some features
        # (e.g. UNSW has no equivalent for active_max, act_data_pkt_fwd etc.)
        # Missing columns are filled with 0.0 so shapes always match.
        df = align_features(df, self.selected_features_, fill_missing=True)

        X, y = _to_numpy(df, self.selected_features_)
        X    = self.scaler_.transform(X)

        log.info(
            "transform '%s': X=%s | benign=%d | attack=%d",
            dataset_name, X.shape,
            int((y == 0).sum()), int((y == 1).sum())
        )
        return X, y

    # =========================================================================
    # SAVE / LOAD PIPELINE ARTEFACTS  (scaler + feature lists)
    # =========================================================================

    def save(self, output_dir: str = None) -> None:
        """Save scaler parameters + feature lists to disk."""
        self._check_fitted()
        out = output_dir or self.artefact_dir
        os.makedirs(out, exist_ok=True)

        self.scaler_.save(os.path.join(out, "iqr_scaler.npz"))
        pd.Series(self.common_features_).to_csv(
            os.path.join(out, "common_features.csv"),
            index=False, header=["feature"]
        )
        pd.Series(self.selected_features_).to_csv(
            os.path.join(out, "selected_features.csv"),
            index=False, header=["feature"]
        )
        log.info("Pipeline artefacts saved -> '%s'", out)

    def load(self, output_dir: str = None) -> "PreprocessingPipeline":
        """Reload saved pipeline artefacts (scaler + feature lists)."""
        src = output_dir or self.artefact_dir

        self.scaler_.load(os.path.join(src, "iqr_scaler.npz"))
        self.common_features_ = pd.read_csv(
            os.path.join(src, "common_features.csv")
        )["feature"].tolist()
        self.selected_features_ = pd.read_csv(
            os.path.join(src, "selected_features.csv")
        )["feature"].tolist()

        self._fitted = True
        log.info("Pipeline artefacts loaded <- '%s' | features: %d",
                 src, len(self.selected_features_))
        return self

    # =========================================================================
    # SAVE / LOAD PROCESSED DATASETS  (scaled numpy arrays -> .npz)
    # =========================================================================

    def save_splits(self,
                    X_train, y_train,
                    X_val,   y_val,
                    X_test,  y_test,
                    X_cic18, y_cic18,
                    X_unsw,  y_unsw,
                    output_dir: str = None) -> None:
        """
        Save all processed + scaled datasets as compressed .npz files.

        Files created
        -------------
        cic2017_train.npz  -> X_train, y_train  (scaled, benign only)
        cic2017_val.npz    -> X_val,   y_val    (scaled, benign only)
        cic2017_test.npz   -> X_test,  y_test   (scaled, benign + attacks)
        cic2018.npz        -> X_cic18, y_cic18  (scaled, benign + attacks)
        unsw.npz           -> X_unsw,  y_unsw   (scaled, benign + attacks)
        """
        out = output_dir or self.artefact_dir
        os.makedirs(out, exist_ok=True)

        _save_npz(out, "cic2017_train", X_train, y_train)
        _save_npz(out, "cic2017_val",   X_val,   y_val)
        _save_npz(out, "cic2017_test",  X_test,  y_test)
        _save_npz(out, "cic2018",       X_cic18, y_cic18)
        _save_npz(out, "unsw",          X_unsw,  y_unsw)

        log.info("All processed datasets saved -> '%s'", out)
        log.info("  cic2017_train : X=%s  y=%s", X_train.shape, y_train.shape)
        log.info("  cic2017_val   : X=%s  y=%s", X_val.shape,   y_val.shape)
        log.info("  cic2017_test  : X=%s  y=%s", X_test.shape,  y_test.shape)
        log.info("  cic2018       : X=%s  y=%s", X_cic18.shape, y_cic18.shape)
        log.info("  unsw          : X=%s  y=%s", X_unsw.shape,  y_unsw.shape)

    @staticmethod
    def load_splits(output_dir: str = ARTEFACT_DIR) -> dict:
        """
        Load all processed datasets from .npz files.
        Use this to skip preprocessing entirely on subsequent runs.

        Parameters
        ----------
        output_dir : folder where .npz files were saved (default: ARTEFACT_DIR)

        Returns
        -------
        dict with keys:
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            X_cic18, y_cic18,
            X_unsw,  y_unsw

        Usage
        -----
        data    = PreprocessingPipeline.load_splits(ARTEFACT_DIR)
        X_train = data["X_train"]
        y_test  = data["y_test"]
        """
        log.info("Loading processed datasets from '%s' ...", output_dir)

        X_train, y_train = _load_npz(output_dir, "cic2017_train")
        X_val,   y_val   = _load_npz(output_dir, "cic2017_val")
        X_test,  y_test  = _load_npz(output_dir, "cic2017_test")
        X_cic18, y_cic18 = _load_npz(output_dir, "cic2018")
        X_unsw,  y_unsw  = _load_npz(output_dir, "unsw")

        log.info("Datasets loaded successfully.")
        log.info("  X_train : %s | X_val  : %s | X_test : %s",
                 X_train.shape, X_val.shape, X_test.shape)
        log.info("  X_cic18 : %s | X_unsw : %s",
                 X_cic18.shape, X_unsw.shape)

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "X_test":  X_test,  "y_test":  y_test,
            "X_cic18": X_cic18, "y_cic18": y_cic18,
            "X_unsw":  X_unsw,  "y_unsw":  y_unsw,
        }

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _load_and_cache_eval(self, eval_paths: dict) -> dict:
        eval_dfs = {}
        for name, path in eval_paths.items():
            log.info("--- Loading eval dataset '%s' ---", name)
            df = load_dataset(path, name)
            df = clean(df)
            self._eval_cache[name] = df
            eval_dfs[name] = df
        return eval_dfs

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "Pipeline not fitted. Call fit_transform() first."
            )

    def _build_combined_benign(self, X_train: np.ndarray) -> np.ndarray:
        """
        Build a combined benign array from CIC2017 + all cached eval datasets.
        Used to fit the IQR scaler on a globally representative benign sample.

        For each eval dataset we take benign rows only, align to selected
        features, and concatenate with CIC2017 X_train (already benign only).
        To keep memory manageable we sample up to 100k benign rows per eval
        dataset.
        """
        MAX_EVAL_BENIGN = 100_000
        parts = [X_train]

        for name, df in self._eval_cache.items():
            # Benign only
            df_benign = df[df["label"] == 0] if "label" in df.columns else df

            # Align to selected features (fill missing with 0)
            df_benign = align_features(df_benign, self.selected_features_,
                                       fill_missing=True)
            X_eval, _ = _to_numpy(df_benign, self.selected_features_)

            # Sample if too large
            if len(X_eval) > MAX_EVAL_BENIGN:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_eval), size=MAX_EVAL_BENIGN,
                                 replace=False)
                X_eval = X_eval[idx]

            parts.append(X_eval)
            log.info("  Combined scaler: added %d benign rows from '%s'.",
                     len(X_eval), name)

        combined = np.concatenate(parts, axis=0)
        log.info("  Combined scaler total: %d rows.", len(combined))
        return combined


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================

def _to_numpy(df: pd.DataFrame, feature_cols: list):
    """Extract feature matrix and label vector as numpy arrays."""
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y


def _save_npz(directory: str, name: str,
              X: np.ndarray, y: np.ndarray) -> None:
    """Save a single X, y pair as a compressed .npz file."""
    path = os.path.join(directory, f"{name}.npz")
    np.savez_compressed(path, X=X, y=y)
    log.info("  Saved '%s.npz'", name)


def _load_npz(directory: str, name: str):
    """Load a single X, y pair from a .npz file."""
    path = os.path.join(directory, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found: '{path}'\n"
            f"Run run_preprocessing.py first to generate it."
        )
    data = np.load(path)
    return data["X"], data["y"]
