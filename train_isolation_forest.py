"""
train_isolation_forest.py
=========================
Trains the Isolation Forest on CIC-IDS2017 benign traffic.
Loads preprocessed .npz splits from artefacts/.

Run from project root:
    python train_isolation_forest.py

Note: IF training is much faster than AE/VAE — typically under 2 minutes
for 1.5M samples with n_estimators=100, max_samples=256.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np

from preprocessing.pipeline import PreprocessingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts"
IF_PATH      = os.path.join(ARTEFACT_DIR, "if_model.pkl")


def run():
    #    Load preprocessed splits                                           
    print("\n" + "=" * 60)
    print("Loading preprocessed datasets ...")
    print("=" * 60)

    data    = PreprocessingPipeline.load_splits(ARTEFACT_DIR)
    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    # Clip scaled values to [-10, 10]
    print("  Clipping scaled values to [-10, 10] ...")
    X_train = np.clip(X_train, -10, 10)
    X_val   = np.clip(X_val,   -10, 10)
    X_test  = np.clip(X_test,  -10, 10)

    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_test  : {X_test.shape}")

    #    Train                                                              
    print("\n" + "=" * 60)
    print("Training Isolation Forest ...")
    print("=" * 60)

    from models.isolation_forest import IsolationForestModel

    iforest = IsolationForestModel(
        n_estimators = 100,
        max_samples  = 256,
        n_jobs       = -1,      # use all CPU cores
        random_state = 42,
    )
    iforest.fit(X_train)

    #    Fit threshold                                                      
    print("\n" + "=" * 60)
    print("Fitting anomaly threshold on validation set ...")
    print("=" * 60)

    tau = iforest.fit_threshold(X_val, n_sigma=3.0, percentile=95.0)
    print(f"  Threshold   = {tau:.6f}")

    #    Quick evaluation                                                   
    print("\n" + "=" * 60)
    print("Quick evaluation on CIC2017 test set ...")
    print("=" * 60)

    y_pred = iforest.predict(X_test)
    _print_metrics(y_test, y_pred, "CIC2017 test")

    #    Save                                                               
    iforest.save(IF_PATH)
    print(f"\n  IF saved -> {IF_PATH}")
    print("\nDone. Next: python train_ensemble.py")


def _print_metrics(y_true, y_pred, label):
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, confusion_matrix)
    acc  = accuracy_score(y_true,  y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true,    y_pred, zero_division=0)
    f1   = f1_score(y_true,        y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n  [{label}]")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")


if __name__ == "__main__":
    run()
