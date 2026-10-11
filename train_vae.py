"""
train_vae.py
============
Trains the Variational Autoencoder on CIC-IDS2017 benign traffic.
Loads preprocessed .npz splits from artefacts/.

Run from project root:
    python train_vae.py
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
VAE_PATH     = os.path.join(ARTEFACT_DIR, "vae_model.pt")


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
    print("Training VAE ...")
    print("=" * 60)

    from models.vae import VAE

    vae = VAE(
        input_dim  = X_train.shape[1],
        latent_dim = 8,
        lr         = 1e-3,
        batch_size = 256,
        max_epochs = 100,
        patience   = 10,
        kl_weight  = 0.1,
    )
    vae.fit(X_train, X_val)

    #    Fit threshold                                                      
    print("\n" + "=" * 60)
    print("Fitting anomaly threshold on validation set ...")
    print("=" * 60)

    tau = vae.fit_threshold(X_val, n_sigma=3.0)
    print(f"  Threshold   = {tau:.6f}")

    #    Quick evaluation                                                   
    print("\n" + "=" * 60)
    print("Quick evaluation on CIC2017 test set ...")
    print("=" * 60)

    y_pred = vae.predict(X_test)
    _print_metrics(y_test, y_pred, "CIC2017 test")

    #    Save                                                               
    vae.save(VAE_PATH)
    print(f"\n  VAE saved -> {VAE_PATH}")
    print("\nDone. Next: python train_isolation_forest.py")


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
