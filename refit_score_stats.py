"""
refit_score_stats.py
====================
Loads already-trained AE and VAE models, computes REN score statistics
(mu, sigma) on X_train, refits thresholds on X_val, and resaves.

Run this INSTEAD of retraining when only score normalisation has changed.
No retraining needed -- model weights are unchanged.

    python refit_score_stats.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np

from preprocessing.pipeline  import PreprocessingPipeline
from models.autoencoder      import Autoencoder
from models.vae              import VAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts"
AE_PATH      = os.path.join(ARTEFACT_DIR, "ae_model.pt")
VAE_PATH     = os.path.join(ARTEFACT_DIR, "vae_model.pt")


def refit(model, name, path, X_train, X_val, X_test, y_test):
    print("")
    print("=" * 60)
    print("Refitting score stats: " + name)
    print("=" * 60)

    # 1. Fit score stats on training data (REN)
    print("  Computing REN score stats on X_train ...")
    model.fit_score_stats(X_train)

    # 1b. Fit latent stats on training data (LSAS)
    print("  Computing LSAS latent stats on X_train ...")
    model.fit_latent_stats(X_train)

    # 2. Refit threshold on val (now uses z-scores internally)
    print("  Refitting threshold on X_val ...")
    tau = model.fit_threshold(X_val, n_sigma=3.0)
    print("  New threshold tau = " + f"{tau:.6f}")

    # 3. Quick eval on CIC2017 test
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, confusion_matrix)
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test,  y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test,    y_pred, zero_division=0)
    f1   = f1_score(y_test,        y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    print("")
    print("  [CIC2017 test]")
    print("  Accuracy  : " + f"{acc:.4f}")
    print("  Precision : " + f"{prec:.4f}")
    print("  Recall    : " + f"{rec:.4f}")
    print("  F1        : " + f"{f1:.4f}")
    print("  TN=" + f"{cm[0,0]:,}" + "  FP=" + f"{cm[0,1]:,}")
    print("  FN=" + f"{cm[1,0]:,}" + "  TP=" + f"{cm[1,1]:,}")

    # 4. Resave with score stats included
    model.save(path)
    print("  Saved -> " + path)


def run():
    # Load datasets
    print("Loading datasets ...")
    data    = PreprocessingPipeline.load_splits(ARTEFACT_DIR)
    X_train = np.clip(data["X_train"], -10, 10)
    X_val   = np.clip(data["X_val"],   -10, 10)
    X_test  = np.clip(data["X_test"],  -10, 10)
    y_test  = data["y_test"]

    #    AE                                                                 
    if os.path.exists(AE_PATH):
        ae = Autoencoder().load(AE_PATH)
        refit(ae, "Autoencoder", AE_PATH, X_train, X_val, X_test, y_test)
    else:
        print("AE model not found at " + AE_PATH + " -- skipping.")

    #    VAE                                                                
    if os.path.exists(VAE_PATH):
        vae = VAE().load(VAE_PATH)
        refit(vae, "VAE", VAE_PATH, X_train, X_val, X_test, y_test)
    else:
        print("VAE model not found at " + VAE_PATH + " -- skipping (still training ).")

    print("")
    print("Done. Run train_ensemble.py next.")


if __name__ == "__main__":
    run()
