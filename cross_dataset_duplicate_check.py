"""
cross_dataset_duplicate_check.py
=================================
Checks for identical flows across datasets
after feature harmonization and IQR scaling.
Uses hash-based comparison for efficiency.

Output: prints counts and saves report
Expected time: ~10 minutes
"""
import os, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"

def hash_rows(X):
    """Hash each row to a string for fast comparison."""
    return set(map(tuple, np.round(X, 4).tolist()))

def run():
    log.info("Loading all partitions...")
    
    d = np.load(os.path.join(DEDUP, "train_dedup.npz"))
    X_train = np.clip(d["X_train"], -10, 10)
    n = len(X_train) // 3
    train_cic17 = X_train[:n]
    train_cic18 = X_train[n:2*n]
    train_unsw  = X_train[2*n:]
    
    d = np.load(os.path.join(DEDUP, "val_dedup.npz"))
    X_val = np.clip(d["X_val"], -10, 10)
    nv = len(X_val) // 3
    val_cic17 = X_val[:nv]
    val_cic18 = X_val[nv:2*nv]
    val_unsw  = X_val[2*nv:]
    
    tests = {}
    for name, f in [("CIC17","test_cic17_dedup.npz"),
                    ("CIC18","test_cic18_dedup.npz"),
                    ("UNSW","test_unsw_dedup.npz")]:
        d = np.load(os.path.join(DEDUP, f))
        tests[name] = np.clip(d["X"], -10, 10)
        log.info("Loaded %s: %d rows", name, len(tests[name]))

    log.info("Hashing training partitions...")
    h_tr_cic17 = hash_rows(train_cic17)
    h_tr_cic18 = hash_rows(train_cic18)
    h_tr_unsw  = hash_rows(train_unsw)

    print("\n" + "="*60)
    print("CROSS-DATASET DUPLICATE CHECK")
    print("="*60)

    # Cross-training-environment duplicates
    pairs = [
        ("Train CIC17", h_tr_cic17, "Train CIC18", h_tr_cic18),
        ("Train CIC17", h_tr_cic17, "Train UNSW",  h_tr_unsw),
        ("Train CIC18", h_tr_cic18, "Train UNSW",  h_tr_unsw),
    ]
    print("\nCross-environment duplicates in training pool:")
    total_cross = 0
    for n1, h1, n2, h2 in pairs:
        overlap = len(h1 & h2)
        total_cross += overlap
        pct = overlap / min(len(h1), len(h2)) * 100
        print(f"  {n1} vs {n2}: {overlap:,} ({pct:.3f}%)")

    # Training vs test duplicates
    print("\nTraining vs test set duplicates:")
    h_train_all = h_tr_cic17 | h_tr_cic18 | h_tr_unsw
    for name, X_test in tests.items():
        h_test = hash_rows(X_test)
        overlap = len(h_train_all & h_test)
        pct = overlap / len(h_test) * 100
        print(f"  Train vs Test-{name}: {overlap:,} ({pct:.3f}%)")

    print(f"\nTotal cross-environment training duplicates: {total_cross:,}")
    if total_cross == 0:
        print("RESULT: No cross-dataset duplicates found after harmonization.")
        print("        The 18-feature intersection does not produce identical")
        print("        flows across different network environments.")
    else:
        print(f"RESULT: {total_cross:,} cross-dataset duplicates found.")
        print("        These should be investigated.")

if __name__ == "__main__":
    run()
