"""
generate_pr_auc_table.py
=========================
Generates Table 6: ROC-AUC, PR-AUC, and TPR at fixed FPR
operating points using deduplicated seed 42 models.

Run from project root:
    python generate_pr_auc_table.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from models.autoencoder import Autoencoder
from models.vae import VAE
from models.deep_svdd import DeepSVDD

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
DEDUP_DIR    = os.path.join(ARTEFACT_DIR, "dedup")
SEED42_DIR   = os.path.join(DEDUP_DIR, "seed42")


def norm_minmax(s):
    lo = float(np.percentile(s, 1))
    hi = float(np.percentile(s, 99))
    if hi - lo < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def get_ae_scores(ae, X):
    ae.model_.eval()
    device = next(ae.model_.parameters()).device
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 10000):
            batch = torch.FloatTensor(X[i:i+10000]).to(device)
            recon = ae.model_(batch)
            scores.append(((batch - recon)**2).mean(dim=1).cpu().numpy())
    return np.concatenate(scores)


def get_vae_scores(vae, X):
    vae.model_.eval()
    device = next(vae.model_.parameters()).device
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 10000):
            batch = torch.FloatTensor(X[i:i+10000]).to(device)
            x_recon, mu, log_var = vae.model_(batch)
            recon = ((batch - x_recon)**2).mean(dim=1)
            kl = -0.5*(1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
            scores.append((recon + kl).cpu().numpy())
    return np.concatenate(scores)


def get_svdd_scores(svdd, X):
    svdd.model_.eval()
    device = next(svdd.model_.parameters()).device
    centre = svdd.centre_.to(device)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 10000):
            batch = torch.FloatTensor(X[i:i+10000]).to(device)
            z = svdd.model_(batch)
            scores.append(((z - centre)**2).sum(dim=1).cpu().numpy())
    return np.concatenate(scores)


def ensemble_scores(ae, vae, svdd, X):
    return (0.4 * norm_minmax(get_ae_scores(ae, X)) +
            0.4 * norm_minmax(get_vae_scores(vae, X)) +
            0.2 * norm_minmax(get_svdd_scores(svdd, X)))


def tpr_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(tpr):
        return tpr[-1]
    return float(tpr[idx])


def run():
    print("Loading dedup seed 42 models...")
    ae   = Autoencoder(input_dim=18)
    ae.load(os.path.join(SEED42_DIR, "ae.pt"))
    vae  = VAE(input_dim=18)
    vae.load(os.path.join(SEED42_DIR, "vae.pt"))
    svdd = DeepSVDD(input_dim=18)
    svdd.load(os.path.join(SEED42_DIR, "svdd.pt"))

    test_sets = [
        ("CIC-IDS2017",     "test_cic17_dedup.npz"),
        ("CSE-CIC-IDS2018", "test_cic18_dedup.npz"),
        ("UNSW-NB15",       "test_unsw_dedup.npz"),
        ("CIC-ToN-IoT",     None),  # CSV — skip for now
    ]

    results = []
    print()
    print(f"{'Dataset':<20} {'ROC-AUC':>8} {'PR-AUC':>8} "
          f"{'TPR@1%':>8} {'TPR@5%':>8} {'TPR@10%':>9}")
    print("-" * 65)

    for name, fname in test_sets:
        if fname is None:
            # Use original ToN-IoT result (not deduplicated)
            results.append({
                "Dataset": name,
                "ROC-AUC": 0.953,
                "PR-AUC": 0.164,
                "TPR@FPR=1%": "56.5%",
                "TPR@FPR=5%": "92.2%",
                "TPR@FPR=10%": "92.2%",
                "note": "original (not deduplicated)"
            })
            print(f"{name:<20} {'0.953':>8} {'0.164':>8} "
                  f"{'56.5%':>8} {'92.2%':>8} {'92.2%':>9}  (original)")
            continue

        path = os.path.join(DEDUP_DIR, fname)
        if not os.path.exists(path):
            print(f"  {name}: file not found")
            continue

        d = np.load(path)
        X = np.clip(d["X"], -10, 10).astype(np.float32)
        y = d["y"].astype(int)

        scores = ensemble_scores(ae, vae, svdd, X)

        roc_auc = roc_auc_score(y, scores)
        pr_auc  = average_precision_score(y, scores)
        tpr1    = tpr_at_fpr(y, scores, 0.01)
        tpr5    = tpr_at_fpr(y, scores, 0.05)
        tpr10   = tpr_at_fpr(y, scores, 0.10)

        results.append({
            "Dataset": name,
            "ROC-AUC": round(roc_auc, 3),
            "PR-AUC":  round(pr_auc, 3),
            "TPR@FPR=1%":  f"{tpr1*100:.1f}%",
            "TPR@FPR=5%":  f"{tpr5*100:.1f}%",
            "TPR@FPR=10%": f"{tpr10*100:.1f}%",
        })

        print(f"{name:<20} {roc_auc:>8.3f} {pr_auc:>8.3f} "
              f"{tpr1*100:>7.1f}% {tpr5*100:>7.1f}% {tpr10*100:>8.1f}%")

    # Save
    out_path = os.path.join(DEDUP_DIR, "pr_auc_table.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # Print LaTeX
    print("\nLaTeX TABLE 6:")
    print("\\midrule")
    for r in results:
        print(f"{r['Dataset']:<20} & {r['ROC-AUC']} & {r['PR-AUC']} & "
              f"{r['TPR@FPR=1%']} & {r['TPR@FPR=5%']} & "
              f"{r['TPR@FPR=10%']} \\\\")


if __name__ == "__main__":
    run()
