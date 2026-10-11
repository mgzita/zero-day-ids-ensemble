"""
generate_roc_curves.py
======================
Generates ROC curves and Precision-Recall curves for the v2 ensemble
across all 5 evaluated datasets.

Output:
  artefacts_v2/figures/roc_curves.png
  artefacts_v2/figures/pr_curves.png

Run from project root:
    python generate_roc_curves.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score)

from models.autoencoder   import Autoencoder
from models.vae           import VAE
from models.deep_svdd     import DeepSVDD
from preprocessing.scaler import IQRScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
OUT_DIR      = os.path.join(ARTEFACT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

TON_IOT_RENAME = {
    'fwd_pkt_len_mean':  'fwd_packet_length_mean',
    'bwd_pkt_len_mean':  'bwd_packet_length_mean',
    'fwd_pkt_len_std':   'fwd_packet_length_std',
    'bwd_pkt_len_std':   'bwd_packet_length_std',
    'flow_byts_s':       'flow_bytes_s',
    'fwd_pkts_s':        'fwd_packets_s',
    'bwd_pkts_s':        'bwd_packets_s',
    'fwd_byts_b_avg':    'fwd_avg_bytes_bulk',
    'bwd_byts_b_avg':    'bwd_avg_bytes_bulk',
    'fwd_pkts_b_avg':    'fwd_avg_packets_bulk',
    'bwd_pkts_b_avg':    'bwd_avg_packets_bulk',
    'init_fwd_win_byts': 'init_win_bytes_forward',
    'init_bwd_win_byts': 'init_win_bytes_backward',
    'subflow_fwd_byts':  'subflow_fwd_bytes',
    'dst_port':          'destination_port',
}

# Dataset configs — colours and labels for plots
DATASETS = [
    {
        "name":    "CIC-IDS2017",
        "label":   "CIC-IDS2017 (same-domain)",
        "path":    r"C:/MLProject/zero_day_project/artefacts_v2/test_cic17.npz",
        "type":    "npz",
        "color":   "#2471A3",
        "ls":      "-",
    },
    {
        "name":    "CSE-CIC-IDS2018",
        "label":   "CSE-CIC-IDS2018 (same-domain)",
        "path":    r"C:/MLProject/zero_day_project/artefacts_v2/test_cic18.npz",
        "type":    "npz",
        "color":   "#27AE60",
        "ls":      "--",
    },
    {
        "name":    "UNSW-NB15",
        "label":   "UNSW-NB15 (same-domain)",
        "path":    r"C:/MLProject/zero_day_project/artefacts_v2/test_unsw.npz",
        "type":    "npz",
        "color":   "#8E44AD",
        "ls":      "-.",
    },
    {
        "name":    "CIC-ToN-IoT",
        "label":   "CIC-ToN-IoT (unseen, local cal.)",
        "path":    r"C:/MLProject/zero_day_project/data/raw/cic_ton_iot/CIC-ToN-IoT.csv",
        "type":    "csv",
        "color":   "#C0392B",
        "ls":      ":",
        "max_rows": 200_000,
    },
]

RNG = np.random.default_rng(42)


def norm_local(s):
    lo = float(np.percentile(s, 1))
    hi = float(np.percentile(s, 99))
    if hi - lo < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_npz(path, scaler=None):
    d = np.load(path)
    X = np.clip(d["X"], -10, 10).astype(np.float32)
    y = d["y"].astype(int)
    return X, y


def load_csv(path, features, scaler, max_rows=200_000):
    frames = []
    total  = 0
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        chunk.columns = (chunk.columns.str.strip().str.lower()
                         .str.replace(r'\s+', '_', regex=True)
                         .str.replace(r'[^a-z0-9_]', '_', regex=True)
                         .str.strip('_'))
        chunk.rename(columns=TON_IOT_RENAME, inplace=True)
        chunk['label'] = (chunk['label'] != 0).astype(int) \
            if 'label' in chunk.columns else 1
        for f in features:
            if f not in chunk.columns:
                chunk[f] = 0.0
        for col in chunk.select_dtypes(exclude=[np.number]).columns:
            if col != 'label':
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0.0, inplace=True)
        frames.append(chunk)
        total += len(chunk)
        if total >= max_rows:
            break
        gc.collect()
    df = pd.concat(frames, ignore_index=True).iloc[:max_rows]
    X  = np.clip(scaler.transform(
        df[features].values.astype(np.float32)), -10, 10)
    y  = df['label'].values.astype(int)
    return X, y


def get_scores(ae, vae, svdd, X):
    z_ae   = norm_local(ae.anomaly_scores(X))
    z_vae  = norm_local(vae.anomaly_scores(X))
    z_svdd = norm_local(svdd.anomaly_scores(X))
    return 0.4 * z_ae + 0.4 * z_vae + 0.2 * z_svdd


def run():
    # ── Load models ───────────────────────────────────────────────────────
    log.info("Loading models ...")
    ae   = Autoencoder().load(os.path.join(ARTEFACT_DIR, "ae_model.pt"))
    vae  = VAE().load(os.path.join(ARTEFACT_DIR, "vae_model.pt"))
    svdd = DeepSVDD().load(os.path.join(ARTEFACT_DIR, "svdd_model.pt"))

    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    features  = pd.read_csv(feat_path, header=None)[0].tolist()
    features  = [f for f in features
                 if not str(f).strip().lstrip("-").isdigit()]

    scaler = IQRScaler()
    scaler.load(os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"))

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('white')

    ax_roc = axes[0]
    ax_pr  = axes[1]

    # Random baseline
    ax_roc.plot([0,1],[0,1], color='#AAAAAA', lw=1,
                linestyle='--', label='Random baseline (AUC = 0.50)')

    results = {}

    for ds in DATASETS:
        log.info("Processing %s ...", ds["name"])

        if ds["type"] == "npz":
            X, y = load_npz(ds["path"])
        else:
            X, y = load_csv(ds["path"], features, scaler,
                            ds.get("max_rows", 200_000))

        if (y == 1).sum() == 0:
            log.warning("  No attack flows — skipping %s", ds["name"])
            continue

        scores = get_scores(ae, vae, svdd, X)

        # ROC
        fpr_arr, tpr_arr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr_arr, tpr_arr)

        # PR
        prec_arr, rec_arr, _ = precision_recall_curve(y, scores)
        pr_auc = average_precision_score(y, scores)

        results[ds["name"]] = {
            "roc_auc": roc_auc,
            "pr_auc":  pr_auc,
        }

        # ── Plot ROC ──────────────────────────────────────────────────────
        ax_roc.plot(fpr_arr, tpr_arr,
                    color=ds["color"], lw=2, linestyle=ds["ls"],
                    label=f"{ds['label']}\n(AUC = {roc_auc:.3f})")

        # ── Plot PR ───────────────────────────────────────────────────────
        ax_pr.plot(rec_arr, prec_arr,
                   color=ds["color"], lw=2, linestyle=ds["ls"],
                   label=f"{ds['label']}\n(AP = {pr_auc:.3f})")

        # TPR at fixed FPR operating points
        for target_fpr in [0.01, 0.05, 0.10]:
            idx = np.searchsorted(fpr_arr, target_fpr)
            if idx < len(tpr_arr):
                tpr_at = tpr_arr[idx]
                results[ds["name"]][f"tpr_at_fpr{int(target_fpr*100)}"] = \
                    round(tpr_at, 4)

        log.info("  AUC=%.4f  PR-AUC=%.4f", roc_auc, pr_auc)
        gc.collect()

    # ── Format ROC axes ───────────────────────────────────────────────────
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate (Detection Rate)", fontsize=12)
    ax_roc.set_title("ROC Curves -- AE-VAE-Deep SVDD Ensemble",
                     fontsize=13, fontweight='bold', pad=12)
    ax_roc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9, framealpha=0.9)
    ax_roc.grid(True, alpha=0.3, linewidth=0.5)
    ax_roc.set_facecolor('#FFFFFF')

    # Mark p95 operating point (FPR ≈ 0.05) with vertical line
    ax_roc.axvline(x=0.05, color='#888888', lw=1, linestyle=':',
                   alpha=0.7)
    ax_roc.text(0.052, 0.05, 'p95 threshold\n(FPR ≈ 5%)',
                fontsize=8, color='#666666', va='bottom')

    # Mark FPR operating points
    for fpr_mark, label in [(0.01,'1%'), (0.05,'5%'), (0.10,'10%')]:
        ax_roc.axvline(x=fpr_mark, color='#CCCCCC', lw=0.8,
                       linestyle='--', alpha=0.5)

    # ── Format PR axes ────────────────────────────────────────────────────
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall (Detection Rate)", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)
    ax_pr.set_title("Precision-Recall Curves -- AE-VAE-Deep SVDD Ensemble",
                    fontsize=13, fontweight='bold', pad=12)
    ax_pr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9, framealpha=0.9)
    ax_pr.grid(True, alpha=0.3, linewidth=0.5)
    ax_pr.set_facecolor('#FFFFFF')

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.22)
    out_path = os.path.join(OUT_DIR, "roc_pr_curves.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    log.info("Saved -> %s", out_path)
    plt.close()

    # ── TPR at fixed FPR summary ──────────────────────────────────────────
    print("")
    print("=" * 65)
    print("TPR at Fixed FPR Operating Points")
    print("=" * 65)
    print(f"{'Dataset':<22} {'AUC':>7} {'PR-AUC':>8} "
          f"{'TPR@1%':>8} {'TPR@5%':>8} {'TPR@10%':>9}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<22} {r['roc_auc']:>7.4f} {r['pr_auc']:>8.4f} "
              f"{r.get('tpr_at_fpr1', float('nan')):>8.4f} "
              f"{r.get('tpr_at_fpr5', float('nan')):>8.4f} "
              f"{r.get('tpr_at_fpr10', float('nan')):>9.4f}")
    print("=" * 65)

    # Save TPR summary
    import json
    json_path = os.path.join(ARTEFACT_DIR, "roc_pr_summary.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info("Summary saved -> %s", json_path)
    log.info("Done.")


if __name__ == "__main__":
    run()
