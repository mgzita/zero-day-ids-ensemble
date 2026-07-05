"""
generate_confusion_matrices.py
===============================
Generates confusion matrices for all 5 datasets in a 2x3 grid.
Row 1: CIC-IDS2017 | CSE-CIC-IDS2018 | UNSW-NB15
Row 2: BoT-IoT     | CIC-ToN-IoT     | [legend]

Output:
  artefacts_v2/figures/confusion_matrices_v5.png

Run from project root:
    python generate_confusion_matrices.py
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
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

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

DATASETS = [
    {
        "name":     "CIC-IDS2017",
        "subtitle": "Training-domain  |  AUC = 0.895",
        "path":     r"C:/MLProject/zero_day_project/artefacts_v2/test_cic17.npz",
        "type":     "npz",
        "color":    "#1A5276",
        "local_cal": False,
    },
    {
        "name":     "CSE-CIC-IDS2018",
        "subtitle": "Training-domain  |  AUC = 0.942",
        "path":     r"C:/MLProject/zero_day_project/artefacts_v2/test_cic18.npz",
        "type":     "npz",
        "color":    "#1A5276",
        "local_cal": False,
    },
    {
        "name":     "UNSW-NB15",
        "subtitle": "Training-domain  |  AUC = 0.879",
        "path":     r"C:/MLProject/zero_day_project/artefacts_v2/test_unsw.npz",
        "type":     "npz",
        "color":    "#1A5276",
        "local_cal": False,
    },
    {
        "name":     "BoT-IoT",
        "subtitle": "Unseen zero-day  |  DR = 91.2%",
        "path":     r"C:/MLProject/zero_day_project/artefacts_v2/test_botiot.npz",
        "type":     "npz",
        "color":    "#1A5276",
        "local_cal": False,
        "no_benign": True,
    },
    {
        "name":     "CIC-ToN-IoT",
        "subtitle": "Unseen zero-day  |  AUC = 0.953",
        "path":     r"C:/MLProject/zero_day_project/data/raw/cic_ton_iot/CIC-ToN-IoT.csv",
        "type":     "csv",
        "color":    "#1A5276",
        "local_cal": True,
        "max_rows":  200_000,
    },
]

RNG = np.random.default_rng(42)
CELL_LABELS = [["TN", "FP"], ["FN", "TP"]]


def norm_local(s):
    lo = float(np.percentile(s, 1))
    hi = float(np.percentile(s, 99))
    if hi - lo < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def get_scores(ae, vae, svdd, X):
    return (0.4 * norm_local(ae.anomaly_scores(X)) +
            0.4 * norm_local(vae.anomaly_scores(X)) +
            0.2 * norm_local(svdd.anomaly_scores(X)))


def load_npz(path):
    d = np.load(path)
    return np.clip(d["X"], -10, 10).astype(np.float32), d["y"].astype(int)


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


def draw_confusion(ax, cm, ds, tau):
    """Draw a single confusion matrix heatmap."""
    color  = ds["color"]
    no_ben = ds.get("no_benign", False)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#FFFFFF", color], N=256)

    ax.imshow(cm_norm, interpolation='nearest', cmap=cmap,
              vmin=0, vmax=1, aspect='auto')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nBenign", "Predicted\nAttack"],
                       fontsize=10, fontweight='bold')
    ax.set_yticklabels(["Actual\nBenign", "Actual\nAttack"],
                       fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', length=0, pad=6)

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            bg    = cm_norm[i, j]
            txt_color = "white" if bg > 0.45 else "#1a1a1a"
            label = CELL_LABELS[i][j]

            if no_ben and i == 0:
                ax.text(j, i, "N/A\n(no benign\ntest data)",
                        ha='center', va='center',
                        color="#888888", fontsize=8, style='italic',
                        fontweight='bold')
                continue

            ax.text(j, i,
                    f"{label}\n{count:,}\n({pct:.1f}%)",
                    ha='center', va='center',
                    color=txt_color,
                    fontsize=10, fontweight='bold',
                    linespacing=1.5)

    ax.set_title(f"{ds['name']}\n{ds['subtitle']}",
                 fontsize=11, fontweight='bold', pad=10, color='#1A1A1A')

    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which='minor', color='white', linewidth=2.5)
    ax.tick_params(which='minor', length=0)

    for spine in ax.spines.values():
        spine.set_edgecolor('#AAAAAA')
        spine.set_linewidth(2)

    # Threshold annotation
    ax.text(1.0, -0.18, f"tau = {tau:.3f}",
            transform=ax.transAxes,
            fontsize=9, color='#000000', ha='right',
            style='italic')


def run():
    # â”€â”€ Load models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # Val set threshold for BoT-IoT
    val_npz    = np.load(os.path.join(ARTEFACT_DIR, "val.npz"))
    X_val      = np.clip(val_npz["X_val"][:30_000], -10, 10).astype(np.float32)
    val_scores = get_scores(ae, vae, svdd, X_val)
    tau_global = float(np.percentile(val_scores, 95))

    # â”€â”€ Figure 2x3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(hspace=0.50, wspace=0.35,
                        top=0.90, bottom=0.08, left=0.06, right=0.72)

    # suptitle removed

    for ax, ds in zip(axes_flat[:5], DATASETS):
        log.info("Processing %s ...", ds["name"])

        if ds["type"] == "npz":
            X, y = load_npz(ds["path"])
        else:
            X, y = load_csv(ds["path"], features, scaler,
                            ds.get("max_rows", 200_000))

        scores = get_scores(ae, vae, svdd, X)

        # Threshold
        if ds.get("no_benign", False):
            tau = tau_global
        elif ds.get("local_cal", False):
            benign_s = scores[y == 0]
            n   = min(10_000, len(benign_s))
            idx = RNG.choice(len(benign_s), size=n, replace=False)
            tau = float(np.percentile(benign_s[idx], 95))
        else:
            tau = float(np.percentile(scores[y == 0], 95))

        y_pred = (scores > tau).astype(int)

        # Build confusion matrix
        TP = int(((y == 1) & (y_pred == 1)).sum())
        TN = int(((y == 0) & (y_pred == 0)).sum())
        FP = int(((y == 0) & (y_pred == 1)).sum())
        FN = int(((y == 1) & (y_pred == 0)).sum())
        cm = np.array([[TN, FP], [FN, TP]])

        log.info("  Ï„=%.4f  TP=%d  TN=%d  FP=%d  FN=%d",
                 tau, TP, TN, FP, FN)

        draw_confusion(axes_flat[list(DATASETS).index(ds)], cm, ds, tau)
        gc.collect()

    # â”€â”€ 6th panel: text legend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax6 = axes_flat[5]
    ax6.set_visible(True)
    ax6.set_facecolor('#FAFAFA')
    for spine in ax6.spines.values():
        spine.set_visible(False)
    ax6.set_xticks([])
    ax6.set_yticks([])

    legend_text = (
        "TP = Attack correctly\n"
        "     detected\n\n"
        "TN = Benign correctly\n"
        "     classified\n\n"
        "FP = Benign flagged\n"
        "     as attack\n\n"
        "FN = Attack missed\n"
        "     (critical)\n\n"
        "tau = p95 threshold\n\n"
        "N/A = no benign\n"
        "      traffic (BoT-IoT)\n\n"
        "Row 2: local benign\n"
        "calibration (ToN-IoT)"
    )
    ax6.text(0.5, 0.5, legend_text,
             ha='center', va='center',
             fontsize=9.5, color='#000000',
             fontfamily='monospace',
             fontweight='bold',
             transform=ax6.transAxes,
             bbox=dict(boxstyle='round,pad=0.8',
                       facecolor='#F8F8F8',
                       edgecolor='#AAAAAA',
                       linewidth=1.0),
             linespacing=1.8)

    out_path = os.path.join(OUT_DIR, "confusion_matrices_v5.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor='#FAFAFA')
    log.info("Saved -> %s", out_path)
    plt.close()
    log.info("Done.")


if __name__ == "__main__":
    run()

