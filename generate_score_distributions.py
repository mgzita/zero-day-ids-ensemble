"""
Generate anomaly score distribution plots for all 5 datasets.

Run from project root:
    python generate_score_distributions.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

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
    "fwd_pkt_len_mean":  "fwd_packet_length_mean",
    "bwd_pkt_len_mean":  "bwd_packet_length_mean",
    "fwd_pkt_len_std":   "fwd_packet_length_std",
    "bwd_pkt_len_std":   "bwd_packet_length_std",
    "flow_byts_s":       "flow_bytes_s",
    "fwd_pkts_s":        "fwd_packets_s",
    "bwd_pkts_s":        "bwd_packets_s",
    "fwd_byts_b_avg":    "fwd_avg_bytes_bulk",
    "bwd_byts_b_avg":    "bwd_avg_bytes_bulk",
    "fwd_pkts_b_avg":    "fwd_avg_packets_bulk",
    "bwd_pkts_b_avg":    "bwd_avg_packets_bulk",
    "init_fwd_win_byts": "init_win_bytes_forward",
    "init_bwd_win_byts": "init_win_bytes_backward",
    "subflow_fwd_byts":  "subflow_fwd_bytes",
    "dst_port":          "destination_port",
}

DATASETS = [
    {"name": "CIC-IDS2017", "label": "CIC-IDS2017\n(same-domain)",
     "path": r"C:/MLProject/zero_day_project/artefacts_v2/test_cic17.npz",
     "type": "npz", "color_benign": "#2471A3", "color_attack": "#C0392B", "has_benign": True},
    {"name": "CSE-CIC-IDS2018", "label": "CSE-CIC-IDS2018\n(same-domain)",
     "path": r"C:/MLProject/zero_day_project/artefacts_v2/test_cic18.npz",
     "type": "npz", "color_benign": "#2471A3", "color_attack": "#C0392B", "has_benign": True},
    {"name": "UNSW-NB15", "label": "UNSW-NB15\n(same-domain)",
     "path": r"C:/MLProject/zero_day_project/artefacts_v2/test_unsw.npz",
     "type": "npz", "color_benign": "#2471A3", "color_attack": "#C0392B", "has_benign": True},
    {"name": "BoT-IoT", "label": "BoT-IoT\n(unseen zero-day)",
     "path": r"C:/MLProject/zero_day_project/artefacts_v2/test_botiot.npz",
     "type": "npz", "color_benign": "#2471A3", "color_attack": "#C0392B", "has_benign": False},
    {"name": "CIC-ToN-IoT", "label": "CIC-ToN-IoT\n(unseen zero-day)",
     "path": r"C:/MLProject/zero_day_project/data/raw/cic_ton_iot/CIC-ToN-IoT.csv",
     "type": "csv", "color_benign": "#2471A3", "color_attack": "#C0392B", "has_benign": True, "max_rows": 100000},
]

RNG = np.random.default_rng(42)


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


def load_csv(path, features, scaler, max_rows=100000):
    frames = []
    total = 0
    for chunk in pd.read_csv(path, chunksize=100000, low_memory=False):
        chunk.columns = (chunk.columns.str.strip().str.lower()
                         .str.replace(r"\s+", "_", regex=True)
                         .str.replace(r"[^a-z0-9_]", "_", regex=True)
                         .str.strip("_"))
        chunk.rename(columns=TON_IOT_RENAME, inplace=True)
        chunk["label"] = (chunk["label"] != 0).astype(int) if "label" in chunk.columns else 1
        for f in features:
            if f not in chunk.columns:
                chunk[f] = 0.0
        for col in chunk.select_dtypes(exclude=[np.number]).columns:
            if col != "label":
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0.0, inplace=True)
        frames.append(chunk)
        total += len(chunk)
        if total >= max_rows:
            break
        gc.collect()
    df = pd.concat(frames, ignore_index=True).iloc[:max_rows]
    X = np.clip(scaler.transform(df[features].values.astype(np.float32)), -10, 10)
    y = df["label"].values.astype(int)
    return X, y


def plot_kde(ax, scores, color, label, alpha=0.55, lw=2):
    if len(scores) < 10:
        return
    if len(scores) > 50000:
        idx = RNG.choice(len(scores), size=50000, replace=False)
        scores = scores[idx]
    try:
        kde = gaussian_kde(scores, bw_method=0.08)
        xs = np.linspace(0, 1, 400)
        dens = kde(xs)
        ax.fill_between(xs, dens, alpha=alpha, color=color, linewidth=0)
        ax.plot(xs, dens, color=color, lw=lw, label=label)
    except Exception as e:
        log.warning("KDE failed for %s: %s", label, e)


def run():
    log.info("Loading models ...")
    ae = Autoencoder().load(os.path.join(ARTEFACT_DIR, "ae_model.pt"))
    vae = VAE().load(os.path.join(ARTEFACT_DIR, "vae_model.pt"))
    svdd = DeepSVDD().load(os.path.join(ARTEFACT_DIR, "svdd_model.pt"))

    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    features = pd.read_csv(feat_path, header=None)[0].tolist()
    features = [f for f in features if not str(f).strip().lstrip("-").isdigit()]

    scaler = IQRScaler()
    scaler.load(os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"))

    val_npz = np.load(os.path.join(ARTEFACT_DIR, "val.npz"))
    X_val = np.clip(val_npz["X_val"][:30000], -10, 10).astype(np.float32)
    val_scores = get_scores(ae, vae, svdd, X_val)
    tau_global = float(np.percentile(val_scores, 95))
    log.info("Global threshold = %.4f", tau_global)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False)
    axes_flat = axes.flatten()
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.45, wspace=0.35, top=0.93, bottom=0.08, left=0.06, right=0.97)

    #fig.suptitle("Ensemble Anomaly Score Distributions -- Benign vs Attack",
               # fontsize=13, fontweight="bold", y=0.98, color="#1A1A1A")

    for ax, ds in zip(axes_flat[:5], DATASETS):
        log.info("Processing %s ...", ds["name"])

        if ds["type"] == "npz":
            X, y = load_npz(ds["path"])
        else:
            X, y = load_csv(ds["path"], features, scaler, ds.get("max_rows", 100000))

        scores = get_scores(ae, vae, svdd, X)

        if (y == 0).sum() == 0:
            tau = tau_global
        else:
            benign_s = scores[y == 0]
            n = min(10000, len(benign_s))
            idx = RNG.choice(len(benign_s), size=n, replace=False)
            tau = float(np.percentile(benign_s[idx], 95))

        benign_scores = scores[y == 0]
        attack_scores = scores[y == 1]

        if len(benign_scores) > 0:
            plot_kde(ax, benign_scores, color=ds["color_benign"],
                     label=f"Benign (n={len(benign_scores):,})")

        if len(attack_scores) > 0:
            plot_kde(ax, attack_scores, color=ds["color_attack"],
                     label=f"Attack (n={len(attack_scores):,})")

        ax.axvline(x=tau, color="#333333", lw=1.5, linestyle="--", alpha=0.8)
        ax.text(tau + 0.02,
                ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 5,
                f"threshold = {tau:.2f}",
                fontsize=8, color="#333333", va="top")

        if len(benign_scores) > 0 and len(attack_scores) > 0:
            ax.text(0.05, 0.97,
                    f"Benign mean: {benign_scores.mean():.2f}\nAttack mean: {attack_scores.mean():.2f}",
                    transform=ax.transAxes, fontsize=7.5, va="top", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#CCCCCC"))

        ax.set_xlim([0, 1])
        ax.set_xlabel("Ensemble anomaly score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(ds["label"], fontsize=10, fontweight="bold", color="#1A1A1A", pad=6)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85, edgecolor="#CCCCCC")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_facecolor("#FFFFFF")
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")
            spine.set_linewidth(0.8)

        gc.collect()

    # ── 6th panel: legend, force-aligned to panel 3's x-position ──────────
    ax6 = axes_flat[5]
    ax3 = axes_flat[2]
    ax5 = axes_flat[4]

    ax6.set_visible(True)
    ax6.set_facecolor("white")
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    for spine in ax6.spines.values():
        spine.set_visible(False)
    ax6.set_xticks([])
    ax6.set_yticks([])

    # Force panel 6 to share the exact same horizontal extent as panel 3
    # (panel 3 sits directly above panel 6 in the 2x3 grid)
    pos3 = ax3.get_position()
    pos6 = ax6.get_position()
    ax6.set_position([pos3.x0, pos6.y0, pos3.width, pos6.height])

    legend_patches = [
        Patch(facecolor="#2471A3", alpha=0.6, label="Benign traffic"),
        Patch(facecolor="#C0392B", alpha=0.6, label="Attack traffic"),
        plt.Line2D([0], [0], color="#333333", lw=1.5, linestyle="--", label="p95 threshold"),
    ]
    ax6.legend(handles=legend_patches, loc="center",
               fontsize=13, framealpha=0.95, edgecolor="#AAAAAA",
               title="Legend", title_fontsize=12)

    out_path = os.path.join(OUT_DIR, "score_distributions_2x3.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    log.info("Saved -> %s", out_path)
    plt.close()
    log.info("Done.")


if __name__ == "__main__":
    run()
