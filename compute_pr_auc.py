"""
compute_pr_auc.py
=================
Computes ROC-AUC, PR-AUC and TPR at fixed FPR for all datasets.
Uses same feature mapping as run_multiseed_v3.py for CIC-ToN-IoT.

Run from project root:
    python compute_pr_auc.py
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve)
from models.autoencoder import Autoencoder
from models.vae import VAE
from models.deep_svdd import DeepSVDD
from preprocessing.scaler import IQRScaler

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
TON_PATH     = r"C:/MLProject/zero_day_project/data/raw/cic_ton_iot/CIC-ToN-IoT.csv"
MAX_ROWS     = 200000

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

def tpr_at_fpr(y, scores, fpr_target):
    fpr_arr, tpr_arr, _ = roc_curve(y, scores)
    idx = np.argmin(np.abs(fpr_arr - fpr_target))
    return tpr_arr[idx]

def run():
    print("Loading models (seed 42)...")
    ae   = Autoencoder().load(os.path.join(ARTEFACT_DIR, "ae_model.pt"))
    vae  = VAE().load(os.path.join(ARTEFACT_DIR, "vae_model.pt"))
    svdd = DeepSVDD().load(os.path.join(ARTEFACT_DIR, "svdd_model.pt"))

    scaler = IQRScaler()
    scaler.load(os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"))

    feats = pd.read_csv(os.path.join(ARTEFACT_DIR, "selected_features.csv"),
                        header=None)[0].tolist()
    feats = [f for f in feats if f != '0']

    # NPZ datasets
    npz_datasets = [
        ("CIC-IDS2017",     "test_cic17.npz"),
        ("CSE-CIC-IDS2018", "test_cic18.npz"),
        ("UNSW-NB15",       "test_unsw.npz"),
    ]

    print()
    print(f"{'Dataset':<20} {'ROC-AUC':>10} {'PR-AUC':>10} "
          f"{'TPR@1%':>10} {'TPR@5%':>10} {'TPR@10%':>10}")
    print("-"*72)

    for name, fname in npz_datasets:
        path = os.path.join(ARTEFACT_DIR, fname)
        d = np.load(path)
        X = np.clip(d["X"], -10, 10).astype(np.float32)
        y = d["y"].astype(int)

        scores = get_scores(ae, vae, svdd, X)
        roc = roc_auc_score(y, scores)
        pr  = average_precision_score(y, scores)
        t1  = tpr_at_fpr(y, scores, 0.01)
        t5  = tpr_at_fpr(y, scores, 0.05)
        t10 = tpr_at_fpr(y, scores, 0.10)
        print(f"{name:<20} {roc:>10.4f} {pr:>10.4f} "
              f"{t1:>9.1%} {t5:>9.1%} {t10:>9.1%}")

    # CIC-ToN-IoT from CSV with rename + local calibration
    print(f"{'CIC-ToN-IoT':<20}", end=" ", flush=True)
    frames = []
    total = 0
    for chunk in pd.read_csv(TON_PATH, chunksize=100000, low_memory=False):
        chunk.columns = (chunk.columns.str.strip().str.lower()
                         .str.replace(r'\s+', '_', regex=True)
                         .str.replace(r'[^a-z0-9_]', '_', regex=True)
                         .str.strip('_'))
        chunk.rename(columns=TON_IOT_RENAME, inplace=True)
        chunk["label"] = (chunk["label"] != 0).astype(int) \
                         if "label" in chunk.columns else 1
        for f in feats:
            if f not in chunk.columns:
                chunk[f] = 0.0
        for col in chunk.select_dtypes(exclude=[np.number]).columns:
            if col != "label":
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0.0, inplace=True)
        frames.append(chunk)
        total += len(chunk)
        if total >= MAX_ROWS:
            break

    df = pd.concat(frames, ignore_index=True).iloc[:MAX_ROWS]
    X  = np.clip(scaler.transform(df[feats].values.astype(np.float32)), -10, 10)
    y  = df["label"].values.astype(int)

    scores = get_scores(ae, vae, svdd, X)

    # Local calibration: use benign scores for threshold
    benign_scores = scores[y == 0]
    tau_local = float(np.percentile(benign_scores, 95))

    roc = roc_auc_score(y, scores)
    pr  = average_precision_score(y, scores)
    t1  = tpr_at_fpr(y, scores, 0.01)
    t5  = tpr_at_fpr(y, scores, 0.05)
    t10 = tpr_at_fpr(y, scores, 0.10)
    print(f"{roc:>10.4f} {pr:>10.4f} "
          f"{t1:>9.1%} {t5:>9.1%} {t10:>9.1%}")
    print(f"  (local tau={tau_local:.4f}, "
          f"recall@tau={( scores[y==1] > tau_local ).mean():.4f})")

    print("\nDone.")

if __name__ == "__main__":
    run()
