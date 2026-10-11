"""
run_multiseed.py
================
Trains the full AE + VAE + Deep SVDD ensemble across 3 random seeds
and evaluates on all 5 datasets. Reports mean +/- std for all metrics.

Seeds: 42 (already done), 123, 456

Results saved to:
  artefacts_v2/multiseed_results.csv
  artefacts_v2/multiseed_summary.json

Run from project root:
    python run_multiseed.py

Expected time: 6-8 hours on CPU (3-4 hours per new seed).
Seed 42 results are loaded from existing artefacts if available,
skipping retraining for that seed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import json
import time
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, recall_score

from models.autoencoder   import Autoencoder
from models.vae           import VAE
from models.deep_svdd     import DeepSVDD
from preprocessing.scaler import IQRScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
SEED_DIR     = r"C:/MLProject/zero_day_project/artefacts_multiseed"
os.makedirs(SEED_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 999]

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

DATASETS_CONFIG = [
    {
        "name":      "CIC-IDS2017",
        "path":      r"C:/MLProject/zero_day_project/artefacts_v2/test_cic17.npz",
        "type":      "npz",
        "local_cal": False,
    },
    {
        "name":      "CSE-CIC-IDS2018",
        "path":      r"C:/MLProject/zero_day_project/artefacts_v2/test_cic18.npz",
        "type":      "npz",
        "local_cal": False,
    },
    {
        "name":      "UNSW-NB15",
        "path":      r"C:/MLProject/zero_day_project/artefacts_v2/test_unsw.npz",
        "type":      "npz",
        "local_cal": False,
    },
    {
        "name":      "BoT-IoT",
        "path":      r"C:/MLProject/zero_day_project/artefacts_v2/test_botiot.npz",
        "type":      "npz",
        "local_cal": False,
        "no_benign": True,
    },
    {
        "name":      "CIC-ToN-IoT",
        "path":      r"C:/MLProject/zero_day_project/data/raw/cic_ton_iot/CIC-ToN-IoT.csv",
        "type":      "csv",
        "local_cal": True,
        "max_rows":  200_000,
    },
]


#    Helpers                                                                    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_local(s):
    lo = float(np.percentile(s, 1))
    hi = float(np.percentile(s, 99))
    if hi - lo < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def get_ensemble_scores(ae, vae, svdd, X):
    return (0.4 * norm_local(ae.anomaly_scores(X)) +
            0.4 * norm_local(vae.anomaly_scores(X)) +
            0.2 * norm_local(svdd.anomaly_scores(X)))


def load_npz(path):
    d = np.load(path)
    return np.clip(d["X"], -10, 10).astype(np.float32), d["y"].astype(int)


def load_csv_ton_iot(features, scaler, max_rows=200_000):
    path = r"C:/MLProject/zero_day_project/data/raw/cic_ton_iot/CIC-ToN-IoT.csv"
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


def evaluate_dataset(scores, y, tau):
    y_pred = (scores > tau).astype(int)
    rec = recall_score(y, y_pred, zero_division=0)
    f1  = f1_score(y, y_pred, zero_division=0)
    fpr = float(((y==0)&(y_pred==1)).sum() / max(1,(y==0).sum()))
    try:    auc = roc_auc_score(y, scores)
    except: auc = float("nan")
    return {"auc": auc, "recall": rec, "f1": f1, "fpr": fpr}


def get_threshold(scores, y, local_cal, val_scores, rng):
    benign = scores[y == 0]
    if len(benign) == 0:
        return float(np.percentile(val_scores, 95))
    if local_cal:
        n   = min(10_000, len(benign))
        idx = rng.choice(len(benign), size=n, replace=False)
        return float(np.percentile(benign[idx], 95))
    return float(np.percentile(benign, 95))


#    Training                                                                   

def train_seed(seed, X_train, X_val, features, scaler):
    """Train AE + VAE + Deep SVDD for a given seed. Returns trained models."""

    seed_path = os.path.join(SEED_DIR, f"seed_{seed}")
    os.makedirs(seed_path, exist_ok=True)

    ae_path   = os.path.join(seed_path, "ae_model.pt")
    vae_path  = os.path.join(seed_path, "vae_model.pt")
    svdd_path = os.path.join(seed_path, "svdd_model.pt")

    #    If seed 42 models exist, load them; otherwise train from scratch   
    if seed == 42:
        ae_path42  = os.path.join(ARTEFACT_DIR, "ae_model.pt")
        vae_path42 = os.path.join(ARTEFACT_DIR, "vae_model.pt")
        svdd_path42= os.path.join(ARTEFACT_DIR, "svdd_model.pt")
        if os.path.exists(ae_path42) and os.path.exists(vae_path42)                 and os.path.exists(svdd_path42):
            log.info("Seed 42: loading existing models from artefacts_v2 ...")
            ae   = Autoencoder().load(ae_path42)
            vae  = VAE().load(vae_path42)
            svdd = DeepSVDD().load(svdd_path42)
            return ae, vae, svdd
        else:
            log.info("Seed 42: no existing models found, training from scratch ...")

    #    If already trained, load                                            
    if os.path.exists(ae_path) and os.path.exists(vae_path) \
            and os.path.exists(svdd_path):
        log.info("Seed %d: models already exist, loading ...", seed)
        ae   = Autoencoder().load(ae_path)
        vae  = VAE().load(vae_path)
        svdd = DeepSVDD().load(svdd_path)
        return ae, vae, svdd

    #    Train from scratch                                                 
    input_dim = X_train.shape[1]
    log.info("  input_dim=%d", input_dim)

    try:
        set_seed(seed)
        log.info("Seed %d: training AE ...", seed)
        t0 = time.time()
        ae = Autoencoder(input_dim=input_dim)
        ae.fit(X_train, X_val)
        ae.fit_latent_stats(X_train)
        ae.fit_score_stats(X_train)
        ae.save(ae_path)
        log.info("  AE done in %.1f min", (time.time()-t0)/60)
    except Exception as e:
        log.error("  AE training failed: %s", e)
        raise

    try:
        set_seed(seed)
        log.info("Seed %d: training VAE ...", seed)
        t0 = time.time()
        vae = VAE(input_dim=input_dim)
        vae.fit(X_train, X_val)
        vae.fit_latent_stats(X_train)
        vae.fit_score_stats(X_train)
        vae.save(vae_path)
        log.info("  VAE done in %.1f min", (time.time()-t0)/60)
    except Exception as e:
        log.error("  VAE training failed: %s", e)
        raise

    try:
        set_seed(seed)
        log.info("Seed %d: training Deep SVDD ...", seed)
        t0 = time.time()
        svdd = DeepSVDD(input_dim=input_dim)
        svdd.fit(X_train, X_val)
        svdd.fit_score_stats(X_train)
        svdd.save(svdd_path)
        log.info("  SVDD done in %.1f min", (time.time()-t0)/60)
    except Exception as e:
        log.error("  SVDD training failed: %s", e)
        raise

    return ae, vae, svdd


#    Main                                                                       

def run():
    #    Load shared data                                                   
    log.info("Loading shared data ...")
    X_train = np.clip(
        np.load(os.path.join(ARTEFACT_DIR, "train.npz"))["X_train"],
        -10, 10).astype(np.float32)
    X_val   = np.clip(
        np.load(os.path.join(ARTEFACT_DIR, "val.npz"))["X_val"],
        -10, 10).astype(np.float32)

    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    features  = pd.read_csv(feat_path, header=None)[0].tolist()
    features  = [f for f in features
                 if not str(f).strip().lstrip("-").isdigit()]

    scaler = IQRScaler()
    scaler.load(os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"))

    log.info("  X_train: %s  X_val: %s", X_train.shape, X_val.shape)

    #    Load test datasets once                                            
    log.info("Loading test datasets ...")
    test_data = {}
    for cfg in DATASETS_CONFIG:
        name = cfg["name"]
        if cfg["type"] == "npz":
            X, y = load_npz(cfg["path"])
        else:
            X, y = load_csv_ton_iot(features, scaler,
                                    cfg.get("max_rows", 200_000))
        test_data[name] = (X, y, cfg["local_cal"],
                           cfg.get("no_benign", False))
        log.info("  %s: %d flows | benign=%d | attack=%d",
                 name, len(y), int((y==0).sum()), int((y==1).sum()))

    #    Run each seed                                                      
    all_results = []

    for seed in SEEDS:
        log.info("")
        log.info("=" * 60)
        log.info("SEED %d", seed)
        log.info("=" * 60)
        rng = np.random.default_rng(seed)

        ae, vae, svdd = train_seed(seed, X_train, X_val, features, scaler)

        # Val scores for BoT-IoT threshold
        val_scores = get_ensemble_scores(ae, vae, svdd, X_val[:30_000])
        tau_val    = float(np.percentile(val_scores, 95))

        for cfg in DATASETS_CONFIG:
            name = cfg["name"]
            X, y, local_cal, no_benign = test_data[name]

            log.info("  Evaluating %s ...", name)
            scores = get_ensemble_scores(ae, vae, svdd, X)

            if no_benign:
                tau = tau_val
            else:
                tau = get_threshold(scores, y, local_cal, val_scores, rng)

            r = evaluate_dataset(scores, y, tau)
            r["seed"]    = seed
            r["dataset"] = name
            all_results.append(r)

            auc_s = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else "—"
            log.info("    AUC=%s  Recall=%.4f  F1=%.4f  FPR=%.4f",
                     auc_s, r["recall"], r["f1"], r["fpr"])

        gc.collect()

    #    Compute mean ± std                                                 
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("MULTI-SEED RESULTS — Mean ± Std (3 seeds: 42, 123, 456)")
    print("=" * 80)

    summary = {}
    ds_order = ["CIC-IDS2017", "CSE-CIC-IDS2018", "UNSW-NB15",
                "BoT-IoT", "CIC-ToN-IoT"]

    print(f"\n{'Dataset':<22} {'AUC (mean±std)':>18} "
          f"{'Recall (mean±std)':>20} {'F1 (mean±std)':>18} "
          f"{'FPR (mean±std)':>18}")
    print("-" * 100)

    for ds in ds_order:
        sub = df[df["dataset"] == ds]
        row = {}
        for metric in ["auc", "recall", "f1", "fpr"]:
            vals = sub[metric].dropna().values
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"]  = float(np.std(vals))
        summary[ds] = row

        auc_s    = f"{row['auc_mean']:.4f}±{row['auc_std']:.4f}" if not np.isnan(row['auc_mean']) else "—"
        rec_s    = f"{row['recall_mean']:.4f}±{row['recall_std']:.4f}"
        f1_s     = f"{row['f1_mean']:.4f}±{row['f1_std']:.4f}"
        fpr_s    = f"{row['fpr_mean']:.4f}±{row['fpr_std']:.4f}"
        print(f"{ds:<22} {auc_s:>18} {rec_s:>20} {f1_s:>18} {fpr_s:>18}")

    print("-" * 100)

    # Per-seed breakdown
    print("\nPer-seed AUC breakdown:")
    print(f"{'Dataset':<22}" +
          "".join(f"{'Seed '+str(s):>12}" for s in SEEDS))
    print("-" * 60)
    for ds in ds_order:
        row = f"{ds:<22}"
        for s in SEEDS:
            val = df[(df["dataset"]==ds) & (df["seed"]==s)]["auc"].values
            if len(val) > 0 and not np.isnan(val[0]):
                row += f"{val[0]:>12.4f}"
            else:
                row += f"{'—':>12}"
        print(row)

    #    Save                                                               
    csv_path  = os.path.join(ARTEFACT_DIR, "multiseed_results.csv")
    json_path = os.path.join(ARTEFACT_DIR, "multiseed_summary.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Results saved -> %s", csv_path)
    log.info("Summary saved -> %s", json_path)
    log.info("Done.")


if __name__ == "__main__":
    run()
