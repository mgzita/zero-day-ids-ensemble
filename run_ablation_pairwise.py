"""
run_ablation_pairwise.py
========================
Runs pairwise ablation across all 7 ensemble combinations:

  Single models:   AE only, VAE only, SVDD only
  Pairs:           AE+VAE, AE+SVDD, VAE+SVDD
  Full ensemble:   AE+VAE+SVDD (reference)

Evaluated on: CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15, BoT-IoT, CIC-ToN-IoT

Results saved to:
  artefacts_v2/ablation_pairwise_results.csv
  artefacts_v2/ablation_pairwise_results.json

Run from project root:
    python run_ablation_pairwise.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import logging
import numpy as np
import pandas as pd
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
DATA_DIR     = r"C:/MLProject/zero_day_project/data/raw"

#    Dataset configs                                                           
DATASETS = {
    "CIC-IDS2017": {
        "path":    r"C:/MLProject/zero_day_project/artefacts_v2/test_cic17.npz",
        "type":    "npz",
    },
    "CSE-CIC-IDS2018": {
        "path":    r"C:/MLProject/zero_day_project/artefacts_v2/test_cic18.npz",
        "type":    "npz",
    },
    "UNSW-NB15": {
        "path":    r"C:/MLProject/zero_day_project/artefacts_v2/test_unsw.npz",
        "type":    "npz",
    },
    "BoT-IoT": {
        "path":    r"C:/MLProject/zero_day_project/artefacts_v2/test_botiot.npz",
        "type":    "npz",
    },
    "CIC-ToN-IoT": {
        "path":    r"C:/MLProject/zero_day_project/data/raw/cic_ton_iot/CIC-ToN-IoT.csv",
        "type":    "csv",
        "rename":  {
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
        },
        "max_rows": 200_000,
        "local_cal": True,
    },
}

#    Ensemble combinations                                                      
COMBOS = [
    ("AE only",       ["ae"],          [1.0, 0.0, 0.0]),
    ("VAE only",      ["vae"],         [0.0, 1.0, 0.0]),
    ("SVDD only",     ["svdd"],        [0.0, 0.0, 1.0]),
    ("AE + VAE",      ["ae", "vae"],   [0.5, 0.5, 0.0]),
    ("AE + SVDD",     ["ae", "svdd"],  [0.67, 0.0, 0.33]),
    ("VAE + SVDD",    ["vae", "svdd"], [0.0, 0.67, 0.33]),
    ("Full ensemble", ["ae","vae","svdd"], [0.4, 0.4, 0.2]),
]

RNG = np.random.default_rng(42)


def norm_local(s):
    lo = float(np.percentile(s, 1))
    hi = float(np.percentile(s, 99))
    if hi - lo < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_dataset(name, cfg, features, scaler):
    """Load dataset, return (X_scaled, y)."""
    if cfg["type"] == "npz":
        if not os.path.exists(cfg["path"]):
            log.warning("  %s: npz not found at %s", name, cfg["path"])
            return None, None
        d = np.load(cfg["path"])
        X = np.clip(d["X"], -10, 10).astype(np.float32)
        y = d["y"].astype(int)
        return X, y

    elif cfg["type"] == "csv":
        frames = []
        total  = 0
        CHUNK  = 100_000
        max_r  = cfg.get("max_rows", 200_000)

        for chunk in pd.read_csv(cfg["path"], chunksize=CHUNK,
                                 low_memory=False):
            chunk.columns = (chunk.columns.str.strip().str.lower()
                             .str.replace(r'\s+', '_', regex=True)
                             .str.replace(r'[^a-z0-9_]', '_', regex=True)
                             .str.strip('_'))
            chunk.rename(columns=cfg.get("rename", {}), inplace=True)

            if 'label' in chunk.columns:
                chunk['label'] = (chunk['label'] != 0).astype(int)
            else:
                chunk['label'] = 1

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
            if total >= max_r:
                break
            gc.collect()

        df = pd.concat(frames, ignore_index=True).iloc[:max_r]
        X  = np.clip(
            scaler.transform(df[features].values.astype(np.float32)),
            -10, 10)
        y  = df['label'].values.astype(int)
        return X, y

    return None, None


def compute_ensemble(scores_dict, models_used, weights):
    """Combine normalised scores with given weights."""
    w_ae, w_vae, w_svdd = weights
    z = np.zeros(len(next(iter(scores_dict.values()))), dtype=np.float32)
    if "ae"   in models_used: z += w_ae   * norm_local(scores_dict["ae"])
    if "vae"  in models_used: z += w_vae  * norm_local(scores_dict["vae"])
    if "svdd" in models_used: z += w_svdd * norm_local(scores_dict["svdd"])
    return z


def get_threshold(S, y, local_cal=False):
    """Derive p95 threshold from benign scores."""
    benign = S[y == 0]
    if len(benign) == 0:
        # No benign — use val set (BoT-IoT case)
        val_path = os.path.join(ARTEFACT_DIR, "val.npz")
        if os.path.exists(val_path):
            X_val = np.clip(np.load(val_path)["X_val"][:30_000],
                            -10, 10).astype(np.float32)
            return None, X_val   # signal to caller: use val benign
        return 0.5, None

    if local_cal:
        n   = min(10_000, len(benign))
        idx = RNG.choice(len(benign), size=n, replace=False)
        return float(np.percentile(benign[idx], 95)), None
    else:
        return float(np.percentile(benign, 95)), None


def run():
    #    Load models                                                        
    log.info("Loading models ...")
    ae   = Autoencoder().load(os.path.join(ARTEFACT_DIR, "ae_model.pt"))
    vae  = VAE().load(os.path.join(ARTEFACT_DIR, "vae_model.pt"))
    svdd = DeepSVDD().load(os.path.join(ARTEFACT_DIR, "svdd_model.pt"))

    # Load training data to fit score stats if not already fitted
    train_path = os.path.join(ARTEFACT_DIR, "train.npz")
    if os.path.exists(train_path):
        X_train_raw = np.load(train_path)["X_train"]
        if ae.score_mu_ is None:
            log.info("Fitting score stats on training data ...")
            ae.fit_score_stats(X_train_raw)
            vae.fit_score_stats(X_train_raw)
            svdd.fit_score_stats(X_train_raw)
        del X_train_raw

    #    Load features & scaler                                             
    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    features  = pd.read_csv(feat_path, header=None)[0].tolist()
    features  = [f for f in features
                 if not str(f).strip().lstrip("-").isdigit()]

    scaler = IQRScaler()
    scaler.load(os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"))

    #    Pre-load val set for BoT-IoT threshold                             
    val_path  = os.path.join(ARTEFACT_DIR, "val.npz")
    X_val_raw = None
    if os.path.exists(val_path):
        X_val_raw = np.clip(
            np.load(val_path)["X_val"][:30_000], -10, 10
        ).astype(np.float32)

    results = []

    for ds_name, cfg in DATASETS.items():
        log.info("Loading %s ...", ds_name)
        X, y = load_dataset(ds_name, cfg, features, scaler)
        if X is None:
            log.warning("  Skipping %s — data not found", ds_name)
            continue

        log.info("  %d flows | benign=%d | attack=%d",
                 len(y), int((y==0).sum()), int((y==1).sum()))

        #    Compute raw scores once                                        
        log.info("  Computing model scores ...")
        raw = {
            "ae":   ae.anomaly_scores(X),
            "vae":  vae.anomaly_scores(X),
            "svdd": svdd.anomaly_scores(X),
        }

        local_cal = cfg.get("local_cal", False)

        for combo_name, models_used, weights in COMBOS:
            S = compute_ensemble(raw, models_used, weights)

            # Threshold
            if (y == 0).sum() == 0 and X_val_raw is not None:
                # BoT-IoT: derive threshold from val benign
                raw_val = {
                    "ae":   ae.anomaly_scores(X_val_raw),
                    "vae":  vae.anomaly_scores(X_val_raw),
                    "svdd": svdd.anomaly_scores(X_val_raw),
                }
                S_val = compute_ensemble(raw_val, models_used, weights)
                tau   = float(np.percentile(S_val, 95))
            else:
                benign_s = S[y == 0]
                if local_cal:
                    n   = min(10_000, len(benign_s))
                    idx = RNG.choice(len(benign_s), size=n, replace=False)
                    tau = float(np.percentile(benign_s[idx], 95))
                else:
                    tau = float(np.percentile(benign_s, 95))

            y_pred = (S > tau).astype(int)

            try:    auc = roc_auc_score(y, S)
            except: auc = float("nan")

            rec = recall_score(y, y_pred, zero_division=0)
            f1  = f1_score(y, y_pred, zero_division=0)
            fpr = float(((y==0)&(y_pred==1)).sum() / max(1,(y==0).sum()))

            results.append({
                "dataset":   ds_name,
                "combo":     combo_name,
                "models":    "+".join(models_used),
                "auc":       round(auc, 4),
                "recall":    round(rec, 4),
                "f1":        round(f1,  4),
                "fpr":       round(fpr, 4),
            })

            log.info("  %-20s  AUC=%.4f  Recall=%.4f  F1=%.4f  FPR=%.4f",
                     combo_name, auc, rec, f1, fpr)

        gc.collect()

    #    Save results                                                       
    df_res = pd.DataFrame(results)

    csv_path  = os.path.join(ARTEFACT_DIR, "ablation_pairwise_results.csv")
    json_path = os.path.join(ARTEFACT_DIR, "ablation_pairwise_results.json")
    df_res.to_csv(csv_path,  index=False)
    df_res.to_json(json_path, orient="records", indent=2)

    log.info("Results saved -> %s", csv_path)

    #    Print summary table                                                
    print("")
    print("=" * 80)
    print("PAIRWISE ABLATION RESULTS — AUC (primary metric)")
    print("=" * 80)

    pivot = df_res.pivot(index="combo", columns="dataset", values="auc")

    # Order combos
    order = [c[0] for c in COMBOS]
    pivot = pivot.reindex(order)

    col_order = ["CIC-IDS2017","CSE-CIC-IDS2018","UNSW-NB15",
                 "BoT-IoT","CIC-ToN-IoT"]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]

    header = f"{'Combination':<22}" + "".join(f"{c:>16}" for c in col_order)
    print(header)
    print("-" * len(header))

    for combo in order:
        if combo not in pivot.index:
            continue
        row   = pivot.loc[combo]
        is_full = combo == "Full ensemble"
        line  = f"{'* '+combo if is_full else combo:<22}"
        line += "".join(
            f"{row[c]:>16.4f}" if not np.isnan(row[c]) else f"{'—':>16}"
            for c in col_order)
        print(line)

    print("-" * len(header))
    print("* Full ensemble = proposed method")
    print("=" * 80)

    log.info("Done.")


if __name__ == "__main__":
    run()
