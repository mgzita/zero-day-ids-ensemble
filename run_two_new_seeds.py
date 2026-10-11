"""
run_two_new_seeds.py
====================
Retrains AE, VAE, Deep SVDD with seeds 789 and 999
and appends results to multiseed_results.csv.

After this script completes, run:
    python run_significance_3seed.py
(rename it or update it to say 5 seeds)

Run from project root:
    python run_two_new_seeds.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

PROJECT_ROOT = r"C:/MLProject/zero_day_project"
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "artefacts_v2")

NEW_SEEDS = [789, 999]

DATASETS = [
    ("CIC-IDS2017",     "test_cic17.npz"),
    ("CSE-CIC-IDS2018", "test_cic18.npz"),
    ("UNSW-NB15",       "test_unsw.npz"),
    ("BoT-IoT",         "test_botiot.npz"),
    ("CIC-ToN-IoT",     None),
]

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


def set_seed(seed):
    import torch
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def norm_scores(scores):
    mu, sigma = scores.mean(), scores.std() + 1e-8
    return np.clip((scores - mu) / sigma, 0, 1).astype(np.float32)


def load_ton_iot(features, scaler_npz, max_rows=200_000):
    import gc
    path = os.path.join(
        PROJECT_ROOT, "data", "raw", "cic_ton_iot", "CIC-ToN-IoT.csv"
    )
    q1  = scaler_npz["q1"].astype(np.float64)
    q2  = scaler_npz["q2"].astype(np.float64)
    q3  = scaler_npz["q3"].astype(np.float64)
    iqr = q3 - q1
    iqr[iqr < 1e-8] = 1.0

    frames, total = [], 0
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        chunk.columns = (
            chunk.columns.str.strip().str.lower()
            .str.replace(r'\s+', '_', regex=True)
            .str.replace(r'[^a-z0-9_]', '_', regex=True)
            .str.strip('_')
        )
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
    X  = (df[features].values.astype(np.float64) - q2) / iqr
    X  = np.clip(X, -10, 10).astype(np.float32)
    y  = df['label'].values.astype(int)
    return X, y


def evaluate_ensemble(ae, vae, svdd, X, y, X_val):
    """Compute ensemble scores and metrics."""
    s_ae   = norm_scores(ae.anomaly_scores(X))
    s_vae  = norm_scores(vae.anomaly_scores(X))
    s_svdd = norm_scores(svdd.anomaly_scores(X))
    ens    = 0.4*s_ae + 0.4*s_vae + 0.2*s_svdd

    # Threshold from val benign
    v_ae   = norm_scores(ae.anomaly_scores(X_val))
    v_vae  = norm_scores(vae.anomaly_scores(X_val))
    v_svdd = norm_scores(svdd.anomaly_scores(X_val))
    v_ens  = 0.4*v_ae + 0.4*v_vae + 0.2*v_svdd
    tau    = float(np.percentile(v_ens, 95))

    try:
        auc = roc_auc_score(y, ens)
    except:
        auc = float("nan")

    y_pred = (ens > tau).astype(int)
    return {
        "auc":    auc,
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1":     f1_score(y, y_pred, zero_division=0),
        "fpr":    float(((y==0) & (y_pred==1)).sum()
                        / max(1, (y==0).sum())),
    }


def run():
    t0_total = time.time()

    #    Load data                                                          
    log.info("Loading training and validation data...")
    train_data = np.load(os.path.join(ARTEFACT_DIR, "train.npz"))
    X_train    = np.clip(
        train_data["X_train"], -10, 10
    ).astype(np.float32)

    val_data = np.load(os.path.join(ARTEFACT_DIR, "val.npz"))
    X_val    = np.clip(val_data["X_val"], -10, 10).astype(np.float32)
    log.info("  X_train=%s  X_val=%s", X_train.shape, X_val.shape)

    #    Load test datasets                                                 
    log.info("Loading test datasets...")
    test_data = {}
    for ds_name, npz_file in DATASETS:
        if npz_file is None:
            continue
        d = np.load(os.path.join(ARTEFACT_DIR, npz_file))
        test_data[ds_name] = (
            np.clip(d["X"], -10, 10).astype(np.float32),
            d["y"].astype(int)
        )
        log.info("  %s: %d flows", ds_name, len(test_data[ds_name][1]))

    #    Load CIC-ToN-IoT                                                   
    log.info("Loading CIC-ToN-IoT...")
    scaler_npz = np.load(
        os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"),
        allow_pickle=True
    )
    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    features  = pd.read_csv(feat_path, header=None)[0].tolist()
    features  = [f for f in features
                 if not str(f).strip().lstrip("-").isdigit()]
    X_ton, y_ton = load_ton_iot(features, scaler_npz)
    test_data["CIC-ToN-IoT"] = (X_ton, y_ton)
    log.info("  CIC-ToN-IoT: %d flows", len(y_ton))

    #    Load model classes                                                 
    from models.autoencoder import Autoencoder
    from models.vae          import VAE
    from models.deep_svdd    import DeepSVDD

    #    Load existing results                                              
    existing_path = os.path.join(ARTEFACT_DIR, "multiseed_results.csv")
    df_existing   = pd.read_csv(existing_path)
    log.info("Existing results: seeds %s",
             df_existing["seed"].unique().tolist())

    new_rows = []

    #    Train and evaluate for each new seed                               
    for seed in NEW_SEEDS:
        log.info("\n" + "="*60)
        log.info("SEED %d", seed)
        log.info("="*60)
        set_seed(seed)
        t0_seed = time.time()

        # Train AE
        log.info("Training AE (seed=%d)...", seed)
        ae = Autoencoder(input_dim=X_train.shape[1])
        ae.fit(X_train, X_val)
        ae.fit_score_stats(X_train)

        # Train VAE
        log.info("Training VAE (seed=%d)...", seed)
        vae = VAE(input_dim=X_train.shape[1])
        vae.fit(X_train, X_val)
        vae.fit_score_stats(X_train)

        # Train Deep SVDD
        log.info("Training Deep SVDD (seed=%d)...", seed)
        svdd = DeepSVDD(input_dim=X_train.shape[1])
        svdd.fit(X_train, X_val)
        svdd.fit_score_stats(X_train)

        # Evaluate on all datasets
        for ds_name, _ in DATASETS:
            if ds_name not in test_data:
                continue
            X_test, y_test = test_data[ds_name]
            log.info("  Evaluating %s...", ds_name)

            # Local calibration for CIC-ToN-IoT
            if ds_name == "CIC-ToN-IoT":
                benign_idx = np.where(y_test == 0)[0]
                cal_idx    = np.random.default_rng(seed).choice(
                    len(benign_idx), size=10_000, replace=False
                )
                X_cal = X_test[benign_idx[cal_idx]]

                s_ae_t   = norm_scores(ae.anomaly_scores(X_test))
                s_vae_t  = norm_scores(vae.anomaly_scores(X_test))
                s_svdd_t = norm_scores(svdd.anomaly_scores(X_test))
                ens_t    = 0.4*s_ae_t + 0.4*s_vae_t + 0.2*s_svdd_t

                s_ae_c   = norm_scores(ae.anomaly_scores(X_cal))
                s_vae_c  = norm_scores(vae.anomaly_scores(X_cal))
                s_svdd_c = norm_scores(svdd.anomaly_scores(X_cal))
                ens_c    = 0.4*s_ae_c + 0.4*s_vae_c + 0.2*s_svdd_c
                tau      = float(np.percentile(ens_c, 95))

                try:
                    auc = roc_auc_score(y_test, ens_t)
                except:
                    auc = float("nan")
                y_pred = (ens_t > tau).astype(int)
                metrics = {
                    "auc":    auc,
                    "recall": recall_score(y_test, y_pred,
                                           zero_division=0),
                    "f1":     f1_score(y_test, y_pred,
                                       zero_division=0),
                    "fpr":    float(
                        ((y_test==0) & (y_pred==1)).sum()
                        / max(1, (y_test==0).sum())
                    ),
                }
            else:
                metrics = evaluate_ensemble(
                    ae, vae, svdd, X_test, y_test, X_val
                )

            row = {
                "auc":     metrics["auc"],
                "recall":  metrics["recall"],
                "f1":      metrics["f1"],
                "fpr":     metrics["fpr"],
                "seed":    seed,
                "dataset": ds_name,
            }
            new_rows.append(row)
            log.info("    AUC=%.4f  Recall=%.4f  F1=%.4f",
                     metrics["auc"] if not
                     np.isnan(metrics["auc"]) else -1,
                     metrics["recall"], metrics["f1"])

        log.info("Seed %d done in %.1f min",
                 seed, (time.time()-t0_seed)/60)

    #    Combine and save                                                   
    df_new = pd.DataFrame(new_rows)
    df_all = pd.concat([df_existing, df_new], ignore_index=True)

    out_path = os.path.join(ARTEFACT_DIR, "multiseed_results.csv")
    df_all.to_csv(out_path, index=False)
    log.info("Updated multiseed_results.csv -> %s", out_path)
    log.info("Total seeds now: %s",
             df_all["seed"].unique().tolist())

    #    Print final summary                                                
    print("\n" + "="*65)
    print("5-SEED SUMMARY")
    print("="*65)
    print(f"  {'Dataset':<22} {'AUC mean':>10} {'AUC std':>10} "
          f"{'Recall mean':>12}")
    print("  " + "-"*58)
    for ds_name, _ in DATASETS:
        rows = df_all[df_all["dataset"]==ds_name]
        auc_mean = rows["auc"].mean()
        auc_std  = rows["auc"].std()
        rec_mean = rows["recall"].mean()
        print(f"  {ds_name:<22} {auc_mean:>10.4f} "
              f"{auc_std:>10.4f} {rec_mean:>12.4f}")

    log.info("Total time: %.1f min",
             (time.time()-t0_total)/60)
    log.info("Done. Now run: python run_significance_3seed.py")


if __name__ == "__main__":
    run()
