"""
run_isolation_forest.py
=======================
Evaluates Isolation Forest on all 5 datasets.
Trained on full 1.8M combined benign samples.

Run from project root:
    python run_isolation_forest.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from preprocessing.scaler import IQRScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
RNG          = np.random.default_rng(42)

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


def load_ton_iot(features, scaler, max_rows=200_000):
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


def evaluate(scores, y, tau):
    y_pred = (scores > tau).astype(int)
    rec = recall_score(y, y_pred, zero_division=0)
    f1  = f1_score(y, y_pred, zero_division=0)
    fpr = float(((y==0)&(y_pred==1)).sum() / max(1,(y==0).sum()))
    try:    auc = roc_auc_score(y, scores)
    except: auc = float("nan")
    return dict(auc=auc, recall=rec, f1=f1, fpr=fpr)


def run():
    #    Load data                                                          
    log.info("Loading data ...")
    X_train = np.clip(
        np.load(os.path.join(ARTEFACT_DIR, "train.npz"))["X_train"],
        -10, 10)
    X_val   = np.clip(
        np.load(os.path.join(ARTEFACT_DIR, "val.npz"))["X_val"][:30_000],
        -10, 10)

    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    features  = pd.read_csv(feat_path, header=None)[0].tolist()
    features  = [f for f in features
                 if not str(f).strip().lstrip("-").isdigit()]

    scaler = IQRScaler()
    scaler.load(os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"))

    d17  = np.load(os.path.join(ARTEFACT_DIR, "test_cic17.npz"))
    d18  = np.load(os.path.join(ARTEFACT_DIR, "test_cic18.npz"))
    dsw  = np.load(os.path.join(ARTEFACT_DIR, "test_unsw.npz"))
    dbot = np.load(os.path.join(ARTEFACT_DIR, "test_botiot.npz"))

    log.info("Loading CIC-ToN-IoT ...")
    X_ton, y_ton = load_ton_iot(features, scaler)
    log.info("  ToN-IoT: %d flows | benign=%d | attack=%d",
             len(y_ton), int((y_ton==0).sum()), int((y_ton==1).sum()))

    datasets = [
        ("CIC-IDS2017",    np.clip(d17["X"],-10,10),  d17["y"],   False),
        ("CSE-CIC-IDS2018",np.clip(d18["X"],-10,10),  d18["y"],   False),
        ("UNSW-NB15",      np.clip(dsw["X"],-10,10),  dsw["y"],   False),
        ("BoT-IoT",        np.clip(dbot["X"],-10,10), dbot["y"],  False),
        ("CIC-ToN-IoT",    X_ton,                     y_ton,       True),
    ]

    #    Train Isolation Forest on full 1.8M samples                        
    log.info("Training Isolation Forest on %d samples ...", len(X_train))
    log.info("  (n_estimators=100, contamination=0.05) — this may take a few minutes")
    clf = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    clf.fit(X_train)
    log.info("  IF fitted.")

    #    Evaluate                                                           
    col_w  = 14
    header = (f"  {'Dataset':<20}" +
              f"{'AUC':>{col_w}}" +
              f"{'Recall':>{col_w}}" +
              f"{'F1':>{col_w}}" +
              f"{'FPR':>{col_w}}")
    divider = "  " + "-" * (20 + col_w * 4)

    print("\n" + "=" * 76)
    print("Isolation Forest — Full 1,800,000 benign training samples")
    print("=" * 76)
    print(header)
    print(divider)

    results = []
    val_scores = -clf.decision_function(X_val).astype(np.float32)

    for ds_name, X, y, local_cal in datasets:
        scores = -clf.decision_function(X).astype(np.float32)

        # Threshold
        benign_s = scores[y == 0]
        if len(benign_s) == 0:
            tau = float(np.percentile(val_scores, 95))
        elif local_cal:
            n   = min(10_000, len(benign_s))
            idx = RNG.choice(len(benign_s), size=n, replace=False)
            tau = float(np.percentile(benign_s[idx], 95))
        else:
            tau = float(np.percentile(benign_s, 95))

        r = evaluate(scores, y, tau)
        r["dataset"] = ds_name
        results.append(r)

        auc_s = f"{r['auc']:>{col_w}.4f}" if not np.isnan(r['auc']) \
                else f"{'—':>{col_w}}"
        print(f"  {ds_name:<20}{auc_s}"
              f"{r['recall']:>{col_w}.4f}"
              f"{r['f1']:>{col_w}.4f}"
              f"{r['fpr']:>{col_w}.4f}")

        gc.collect()

    print(divider)

    #    Summary                                                            
    print("\n" + "=" * 76)
    print("SUMMARY — AUC and F1 for paper table")
    print("=" * 76)
    for r in results:
        auc_s = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else "—"
        print(f"  {r['dataset']:<22}  AUC={auc_s}  "
              f"Recall={r['recall']:.4f}  F1={r['f1']:.4f}  "
              f"FPR={r['fpr']:.4f}")

    #    Save                                                               
    out = os.path.join(ARTEFACT_DIR, "if_results.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    log.info("Saved -> %s", out)
    log.info("Done.")


if __name__ == "__main__":
    run()
