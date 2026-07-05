"""
run_baselines_v2.py
===================
Evaluates HBOS, OC-SVM, and LOF baselines on all 5 datasets
including CIC-ToN-IoT as the second unseen zero-day test.

Run from project root:
    python run_baselines_v2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import logging
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
RNG          = np.random.default_rng(42)

OCSVM_TRAIN_SIZE = 10_000
LOF_TRAIN_SIZE   = 10_000

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


class HBOS:
    def __init__(self, n_bins=50):
        self.n_bins = n_bins
        self.histograms_ = []

    def fit(self, X):
        self.histograms_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            counts, edges = np.histogram(col, bins=self.n_bins, density=True)
            counts = np.maximum(counts, 1e-10)
            self.histograms_.append((counts, edges))
        return self

    def score_samples(self, X):
        scores = np.zeros(len(X), dtype=np.float32)
        for j, (counts, edges) in enumerate(self.histograms_):
            col = X[:, j]
            idx = np.searchsorted(edges[1:-1], col)
            idx = np.clip(idx, 0, len(counts) - 1)
            scores += -np.log(counts[idx])
        return scores


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
    from preprocessing.scaler import IQRScaler
    X = np.clip(scaler.transform(
        df[features].values.astype(np.float32)), -10, 10)
    y = df['label'].values.astype(int)
    return X, y


def evaluate(scores, y, tau):
    y_pred = (scores > tau).astype(int)
    rec = recall_score(y, y_pred, zero_division=0)
    f1  = f1_score(y, y_pred, zero_division=0)
    fpr = float(((y==0)&(y_pred==1)).sum() / max(1,(y==0).sum()))
    try:    auc = roc_auc_score(y, scores)
    except: auc = float("nan")
    return dict(auc=auc, recall=rec, f1=f1, fpr=fpr)


def get_tau(scores, y, local_cal=False, X_val_scores=None):
    benign = scores[y == 0]
    if len(benign) == 0:
        return float(np.percentile(X_val_scores, 95))
    if local_cal:
        n = min(10_000, len(benign))
        idx = RNG.choice(len(benign), size=n, replace=False)
        return float(np.percentile(benign[idx], 95))
    return float(np.percentile(benign, 95))


def run():
    #    Load data                                                          
    log.info("Loading preprocessed data ...")
    X_train = np.clip(np.load(os.path.join(ARTEFACT_DIR,"train.npz"))["X_train"],
                      -10, 10)
    X_val   = np.clip(np.load(os.path.join(ARTEFACT_DIR,"val.npz"))["X_val"],
                      -10, 10)
    d17  = np.load(os.path.join(ARTEFACT_DIR,"test_cic17.npz"))
    d18  = np.load(os.path.join(ARTEFACT_DIR,"test_cic18.npz"))
    dsw  = np.load(os.path.join(ARTEFACT_DIR,"test_unsw.npz"))
    dbot = np.load(os.path.join(ARTEFACT_DIR,"test_botiot.npz"))

    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    features  = pd.read_csv(feat_path, header=None)[0].tolist()
    features  = [f for f in features
                 if not str(f).strip().lstrip("-").isdigit()]

    from preprocessing.scaler import IQRScaler
    scaler = IQRScaler()
    scaler.load(os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"))

    log.info("Loading CIC-ToN-IoT ...")
    X_ton, y_ton = load_ton_iot(features, scaler)
    log.info("  ToN-IoT: %d flows | benign=%d | attack=%d",
             len(y_ton), int((y_ton==0).sum()), int((y_ton==1).sum()))

    datasets = [
        ("CIC-IDS2017",    np.clip(d17["X"],-10,10), d17["y"],   False),
        ("CIC-IDS2018",    np.clip(d18["X"],-10,10), d18["y"],   False),
        ("UNSW-NB15",      np.clip(dsw["X"],-10,10), dsw["y"],   False),
        ("BoT-IoT",        np.clip(dbot["X"],-10,10),dbot["y"],  False),
        ("CIC-ToN-IoT",    X_ton,                    y_ton,       True),
    ]

    col_w = 16
    header = (f"  {'Dataset':<18}" +
              f"{'AUC':>{col_w}}" +
              f"{'Recall':>{col_w}}" +
              f"{'F1':>{col_w}}" +
              f"{'FPR':>{col_w}}")
    divider = "  " + "-" * (18 + col_w*4)

    all_results = []

    #    HBOS                                                              
    print("\n" + "="*70)
    print("Baseline 1: HBOS  (full 1.8M training samples)")
    print("="*70)
    hbos = HBOS(n_bins=50)
    hbos.fit(X_train)
    val_hbos = hbos.score_samples(X_val[:50_000])

    print(header); print(divider)
    for ds_name, X, y, local_cal in datasets:
        s   = hbos.score_samples(X)
        tau = get_tau(s, y, local_cal,
                      X_val_scores=val_hbos if (y==0).sum()==0 else None)
        r   = evaluate(s, y, tau)
        r.update({"dataset": ds_name, "baseline": "HBOS"})
        all_results.append(r)
        auc_s = f"{r['auc']:>{col_w}.4f}" if not np.isnan(r['auc']) \
                else f"{'—':>{col_w}}"
        print(f"  {ds_name:<18}{auc_s}"
              f"{r['recall']:>{col_w}.4f}"
              f"{r['f1']:>{col_w}.4f}"
              f"{r['fpr']:>{col_w}.4f}")
    print(divider)

    #    OC-SVM                                                            
    print("\n" + "="*70)
    print(f"Baseline 2: OC-SVM  ({OCSVM_TRAIN_SIZE:,} samples, RBF kernel)")
    print("="*70)
    idx_oc = RNG.choice(len(X_train), size=OCSVM_TRAIN_SIZE, replace=False)
    ocsvm  = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X_train[idx_oc])
    val_oc = -ocsvm.decision_function(X_val[:10_000]).astype(np.float32)

    print(header); print(divider)
    for ds_name, X, y, local_cal in datasets:
        s   = -ocsvm.decision_function(X).astype(np.float32)
        tau = get_tau(s, y, local_cal,
                      X_val_scores=val_oc if (y==0).sum()==0 else None)
        r   = evaluate(s, y, tau)
        r.update({"dataset": ds_name, "baseline": "OC-SVM"})
        all_results.append(r)
        auc_s = f"{r['auc']:>{col_w}.4f}" if not np.isnan(r['auc']) \
                else f"{'—':>{col_w}}"
        print(f"  {ds_name:<18}{auc_s}"
              f"{r['recall']:>{col_w}.4f}"
              f"{r['f1']:>{col_w}.4f}"
              f"{r['fpr']:>{col_w}.4f}")
    print(divider)

    #    LOF                                                               
    print("\n" + "="*70)
    print(f"Baseline 3: LOF  ({LOF_TRAIN_SIZE:,} samples, k=20)")
    print("="*70)
    idx_lof = RNG.choice(len(X_train), size=LOF_TRAIN_SIZE, replace=False)
    lof     = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    lof.fit(X_train[idx_lof])
    val_lof = -lof.decision_function(X_val[:5_000]).astype(np.float32)

    print(header); print(divider)
    for ds_name, X, y, local_cal in datasets:
        s   = -lof.decision_function(X).astype(np.float32)
        tau = get_tau(s, y, local_cal,
                      X_val_scores=val_lof if (y==0).sum()==0 else None)
        r   = evaluate(s, y, tau)
        r.update({"dataset": ds_name, "baseline": "LOF"})
        all_results.append(r)
        auc_s = f"{r['auc']:>{col_w}.4f}" if not np.isnan(r['auc']) \
                else f"{'—':>{col_w}}"
        print(f"  {ds_name:<18}{auc_s}"
              f"{r['recall']:>{col_w}.4f}"
              f"{r['f1']:>{col_w}.4f}"
              f"{r['fpr']:>{col_w}.4f}")
    print(divider)

    #    Summary table                                                      
    print("\n" + "="*80)
    print("BASELINE COMPARISON — AUC Summary")
    print("="*80)
    df_res = pd.DataFrame(all_results)

    ds_order = ["CIC-IDS2017","CIC-IDS2018","UNSW-NB15",
                "BoT-IoT","CIC-ToN-IoT"]
    col_w2 = 14
    print("  " + f"{'Method':<12}" +
          "".join(f"{d:>{col_w2}}" for d in ds_order))
    print("  " + "-" * (12 + col_w2 * len(ds_order)))

    for bl in ["HBOS", "OC-SVM", "LOF"]:
        sub = df_res[df_res["baseline"] == bl]
        row = f"  {bl:<12}"
        for ds in ds_order:
            r = sub[sub["dataset"] == ds]
            if len(r) > 0:
                auc = r.iloc[0]["auc"]
                row += f"{auc:>{col_w2}.4f}" if not np.isnan(auc) \
                       else f"{'—':>{col_w2}}"
            else:
                row += f"{'N/A':>{col_w2}}"
        print(row)

    # Ensemble reference row
    ens_auc = {
        "CIC-IDS2017":    0.9370,
        "CIC-IDS2018":    0.9564,
        "UNSW-NB15":      0.8657,
        "BoT-IoT":        float("nan"),
        "CIC-ToN-IoT":    0.9476,
    }
    row = f"  {'Ensemble':<12}"
    for ds in ds_order:
        auc = ens_auc.get(ds, float("nan"))
        row += f"{auc:>{col_w2}.4f}" if not np.isnan(auc) \
               else f"{'—':>{col_w2}}"
    print(row)
    print("  " + "-" * (12 + col_w2 * len(ds_order)))

    #    Save                                                               
    out = os.path.join(ARTEFACT_DIR, "baseline_results_v2.csv")
    df_res.to_csv(out, index=False)
    log.info("Results saved -> %s", out)
    log.info("Done.")


if __name__ == "__main__":
    run()
