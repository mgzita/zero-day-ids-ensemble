"""
run_lodo.py
===========
Leave-One-Dataset-Out (LODO) evaluation.

For each experiment, one dataset is held out completely from training.
Models are trained on the remaining two datasets' benign traffic,
then evaluated on all four datasets including BoT-IoT.

Experiments:
  LODO-1: Train on CIC2018 + UNSW     Test on CIC2017 (held out) + BoT-IoT
  LODO-2: Train on CIC2017 + UNSW     Test on CIC2018 (held out) + BoT-IoT
  LODO-3: Train on CIC2017 + CIC2018   Test on UNSW   (held out) + BoT-IoT

This is the strongest generalisation test: the held-out dataset's benign
patterns are completely unseen during training and threshold calibration.

All results are saved to artefacts_lodo/ and printed as a summary table.

Run from project root:
    python run_lodo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from models.autoencoder  import Autoencoder
from models.vae          import VAE
from models.deep_svdd    import DeepSVDD
from preprocessing.scaler import IQRScaler
from preprocessing_v2.config import (
    DATASET_PATHS, UNSW_COLUMNS, UNSW_DATA_FILES,
    BOTIOT_DATA_FILES, BOTIOT_RENAME, CIC2018_RENAME, UNSW_RENAME,
    NON_NUMERIC_COLS, LOG_TRANSFORM_COLS, CORRELATION_THRESHOLD,
)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

LODO_DIR  = r"C:/MLProject/zero_day_project/artefacts_lodo"
CHUNK     = 100_000
MAX_BENIGN= 600_000
MAX_EVAL  = 500_000
MAX_BOTIOT= 500_000
RNG       = np.random.default_rng(42)

os.makedirs(LODO_DIR, exist_ok=True)


# =============================================================================
# SHARED CLEANING  (same as run_preprocessing_v2.py)
# =============================================================================

def _clean(df, rename=None):
    df.columns = (df.columns.str.strip().str.lower()
                             .str.replace(r"[^a-z0-9_]","_",regex=True))
    if rename:
        df.rename(columns=rename, inplace=True)
    df.drop(columns=[c for c in NON_NUMERIC_COLS if c in df.columns],
            inplace=True, errors="ignore")
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if col != "label":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    log_cols = [c for c in LOG_TRANSFORM_COLS if c in df.columns]
    if log_cols:
        df[log_cols] = np.log1p(df[log_cols].clip(lower=0))
    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    return df

def _label_cic(df):
    lc = next((c for c in ["label","flow_label"] if c in df.columns), None)
    if lc:
        df["label"] = (df[lc].astype(str).str.strip().str.upper()!="BENIGN").astype(int)
        if lc != "label": df.drop(columns=[lc], inplace=True)
    else:
        df["label"] = 0
    return df

def _label_num(df, col="label"):
    df["label"] = pd.to_numeric(df.get(col,0),errors="coerce").fillna(0).astype(int)
    return df


# =============================================================================
# LOADERS
# =============================================================================

def load_cic_benign(folder, rename=None, max_rows=MAX_BENIGN):
    files = sorted(Path(folder).glob("*.csv"))
    frames=[]; total=0
    for f in files:
        for chunk in pd.read_csv(f, chunksize=CHUNK, low_memory=False):
            chunk = _clean(chunk, rename)
            chunk = _label_cic(chunk)
            chunk = chunk[chunk["label"]==0].copy()
            frames.append(chunk); total+=len(chunk)
            if total>=max_rows: break
        if total>=max_rows: break
        gc.collect()
    df = pd.concat(frames, ignore_index=True).iloc[:max_rows]
    log.info("  Loaded %d benign rows", len(df))
    return df

def load_cic_all(folder, rename=None, max_rows=MAX_EVAL):
    files = sorted(Path(folder).glob("*.csv"))
    frames=[]; total=0
    for f in files:
        for chunk in pd.read_csv(f, chunksize=CHUNK, low_memory=False):
            chunk = _clean(chunk, rename)
            chunk = _label_cic(chunk)
            frames.append(chunk); total+=len(chunk)
            if total>=max_rows: break
        if total>=max_rows: break
        gc.collect()
    df = pd.concat(frames, ignore_index=True).iloc[:max_rows]
    log.info("  Loaded %d rows | benign=%d attack=%d",
             len(df), int((df["label"]==0).sum()), int((df["label"]==1).sum()))
    return df

def load_unsw_benign(max_rows=MAX_BENIGN):
    folder=DATASET_PATHS["unsw"]; frames=[]; total=0
    for fname in UNSW_DATA_FILES:
        for chunk in pd.read_csv(os.path.join(folder,fname), chunksize=CHUNK,
                                  header=None, names=UNSW_COLUMNS, low_memory=False):
            chunk=_clean(chunk,UNSW_RENAME); chunk=_label_num(chunk)
            chunk=chunk[chunk["label"]==0].copy()
            frames.append(chunk); total+=len(chunk)
            if total>=max_rows: break
        if total>=max_rows: break
        gc.collect()
    df=pd.concat(frames,ignore_index=True).iloc[:max_rows]
    log.info("  Loaded %d UNSW benign rows",len(df)); return df

def load_unsw_all(max_rows=MAX_EVAL):
    folder=DATASET_PATHS["unsw"]; frames=[]; total=0
    for fname in UNSW_DATA_FILES:
        for chunk in pd.read_csv(os.path.join(folder,fname), chunksize=CHUNK,
                                  header=None, names=UNSW_COLUMNS, low_memory=False):
            chunk=_clean(chunk,UNSW_RENAME); chunk=_label_num(chunk)
            frames.append(chunk); total+=len(chunk)
            if total>=max_rows: break
        if total>=max_rows: break
        gc.collect()
    df=pd.concat(frames,ignore_index=True).iloc[:max_rows]
    log.info("  Loaded %d UNSW rows | benign=%d attack=%d",
             len(df),int((df["label"]==0).sum()),int((df["label"]==1).sum()))
    return df

def load_botiot(max_rows=MAX_BOTIOT):
    folder=DATASET_PATHS["botiot"]; frames=[]; total=0
    for fname in BOTIOT_DATA_FILES:
        for chunk in pd.read_csv(os.path.join(folder,fname),chunksize=CHUNK,low_memory=False):
            chunk=_clean(chunk,BOTIOT_RENAME); chunk=_label_num(chunk)
            frames.append(chunk); total+=len(chunk)
            if total>=max_rows: break
        if total>=max_rows: break
        gc.collect()
    df=pd.concat(frames,ignore_index=True).iloc[:max_rows]
    log.info("  Loaded %d BoT-IoT rows | benign=%d attack=%d",
             len(df),int((df["label"]==0).sum()),int((df["label"]==1).sum()))
    return df


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def spearman_select(X, feature_names, threshold=CORRELATION_THRESHOLD):
    from scipy.stats import rankdata
    X_r  = np.apply_along_axis(rankdata,0,X)
    corr = np.corrcoef(X_r.T)
    n    = len(feature_names); drop=set()
    for i in range(n):
        if feature_names[i] in drop: continue
        for j in range(i+1,n):
            if feature_names[j] in drop: continue
            if abs(corr[i,j])>threshold: drop.add(feature_names[j])
    selected=[f for f in feature_names if f not in drop]
    log.info("  Spearman: %d -> %d features",n,len(selected))
    return selected


# =============================================================================
# ENSEMBLE SCORING
# =============================================================================

def norm(s):
    lo=float(np.percentile(s,1)); hi=float(np.percentile(s,99))
    if hi-lo<1e-8: return np.zeros_like(s)
    return np.clip((s-lo)/(hi-lo),0.0,1.0)

def ens_scores(ae,vae,svdd,X,w_ae=0.4,w_vae=0.4,w_svdd=0.2):
    return (w_ae*norm(ae.anomaly_scores(X)) +
            w_vae*norm(vae.anomaly_scores(X)) +
            w_svdd*norm(svdd.anomaly_scores(X)))

def eval_dataset(ae,vae,svdd,X,y,name,tau=None):
    scores = ens_scores(ae,vae,svdd,X)
    benign = scores[y==0]
    if tau is None:
        tau = float(np.percentile(benign,95)) if len(benign)>0 else 0.5
    y_pred = (scores>tau).astype(int)
    acc  = accuracy_score(y,y_pred)
    prec = precision_score(y,y_pred,zero_division=0)
    rec  = recall_score(y,y_pred,zero_division=0)
    f1   = f1_score(y,y_pred,zero_division=0)
    try:    auc=roc_auc_score(y,scores)
    except: auc=float("nan")
    return dict(name=name,acc=acc,prec=prec,rec=rec,f1=f1,auc=auc,tau=tau)


# =============================================================================
# SINGLE LODO EXPERIMENT
# =============================================================================

def run_lodo_experiment(exp_name, train_dfs, eval_datasets, val_tau_X=None):
    """
    train_dfs    : list of benign DataFrames to train on
    eval_datasets: list of (name, df_all) for evaluation
    """
    log.info("")
    log.info("=" * 60)
    log.info("EXPERIMENT: %s", exp_name)
    log.info("=" * 60)

    #    Common features                                                    
    def num_cols(df):
        return set(df.select_dtypes(include=[np.number]).columns)-{"label"}
    common = sorted(set.intersection(*[num_cols(df) for df in train_dfs] +
                                       [num_cols(df) for _,df in eval_datasets]))
    log.info("Common features: %d", len(common))

    #    Spearman selection on first training dataset benign                
    df0_benign = train_dfs[0][train_dfs[0]["label"]==0] if "label" in train_dfs[0].columns else train_dfs[0]
    n_sel = min(50_000, len(df0_benign))
    idx   = RNG.choice(len(df0_benign), size=n_sel, replace=False)
    X_sel = df0_benign.iloc[idx][common].values.astype(np.float32)
    selected = spearman_select(X_sel, common)
    n_feat   = len(selected)

    #    Build combined training set                                        
    def to_X(df):
        for f in selected:
            if f not in df.columns: df[f]=0.0
        return df[selected].values.astype(np.float32)

    benign_parts = []
    for df in train_dfs:
        ben = df[df["label"]==0] if "label" in df.columns else df
        X_b = to_X(ben.copy())
        idx_b = RNG.choice(len(X_b), size=min(len(X_b),600_000), replace=False)
        benign_parts.append(X_b[idx_b])
    X_train = np.concatenate(benign_parts, axis=0)

    # 80/20 split for val
    idx_all = np.arange(len(X_train)); RNG.shuffle(idx_all)
    n_tr    = int(len(idx_all)*0.8)
    X_tr    = X_train[idx_all[:n_tr]]
    X_val   = X_train[idx_all[n_tr:]]
    log.info("X_train=%s  X_val=%s  features=%d", X_tr.shape, X_val.shape, n_feat)

    #    Fit scaler                                                         
    scaler = IQRScaler()
    scaler.fit(X_tr)
    def scale(X): return np.clip(scaler.transform(X),-10,10).astype(np.float32)
    X_tr  = scale(X_tr)
    X_val = scale(X_val)

    #    Train models                                                       
    log.info("Training AE ...")
    ae = Autoencoder(input_dim=n_feat)
    ae.fit(X_tr, X_val)
    ae.fit_score_stats(X_tr)
    ae.fit_latent_stats(X_tr)
    ae.fit_threshold(X_val)

    log.info("Training VAE ...")
    vae = VAE(input_dim=n_feat, latent_dim=8)
    vae.fit(X_tr, X_val)
    vae.fit_score_stats(X_tr)
    vae.fit_latent_stats(X_tr)
    vae.fit_threshold(X_val)

    log.info("Training Deep SVDD ...")
    svdd = DeepSVDD(input_dim=n_feat, latent_dim=16)
    svdd.fit(X_tr, X_val)
    svdd.fit_score_stats(X_tr)
    svdd.fit_threshold(X_val)

    # Global val threshold
    val_scores = ens_scores(ae, vae, svdd, X_val)
    tau_global = float(np.percentile(val_scores, 95))

    #    Evaluate                                                           
    results = []
    for ds_name, df_eval in eval_datasets:
        for f in selected:
            if f not in df_eval.columns: df_eval[f]=0.0
        X_ev = scale(df_eval[selected].values.astype(np.float32))
        y_ev = df_eval["label"].values
        benign_ev = X_ev[y_ev==0]
        if len(benign_ev)>0:
            n_cal = max(1000, int(len(benign_ev)*0.10))
            idx_c = RNG.choice(len(benign_ev), size=n_cal, replace=False)
            s_cal = ens_scores(ae,vae,svdd,benign_ev[idx_c])
            tau   = float(np.percentile(s_cal,95))
        else:
            tau = tau_global
        r = eval_dataset(ae,vae,svdd,X_ev,y_ev,ds_name,tau=tau)
        r["experiment"] = exp_name
        results.append(r)
        log.info("  %-25s AUC=%.4f  F1=%.4f  Rec=%.4f",
                 ds_name, r["auc"], r["f1"], r["rec"])

    # Save models
    exp_safe = exp_name.replace(" ","_").replace("/","_")
    ae.save(os.path.join(LODO_DIR,   f"ae_{exp_safe}.pt"))
    vae.save(os.path.join(LODO_DIR,  f"vae_{exp_safe}.pt"))
    svdd.save(os.path.join(LODO_DIR, f"svdd_{exp_safe}.pt"))

    return results


# =============================================================================
# MAIN
# =============================================================================

def run():
    log.info("Loading all datasets ...")

    # Load all datasets once — reuse across experiments
    log.info("CIC2017 benign ...")
    df17b = load_cic_benign(DATASET_PATHS["cic2017"])
    df17b["label"] = 0

    log.info("CIC2017 all ...")
    df17  = load_cic_all(DATASET_PATHS["cic2017"])

    log.info("CIC2018 benign ...")
    df18b = load_cic_benign(DATASET_PATHS["cic2018"], rename=CIC2018_RENAME)
    df18b["label"] = 0

    log.info("CIC2018 all ...")
    df18  = load_cic_all(DATASET_PATHS["cic2018"], rename=CIC2018_RENAME)

    log.info("UNSW benign ...")
    dfswb = load_unsw_benign()
    dfswb["label"] = 0

    log.info("UNSW all ...")
    dfsw  = load_unsw_all()

    log.info("BoT-IoT ...")
    dfbot = load_botiot()

    all_results = []

    #    LODO-1: Train CIC2018+UNSW   Test CIC2017 (held out)              
    r1 = run_lodo_experiment(
        exp_name="LODO-1 (hold out CIC2017)",
        train_dfs=[df18b, dfswb],
        eval_datasets=[
            ("CIC2017 (held out)", df17),
            ("CIC2018 (train)",    df18),
            ("UNSW (train)",       dfsw),
            ("BoT-IoT (unseen)",   dfbot),
        ]
    )
    all_results.extend(r1)
    gc.collect()

    #    LODO-2: Train CIC2017+UNSW   Test CIC2018 (held out)              
    r2 = run_lodo_experiment(
        exp_name="LODO-2 (hold out CIC2018)",
        train_dfs=[df17b, dfswb],
        eval_datasets=[
            ("CIC2017 (train)",    df17),
            ("CIC2018 (held out)", df18),
            ("UNSW (train)",       dfsw),
            ("BoT-IoT (unseen)",   dfbot),
        ]
    )
    all_results.extend(r2)
    gc.collect()

    #    LODO-3: Train CIC2017+CIC2018   Test UNSW (held out)              
    r3 = run_lodo_experiment(
        exp_name="LODO-3 (hold out UNSW)",
        train_dfs=[df17b, df18b],
        eval_datasets=[
            ("CIC2017 (train)",    df17),
            ("CIC2018 (train)",    df18),
            ("UNSW (held out)",    dfsw),
            ("BoT-IoT (unseen)",   dfbot),
        ]
    )
    all_results.extend(r3)

    #    Summary table                                                      
    print("")
    print("=" * 78)
    print("LODO SUMMARY — AUC (primary) and F1")
    print("=" * 78)
    print("  " + f"{'Experiment':<32}" +
          f"{'Dataset':<26}" +
          f"{'AUC':>7}" + f"{'F1':>7}" + f"{'Recall':>8}")
    print("  " + "-" * 82)

    for r in all_results:
        held = "(held out)" in r["name"]
        unseen = "BoT-IoT" in r["name"]
        tag = " *" if held else (" †" if unseen else "")
        auc_s = f"{r['auc']:>7.4f}" if not (isinstance(r['auc'],float) and
                                              r['auc']!=r['auc']) else "    nan"
        print("  " + f"{r['experiment']:<32}" +
              f"{r['name']+tag:<26}" +
              auc_s + f"{r['f1']:>7.4f}" + f"{r['rec']:>8.4f}")

    print("  " + "-" * 82)
    print("  * = held-out domain (never seen during training)")
    print("  † = BoT-IoT (fully unseen in all experiments)")

    # Save results CSV
    df_res = pd.DataFrame(all_results)
    csv_path = os.path.join(LODO_DIR, "lodo_results.csv")
    df_res.to_csv(csv_path, index=False)
    log.info("Results saved -> %s", csv_path)
    log.info("Done.")


if __name__ == "__main__":
    run()
