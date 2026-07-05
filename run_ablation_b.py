"""
run_ablation_b.py
=================
Ablation condition (b): Combined benign training + Isolation Forest.

This is the missing condition needed to isolate the effect of:
  - Combined training alone (vs v1 CIC2017-only)
  - Deep SVDD replacement alone (vs IF)

Ablation table:
  (a) CIC2017-only  + IF      v1 (artefacts/)
  (b) Combined      + IF      THIS SCRIPT
  (c) Combined      + SVDD    v2 (artefacts_v2/)

Outputs saved to artefacts_ablation/

Run from project root:
    python run_ablation_b.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from models.autoencoder      import Autoencoder
from models.vae              import VAE
from models.isolation_forest import IsolationForestModel
from preprocessing.scaler    import IQRScaler
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

ABLATION_DIR = r"C:/MLProject/zero_day_project/artefacts_ablation"
CHUNK        = 100_000
MAX_BENIGN   = 600_000
MAX_EVAL     = 500_000
RNG          = np.random.default_rng(42)

os.makedirs(ABLATION_DIR, exist_ok=True)


# =============================================================================
# SHARED HELPERS
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

def load_cic_benign(folder, rename=None, max_rows=MAX_BENIGN):
    files=sorted(Path(folder).glob("*.csv")); frames=[]; total=0
    for f in files:
        for chunk in pd.read_csv(f, chunksize=CHUNK, low_memory=False):
            chunk=_clean(chunk,rename); chunk=_label_cic(chunk)
            chunk=chunk[chunk["label"]==0].copy()
            frames.append(chunk); total+=len(chunk)
            if total>=max_rows: break
        if total>=max_rows: break
        gc.collect()
    return pd.concat(frames,ignore_index=True).iloc[:max_rows]

def load_cic_all(folder, rename=None, max_rows=MAX_EVAL):
    files=sorted(Path(folder).glob("*.csv")); frames=[]; total=0
    for f in files:
        for chunk in pd.read_csv(f, chunksize=CHUNK, low_memory=False):
            chunk=_clean(chunk,rename); chunk=_label_cic(chunk)
            frames.append(chunk); total+=len(chunk)
            if total>=max_rows: break
        if total>=max_rows: break
        gc.collect()
    return pd.concat(frames,ignore_index=True).iloc[:max_rows]

def load_unsw_benign(max_rows=MAX_BENIGN):
    folder=DATASET_PATHS["unsw"]; frames=[]; total=0
    for fname in UNSW_DATA_FILES:
        for chunk in pd.read_csv(os.path.join(folder,fname),chunksize=CHUNK,
                                  header=None,names=UNSW_COLUMNS,low_memory=False):
            chunk=_clean(chunk,UNSW_RENAME); chunk=_label_num(chunk)
            chunk=chunk[chunk["label"]==0].copy()
            frames.append(chunk); total+=len(chunk)
            if total>=max_rows: break
        if total>=max_rows: break
        gc.collect()
    return pd.concat(frames,ignore_index=True).iloc[:max_rows]

def load_unsw_all(max_rows=MAX_EVAL):
    folder=DATASET_PATHS["unsw"]; frames=[]; total=0
    for fname in UNSW_DATA_FILES:
        for chunk in pd.read_csv(os.path.join(folder,fname),chunksize=CHUNK,
                                  header=None,names=UNSW_COLUMNS,low_memory=False):
            chunk=_clean(chunk,UNSW_RENAME); chunk=_label_num(chunk)
            frames.append(chunk); total+=len(chunk)
            if total>=max_rows: break
        if total>=max_rows: break
        gc.collect()
    return pd.concat(frames,ignore_index=True).iloc[:max_rows]

def load_botiot(max_rows=MAX_EVAL):
    folder=DATASET_PATHS["botiot"]; frames=[]; total=0
    for fname in BOTIOT_DATA_FILES:
        for chunk in pd.read_csv(os.path.join(folder,fname),chunksize=CHUNK,low_memory=False):
            chunk=_clean(chunk,BOTIOT_RENAME); chunk=_label_num(chunk)
            frames.append(chunk); total+=len(chunk)
            if total>=max_rows: break
        if total>=max_rows: break
        gc.collect()
    return pd.concat(frames,ignore_index=True).iloc[:max_rows]

def norm(s):
    lo=float(np.percentile(s,1)); hi=float(np.percentile(s,99))
    if hi-lo<1e-8: return np.zeros_like(s)
    return np.clip((s-lo)/(hi-lo),0.0,1.0)

def ens_scores(ae, vae, iforest, X):
    return (0.4*norm(ae.anomaly_scores(X)) +
            0.4*norm(vae.anomaly_scores(X)) +
            0.2*norm(iforest.anomaly_scores(X)))

def evaluate(ae, vae, iforest, X, y, name, tau_global):
    scores = ens_scores(ae, vae, iforest, X)
    benign = scores[y==0]
    tau    = float(np.percentile(benign,95)) if len(benign)>0 else tau_global
    y_pred = (scores>tau).astype(int)
    acc  = accuracy_score(y,y_pred)
    prec = precision_score(y,y_pred,zero_division=0)
    rec  = recall_score(y,y_pred,zero_division=0)
    f1   = f1_score(y,y_pred,zero_division=0)
    try:    auc=roc_auc_score(y,scores)
    except: auc=float("nan")
    auc_s = f"{auc:.4f}" if not (isinstance(auc,float) and auc!=auc) else "   nan"
    log.info("  %-25s AUC=%s  F1=%.4f  Rec=%.4f", name, auc_s, f1, rec)
    return dict(dataset=name,acc=acc,prec=prec,rec=rec,f1=f1,auc=auc)


# =============================================================================
# MAIN
# =============================================================================

def run():
    log.info("=" * 60)
    log.info("Ablation (b): Combined training + Isolation Forest")
    log.info("=" * 60)

    #    Load training data                                                 
    log.info("Loading benign training data ...")
    df17b = load_cic_benign(DATASET_PATHS["cic2017"])
    df18b = load_cic_benign(DATASET_PATHS["cic2018"], rename=CIC2018_RENAME)
    dfswb = load_unsw_benign()

    #    Common features                                                    
    def num_cols(df):
        return set(df.select_dtypes(include=[np.number]).columns)-{"label"}
    common = sorted(num_cols(df17b) & num_cols(df18b) & num_cols(dfswb))
    log.info("Common features: %d", len(common))

    #    Spearman selection                                                 
    from scipy.stats import rankdata
    n_sel = min(50_000, len(df17b[df17b["label"]==0]))
    idx   = RNG.choice(len(df17b), size=n_sel, replace=False)
    X_sel = df17b.iloc[idx][common].values.astype(np.float32)
    X_r   = np.apply_along_axis(rankdata,0,X_sel)
    corr  = np.corrcoef(X_r.T)
    drop  = set()
    for i in range(len(common)):
        if common[i] in drop: continue
        for j in range(i+1,len(common)):
            if common[j] in drop: continue
            if abs(corr[i,j])>CORRELATION_THRESHOLD: drop.add(common[j])
    selected = [f for f in common if f not in drop]
    n_feat   = len(selected)
    log.info("Selected features: %d", n_feat)

    # Save feature list
    pd.Series(selected).to_csv(
        os.path.join(ABLATION_DIR,"selected_features.csv"), index=False)

    #    Build combined training set                                        
    def to_X(df):
        for f in selected:
            if f not in df.columns: df[f]=0.0
        X   = df[selected].values.astype(np.float32)
        idx = RNG.choice(len(X), size=min(len(X),600_000), replace=False)
        return X[idx]

    X_train = np.concatenate([to_X(df17b), to_X(df18b), to_X(dfswb)], axis=0)
    del df17b, df18b, dfswb; gc.collect()

    # 80/20 split
    idx_all = np.arange(len(X_train)); RNG.shuffle(idx_all)
    n_tr    = int(len(idx_all)*0.8)
    X_tr    = X_train[idx_all[:n_tr]]
    X_val   = X_train[idx_all[n_tr:]]
    del X_train; gc.collect()

    #    Fit IQR scaler                                                     
    scaler = IQRScaler()
    scaler.fit(X_tr)
    scaler.save(os.path.join(ABLATION_DIR,"iqr_scaler.npz"))

    def scale(X):
        return np.clip(scaler.transform(X),-10,10).astype(np.float32)
    X_tr  = scale(X_tr)
    X_val = scale(X_val)

    #    Train AE                                                           
    log.info("Training AE ...")
    ae = Autoencoder(input_dim=n_feat)
    ae.fit(X_tr, X_val)
    ae.fit_score_stats(X_tr)
    ae.fit_latent_stats(X_tr)
    ae.fit_threshold(X_val)
    ae.save(os.path.join(ABLATION_DIR,"ae_model.pt"))

    #    Train VAE                                                          
    log.info("Training VAE ...")
    vae = VAE(input_dim=n_feat, latent_dim=8)
    vae.fit(X_tr, X_val)
    vae.fit_score_stats(X_tr)
    vae.fit_latent_stats(X_tr)
    vae.fit_threshold(X_val)
    vae.save(os.path.join(ABLATION_DIR,"vae_model.pt"))

    #    Train IF                                                           
    log.info("Training Isolation Forest ...")
    iforest = IsolationForestModel(n_estimators=100, max_samples=256)
    iforest.fit(X_tr)
    iforest.fit_threshold(X_val)
    iforest.save(os.path.join(ABLATION_DIR,"if_model.pkl"))

    # Global val threshold
    val_scores = ens_scores(ae, vae, iforest, X_val)
    tau_global = float(np.percentile(val_scores, 95))
    del X_tr, X_val; gc.collect()

    #    Evaluate                                                           
    log.info("=" * 60)
    log.info("Ablation (b) Evaluation")
    log.info("=" * 60)

    results = []
    eval_sets = [
        ("CIC2017",  lambda: load_cic_all(DATASET_PATHS["cic2017"])),
        ("CIC2018",  lambda: load_cic_all(DATASET_PATHS["cic2018"],rename=CIC2018_RENAME)),
        ("UNSW",     lambda: load_unsw_all()),
        ("BoT-IoT",  lambda: load_botiot()),
    ]

    for name, loader in eval_sets:
        log.info("Evaluating %s ...", name)
        df = loader()
        for f in selected:
            if f not in df.columns: df[f]=0.0
        X_ev = scale(df[selected].values.astype(np.float32))
        y_ev = df["label"].values
        r    = evaluate(ae, vae, iforest, X_ev, y_ev, name, tau_global)
        results.append(r)
        del df, X_ev; gc.collect()

    #    Save results                                                       
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(ABLATION_DIR,"ablation_b_results.csv"), index=False)

    #    Print ablation summary                                             
    print("")
    print("=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print("  Condition (a): CIC2017-only training  + IF   [v1 results]")
    print("  Condition (b): Combined training      + IF   [this script]")
    print("  Condition (c): Combined training      + SVDD [v2 results]")
    print("")
    print("  " + f"{'Dataset':<20}" +
          f"{'(a) AUC':>10}" + f"{'(b) AUC':>10}" + f"{'(c) AUC':>10}" +
          f"{' (a b)':>10}" + f"{' (b c)':>10}")
    print("  " + "-" * 72)

    # v1 and v2 AUCs from previous results
    v1_auc = {"CIC2017":0.8803,"CIC2018":0.5019,"UNSW":0.6464,"BoT-IoT":float("nan")}
    v2_auc = {"CIC2017":0.9370,"CIC2018":0.9564,"UNSW":0.8657,"BoT-IoT":float("nan")}

    for r in results:
        ds   = r["dataset"]
        a    = v1_auc.get(ds, float("nan"))
        b    = r["auc"]
        c    = v2_auc.get(ds, float("nan"))
        d_ab = b-a if not (b!=b or a!=a) else float("nan")
        d_bc = c-b if not (c!=c or b!=b) else float("nan")
        def fmt(v): return f"{v:>10.4f}" if not (isinstance(v,float) and v!=v) else "       nan"
        def fmtd(v): return f"{v:>+10.4f}" if not (isinstance(v,float) and v!=v) else "       nan"
        print("  " + f"{ds:<20}" + fmt(a) + fmt(b) + fmt(c) + fmtd(d_ab) + fmtd(d_bc))

    print("  " + "-" * 72)
    print("   (a b) = effect of combined training (IF unchanged)")
    print("   (b c) = effect of replacing IF with Deep SVDD")
    print("")
    log.info("Done. Results saved -> %s", ABLATION_DIR)


if __name__ == "__main__":
    run()
