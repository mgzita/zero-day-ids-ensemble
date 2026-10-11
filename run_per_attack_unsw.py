"""
compute_per_attack_b.py
=======================
Per-attack-category recall on UNSW-NB15
using UNSW_NB15_testing-set.csv directly.

Applies the same IQR scaler fitted on training
data (iqr_scaler.npz) to the testing-set.csv
features, then scores with DAE-Multi-full.

This is the working condition (AUC~0.93)
because UNSW benign IS in the training pool.

Output:
  artefacts_v2/dedup/per_attack_results_b.csv

Expected time: ~3.5 hours (5 seeds x ~40 min)
"""
import os, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP    = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
ART_DIR  = r"C:/MLProject/zero_day_project/artefacts_v2"
UNSW_CSV = r"C:/MLProject/zero_day_project/data/raw/UNSW/Training and Testing Sets/UNSW_NB15_testing-set.csv"
SEEDS    = [42, 123, 456, 789, 999]
NOISE    = 0.1

FEATURES = [
    'dur', 'sbytes', 'dbytes', 'rate', 'sinpkt', 'dinpkt',
    'sjit', 'djit', 'smean', 'dmean', 'swin', 'dwin',
    'ct_srv_src', 'ct_srv_dst', 'spkts', 'dpkts',
    'ct_src_ltm', 'dttl'
]

FEAT_RENAME = {
    'dur':       0,  'sbytes':    1,  'dbytes':    2,
    'rate':      3,  'sinpkt':    4,  'dinpkt':    5,
    'sjit':      6,  'djit':      7,  'smean':     8,
    'dmean':     9,  'swin':      10, 'dwin':      11,
    'ct_srv_src':12, 'ct_srv_dst':13, 'spkts':     14,
    'dpkts':     15, 'ct_src_ltm':16, 'dttl':      17,
}


class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(18,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(8,16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),
            nn.Linear(32,64),nn.ReLU(),nn.Linear(64,18))
    def forward(self,x): return self.dec(self.enc(x))


def train_dae(Xtr, Xva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m=DAE(); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5)
    crit=nn.HuberLoss(delta=1.0)
    Xt=torch.FloatTensor(Xtr); Xv=torch.FloatTensor(Xva)
    best,pat,bs=float('inf'),0,None
    for ep in range(100):
        m.train()
        idx=torch.randperm(len(Xt))
        for i in range(0,len(Xt),256):
            b=Xt[idx[i:i+256]]
            n=torch.clamp(b+torch.randn_like(b)*NOISE,-10,10)
            opt.zero_grad(); crit(m(n),b).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            nv=torch.clamp(Xv+torch.randn_like(Xv)*NOISE,-10,10)
            vl=crit(m(nv),Xv).item()
        sch.step(vl)
        if vl<best: best=vl; bs={k:v.clone() for k,v in m.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=10: log.info("  Early stop ep %d",ep+1); break
    m.load_state_dict(bs); return m


def get_scores(m, X):
    m.eval(); out=[]
    with torch.no_grad():
        Xt=torch.FloatTensor(X)
        for i in range(0,len(Xt),10000):
            b=Xt[i:i+10000]
            out.append(((b-m(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def norm(s,p1,p99):
    return np.clip((s-p1)/(p99-p1+1e-8),0,1)


def load_unsw_testset():
    """Load testing-set.csv and apply training IQR scaler."""
    log.info("Loading UNSW_NB15_testing-set.csv...")
    df = pd.read_csv(UNSW_CSV)
    df.columns = df.columns.str.strip().str.lower()

    # Load IQR scaler fitted on training data
    scaler = np.load(os.path.join(ART_DIR, "iqr_scaler.npz"))
    q2  = scaler['q2']
    iqr = scaler['iqr']
    iqr[iqr==0] = 1.0

    # Extract features — use available columns
    X = np.zeros((len(df), 18), dtype=np.float32)
    for feat, idx in FEAT_RENAME.items():
        if feat in df.columns:
            col = pd.to_numeric(df[feat], errors='coerce').fillna(0).values
            X[:, idx] = col

    X = np.where(np.isfinite(X), X, 0.0)
    X = np.clip((X - q2) / iqr, -10, 10).astype(np.float32)

    y    = df['label'].values.astype(int)
    cats = df['attack_cat'].fillna('Normal').values

    log.info("Testing-set: %d rows, %d benign, %d attack",
             len(y), (y==0).sum(), (y==1).sum())
    log.info("Categories: %s",
             dict(pd.Series(cats[y==1]).value_counts()))
    return X, y, cats


def run():
    log.info("Loading training data (DAE-Multi-full)...")
    d=np.load(os.path.join(DEDUP,"train_dedup.npz"))
    Xtr=np.clip(d["X_train"],-10,10).astype(np.float32)
    d=np.load(os.path.join(DEDUP,"val_dedup.npz"))
    Xva=np.clip(d["X_val"],-10,10).astype(np.float32)

    # Load testing-set.csv directly — has both X and attack categories
    log.info("Loading UNSW_NB15_testing-set.csv...")
    df_csv = pd.read_csv(UNSW_CSV)
    df_csv.columns = df_csv.columns.str.strip().str.lower()

    scaler = np.load(os.path.join(
        os.path.dirname(DEDUP), "iqr_scaler.npz"))
    q2 = scaler["q2"]; iqr = scaler["iqr"]; iqr[iqr==0]=1.0

    X_raw = np.zeros((len(df_csv),18),dtype=np.float32)
    for feat,col_idx in FEAT_RENAME.items():
        if feat in df_csv.columns:
            X_raw[:,col_idx] = pd.to_numeric(
                df_csv[feat],errors="coerce").fillna(0).values
    X_raw = np.where(np.isfinite(X_raw),X_raw,0.0)
    X_test = np.clip((X_raw-q2)/iqr,-10,10).astype(np.float32)
    y_test = df_csv["label"].values.astype(int)
    cats   = df_csv["attack_cat"].fillna("Normal").values
    log.info("Testing-set: %d rows, %d benign, %d attack",
             len(y_test),(y_test==0).sum(),(y_test==1).sum())

    attack_cats = sorted([c for c in np.unique(cats)
                         if c not in ("Normal","normal","")])
    log.info("Attack categories: %s", attack_cats)

    rows = []

    for seed in SEEDS:
        log.info("Seed %d...",seed); t0=time.time()
        m = train_dae(Xtr,Xva,seed)
        trs = get_scores(m,Xtr[:50000])
        p1=float(np.percentile(trs,1)); p99=float(np.percentile(trs,99))
        vs = norm(get_scores(m,Xva),p1,p99)
        tau = float(np.percentile(vs,95))
        s = norm(get_scores(m,X_test),p1,p99)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test,s)
        log.info("  AUC=%.4f tau=%.4f",auc,tau)

        for cat in attack_cats:
            mask = (cats==cat)&(y_test==1)
            if mask.sum()==0: continue
            recall = float((s[mask]>tau).mean())
            rows.append({"seed":seed,"category":cat,
                         "n_flows":int(mask.sum()),
                         "recall":recall,"auc":auc})
            log.info("    %s: recall=%.3f (n=%d)",cat,recall,mask.sum())
        log.info("  Done %.1f min",(time.time()-t0)/60)

    df = pd.DataFrame(rows)
    out = os.path.join(DEDUP,"per_attack_results_b.csv")
    df.to_csv(out,index=False)
    log.info("Saved -> %s",out)

    print("\n"+"="*60)
    print("PER-ATTACK RECALL (DAE-Multi-full, 5 seeds)")
    print("="*60)
    print(f"{'Category':<20}{'N':>6}{'Recall':>10}{'Std':>8}")
    print("-"*46)
    for cat in attack_cats:
        sub=df[df["category"]==cat]
        if len(sub)==0: continue
        print(f"{cat:<20}{sub['n_flows'].iloc[0]:>6}"
              f"{sub['recall'].mean():>10.3f}"
              f"{sub['recall'].std(ddof=0):>8.3f}")



if __name__=="__main__":
    run()
