"""
compute_additional_metrics.py
=============================
Computes three sets of additional metrics
from existing DAE-Multi-full models:

1. PR-AUC (Precision-Recall AUC)
2. Threshold sensitivity (p90, p95, p97.5, p99)
3. MMD and Wasserstein distance between
   training environment feature distributions

Uses existing dedup npz files — no retraining.
Loads saved DAE-Multi seed 42 model if available,
otherwise retrains once.

Output:
  artefacts_v2/dedup/pr_auc_results.csv
  artefacts_v2/dedup/threshold_sensitivity.csv
  artefacts_v2/dedup/distribution_distances.csv

Expected time: ~1.5 hours
"""
import os, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, f1_score)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
SEEDS = [42, 123, 456, 789, 999]
NOISE = 0.1


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


def mmd_rbf(X, Y, sigma=1.0, n=2000):
    """Maximum Mean Discrepancy with RBF kernel."""
    rng = np.random.default_rng(42)
    if len(X) > n: X = X[rng.choice(len(X),n,replace=False)]
    if len(Y) > n: Y = Y[rng.choice(len(Y),n,replace=False)]
    XX = np.sum((X[:,None,:]-X[None,:,:])**2,axis=-1)
    YY = np.sum((Y[:,None,:]-Y[None,:,:])**2,axis=-1)
    XY = np.sum((X[:,None,:]-Y[None,:,:])**2,axis=-1)
    s2 = 2*sigma**2
    return (np.exp(-XX/s2).mean() +
            np.exp(-YY/s2).mean() -
            2*np.exp(-XY/s2).mean())


def wasserstein1d(X, Y):
    """Mean per-feature 1D Wasserstein distance."""
    dists = []
    for j in range(X.shape[1]):
        xs = np.sort(X[:,j])
        ys = np.sort(Y[:,j])
        # Interpolate to same length
        t = np.linspace(0,1,min(len(xs),len(ys)))
        xi = np.interp(t, np.linspace(0,1,len(xs)), xs)
        yi = np.interp(t, np.linspace(0,1,len(ys)), ys)
        dists.append(np.mean(np.abs(xi-yi)))
    return np.mean(dists)


def run():
    log.info("Loading data...")
    d=np.load(os.path.join(DEDUP,"train_dedup.npz"))
    Xfull=np.clip(d["X_train"],-10,10).astype(np.float32)
    n=len(Xfull)//3
    env={'CIC17':Xfull[:n], 'CIC18':Xfull[n:2*n], 'UNSW':Xfull[2*n:]}

    d=np.load(os.path.join(DEDUP,"val_dedup.npz"))
    Vfull=np.clip(d["X_val"],-10,10).astype(np.float32)

    tests={}
    for name,f in [("CIC17","test_cic17_dedup.npz"),
                   ("CIC18","test_cic18_dedup.npz"),
                   ("UNSW","test_unsw_dedup.npz")]:
        d=np.load(os.path.join(DEDUP,f))
        tests[name]=(np.clip(d["X"],-10,10).astype(np.float32),
                     d["y"].astype(int))

    # ── 1. Distribution distances ──────────────────────────────────────────
    log.info("Computing distribution distances...")
    pairs = [('CIC17','CIC18'), ('CIC17','UNSW'), ('CIC18','UNSW')]
    dist_rows = []
    for a,b in pairs:
        mmd  = mmd_rbf(env[a], env[b])
        wass = wasserstein1d(env[a], env[b])
        log.info("  %s vs %s: MMD=%.4f Wasserstein=%.4f", a, b, mmd, wass)
        dist_rows.append({'pair':f'{a} vs {b}', 'MMD':mmd, 'Wasserstein':wass})
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(DEDUP,"distribution_distances.csv"), index=False)
    log.info("Distribution distances saved")

    # ── 2. PR-AUC and threshold sensitivity ───────────────────────────────
    log.info("Training DAE-Multi-full (5 seeds) for PR-AUC...")
    pr_rows = []; thr_rows = []
    thresholds = [0.90, 0.925, 0.95, 0.975, 0.99]

    for seed in SEEDS:
        log.info("  Seed %d...", seed); t0=time.time()
        m = train_dae(Xfull, Vfull, seed)
        trs = get_scores(m, Xfull[:50000])
        p1=float(np.percentile(trs,1)); p99=float(np.percentile(trs,99))
        vs = norm(get_scores(m,Vfull),p1,p99)

        for ds,(Xt,yt) in tests.items():
            s = norm(get_scores(m,Xt),p1,p99)
            roc = roc_auc_score(yt,s)
            pr  = average_precision_score(yt,s)
            pr_rows.append({'seed':seed,'dataset':ds,
                            'ROC_AUC':roc,'PR_AUC':pr})
            log.info("    %s: ROC=%.4f PR=%.4f", ds, roc, pr)

            # Threshold sensitivity
            for pct in thresholds:
                tau = float(np.percentile(vs, pct*100))
                yp  = (s>tau).astype(int)
                f1  = f1_score(yt,yp,zero_division=0)
                fpr = float(((yt==0)&(yp==1)).sum()/max(1,(yt==0).sum()))
                tpr = float(((yt==1)&(yp==1)).sum()/max(1,(yt==1).sum()))
                thr_rows.append({'seed':seed,'dataset':ds,
                                 'threshold_pct':pct,
                                 'tau':tau,'F1':f1,
                                 'FPR':fpr,'TPR':tpr})

        log.info("  Seed %d done %.1f min", seed, (time.time()-t0)/60)

    pd.DataFrame(pr_rows).to_csv(
        os.path.join(DEDUP,"pr_auc_results.csv"), index=False)
    pd.DataFrame(thr_rows).to_csv(
        os.path.join(DEDUP,"threshold_sensitivity.csv"), index=False)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n"+"="*60)
    print("PR-AUC RESULTS (mean ± std, 5 seeds)")
    print("="*60)
    df=pd.DataFrame(pr_rows)
    for ds in ['CIC17','CIC18','UNSW']:
        sub=df[df['dataset']==ds]
        roc=sub['ROC_AUC'].values; pr=sub['PR_AUC'].values
        print(f"{ds}: ROC={roc.mean():.3f}±{roc.std(ddof=0):.3f}  "
              f"PR={pr.mean():.3f}±{pr.std(ddof=0):.3f}")

    print("\n"+"="*60)
    print("THRESHOLD SENSITIVITY (mean F1, 5 seeds, UNSW)")
    print("="*60)
    df2=pd.DataFrame(thr_rows)
    unsw=df2[df2['dataset']=='UNSW']
    for pct in thresholds:
        sub=unsw[unsw['threshold_pct']==pct]
        print(f"  p{pct*100:.0f}: F1={sub['F1'].mean():.3f}  "
              f"FPR={sub['FPR'].mean():.3f}  "
              f"TPR={sub['TPR'].mean():.3f}")

    print("\n"+"="*60)
    print("DISTRIBUTION DISTANCES")
    print("="*60)
    for r in dist_rows:
        print(f"  {r['pair']}: MMD={r['MMD']:.4f}  "
              f"Wasserstein={r['Wasserstein']:.4f}")

    print("\nRelate to LODO performance:")
    print("  LODO-1 held-out CIC17 F1=0.826 (CIC17 vs CIC18+UNSW)")
    print("  LODO-2 held-out CIC18 F1=0.882 (CIC18 vs CIC17+UNSW)")
    print("  LODO-3 held-out UNSW  F1=0.049 (UNSW vs CIC17+CIC18)")
    print("  → Higher distance from CIC17/CIC18 may explain UNSW failure")


if __name__=="__main__":
    run()
