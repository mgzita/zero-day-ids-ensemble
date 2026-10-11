"""
yj_quick.py
===========
Quick single-seed comparison: DAE-Multi with vs without YJ.
Uses existing seed 42 NoYJ result from CSV.
Only trains YJ version (~40 min).

Run: python yj_quick.py
"""
import numpy as np, torch, torch.nn as nn, time
from scipy import stats as sp
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    level=logging.INFO)
log = logging.getLogger(__name__)

DEDUP = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
SEEDS = [42, 123, 456, 789, 999]
NOISE = 0.1

# Existing NoYJ 5-seed results (from ablation_dae_dedup.csv)
NOYJ_SEEDS = {
    42:  {'CIC17': 0.9477, 'CIC18': 0.9065, 'UNSW': 0.9394, 'BoT': 0.7810},
    123: {'CIC17': 0.9375, 'CIC18': 0.9519, 'UNSW': 0.9403, 'BoT': 0.5901},
    456: {'CIC17': 0.9192, 'CIC18': 0.9972, 'UNSW': 0.9573, 'BoT': 0.5810},
    789: {'CIC17': 0.9121, 'CIC18': 0.9713, 'UNSW': 0.9448, 'BoT': 0.7795},
    999: {'CIC17': 0.9078, 'CIC18': 0.9959, 'UNSW': 0.9342, 'BoT': 0.6631},
}

class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(18,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,16),nn.ReLU(),nn.Linear(16,8), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(8,16), nn.ReLU(),nn.Linear(16,32),nn.ReLU(),
            nn.Linear(32,64),nn.ReLU(),nn.Linear(64,18))
    def forward(self,x): return self.dec(self.enc(x))

def yj_fit_transform(X_tr, *others):
    lams = []
    out_tr = np.zeros_like(X_tr)
    for j in range(X_tr.shape[1]):
        _, lam = sp.yeojohnson(X_tr[:,j])
        lams.append(lam)
        out_tr[:,j] = sp.yeojohnson(X_tr[:,j], lmbda=lam)
    results = [np.clip(out_tr,-10,10).astype(np.float32)]
    for X in others:
        out = np.zeros_like(X)
        for j in range(X.shape[1]):
            out[:,j] = sp.yeojohnson(X[:,j], lmbda=lams[j])
        results.append(np.clip(out,-10,10).astype(np.float32))
    return results

def score(model, X):
    model.eval(); out=[]
    with torch.no_grad():
        Xt=torch.FloatTensor(X)
        for i in range(0,len(Xt),10000):
            b=Xt[i:i+10000]
            out.append(((b-model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)

def norm(s, p1, p99):
    return np.clip((s-p1)/(p99-p1+1e-8),0,1)

log.info("Loading data...")
d=np.load(f'{DEDUP}/train_dedup.npz')
X_tr=np.clip(d['X_train'],-10,10).astype(np.float32)
d=np.load(f'{DEDUP}/val_dedup.npz')
X_va=np.clip(d['X_val'],-10,10).astype(np.float32)

tests={}
for name,f in [('CIC17','test_cic17_dedup.npz'),('CIC18','test_cic18_dedup.npz'),
               ('UNSW','test_unsw_dedup.npz'),('BoT','test_botiot_dedup.npz')]:
    d=np.load(f'{DEDUP}/{f}')
    tests[name]=(np.clip(d['X'],-10,10).astype(np.float32), d['y'].astype(int))

log.info("Fitting Yeo-Johnson on training data...")
X_tr_yj, X_va_yj, *X_ts_yj = yj_fit_transform(
    X_tr, X_va,
    tests['CIC17'][0], tests['CIC18'][0],
    tests['UNSW'][0],  tests['BoT'][0])
X_ts_yj_dict = dict(zip(['CIC17','CIC18','UNSW','BoT'], X_ts_yj))
log.info("YJ complete. Training DAE-Multi with YJ (5 seeds)...")

all_yj = {name: [] for name in ['CIC17','CIC18','UNSW','BoT']}

for SEED in SEEDS:
    log.info("  Seed %d...", SEED)
    torch.manual_seed(SEED); np.random.seed(SEED)
    model=DAE()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5)
    crit=nn.HuberLoss(delta=1.0)
    Xt=torch.FloatTensor(X_tr_yj); Xv=torch.FloatTensor(X_va_yj)
    best,pat,bs=float('inf'),0,None
    t0=time.time()
    for ep in range(100):
        model.train()
        idx=torch.randperm(len(Xt))
        for i in range(0,len(Xt),256):
            b=Xt[idx[i:i+256]]
            n=torch.clamp(b+torch.randn_like(b)*NOISE,-10,10)
            opt.zero_grad(); crit(model(n),b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            nv=torch.clamp(Xv+torch.randn_like(Xv)*NOISE,-10,10)
            vl=crit(model(nv),Xv).item()
        sched.step(vl)
        if vl<best: best=vl; bs={k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=10: log.info("    Early stop ep %d",ep+1); break
    model.load_state_dict(bs)
    log.info("    Done in %.1f min", (time.time()-t0)/60)

    tr_s = score(model, X_tr_yj[:50000])
    p1=float(np.percentile(tr_s,1)); p99=float(np.percentile(tr_s,99))
    val_s = norm(score(model, X_va_yj), p1, p99)
    tau = float(np.percentile(val_s, 95))

    for name in ['CIC17','CIC18','UNSW']:
        s = norm(score(model, X_ts_yj_dict[name]), p1, p99)
        auc = roc_auc_score(tests[name][1], s)
        all_yj[name].append(auc)
        log.info("    %s AUC=%.4f", name, auc)

    s_bot = norm(score(model, X_ts_yj_dict['BoT']), p1, p99)
    dr = float((s_bot > tau).mean())
    all_yj['BoT'].append(dr)
    log.info("    BoT DR=%.4f tau=%.4f", dr, tau)

yj_results = {k: np.mean(v) for k,v in all_yj.items()}
yj_std     = {k: np.std(v, ddof=0) for k,v in all_yj.items()}

print("\n" + "="*55)
print("YEO-JOHNSON ABLATION (seed 42, DAE-Multi-full)")
print("="*55)
# results displayed below

# Compute NoYJ means from CSV values
noyj_mean = {
    'CIC17': np.mean([NOYJ_SEEDS[s]['CIC17'] for s in SEEDS]),
    'CIC18': np.mean([NOYJ_SEEDS[s]['CIC18'] for s in SEEDS]),
    'UNSW':  np.mean([NOYJ_SEEDS[s]['UNSW']  for s in SEEDS]),
    'BoT':   np.mean([NOYJ_SEEDS[s]['BoT']   for s in SEEDS]),
}
noyj_std = {
    'CIC17': np.std([NOYJ_SEEDS[s]['CIC17'] for s in SEEDS], ddof=0),
    'CIC18': np.std([NOYJ_SEEDS[s]['CIC18'] for s in SEEDS], ddof=0),
    'UNSW':  np.std([NOYJ_SEEDS[s]['UNSW']  for s in SEEDS], ddof=0),
    'BoT':   np.std([NOYJ_SEEDS[s]['BoT']   for s in SEEDS], ddof=0),
}

print(f"{'Dataset':<10} {'NoYJ mean±std':>16} {'YJ mean±std':>16} {'Delta':>8}")
print("-"*55)
for name in ['CIC17','CIC18','UNSW','BoT']:
    nm = noyj_mean[name]; ns = noyj_std[name]
    ym = yj_results[name]; ys = yj_std[name]
    delta = ym - nm
    print(f"{name:<10} {nm:.3f}±{ns:.3f}     {ym:.3f}±{ys:.3f}   {delta:>+8.4f}")

print()
unsw_delta = yj_results['UNSW'] - noyj_mean['UNSW']
if abs(unsw_delta) < 0.01:
    print("VERDICT: YJ does NOT materially improve results.")
    print(f"UNSW delta = {unsw_delta:+.4f} (< 0.01 threshold)")
    print("The diversity effect is robust to preprocessing choice.")
elif abs(unsw_delta) < 0.02:
    print(f"VERDICT: YJ provides marginal improvement (+{unsw_delta:.4f} UNSW AUC).")
    print("Improvement is small relative to diversity effect (+0.436).")
else:
    print(f"VERDICT: YJ provides meaningful improvement ({unsw_delta:+.4f} UNSW AUC).")
