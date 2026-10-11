"""
train_vae_if.py - VAE and IF under same conditions as AE/DAE/SVDD
5 seeds, single-env and multi-env full conditions
"""
import os, time, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
SEEDS = [42, 123, 456, 789, 999]

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(18,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU(),
                                 nn.Linear(32,16),nn.ReLU())
        self.mu  = nn.Linear(16,8)
        self.lv  = nn.Linear(16,8)
        self.dec = nn.Sequential(nn.Linear(8,16),nn.ReLU(),
                                 nn.Linear(16,32),nn.ReLU(),
                                 nn.Linear(32,64),nn.ReLU(),
                                 nn.Linear(64,18))
    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.exp(0.5*lv)*torch.randn_like(mu)
        return self.dec(z), mu, lv

def train_vae(Xtr, Xva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = VAE()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    Xt = torch.FloatTensor(Xtr); Xv = torch.FloatTensor(Xva)
    best, pat, bs = float('inf'), 0, None
    for ep in range(100):
        m.train()
        idx = torch.randperm(len(Xt))
        for i in range(0,len(Xt),256):
            b = Xt[idx[i:i+256]]; opt.zero_grad()
            r,mu,lv = m(b)
            loss = nn.functional.huber_loss(r,b,delta=1.0) + \
                   (-0.5*(1+lv-mu.pow(2)-lv.exp())).mean()
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            r,mu,lv = m(Xv)
            vl = (nn.functional.huber_loss(r,Xv,delta=1.0) +
                  (-0.5*(1+lv-mu.pow(2)-lv.exp())).mean()).item()
        sch.step(vl)
        if vl < best: best=vl; bs={k:v.clone() for k,v in m.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=10: log.info("  Early stop ep %d",ep+1); break
    m.load_state_dict(bs)
    return m

def score_vae(m, X):
    m.eval(); out=[]
    with torch.no_grad():
        Xt=torch.FloatTensor(X)
        for i in range(0,len(Xt),10000):
            b=Xt[i:i+10000]; r,_,_=m(b)
            out.append(((b-r)**2).mean(dim=1).numpy())
    return np.concatenate(out)

def norm(s, p1, p99):
    return np.clip((s-p1)/(p99-p1+1e-8),0,1)

def run():
    log.info("Loading data...")
    d=np.load(os.path.join(DEDUP,"train_dedup.npz"))
    Xfull=np.clip(d["X_train"],-10,10).astype(np.float32)
    Xsingle=Xfull[:len(Xfull)//3]
    d=np.load(os.path.join(DEDUP,"val_dedup.npz"))
    Vfull=np.clip(d["X_val"],-10,10).astype(np.float32)
    Vsingle=Vfull[:len(Vfull)//3]

    tests={}
    for n,f in [("CIC17","test_cic17_dedup.npz"),("CIC18","test_cic18_dedup.npz"),
                ("UNSW","test_unsw_dedup.npz"),("BoT","test_botiot_dedup.npz")]:
        d=np.load(os.path.join(DEDUP,f))
        tests[n]=(np.clip(d["X"],-10,10).astype(np.float32),d["y"].astype(int))
        log.info("Loaded %s: %d rows",n,len(tests[n][0]))

    rows=[]

    # VAE experiments
    for cond,Xtr,Xva in [("VAE-Single",Xsingle,Vsingle),("VAE-Multi",Xfull,Vfull)]:
        log.info("="*50); log.info("%s",cond)
        for seed in SEEDS:
            log.info("  Seed %d...",seed); t0=time.time()
            m=train_vae(Xtr,Xva,seed)
            trs=score_vae(m,Xtr[:50000])
            p1=float(np.percentile(trs,1)); p99=float(np.percentile(trs,99))
            vs=norm(score_vae(m,Xva),p1,p99); tau=float(np.percentile(vs,95))
            row={"condition":cond,"seed":seed}
            for n,(Xt,yt) in tests.items():
                s=norm(score_vae(m,Xt),p1,p99)
                if n=="BoT": row["BoT_DR"]=float((s>tau).mean())
                else: row[n+"_AUC"]=roc_auc_score(yt,s)
                log.info("    %s: %.4f",n,row.get(n+"_AUC",row.get("BoT_DR")))
            log.info("  Done %.1f min",(time.time()-t0)/60); rows.append(row)

    # IF experiments
    for cond,Xtr,Xva in [("IF-Single",Xsingle,Vsingle),("IF-Multi",Xfull,Vfull)]:
        log.info("="*50); log.info("%s",cond)
        for seed in SEEDS:
            log.info("  Seed %d...",seed); t0=time.time()
            m=IsolationForest(n_estimators=100,contamination=0.05,
                              random_state=seed,n_jobs=-1)
            m.fit(Xtr)
            trs=-m.score_samples(Xtr[:50000])
            p1=float(np.percentile(trs,1)); p99=float(np.percentile(trs,99))
            vs=norm(-m.score_samples(Xva),p1,p99); tau=float(np.percentile(vs,95))
            row={"condition":cond,"seed":seed}
            for n,(Xt,yt) in tests.items():
                s=norm(-m.score_samples(Xt),p1,p99)
                if n=="BoT": row["BoT_DR"]=float((s>tau).mean())
                else: row[n+"_AUC"]=roc_auc_score(yt,s)
                log.info("    %s: %.4f",n,row.get(n+"_AUC",row.get("BoT_DR")))
            log.info("  Done %.1f min",(time.time()-t0)/60); rows.append(row)

    df=pd.DataFrame(rows)
    out=os.path.join(DEDUP,"vae_if_results.csv")
    df.to_csv(out,index=False)
    log.info("Saved -> %s",out)

    print("\n"+"="*60)
    print("RESULTS (mean +/- pop std, 5 seeds)")
    print("="*60)
    for cond in ["VAE-Single","VAE-Multi","IF-Single","IF-Multi"]:
        sub=df[df["condition"]==cond]
        c17=sub["CIC17_AUC"].values; c18=sub["CIC18_AUC"].values
        uns=sub["UNSW_AUC"].values;  bot=sub["BoT_DR"].values
        print(f"{cond}:")
        print(f"  CIC17={c17.mean():.3f}+/-{c17.std(ddof=0):.3f}  "
              f"CIC18={c18.mean():.3f}+/-{c18.std(ddof=0):.3f}  "
              f"UNSW={uns.mean():.3f}+/-{uns.std(ddof=0):.3f}  "
              f"BoT={bot.mean():.3f}+/-{bot.std(ddof=0):.3f}")

    print("\nDiversity effect (UNSW AUC):")
    for base in ["VAE","IF"]:
        s=df[df["condition"]==base+"-Single"]["UNSW_AUC"].mean()
        m=df[df["condition"]==base+"-Multi"]["UNSW_AUC"].mean()
        print(f"  {base}: {s:.3f} -> {m:.3f} ({m-s:+.3f})")

if __name__=="__main__":
    run()
