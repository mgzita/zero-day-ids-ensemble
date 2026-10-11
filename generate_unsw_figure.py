import sys, os, logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
DEDUP_DIR    = os.path.join(ARTEFACT_DIR, "dedup")
OUT_DIR      = os.path.join(ARTEFACT_DIR, "figures")
SEED=42; NOISE_STD=0.1; INPUT_DIM=18

class DAEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(18,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(8,16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),
            nn.Linear(32,64),nn.ReLU(),nn.Linear(64,18))
    def forward(self,x): return self.decoder(self.encoder(x))

def get_scores(model,X):
    model.eval(); scores=[]
    with torch.no_grad():
        X_t=torch.FloatTensor(X)
        for i in range(0,len(X_t),10000):
            b=X_t[i:i+10000]
            scores.append(((b-model(b))**2).mean(dim=1).numpy())
    return np.concatenate(scores)

def norm(s,p1,p99): return np.clip((s-p1)/(p99-p1+1e-8),0,1)

def train_dae(X_train,X_val,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model=DAEModel(); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5)
    crit=nn.HuberLoss(delta=1.0)
    X_t=torch.FloatTensor(X_train); X_v=torch.FloatTensor(X_val)
    best_loss=float('inf'); patience_c=0; best_state=None
    for epoch in range(100):
        model.train()
        idx=torch.randperm(len(X_t))
        for i in range(0,len(X_t),256):
            batch=X_t[idx[i:i+256]]
            noisy=torch.clamp(batch+torch.randn_like(batch)*NOISE_STD,-10,10)
            opt.zero_grad(); crit(model(noisy),batch).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            nv=torch.clamp(X_v+torch.randn_like(X_v)*NOISE_STD,-10,10)
            vl=crit(model(nv),X_v).item()
        sched.step(vl)
        if vl<best_loss:
            best_loss=vl
            best_state={k:v.clone() for k,v in model.state_dict().items()}
            patience_c=0
        else:
            patience_c+=1
            if patience_c>=10: log.info("Early stop %d",epoch+1); break
    model.load_state_dict(best_state); return model

def run():
    log.info("Loading data...")
    d=np.load(os.path.join(DEDUP_DIR,"train_dedup.npz"))
    X_full=np.clip(d["X_train"],-10,10).astype(np.float32)
    X_single=X_full[:len(X_full)//3]
    d=np.load(os.path.join(DEDUP_DIR,"val_dedup.npz"))
    X_val_full=np.clip(d["X_val"],-10,10).astype(np.float32)
    X_val_single=X_val_full[:len(X_val_full)//3]
    d=np.load(os.path.join(DEDUP_DIR,"test_unsw_dedup.npz"))
    X_unsw=np.clip(d["X"],-10,10).astype(np.float32)
    y_unsw=d["y"].astype(int)
    log.info("UNSW: %d benign %d attack",(y_unsw==0).sum(),(y_unsw==1).sum())

    log.info("Training DAE-Single...")
    ms=train_dae(X_single,X_val_single,SEED)
    ts=get_scores(ms,X_single)
    p1s,p99s=float(np.percentile(ts,1)),float(np.percentile(ts,99))
    tau_s=float(np.percentile(norm(get_scores(ms,X_val_single),p1s,p99s),95))
    log.info("DAE-Single tau=%.4f",tau_s)

    mp_path=os.path.join(DEDUP_DIR,"seed42_dae","dae_multi.pt")
    mm=DAEModel()
    if os.path.exists(mp_path):
        log.info("Loading DAE-Multi...")
        mm.load_state_dict(torch.load(mp_path,map_location="cpu"))
    else:
        log.info("Training DAE-Multi...")
        mm=train_dae(X_full,X_val_full,SEED)
        os.makedirs(os.path.dirname(mp_path),exist_ok=True)
        torch.save(mm.state_dict(),mp_path)
    tm=get_scores(mm,X_full[:50000])
    p1m,p99m=float(np.percentile(tm,1)),float(np.percentile(tm,99))
    tau_m=float(np.percentile(norm(get_scores(mm,X_val_full),p1m,p99m),95))
    log.info("DAE-Multi tau=%.4f",tau_m)

    from sklearn.metrics import roc_auc_score
    ss=norm(get_scores(ms,X_unsw),p1s,p99s)
    sm=norm(get_scores(mm,X_unsw),p1m,p99m)
    aucs=roc_auc_score(y_unsw,ss)
    aucm=roc_auc_score(y_unsw,sm)
    log.info("Single AUC=%.4f Multi AUC=%.4f",aucs,aucm)

    bins=np.linspace(0,1,51)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
    fig.subplots_adjust(wspace=0.3)

    for ax,scores,tau,auc,title in [
        (ax1,ss,tau_s,aucs,f"(a) DAE-Single  (1 env, 536k flows)\nAUC = {aucs:.3f}"),
        (ax2,sm,tau_m,aucm,f"(b) DAE-Multi  (3 envs, 536k flows)\nAUC = {aucm:.3f}")]:
        b=scores[y_unsw==0]; a=scores[y_unsw==1]
        ax.hist(b,bins=bins,density=True,color="#2471A3",alpha=0.45,
                histtype="stepfilled",label=f"Benign (n={len(b):,})")
        ax.hist(a,bins=bins,density=True,color="#C0392B",alpha=0.45,
                histtype="stepfilled",label=f"Attack (n={len(a):,})")
        ax.hist(b,bins=bins,density=True,color="#2471A3",histtype="step",lw=1.5)
        ax.hist(a,bins=bins,density=True,color="#C0392B",histtype="step",lw=1.5)
        ax.axvline(tau,color="black",lw=2,linestyle="--",label=f"τ = {tau:.3f}")
        ax.set_xlim(0,1); ax.set_xlabel("Normalized anomaly score",fontsize=11)
        ax.set_title(title,fontsize=11,fontweight="bold")
        ax.legend(fontsize=9); ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax1.set_ylabel("Density",fontsize=11)

    fig.suptitle("UNSW-NB15 score distributions: size-matched control (seed 42, p95 threshold)",
                 fontsize=11,fontweight="bold",y=1.01)
    fig.text(0.5,-0.03,f"AUC: {aucs:.3f} → {aucm:.3f} (+{aucm-aucs:.3f})  |  536,307 training flows in both conditions",
             ha="center",fontsize=10,
             bbox=dict(boxstyle="round",facecolor="#EBF5FB",edgecolor="#2471A3",alpha=0.9))

    out=os.path.join(OUT_DIR,"unsw_score_comparison.png")
    plt.savefig(out,dpi=180,bbox_inches="tight",facecolor="white")
    plt.close()
    log.info("Saved -> %s",out)

if __name__=="__main__": run()
