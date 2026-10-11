"""
run_lodo_multiseed.py
=====================
Runs LODO experiments across all 5 seeds.

Output:
  artefacts_v2/dedup/lodo_multiseed_results.csv

Run from project root:
    python run_lodo_multiseed.py

Expected time: ~6 hours (3 LODO x 5 seeds)
"""
import sys, os, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
DEDUP_DIR    = os.path.join(ARTEFACT_DIR, "dedup")
SEEDS        = [42, 123, 456, 789, 999]
INPUT_DIM    = 18
NOISE_STD    = 0.1

# LODO configurations
LODO_CONFIGS = [
    {
        'name':     'LODO-1 (no CIC17)',
        'held_out': 'CIC17',
        'envs':     [1, 2],   # CIC18 + UNSW (indices in train_dedup.npz)
    },
    {
        'name':     'LODO-2 (no CIC18)',
        'held_out': 'CIC18',
        'envs':     [0, 2],   # CIC17 + UNSW
    },
    {
        'name':     'LODO-3 (no UNSW)',
        'held_out': 'UNSW',
        'envs':     [0, 1],   # CIC17 + CIC18
    },
]


class DAEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(18,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,8),  nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8,16),  nn.ReLU(),
            nn.Linear(16,32), nn.ReLU(),
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64,18),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_dae(X_train, X_val, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = DAEModel()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    crit  = nn.HuberLoss(delta=1.0)
    X_t   = torch.FloatTensor(X_train)
    X_v   = torch.FloatTensor(X_val)
    best, pat, bs = float('inf'), 0, None
    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            b = X_t[idx[i:i+256]]
            n = torch.clamp(b+torch.randn_like(b)*NOISE_STD,-10,10)
            opt.zero_grad(); crit(model(n),b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            nv = torch.clamp(X_v+torch.randn_like(X_v)*NOISE_STD,-10,10)
            vl = crit(model(nv),X_v).item()
        sched.step(vl)
        if vl < best:
            best=vl; bs={k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=10: log.info("  Early stop ep %d",epoch+1); break
    model.load_state_dict(bs)
    return model


def get_scores(model, X):
    model.eval(); out=[]
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        for i in range(0,len(X_t),10000):
            b=X_t[i:i+10000]
            out.append(((b-model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def norm(s, p1, p99):
    return np.clip((s-p1)/(p99-p1+1e-8), 0, 1)


def run():
    log.info("Loading dedup data...")
    d = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_full = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n_per  = len(X_full) // 3
    # Split by environment
    env_train = {
        0: X_full[:n_per],           # CIC17
        1: X_full[n_per:2*n_per],    # CIC18
        2: X_full[2*n_per:],         # UNSW
    }

    d = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full = np.clip(d["X_val"], -10, 10).astype(np.float32)
    n_val = len(X_val_full) // 3
    env_val = {
        0: X_val_full[:n_val],
        1: X_val_full[n_val:2*n_val],
        2: X_val_full[2*n_val:],
    }

    # Test sets
    test_files = {
        'CIC17': 'test_cic17_dedup.npz',
        'CIC18': 'test_cic18_dedup.npz',
        'UNSW':  'test_unsw_dedup.npz',
    }
    test_sets = {}
    for name, fname in test_files.items():
        d = np.load(os.path.join(DEDUP_DIR, fname))
        test_sets[name] = (
            np.clip(d["X"], -10, 10).astype(np.float32),
            d["y"].astype(int))
        log.info("Loaded %s: %d rows", name, len(test_sets[name][0]))

    results = []

    for cfg in LODO_CONFIGS:
        lodo_name = cfg['name']
        held_out  = cfg['held_out']
        envs      = cfg['envs']

        log.info("="*55)
        log.info("%s", lodo_name)

        # Build training and validation sets
        X_tr = np.concatenate([env_train[e] for e in envs])
        X_va = np.concatenate([env_val[e]   for e in envs])
        log.info("  Train: %d  Val: %d", len(X_tr), len(X_va))

        for seed in SEEDS:
            log.info("  Seed %d...", seed)
            t0 = time.time()

            model = train_dae(X_tr, X_va, seed)
            train_s = get_scores(model, X_tr[:50000])
            p1  = float(np.percentile(train_s, 1))
            p99 = float(np.percentile(train_s, 99))

            val_s = norm(get_scores(model, X_va), p1, p99)
            tau   = float(np.percentile(val_s, 95))

            row = {
                'experiment': lodo_name,
                'held_out':   held_out,
                'seed':       seed,
                'tau':        tau,
            }

            # Evaluate all three test sets
            for ds_name, (X_test, y_test) in test_sets.items():
                s = norm(get_scores(model, X_test), p1, p99)
                auc = roc_auc_score(y_test, s)
                y_pred = (s > tau).astype(int)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                row[f'{ds_name}_AUC'] = auc
                row[f'{ds_name}_F1']  = f1

                is_held = (ds_name == held_out)
                if is_held:
                    log.info("    [HELD-OUT] %s: AUC=%.4f F1=%.4f",
                             ds_name, auc, f1)
                else:
                    log.info("    %s: AUC=%.4f F1=%.4f", ds_name, auc, f1)

            log.info("  Seed %d done in %.1f min", seed, (time.time()-t0)/60)
            results.append(row)

    df = pd.DataFrame(results)
    out = os.path.join(DEDUP_DIR, "lodo_multiseed_results.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)

    # Summary
    print("\n" + "="*65)
    print("LODO MULTI-SEED SUMMARY (mean ± std, 5 seeds)")
    print("="*65)

    for cfg in LODO_CONFIGS:
        name     = cfg['name']
        held_out = cfg['held_out']
        sub      = df[df['experiment']==name]
        auc_col  = f'{held_out}_AUC'
        f1_col   = f'{held_out}_F1'
        print(f"\n{name}:")
        print(f"  Held-out {held_out}:")
        print(f"    AUC: {sub[auc_col].mean():.4f} ± {sub[auc_col].std(ddof=0):.4f}")
        print(f"    F1:  {sub[f1_col].mean():.4f} ± {sub[f1_col].std(ddof=0):.4f}")
        print(f"    Seed values: {sub[auc_col].values.round(3).tolist()}")


if __name__ == "__main__":
    run()
