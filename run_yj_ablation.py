"""
run_yj_ablation.py
==================
Ablation: DAE-Multi-full with vs without
Yeo-Johnson variance-stabilizing transformation.

Same 5 seeds, same architecture, same hyperparameters.
Only difference: YJ transform applied after IQR scaling.

Output:
  artefacts_v2/dedup/yj_ablation_results.csv

Run from project root:
    python run_yj_ablation.py

Expected time: ~4 hours (2 conditions x 5 seeds)
"""
import sys, os, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEDUP_DIR = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
SEEDS     = [42, 123, 456, 789, 999]
NOISE_STD = 0.1


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


def apply_yj(X_train, X_val, X_tests):
    """
    Fit Yeo-Johnson lambda on training data (per feature),
    apply to all splits. Returns transformed arrays.
    """
    lambdas = []
    X_train_yj = np.zeros_like(X_train)
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        # scipy yeojohnson finds optimal lambda via MLE
        _, lam = scipy_stats.yeojohnson(col)
        lambdas.append(lam)
        X_train_yj[:, j] = scipy_stats.yeojohnson(col, lmbda=lam)

    X_val_yj = np.zeros_like(X_val)
    for j in range(X_val.shape[1]):
        X_val_yj[:, j] = scipy_stats.yeojohnson(X_val[:, j], lmbda=lambdas[j])

    X_tests_yj = []
    for X_test in X_tests:
        X_t = np.zeros_like(X_test)
        for j in range(X_test.shape[1]):
            X_t[:, j] = scipy_stats.yeojohnson(X_test[:, j], lmbda=lambdas[j])
        X_tests_yj.append(X_t)

    # Clip after YJ to same range
    X_train_yj = np.clip(X_train_yj, -10, 10).astype(np.float32)
    X_val_yj   = np.clip(X_val_yj,   -10, 10).astype(np.float32)
    X_tests_yj = [np.clip(x, -10, 10).astype(np.float32) for x in X_tests_yj]

    return X_train_yj, X_val_yj, X_tests_yj


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
    X_train = np.clip(d["X_train"], -10, 10).astype(np.float32)

    d = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val = np.clip(d["X_val"], -10, 10).astype(np.float32)

    # Load test sets
    test_names = ['CIC17', 'CIC18', 'UNSW', 'BoT']
    test_files = [
        'test_cic17_dedup.npz',
        'test_cic18_dedup.npz',
        'test_unsw_dedup.npz',
        'test_botiot_dedup.npz',
    ]
    X_tests, y_tests = [], []
    for fname in test_files:
        d = np.load(os.path.join(DEDUP_DIR, fname))
        X_tests.append(np.clip(d["X"], -10, 10).astype(np.float32))
        y_tests.append(d["y"].astype(int))
        log.info("Loaded %s: %d rows", fname, len(X_tests[-1]))

    log.info("Train: %d  Val: %d", len(X_train), len(X_val))

    # Apply YJ to get transformed versions
    log.info("Fitting Yeo-Johnson on training data...")
    X_train_yj, X_val_yj, X_tests_yj = apply_yj(X_train, X_val, X_tests)
    log.info("YJ transformation complete")

    results = []

    for use_yj in [False, True]:
        condition = "DAE-Multi-YJ" if use_yj else "DAE-Multi-NoYJ"
        X_tr = X_train_yj if use_yj else X_train
        X_va = X_val_yj   if use_yj else X_val
        X_ts = X_tests_yj if use_yj else X_tests

        log.info("="*55)
        log.info("Condition: %s", condition)

        seed_results = {n: [] for n in test_names}

        for seed in SEEDS:
            log.info("  Seed %d...", seed)
            t0 = time.time()

            model   = train_dae(X_tr, X_va, seed)
            train_s = get_scores(model, X_tr[:50000])
            p1      = float(np.percentile(train_s, 1))
            p99     = float(np.percentile(train_s, 99))

            val_s = norm(get_scores(model, X_va), p1, p99)
            tau   = float(np.percentile(val_s, 95))

            row = {'condition': condition, 'seed': seed}
            for name, X_test, y_test in zip(test_names, X_ts, y_tests):
                s   = norm(get_scores(model, X_test), p1, p99)
                if name == 'BoT':
                    # DR only
                    dr = float((s > tau).mean())
                    row[f'{name}_DR'] = dr
                    seed_results[name].append(dr)
                    log.info("    %s: DR=%.4f tau=%.4f", name, dr, tau)
                else:
                    auc = roc_auc_score(y_test, s)
                    row[f'{name}_AUC'] = auc
                    seed_results[name].append(auc)
                    log.info("    %s: AUC=%.4f", name, auc)

            log.info("  Seed %d done in %.1f min", seed, (time.time()-t0)/60)
            results.append(row)

        log.info("%s SUMMARY:", condition)
        for name in test_names:
            vals = np.array(seed_results[name])
            log.info("  %s: %.4f ± %.4f (CV=%.1f%%)",
                     name, vals.mean(), vals.std(ddof=0),
                     vals.std(ddof=0)/vals.mean()*100)

    df = pd.DataFrame(results)
    out = os.path.join(DEDUP_DIR, "yj_ablation_results.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)

    print("\n" + "="*65)
    print("YEO-JOHNSON ABLATION RESULTS (mean ± pop std, 5 seeds)")
    print("="*65)
    for cond in ["DAE-Multi-NoYJ", "DAE-Multi-YJ"]:
        sub = df[df['condition']==cond]
        print(f"\n{cond}:")
        for col in ['CIC17_AUC','CIC18_AUC','UNSW_AUC','BoT_DR']:
            if col in sub.columns:
                vals = sub[col].values
                cv = vals.std(ddof=0)/vals.mean()*100
                print(f"  {col}: {vals.mean():.4f} ± {vals.std(ddof=0):.4f} "
                      f"(CV={cv:.1f}%)")

    print("\nDELTA (YJ minus NoYJ):")
    for col in ['CIC17_AUC','CIC18_AUC','UNSW_AUC','BoT_DR']:
        if col in df.columns:
            no_yj = df[df['condition']=='DAE-Multi-NoYJ'][col].mean()
            yj    = df[df['condition']=='DAE-Multi-YJ'][col].mean()
            print(f"  {col}: {yj-no_yj:+.4f}")


if __name__ == "__main__":
    run()
