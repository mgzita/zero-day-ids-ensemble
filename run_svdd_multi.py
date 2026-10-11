"""
run_svdd_multi.py
=================
Trains Deep SVDD on ALL THREE environments (multi-environment)
across 5 seeds. Completes the statistical parity with AE/DAE.

Output:
  artefacts_v2/dedup/svdd_multi_results.csv

Run from project root:
    python run_svdd_multi.py

Expected time: ~4 hours (5 seeds x ~45 min each)
"""
import sys, os, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
LATENT_DIM   = 16


class SVDDNetwork(nn.Module):
    def __init__(self, input_dim=18, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64, bias=False), nn.ReLU(),
            nn.Linear(64, 32, bias=False),        nn.ReLU(),
            nn.Linear(32, latent_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def init_center(model, X_train, device, eps=0.01):
    model.eval()
    X_t = torch.FloatTensor(X_train).to(device)
    with torch.no_grad():
        outputs = []
        for i in range(0, len(X_t), 1000):
            outputs.append(model(X_t[i:i+1000]))
        z = torch.cat(outputs, dim=0)
    c = z.mean(dim=0)
    c[(abs(c) < eps)] = eps
    return c


def train_svdd(X_train, X_val, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    model  = SVDDNetwork(INPUT_DIM, LATENT_DIM).to(device)

    log.info("  Initializing center...")
    c = init_center(model, X_train, device).to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=1e-3,
                              weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5)

    X_t = torch.FloatTensor(X_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)

    best_loss  = float('inf')
    patience_c = 0
    best_state = None

    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            batch = X_t[idx[i:i+256]]
            opt.zero_grad()
            loss = ((model(batch) - c) ** 2).sum(dim=1).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = ((model(X_v) - c)**2).sum(dim=1).mean().item()

        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= 10:
                log.info("  Early stop epoch %d", epoch + 1)
                break

    model.load_state_dict(best_state)
    return model, c


def get_scores(model, c, X):
    model.eval()
    X_t = torch.FloatTensor(X)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_t), 10000):
            z = model(X_t[i:i+10000])
            scores.append(((z - c)**2).sum(dim=1).numpy())
    return np.concatenate(scores)


def norm_scores(scores, p1, p99):
    return np.clip((scores - p1) / (p99 - p1 + 1e-8), 0, 1)


def run():
    log.info("Loading deduplicated data...")

    # Full multi-environment training data
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train = np.clip(d_train["X_train"], -10, 10).astype(np.float32)
    log.info("Multi-env training: %d flows", len(X_train))

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    log.info("Validation: %d flows", len(X_val))

    # Test sets
    test_sets = {}
    for name, fname in [
        ('CIC17', 'test_cic17_dedup.npz'),
        ('CIC18', 'test_cic18_dedup.npz'),
        ('UNSW',  'test_unsw_dedup.npz'),
        ('BoT',   'test_botiot_dedup.npz'),
    ]:
        path = os.path.join(DEDUP_DIR, fname)
        if os.path.exists(path):
            d = np.load(path)
            test_sets[name] = (
                np.clip(d["X"], -10, 10).astype(np.float32),
                d["y"].astype(int))
            log.info("Loaded %s: %d rows", name, len(test_sets[name][0]))

    results = []

    for seed in SEEDS:
        log.info("=" * 55)
        log.info("SEED %d", seed)
        log.info("=" * 55)
        t0 = time.time()

        log.info("Training SVDD-Multi (3 envs, seed %d)...", seed)
        model, c = train_svdd(X_train, X_val, seed)

        train_scores = get_scores(model, c, X_train[:50000])
        p1  = float(np.percentile(train_scores, 1))
        p99 = float(np.percentile(train_scores, 99))

        val_scores = get_scores(model, c, X_val)
        val_norm   = norm_scores(val_scores, p1, p99)
        tau        = float(np.percentile(val_norm, 95))
        log.info("  tau=%.4f", tau)

        row = {'seed': seed, 'condition': 'SVDD-Multi', 'tau': tau}

        for name, (X_test, y_test) in test_sets.items():
            scores      = get_scores(model, c, X_test)
            scores_norm = norm_scores(scores, p1, p99)

            if name == 'BoT':
                dr = float((scores_norm > tau).mean())
                row['BoT-IoT_DR'] = dr
                log.info("  BoT DR=%.4f", dr)
            else:
                if len(np.unique(y_test)) < 2:
                    continue
                auc = roc_auc_score(y_test, scores_norm)
                col = {
                    'CIC17': 'CIC-IDS2017_AUC',
                    'CIC18': 'CSE-CIC-IDS2018_AUC',
                    'UNSW':  'UNSW-NB15_AUC',
                }[name]
                row[col] = auc
                log.info("  %s AUC=%.4f", name, auc)

        elapsed = time.time() - t0
        log.info("Seed %d done in %.1f min", seed, elapsed / 60)
        results.append(row)

    df = pd.DataFrame(results)
    out_path = os.path.join(DEDUP_DIR, "svdd_multi_results.csv")
    df.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)

    print("\n" + "=" * 65)
    print("SVDD-MULTI RESULTS (3 environments, 5 seeds)")
    print("=" * 65)
    print(df.to_string(index=False))

    cols = {
        'CIC17': 'CIC-IDS2017_AUC',
        'CIC18': 'CSE-CIC-IDS2018_AUC',
        'UNSW':  'UNSW-NB15_AUC',
        'BoT':   'BoT-IoT_DR',
    }

    svdd_single = {'CIC17': 0.9013, 'CIC18': 0.5106,
                   'UNSW':  0.4988, 'BoT':   0.9636}
    dae_multi   = {'CIC17': 0.9249, 'CIC18': 0.9646,
                   'UNSW':  0.9432, 'BoT':   0.6789}

    print(f"\n{'Dataset':<8} {'SVDD-Single':>14} "
          f"{'SVDD-Multi':>16} {'DAE-Multi':>12}")
    print(f"{'':8} {'(5 seeds)':>14} "
          f"{'(5 seeds)':>16} {'(5 seeds)':>12}")
    print("-" * 55)

    for name, col in cols.items():
        if col in df.columns:
            m = df[col].mean()
            s = df[col].std()
            print(f"{name:<8} {svdd_single[name]:>14.4f} "
                  f"{m:>8.4f}+/-{s:.4f} "
                  f"{dae_multi[name]:>12.4f}")

    print()
    print("KEY: Does SVDD-Multi beat SVDD-Single on UNSW?")
    if 'UNSW-NB15_AUC' in df.columns:
        gain = df['UNSW-NB15_AUC'].mean() - svdd_single['UNSW']
        print(f"SVDD UNSW gain: {gain:+.4f}")


if __name__ == "__main__":
    run()
