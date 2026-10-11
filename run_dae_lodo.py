"""
run_dae_lodo.py
================
Leave-One-Dataset-Out (LODO) experiments for DAE.

Three conditions:
LODO-1: Train on CIC18+UNSW, test on CIC17
LODO-2: Train on CIC17+UNSW, test on CIC18
LODO-3: Train on CIC17+CIC18, test on UNSW

Output:
  artefacts_v2/dedup/dae_lodo_results.csv

Run from project root:
    python run_dae_lodo.py

Expected time: ~3 hours (3 experiments x seed 42)
"""
import sys, os, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
DEDUP_DIR    = os.path.join(ARTEFACT_DIR, "dedup")
SEED         = 42
NOISE_STD    = 0.1
INPUT_DIM    = 18


class DAEModel(nn.Module):
    def __init__(self, input_dim=18):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, 8),         nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),         nn.ReLU(),
            nn.Linear(16, 32),        nn.ReLU(),
            nn.Linear(32, 64),        nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_dae(X_train, X_val, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    model  = DAEModel(INPUT_DIM).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5)
    crit   = nn.HuberLoss(delta=1.0)

    X_t = torch.FloatTensor(X_train)
    X_v = torch.FloatTensor(X_val)

    best_loss  = float('inf')
    patience_c = 0
    best_state = None

    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            batch = X_t[idx[i:i+256]]
            noisy = torch.clamp(
                batch + torch.randn_like(batch) * NOISE_STD, -10, 10)
            opt.zero_grad()
            crit(model(noisy), batch).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            noisy_v = torch.clamp(
                X_v + torch.randn_like(X_v) * NOISE_STD, -10, 10)
            val_loss = crit(model(noisy_v), X_v).item()

        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= 10:
                log.info("  Early stop epoch %d", epoch + 1)
                break

    model.load_state_dict(best_state)
    return model


def get_scores(model, X):
    model.eval()
    X_t = torch.FloatTensor(X)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_t), 10000):
            b = X_t[i:i+10000]
            r = model(b)
            scores.append(((b - r)**2).mean(dim=1).numpy())
    return np.concatenate(scores)


def norm_scores(scores, p1, p99):
    return np.clip((scores - p1) / (p99 - p1 + 1e-8), 0, 1)


def run():
    log.info("Loading data...")

    # Load all training data
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(
        d_train["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X_train_full) // 3

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full = np.clip(
        d_val["X_val"], -10, 10).astype(np.float32)
    n_val_per_env = len(X_val_full) // 3

    # Split by environment
    envs = {
        'CIC17': (X_train_full[:n_per_env],
                  X_val_full[:n_val_per_env]),
        'CIC18': (X_train_full[n_per_env:2*n_per_env],
                  X_val_full[n_val_per_env:2*n_val_per_env]),
        'UNSW':  (X_train_full[2*n_per_env:],
                  X_val_full[2*n_val_per_env:]),
    }

    # Test sets
    test_npz = {
        'CIC17': 'test_cic17_dedup.npz',
        'CIC18': 'test_cic18_dedup.npz',
        'UNSW':  'test_unsw_dedup.npz',
        'BoT':   'test_botiot_dedup.npz',
    }
    test_sets = {}
    for name, fname in test_npz.items():
        path = os.path.join(DEDUP_DIR, fname)
        if os.path.exists(path):
            d = np.load(path)
            test_sets[name] = (
                np.clip(d["X"], -10, 10).astype(np.float32),
                d["y"].astype(int))
            log.info("Loaded %s: %d rows", name, len(test_sets[name][0]))

    # LODO experiments
    lodo_configs = [
        ('LODO-1 (no CIC17)', ['CIC18', 'UNSW'], 'CIC17'),
        ('LODO-2 (no CIC18)', ['CIC17', 'UNSW'], 'CIC18'),
        ('LODO-3 (no UNSW)',  ['CIC17', 'CIC18'], 'UNSW'),
    ]

    results = []

    for exp_name, train_envs, held_out in lodo_configs:
        log.info("=" * 55)
        log.info("%s", exp_name)
        log.info("Training on: %s", train_envs)
        log.info("Held out:    %s", held_out)
        log.info("=" * 55)

        # Combine training environments
        X_train = np.vstack([envs[e][0] for e in train_envs])
        X_val   = np.vstack([envs[e][1] for e in train_envs])
        log.info("Training flows: %d", len(X_train))

        t0 = time.time()
        model = train_dae(X_train, X_val, seed=SEED)

        # Normalization from training scores
        train_scores = get_scores(model, X_train[:50000])
        p1  = float(np.percentile(train_scores, 1))
        p99 = float(np.percentile(train_scores, 99))

        # Threshold from validation
        val_scores = get_scores(model, X_val)
        val_norm   = norm_scores(val_scores, p1, p99)
        tau        = float(np.percentile(val_norm, 95))
        log.info("  tau=%.4f", tau)

        row = {'experiment': exp_name, 'held_out': held_out, 'tau': tau}

        # Evaluate on all test sets
        for name, (X_test, y_test) in test_sets.items():
            scores      = get_scores(model, X_test)
            scores_norm = norm_scores(scores, p1, p99)

            if name == 'BoT':
                dr = float((scores_norm > tau).mean())
                row['BoT-IoT_DR'] = dr
                log.info("  BoT DR=%.4f", dr)
            else:
                if len(np.unique(y_test)) < 2:
                    continue
                auc = roc_auc_score(y_test, scores_norm)
                y_pred = (scores_norm > tau).astype(int)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                col_auc = f'{"CIC-IDS2017" if name=="CIC17" else "CSE-CIC-IDS2018" if name=="CIC18" else "UNSW-NB15"}_AUC'
                col_f1  = f'{"CIC-IDS2017" if name=="CIC17" else "CSE-CIC-IDS2018" if name=="CIC18" else "UNSW-NB15"}_F1'
                row[col_auc] = auc
                row[col_f1]  = f1
                log.info("  %s AUC=%.4f F1=%.4f", name, auc, f1)

        elapsed = time.time() - t0
        log.info("%s done in %.1f min", exp_name, elapsed / 60)
        results.append(row)

    # Save
    df = pd.DataFrame(results)
    out_path = os.path.join(DEDUP_DIR, "dae_lodo_results.csv")
    df.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)

    # Print summary
    print("\n" + "="*65)
    print("DAE LODO RESULTS SUMMARY (seed 42)")
    print("="*65)
    print(df[['experiment', 'CIC-IDS2017_AUC',
              'CSE-CIC-IDS2018_AUC', 'UNSW-NB15_AUC',
              'BoT-IoT_DR']].to_string(index=False))
    print()
    print("Held-out F1 scores:")
    for _, row in df.iterrows():
        ho = row['held_out']
        col = f'{"CIC-IDS2017" if ho=="CIC17" else "CSE-CIC-IDS2018" if ho=="CIC18" else "UNSW-NB15"}_F1'
        if col in row:
            print(f"  {row['experiment']}: {col}={row[col]:.4f}")


if __name__ == "__main__":
    run()
