"""
run_dae_size_matched.py
========================
Size-matched diversity control experiment.

Compares:
A. DAE-Single:        536,307 CIC17 dedup flows
B. DAE-Multi-matched: 536,307 diverse flows
                      (178,769 per environment)

Same total training size in both conditions.
Only diversity changes.

If B >> A: diversity explains the gain, not size.

Output:
  artefacts_v2/dedup/dae_size_matched_results.csv

Run from project root:
    python run_dae_size_matched.py

Expected time: ~2 hours (5 seeds x ~25 min each)
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


def train_dae(X_train, X_val, seed):
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
                batch + torch.randn_like(batch) * NOISE_STD,
                -10, 10)
            opt.zero_grad()
            crit(model(noisy), batch).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            noisy_v = torch.clamp(
                X_v + torch.randn_like(X_v) * NOISE_STD,
                -10, 10)
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
    log.info("Loading deduplicated data...")

    # Full dedup training data
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(
        d_train["X_train"], -10, 10).astype(np.float32)
    n_total   = len(X_train_full)
    n_per_env = n_total // 3

    # CIC17 single-env (536,307 flows)
    X_single = X_train_full[:n_per_env]
    n_single  = len(X_single)
    log.info("DAE-Single training size: %d", n_single)

    # Size-matched multi-env
    # Take equal portion from each environment
    n_per_env_matched = n_single // 3
    X_cic17 = X_train_full[:n_per_env][:n_per_env_matched]
    X_cic18 = X_train_full[n_per_env:2*n_per_env][:n_per_env_matched]
    X_unsw  = X_train_full[2*n_per_env:][:n_per_env_matched]
    X_multi_matched = np.vstack([X_cic17, X_cic18, X_unsw])
    log.info("DAE-Multi-matched training size: %d "
             "(%d per env)", len(X_multi_matched), n_per_env_matched)
    log.info("  CIC17: %d, CIC18: %d, UNSW: %d",
             len(X_cic17), len(X_cic18), len(X_unsw))

    # Validation: size-matched too
    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full = np.clip(
        d_val["X_val"], -10, 10).astype(np.float32)
    n_val_per_env = len(X_val_full) // 3
    n_val_matched = n_val_per_env // 3

    X_val_single = X_val_full[:n_val_per_env]
    X_val_matched = np.vstack([
        X_val_full[:n_val_per_env][:n_val_matched],
        X_val_full[n_val_per_env:2*n_val_per_env][:n_val_matched],
        X_val_full[2*n_val_per_env:][:n_val_matched],
    ])
    log.info("Val single: %d, Val matched: %d",
             len(X_val_single), len(X_val_matched))

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
                d["y"].astype(int),
            )
            log.info("Test %s: %d rows", name,
                     len(test_sets[name][0]))

    results = []

    for seed in SEEDS:
        log.info("=" * 55)
        log.info("SEED %d", seed)
        log.info("=" * 55)

        # ── DAE-Multi-matched ─────────────────────────────────────
        log.info("Training DAE-Multi-matched "
                 "(%d flows, seed %d)...",
                 len(X_multi_matched), seed)
        t0 = time.time()

        # Shuffle combined data
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_multi_matched))
        X_train_shuffled = X_multi_matched[idx]

        model = train_dae(X_train_shuffled, X_val_matched, seed)

        train_scores = get_scores(model, X_train_shuffled[:50000])
        p1  = float(np.percentile(train_scores, 1))
        p99 = float(np.percentile(train_scores, 99))

        val_scores = get_scores(model, X_val_matched)
        val_norm   = norm_scores(val_scores, p1, p99)
        tau        = float(np.percentile(val_norm, 95))
        log.info("  tau=%.4f", tau)

        row = {
            'seed':      seed,
            'condition': 'DAE-Multi-matched',
            'n_train':   len(X_multi_matched),
            'n_per_env': n_per_env_matched,
            'tau':       tau,
        }

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

    # Save
    df = pd.DataFrame(results)
    out_path = os.path.join(DEDUP_DIR, "dae_size_matched_results.csv")
    df.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)

    # Summary
    print("\n" + "=" * 70)
    print("SIZE-MATCHED DIVERSITY CONTROL RESULTS")
    print("Same total training size (~536k), different diversity")
    print("=" * 70)

    cols = {
        'CIC17': 'CIC-IDS2017_AUC',
        'CIC18': 'CSE-CIC-IDS2018_AUC',
        'UNSW':  'UNSW-NB15_AUC',
        'BoT':   'BoT-IoT_DR',
    }

    # Known results
    dae_single = {
        'CIC17': 0.9252, 'CIC18': 0.8419,
        'UNSW':  0.4989, 'BoT':   None,
    }
    dae_multi_full = {
        'CIC17': 0.9249, 'CIC18': 0.9646,
        'UNSW':  0.9432, 'BoT':   0.6789,
    }

    print(f"\n{'Dataset':<8} {'DAE-Single':>14} "
          f"{'DAE-Multi-matched':>20} {'DAE-Multi-full':>16}")
    print(f"{'':8} {'(536k, 1 env)':>14} "
          f"{'(536k, 3 envs)':>20} {'(1.6M, 3 envs)':>16}")
    print("-" * 62)

    for name, col in cols.items():
        if col in df.columns:
            m = df[col].mean()
            s = df[col].std()
            single = dae_single.get(name)
            full   = dae_multi_full.get(name)

            single_str = f"{single:.4f}" if single else "N/A"
            full_str   = f"{full:.4f}"   if full   else "N/A"

            print(f"{name:<8} {single_str:>14} "
                  f"{m:>10.4f}+/-{s:.4f}     "
                  f"{full_str:>16}")

    print()
    print("KEY QUESTION:")
    print("Is DAE-Multi-matched >> DAE-Single?")
    print("If YES: diversity (not size) drives generalization")

    # Check UNSW specifically
    if 'UNSW-NB15_AUC' in df.columns:
        unsw_matched = df['UNSW-NB15_AUC'].mean()
        unsw_single  = dae_single['UNSW']
        print(f"\nUNSW-NB15 result:")
        print(f"  DAE-Single:        {unsw_single:.4f}")
        print(f"  DAE-Multi-matched: {unsw_matched:.4f}")
        print(f"  Gain:              {unsw_matched-unsw_single:+.4f}")
        if unsw_matched > unsw_single + 0.1:
            print("  VERDICT: DIVERSITY explains the gain OK")
        else:
            print("  VERDICT: Unclear - further analysis needed")


if __name__ == "__main__":
    run()
