"""
generate_score_distributions_all.py
=====================================
Generates score distribution figure comparing
DAE-Single vs DAE-Multi across all test datasets.
Shows WHY multi-environment training works:
benign/attack score separation improves dramatically.

Output:
  artefacts_v2/figures/score_distributions_all.png

Run from project root:
    python generate_score_distributions_all.py

Expected time: ~45 minutes (trains DAE-Single + loads DAE-Multi)
"""
import sys, os, logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
DEDUP_DIR    = os.path.join(ARTEFACT_DIR, "dedup")
OUT_DIR      = os.path.join(ARTEFACT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

SEED      = 42
NOISE_STD = 0.1
INPUT_DIM = 18


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
            nn.Linear(8, 16),  nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_dae(X_train, X_val, seed, noise_std=NOISE_STD):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cpu')
    model  = DAEModel(INPUT_DIM).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    crit   = nn.HuberLoss(delta=1.0)
    X_t    = torch.FloatTensor(X_train)
    X_v    = torch.FloatTensor(X_val)
    best_loss, patience_c, best_state = float('inf'), 0, None
    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            batch = X_t[idx[i:i+256]]
            noisy = torch.clamp(batch + torch.randn_like(batch)*noise_std, -10, 10)
            opt.zero_grad()
            crit(model(noisy), batch).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            noisy_v = torch.clamp(X_v + torch.randn_like(X_v)*noise_std, -10, 10)
            val_loss = crit(model(noisy_v), X_v).item()
        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= 10:
                log.info("  Early stop epoch %d", epoch+1)
                break
    model.load_state_dict(best_state)
    return model


def get_scores(model, X):
    model.eval()
    scores = []
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        for i in range(0, len(X_t), 10000):
            b = X_t[i:i+10000]
            r = model(b)
            scores.append(((b-r)**2).mean(dim=1).numpy())
    return np.concatenate(scores)


def norm(scores, p1, p99):
    return np.clip((scores - p1) / (p99 - p1 + 1e-8), 0, 1)


def run():
    log.info("Loading data...")

    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(d_train["X_train"], -10, 10).astype(np.float32)
    n_per_env    = len(X_train_full) // 3
    X_single     = X_train_full[:n_per_env]

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full   = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    n_val_per_env = len(X_val_full) // 3
    X_val_single  = X_val_full[:n_val_per_env]

    # Load test sets
    datasets = [
        ("CIC-IDS2017",     "test_cic17_dedup.npz",  True),
        ("CSE-CIC-IDS2018", "test_cic18_dedup.npz",  True),
        ("UNSW-NB15",       "test_unsw_dedup.npz",   True),
        ("BoT-IoT",         "test_botiot_dedup.npz", False),
    ]
    test_sets = {}
    for name, fname, has_benign in datasets:
        path = os.path.join(DEDUP_DIR, fname)
        if os.path.exists(path):
            d = np.load(path)
            test_sets[name] = (
                np.clip(d["X"], -10, 10).astype(np.float32),
                d["y"].astype(int),
                has_benign
            )
            log.info("Loaded %s: %d rows", name, len(test_sets[name][0]))

    # Train DAE-Single
    log.info("Training DAE-Single (CIC17 only, seed %d)...", SEED)
    model_single = train_dae(X_single, X_val_single, SEED)
    train_s_single = get_scores(model_single, X_single)
    p1_s  = float(np.percentile(train_s_single, 1))
    p99_s = float(np.percentile(train_s_single, 99))
    val_s_single = norm(get_scores(model_single, X_val_single), p1_s, p99_s)
    tau_single   = float(np.percentile(val_s_single, 95))
    log.info("  DAE-Single tau=%.4f", tau_single)

    # Load or train DAE-Multi
    multi_path = os.path.join(DEDUP_DIR, "seed42_dae", "dae_multi.pt")
    model_multi = DAEModel(INPUT_DIM)
    if os.path.exists(multi_path):
        log.info("Loading saved DAE-Multi...")
        model_multi.load_state_dict(torch.load(multi_path, map_location='cpu'))
    else:
        log.info("Training DAE-Multi (3 envs, seed %d)...", SEED)
        model_multi = train_dae(X_train_full, X_val_full, SEED)
        os.makedirs(os.path.dirname(multi_path), exist_ok=True)
        torch.save(model_multi.state_dict(), multi_path)

    train_s_multi = get_scores(model_multi, X_train_full[:50000])
    p1_m  = float(np.percentile(train_s_multi, 1))
    p99_m = float(np.percentile(train_s_multi, 99))
    val_s_multi = norm(get_scores(model_multi, X_val_full), p1_m, p99_m)
    tau_multi   = float(np.percentile(val_s_multi, 95))
    log.info("  DAE-Multi tau=%.4f", tau_multi)

    # Plot
    n_ds  = len(test_sets)
    fig   = plt.figure(figsize=(16, 8))
    gs    = gridspec.GridSpec(2, n_ds, hspace=0.45, wspace=0.3)

    colors = {
        'benign': '#2471A3',
        'attack': '#C0392B',
        'single': '#7D3C98',
        'multi':  '#1E8449',
    }

    titles = list(test_sets.keys())

    for col, name in enumerate(titles):
        X_test, y_test, has_benign = test_sets[name]

        # Row 0: DAE-Single
        ax0 = fig.add_subplot(gs[0, col])
        s_single = norm(get_scores(model_single, X_test), p1_s, p99_s)
        if has_benign:
            ax0.hist(s_single[y_test==0], bins=50, alpha=0.6,
                     color=colors['benign'], density=True, label='Benign')
            ax0.hist(s_single[y_test==1], bins=50, alpha=0.6,
                     color=colors['attack'], density=True, label='Attack')
        else:
            ax0.hist(s_single, bins=50, alpha=0.7,
                     color=colors['attack'], density=True, label='Attack only')
        ax0.axvline(tau_single, color='k', linestyle='--', lw=1.2,
                    label=f'τ={tau_single:.2f}')
        ax0.set_title(name, fontsize=10, fontweight='bold')
        if col == 0:
            ax0.set_ylabel('DAE-Single\nDensity', fontsize=9)
        ax0.tick_params(labelsize=8)
        ax0.legend(fontsize=7, loc='upper right')

        # Row 1: DAE-Multi
        ax1 = fig.add_subplot(gs[1, col])
        s_multi = norm(get_scores(model_multi, X_test), p1_m, p99_m)
        if has_benign:
            ax1.hist(s_multi[y_test==0], bins=50, alpha=0.6,
                     color=colors['benign'], density=True, label='Benign')
            ax1.hist(s_multi[y_test==1], bins=50, alpha=0.6,
                     color=colors['attack'], density=True, label='Attack')
        else:
            ax1.hist(s_multi, bins=50, alpha=0.7,
                     color=colors['attack'], density=True, label='Attack only')
        ax1.axvline(tau_multi, color='k', linestyle='--', lw=1.2,
                    label=f'τ={tau_multi:.2f}')
        if col == 0:
            ax1.set_ylabel('DAE-Multi\nDensity', fontsize=9)
        ax1.set_xlabel('Normalized score', fontsize=9)
        ax1.tick_params(labelsize=8)
        ax1.legend(fontsize=7, loc='upper right')

    fig.suptitle(
        'Anomaly score distributions: DAE-Single vs DAE-Multi (seed 42, $p_{95}$ threshold)\n'
        'Benign=blue, Attack=red, dashed line=$\\tau$',
        fontsize=11, fontweight='bold', y=1.01
    )

    out_path = os.path.join(OUT_DIR, "score_distributions_all.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    log.info("Saved -> %s", out_path)
    log.info("Done.")


if __name__ == "__main__":
    run()
