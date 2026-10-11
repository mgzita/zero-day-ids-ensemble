"""
generate_dae_score_distributions.py
=====================================
Generates score distribution figure from DAE-Multi models.
Uses seed 42 DAE model saved during dae_single run,
or retrains DAE-Multi seed 42 if not found.

Output:
  artefacts_v2/figures/score_distributions_dae.png

Run from project root:
    python generate_dae_score_distributions.py

Expected time: ~30 minutes
"""
import sys, os, logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

TON_IOT_RENAME = {
    'fwd_pkt_len_mean':  'fwd_packet_length_mean',
    'bwd_pkt_len_mean':  'bwd_packet_length_mean',
    'fwd_pkt_len_std':   'fwd_packet_length_std',
    'bwd_pkt_len_std':   'bwd_packet_length_std',
    'flow_byts_s':       'flow_bytes_s',
    'fwd_pkts_s':        'fwd_packets_s',
    'bwd_pkts_s':        'bwd_packets_s',
    'fwd_byts_b_avg':    'fwd_avg_bytes_bulk',
    'bwd_byts_b_avg':    'bwd_avg_bytes_bulk',
    'fwd_pkts_b_avg':    'fwd_avg_packets_bulk',
    'bwd_pkts_b_avg':    'bwd_avg_packets_bulk',
    'init_fwd_win_byts': 'init_win_bytes_forward',
    'init_bwd_win_byts': 'init_win_bytes_backward',
    'subflow_fwd_byts':  'subflow_fwd_bytes',
    'dst_port':          'destination_port',
}


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


def train_dae_multi(seed):
    """Train DAE-Multi (3 environments) for seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train = np.clip(d_train["X_train"], -10, 10).astype(np.float32)

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val = np.clip(d_val["X_val"], -10, 10).astype(np.float32)

    device = torch.device('cpu')
    model  = DAEModel(INPUT_DIM).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
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
            noisy = torch.clamp(batch + torch.randn_like(batch)*NOISE_STD, -10, 10)
            opt.zero_grad()
            crit(model(noisy), batch).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            noisy_v = torch.clamp(X_v + torch.randn_like(X_v)*NOISE_STD, -10, 10)
            val_loss = crit(model(noisy_v), X_v).item()

        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss  = val_loss
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
    X_t = torch.FloatTensor(X)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_t), 10000):
            b = X_t[i:i+10000]
            r = model(b)
            scores.append(((b - r)**2).mean(dim=1).numpy())
    return np.concatenate(scores)


def norm_scores(scores):
    p1  = float(np.percentile(scores, 1))
    p99 = float(np.percentile(scores, 99))
    if p99 - p1 < 1e-8:
        return np.zeros_like(scores)
    return np.clip((scores - p1) / (p99 - p1), 0, 1).astype(np.float32)


def run():
    # Try to load saved DAE-Multi seed42 model
    model_path = os.path.join(DEDUP_DIR, "seed42_dae", "dae_multi.pt")
    model = DAEModel(INPUT_DIM)

    if os.path.exists(model_path):
        log.info("Loading saved DAE-Multi model...")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        log.info("Training DAE-Multi (seed 42)...")
        model = train_dae_multi(SEED)
        # Save for future use
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        log.info("Saved DAE-Multi model -> %s", model_path)

    # Load training scores for normalization
    log.info("Computing normalization parameters...")
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train = np.clip(d_train["X_train"], -10, 10).astype(np.float32)
    train_scores = get_scores(model, X_train)
    p1  = float(np.percentile(train_scores, 1))
    p99 = float(np.percentile(train_scores, 99))
    del X_train

    def norm(s):
        return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)

    # Datasets to plot
    datasets = [
        {"name": "CIC-IDS2017",
         "path": os.path.join(DEDUP_DIR, "test_cic17_dedup.npz"),
         "type": "npz", "has_benign": True},
        {"name": "CSE-CIC-IDS2018",
         "path": os.path.join(DEDUP_DIR, "test_cic18_dedup.npz"),
         "type": "npz", "has_benign": True},
        {"name": "UNSW-NB15",
         "path": os.path.join(DEDUP_DIR, "test_unsw_dedup.npz"),
         "type": "npz", "has_benign": True},
        {"name": "BoT-IoT",
         "path": os.path.join(DEDUP_DIR, "test_botiot_dedup.npz"),
         "type": "npz", "has_benign": False},
        {"name": "CIC-ToN-IoT",
         "path": os.path.join(
             ARTEFACT_DIR, "..", "data", "raw",
             "cic_ton_iot", "CIC-ToN-IoT.csv"),
         "type": "csv", "has_benign": True},
    ]

    # Get features list
    feat_path = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    import pandas as pd
    features = pd.read_csv(feat_path)["0"].tolist() \
        if os.path.exists(feat_path) else None

    from preprocessing.scaler import IQRScaler
    scaler_path = os.path.join(ARTEFACT_DIR, "iqr_scaler.npz")
    scaler = IQRScaler()
    scaler.load(scaler_path)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(
        "DAE-Multi Anomaly Score Distributions "
        "(seed 42, $p_{95}$ threshold, deduplicated test sets)",
        fontsize=13, fontweight='bold')

    color_benign = "#2471A3"
    color_attack = "#C0392B"

    for ax, ds in zip(axes, datasets):
        name = ds["name"]
        log.info("Processing %s...", name)

        if ds["type"] == "npz":
            d = np.load(ds["path"])
            X = np.clip(d["X"], -10, 10).astype(np.float32)
            y = d["y"].astype(int)
        else:
            # CSV (ToN-IoT)
            df = pd.read_csv(ds["path"], nrows=200000, low_memory=False)
            df.columns = (df.columns.str.strip().str.lower()
                         .str.replace(r'\s+', '_', regex=True)
                         .str.replace(r'[^a-z0-9_]', '_', regex=True)
                         .str.strip('_'))
            df.rename(columns=TON_IOT_RENAME, inplace=True)
            df['label'] = (df['label'] != 0).astype(int) \
                if 'label' in df.columns else 1
            if features:
                for f in features:
                    if f not in df.columns:
                        df[f] = 0.0
                X_raw = df[features].values.astype(np.float32)
            else:
                X_raw = df.select_dtypes(include=[np.number]).drop(
                    columns=['label'], errors='ignore').values.astype(np.float32)
            X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
            X = np.clip(scaler.transform(X_raw), -10, 10)
            y = df['label'].values.astype(int)

        scores = norm(get_scores(model, X))

        # Compute p95 threshold from validation
        d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
        X_val = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
        val_scores = norm(get_scores(model, X_val))
        tau = float(np.percentile(val_scores, 95))
        del d_val, X_val

        if ds["has_benign"]:
            benign_scores = scores[y == 0]
            attack_scores = scores[y == 1]
            ax.hist(benign_scores, bins=60, alpha=0.6,
                    color=color_benign, label='Benign', density=True)
            ax.hist(attack_scores, bins=60, alpha=0.6,
                    color=color_attack, label='Attack', density=True)
        else:
            ax.hist(scores, bins=60, alpha=0.7,
                    color=color_attack, label='Attack only', density=True)

        ax.axvline(tau, color='black', linestyle='--',
                   linewidth=1.5, label=f'$\\tau$={tau:.2f}')
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel("Normalized Score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#AAAAAA')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "score_distributions_dae.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    log.info("Saved -> %s", out_path)
    log.info("Done.")


if __name__ == "__main__":
    run()
