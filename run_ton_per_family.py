"""
run_ton_per_family.py
=====================
Evaluates DAE-Multi on CIC-ToN-IoT and reports
per-attack-family recall and ZDDR.

Output:
  artefacts_v2/dedup/ton_per_family.csv

Run from project root:
    python run_ton_per_family.py

Expected time: ~45 minutes
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
TON_CSV      = r"C:/MLProject/zero_day_project/data/raw/cic_ton_iot/CIC-ToN-IoT.csv"
SEED         = 42
N_CAL        = 10000
INPUT_DIM    = 18
NOISE_STD    = 0.1

TON_RENAME = {
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

FEATURES = [
    'destination_port', 'flow_bytes_s', 'flow_duration',
    'fwd_avg_bytes_bulk', 'fwd_avg_packets_bulk',
    'fwd_iat_mean', 'fwd_iat_min', 'fwd_iat_std',
    'fwd_packet_length_mean', 'init_win_bytes_backward',
    'init_win_bytes_forward', 'subflow_fwd_bytes',
    'bwd_avg_bytes_bulk', 'bwd_iat_mean', 'bwd_iat_std',
    'bwd_packet_length_mean', 'bwd_packet_length_std',
    'bwd_packets_s'
]


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


def train_dae(X_train, X_val, seed):
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
            noisy = torch.clamp(
                batch + torch.randn_like(batch)*NOISE_STD, -10, 10)
            opt.zero_grad()
            crit(model(noisy), batch).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            noisy_v = torch.clamp(
                X_v + torch.randn_like(X_v)*NOISE_STD, -10, 10)
            val_loss = crit(model(noisy_v), X_v).item()
        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
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
            scores.append(((b - model(b))**2).mean(dim=1).numpy())
    return np.concatenate(scores)


def norm(scores, p1, p99):
    return np.clip((scores - p1) / (p99 - p1 + 1e-8), 0, 1)


def run():
    log.info("Loading training data...")
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train = np.clip(d_train["X_train"], -10, 10).astype(np.float32)

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val = np.clip(d_val["X_val"], -10, 10).astype(np.float32)

    # Train or load DAE-Multi
    model_path = os.path.join(DEDUP_DIR, "seed42_dae", "dae_multi.pt")
    model = DAEModel(INPUT_DIM)
    if os.path.exists(model_path):
        log.info("Loading saved DAE-Multi...")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        log.info("Training DAE-Multi (seed %d)...", SEED)
        model = train_dae(X_train, X_val, SEED)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

    # Normalization
    train_scores = get_scores(model, X_train[:50000])
    p1  = float(np.percentile(train_scores, 1))
    p99 = float(np.percentile(train_scores, 99))

    # Load ToN-IoT with attack type labels
    log.info("Loading CIC-ToN-IoT...")
    df = pd.read_csv(TON_CSV, nrows=500000, low_memory=False)
    df.columns = (df.columns.str.strip().str.lower()
                 .str.replace(r'\s+', '_', regex=True)
                 .str.replace(r'[^a-z0-9_]', '_', regex=True)
                 .str.strip('_'))
    df.rename(columns=TON_RENAME, inplace=True)

    # Get attack type column
    type_col = None
    for c in ['type', 'attack_type', 'label', 'attack']:
        if c in df.columns:
            type_col = c
            break
    log.info("Type column: %s", type_col)
    log.info("Unique types: %s", df[type_col].unique()[:15])

    # Binary label
    if type_col:
        normal_vals = ['normal', 'Normal', 'NORMAL', '0', 0, 'benign', 'Benign']
        df['y'] = (~df[type_col].isin(normal_vals)).astype(int)
    else:
        df['y'] = 1

    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    # Scale
    scaler_data = np.load(os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"))
    q1  = scaler_data['q1']
    iqr = scaler_data['iqr']
    iqr[iqr == 0] = 1.0
    X_raw = df[FEATURES].values.astype(np.float32)
    X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
    X_scaled = np.clip((X_raw - q1) / iqr, -10, 10)

    # Local calibration from benign flows
    benign_idx = np.where(df['y'].values == 0)[0]
    rng = np.random.default_rng(SEED)
    cal_idx = rng.choice(benign_idx, min(N_CAL, len(benign_idx)), replace=False)
    X_cal = X_scaled[cal_idx]
    cal_scores = norm(get_scores(model, X_cal), p1, p99)
    tau = float(np.percentile(cal_scores, 95))
    log.info("Local calibration tau=%.4f", tau)

    # Score all flows
    all_scores = norm(get_scores(model, X_scaled), p1, p99)
    y_pred = (all_scores > tau).astype(int)
    y_true = df['y'].values

    # Overall AUC
    auc = roc_auc_score(y_true, all_scores)
    log.info("Overall AUC=%.4f", auc)

    # Per-family recall
    if type_col:
        attack_types = df[type_col].unique()
        normal_vals  = ['normal', 'Normal', 'NORMAL', '0', 0, 'benign', 'Benign']
        attack_families = [t for t in attack_types if t not in normal_vals]
        log.info("Attack families: %s", attack_families)

        results = []
        for family in sorted(attack_families):
            mask    = (df[type_col] == family).values
            n_flows = mask.sum()
            if n_flows == 0:
                continue
            recall  = y_pred[mask].mean()
            results.append({
                'family':  family,
                'n_flows': n_flows,
                'recall':  recall,
            })
            log.info("  %s: n=%d recall=%.4f (%.1f%%)",
                     family, n_flows, recall, recall*100)

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('recall', ascending=False)

        # ZDDR = macro-averaged recall
        zddr = df_results['recall'].mean()
        log.info("ZDDR (macro avg recall): %.4f (%.1f%%)", zddr, zddr*100)

        # Micro recall
        micro = y_pred[y_true==1].mean()
        log.info("Micro recall (weighted): %.4f (%.1f%%)", micro, micro*100)

        # Save
        out_path = os.path.join(DEDUP_DIR, "ton_per_family.csv")
        df_results.to_csv(out_path, index=False)
        log.info("Saved -> %s", out_path)

        print("\n" + "="*55)
        print("PER-ATTACK-FAMILY RESULTS (DAE-Multi, seed 42)")
        print("="*55)
        print(f"Overall AUC: {auc:.4f}")
        print(f"ZDDR (macro): {zddr:.4f} ({zddr*100:.1f}%)")
        print(f"Micro recall: {micro:.4f} ({micro*100:.1f}%)")
        print()
        print(df_results.to_string(index=False))


if __name__ == "__main__":
    run()
