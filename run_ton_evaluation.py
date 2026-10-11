"""
run_ton_evaluation.py
=====================
Evaluates all five models on CIC-ToN-IoT
using local benign calibration (10,000 flows).

Models evaluated:
- AE-Single  (retrained from scratch)
- DAE-Single (retrained from scratch)
- SVDD-Single (retrained from scratch)
- AE-Multi   (retrained from scratch)
- SVDD-Multi (retrained from scratch)
- DAE-Multi  (retrained from scratch)

Output:
  artefacts_v2/dedup/ton_all_results.csv

Run from project root:
    python run_ton_evaluation.py

Expected time: ~6 hours (6 models x seed 42)
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
SEED         = 42
INPUT_DIM    = 18
NOISE_STD    = 0.1
N_CAL        = 10000  # local calibration flows

# Update this path to your ToN-IoT CSV location
TON_CSV = r"C:/MLProject/zero_day_project/data/raw/cic_ton_iot/CIC-ToN-IoT.csv"

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


# ── Model definitions ──────────────────────────────────────────────────────

class AEModel(nn.Module):
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


# ── Training functions ─────────────────────────────────────────────────────

def train_ae(X_train, X_val, seed, noise=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cpu')
    model  = AEModel(INPUT_DIM).to(device)
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
            inp   = torch.clamp(batch + torch.randn_like(batch)*NOISE_STD,
                                -10, 10) if noise else batch
            opt.zero_grad()
            crit(model(inp), batch).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            inp_v = torch.clamp(X_v + torch.randn_like(X_v)*NOISE_STD,
                                -10, 10) if noise else X_v
            val_loss = crit(model(inp_v), X_v).item()
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


def train_svdd(X_train, X_val, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cpu')
    model  = SVDDNetwork(INPUT_DIM).to(device)
    # Init center
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_train[:5000])
        z   = model(X_t)
        c   = z.mean(dim=0)
        c[(abs(c) < 0.01)] = 0.01
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    X_t   = torch.FloatTensor(X_train)
    X_v   = torch.FloatTensor(X_val)
    best_loss, patience_c, best_state = float('inf'), 0, None
    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            batch = X_t[idx[i:i+256]]
            opt.zero_grad()
            ((model(batch) - c)**2).sum(dim=1).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = ((model(X_v) - c)**2).sum(dim=1).mean().item()
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
    return model, c


# ── Scoring helpers ────────────────────────────────────────────────────────

def ae_scores(model, X):
    model.eval()
    scores = []
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        for i in range(0, len(X_t), 10000):
            b = X_t[i:i+10000]
            scores.append(((b - model(b))**2).mean(dim=1).numpy())
    return np.concatenate(scores)


def svdd_scores(model, c, X):
    model.eval()
    scores = []
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        for i in range(0, len(X_t), 10000):
            z = model(X_t[i:i+10000])
            scores.append(((z - c)**2).sum(dim=1).numpy())
    return np.concatenate(scores)


def norm(scores, p1, p99):
    return np.clip((scores - p1) / (p99 - p1 + 1e-8), 0, 1)


# ── Load ToN-IoT ───────────────────────────────────────────────────────────

def load_ton(csv_path, scaler_path, feat_path):
    log.info("Loading CIC-ToN-IoT from %s", csv_path)
    df = pd.read_csv(csv_path, nrows=500000, low_memory=False)
    df.columns = (df.columns.str.strip().str.lower()
                 .str.replace(r'\s+', '_', regex=True)
                 .str.replace(r'[^a-z0-9_]', '_', regex=True)
                 .str.strip('_'))
    df.rename(columns=TON_RENAME, inplace=True)

    # Label
    if 'label' in df.columns:
        df['y'] = (df['label'] != 0).astype(int)
    elif 'type' in df.columns:
        df['y'] = (df['type'] != 'Normal').astype(int)
    else:
        df['y'] = 1

    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    X_raw = df[FEATURES].values.astype(np.float32)
    X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
    y     = df['y'].values.astype(int)

    # Apply IQR scaler
    scaler_data = np.load(scaler_path)
    q1  = scaler_data['q1']
    iqr = scaler_data['iqr']
    iqr[iqr == 0] = 1.0
    X_scaled = np.clip((X_raw - q1) / iqr, -10, 10)

    log.info("ToN-IoT: %d rows, %d benign, %d attack",
             len(y), (y==0).sum(), (y==1).sum())
    return X_scaled, y


def run():
    log.info("Loading training data...")
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(d_train["X_train"], -10, 10).astype(np.float32)
    n_per_env    = len(X_train_full) // 3
    X_single     = X_train_full[:n_per_env]

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full    = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    n_val_per_env = len(X_val_full) // 3
    X_val_single  = X_val_full[:n_val_per_env]

    # Load ToN-IoT
    scaler_path = os.path.join(ARTEFACT_DIR, "iqr_scaler.npz")
    feat_path   = os.path.join(ARTEFACT_DIR, "selected_features.csv")
    X_ton, y_ton = load_ton(TON_CSV, scaler_path, feat_path)

    # Split ToN into calibration (benign) and test
    benign_idx  = np.where(y_ton == 0)[0]
    attack_idx  = np.where(y_ton == 1)[0]
    rng         = np.random.default_rng(SEED)
    cal_idx     = rng.choice(benign_idx, min(N_CAL, len(benign_idx)),
                             replace=False)
    X_cal       = X_ton[cal_idx]
    log.info("Calibration benign: %d flows", len(X_cal))

    # Test set: remaining benign + all attack
    remaining_benign = np.setdiff1d(benign_idx, cal_idx)
    test_idx  = np.concatenate([remaining_benign, attack_idx])
    rng.shuffle(test_idx)
    X_test    = X_ton[test_idx]
    y_test    = y_ton[test_idx]
    log.info("Test set: %d rows (%d benign, %d attack)",
             len(y_test), (y_test==0).sum(), (y_test==1).sum())

    results = []

    models_to_run = [
        ("AE-Single",   "ae",   False, X_single,     X_val_single),
        ("DAE-Single",  "dae",  False, X_single,     X_val_single),
        ("SVDD-Single", "svdd", False, X_single,     X_val_single),
        ("AE-Multi",    "ae",   False, X_train_full, X_val_full),
        ("DAE-Multi",   "dae",  False, X_train_full, X_val_full),
        ("SVDD-Multi",  "svdd", False, X_train_full, X_val_full),
    ]

    for name, mtype, _, X_tr, X_va in models_to_run:
        log.info("=" * 55)
        log.info("Training %s...", name)
        t0 = time.time()

        is_dae = "DAE" in name

        if mtype == "svdd":
            model, c = train_svdd(X_tr, X_va, SEED)
            train_scores = svdd_scores(model, c, X_tr[:50000])
        else:
            model = train_ae(X_tr, X_va, SEED, noise=is_dae)
            train_scores = ae_scores(model, X_tr[:50000])

        p1  = float(np.percentile(train_scores, 1))
        p99 = float(np.percentile(train_scores, 99))

        # Local calibration threshold from ToN benign flows
        if mtype == "svdd":
            cal_scores = norm(svdd_scores(model, c, X_cal), p1, p99)
            test_scores = norm(svdd_scores(model, c, X_test), p1, p99)
        else:
            cal_scores = norm(ae_scores(model, X_cal), p1, p99)
            test_scores = norm(ae_scores(model, X_test), p1, p99)

        tau = float(np.percentile(cal_scores, 95))
        auc = roc_auc_score(y_test, test_scores)
        log.info("  %s: AUC=%.4f, tau=%.4f", name, auc, tau)

        elapsed = time.time() - t0
        log.info("  Done in %.1f min", elapsed/60)
        results.append({'model': name, 'seed': SEED,
                        'ToN_AUC': auc, 'tau': tau})

    df = pd.DataFrame(results)
    out_path = os.path.join(DEDUP_DIR, "ton_all_results.csv")
    df.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)

    print("\n" + "="*55)
    print("TON-IOT RESULTS (seed 42, local calibration)")
    print("="*55)
    print(df[['model', 'ToN_AUC']].to_string(index=False))


if __name__ == "__main__":
    run()
