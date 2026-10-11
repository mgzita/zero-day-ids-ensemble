"""
run_ton_all_models.py
=====================
Evaluates ALL SIX models on CIC-ToN-IoT
using the DEDUP pipeline (train_dedup.npz,
val_dedup.npz) — same pipeline as
ablation_dae_dedup.csv that gives UNSW=0.943.

The previous script had tau=1.000 because
it retrained with wrong normalization.
This script uses identical pipeline to
the dedup ablation results.

Models:
  AE-Single, DAE-Single, SVDD-Single
  AE-Multi, DAE-Multi, SVDD-Multi

Output:
  artefacts_v2/dedup/ton_all_models_results.csv

Run:
  python run_ton_all_models.py

Time: ~6 hours
"""
import sys, os, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"
DEDUP_DIR    = os.path.join(ARTEFACT_DIR, "dedup")
TON_CSV      = r"C:/MLProject/zero_day_project/data/raw/cic_ton_iot/CIC-ToN-IoT.csv"
SEEDS        = [42, 123, 456, 789, 999]
N_CAL        = 10000
INPUT_DIM    = 18
NOISE_STD    = 0.1

# Feature list must match dedup pipeline exactly
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


# ── Model definitions ──────────────────────────────────────────────────────

class AEModel(nn.Module):
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


class SVDDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18,64,bias=False), nn.ReLU(),
            nn.Linear(64,32,bias=False), nn.ReLU(),
            nn.Linear(32,16,bias=False),
        )
    def forward(self, x):
        return self.net(x)


# ── Training ───────────────────────────────────────────────────────────────

def train_ae(X_tr, X_va, seed, noise=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AEModel()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    crit  = nn.HuberLoss(delta=1.0)
    X_t   = torch.FloatTensor(X_tr)
    X_v   = torch.FloatTensor(X_va)
    best_loss, pat, best_state = float('inf'), 0, None
    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            b = X_t[idx[i:i+256]]
            inp = torch.clamp(
                b + torch.randn_like(b)*NOISE_STD, -10, 10) if noise else b
            opt.zero_grad(); crit(model(inp), b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            inp_v = torch.clamp(
                X_v + torch.randn_like(X_v)*NOISE_STD,
                -10, 10) if noise else X_v
            vl = crit(model(inp_v), X_v).item()
        sched.step(vl)
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 10:
                log.info("  Early stop epoch %d", epoch+1)
                break
    model.load_state_dict(best_state)
    return model


def train_svdd(X_tr, X_va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SVDDNet()
    model.eval()
    with torch.no_grad():
        z = model(torch.FloatTensor(X_tr[:5000]))
        c = z.mean(dim=0)
        c[(abs(c) < 0.01)] = 0.01
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    X_t   = torch.FloatTensor(X_tr)
    X_v   = torch.FloatTensor(X_va)
    best_loss, pat, best_state = float('inf'), 0, None
    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            b = X_t[idx[i:i+256]]
            opt.zero_grad()
            ((model(b)-c)**2).sum(dim=1).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = ((model(X_v)-c)**2).sum(dim=1).mean().item()
        sched.step(vl)
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 10:
                log.info("  Early stop epoch %d", epoch+1)
                break
    model.load_state_dict(best_state)
    return model, c


# ── Scoring ────────────────────────────────────────────────────────────────

def ae_scores(model, X):
    model.eval(); out = []
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        for i in range(0, len(X_t), 10000):
            b = X_t[i:i+10000]
            out.append(((b-model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def svdd_scores(model, c, X):
    model.eval(); out = []
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        for i in range(0, len(X_t), 10000):
            z = model(X_t[i:i+10000])
            out.append(((z-c)**2).sum(dim=1).numpy())
    return np.concatenate(out)


def norm(s, p1, p99):
    return np.clip((s-p1)/(p99-p1+1e-8), 0, 1)


# ── Load ToN-IoT using DEDUP IQR scaler ───────────────────────────────────

def load_ton():
    log.info("Loading CIC-ToN-IoT...")
    df = pd.read_csv(TON_CSV, nrows=500000, low_memory=False)
    df.columns = (df.columns.str.strip().str.lower()
                 .str.replace(r'\s+', '_', regex=True)
                 .str.replace(r'[^a-z0-9_]', '_', regex=True)
                 .str.strip('_'))
    df.rename(columns=TON_RENAME, inplace=True)

    # Label
    for col in ['type', 'attack_type', 'label']:
        if col in df.columns:
            normal = ['normal','Normal','NORMAL','0',0,'benign','Benign']
            df['y'] = (~df[col].isin(normal)).astype(int)
            log.info("Label from column: %s", col)
            break

    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    X_raw = df[FEATURES].values.astype(np.float32)
    X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
    y     = df['y'].values.astype(int)

    # Apply DEDUP IQR scaler — same scaler used for all dedup experiments
    scaler = np.load(os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"))
    q1  = scaler['q1']
    iqr = scaler['iqr']
    iqr[iqr == 0] = 1.0
    X = np.clip((X_raw - q1) / iqr, -10, 10).astype(np.float32)

    log.info("ToN: %d rows, %d benign, %d attack",
             len(y), (y==0).sum(), (y==1).sum())
    return X, y


def run():
    # ── Load dedup training data (same as ablation_dae_dedup.csv) ──────────
    log.info("Loading dedup training data...")
    d = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_full = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n_per  = len(X_full) // 3
    X_single = X_full[:n_per]
    log.info("Single: %d  Multi: %d flows", len(X_single), len(X_full))

    d = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full   = np.clip(d["X_val"], -10, 10).astype(np.float32)
    X_val_single = X_val_full[:len(X_val_full)//3]

    # ── Load ToN ────────────────────────────────────────────────────────────
    X_ton, y_ton = load_ton()

    # Split calibration from test
    benign_idx = np.where(y_ton == 0)[0]
    attack_idx = np.where(y_ton == 1)[0]
    rng        = np.random.default_rng(42)
    cal_idx    = rng.choice(benign_idx, min(N_CAL, len(benign_idx)),
                            replace=False)
    X_cal      = X_ton[cal_idx]
    test_idx   = np.concatenate(
        [np.setdiff1d(benign_idx, cal_idx), attack_idx])
    rng.shuffle(test_idx)
    X_test = X_ton[test_idx]
    y_test = y_ton[test_idx]
    log.info("Cal: %d  Test: %d (%d benign, %d attack)",
             len(X_cal), len(y_test),
             (y_test==0).sum(), (y_test==1).sum())

    # ── Models to evaluate ──────────────────────────────────────────────────
    models_cfg = [
        ("AE-Single",   "ae",   X_single,  X_val_single),
        ("DAE-Single",  "dae",  X_single,  X_val_single),
        ("SVDD-Single", "svdd", X_single,  X_val_single),
        ("AE-Multi",    "ae",   X_full,    X_val_full),
        ("DAE-Multi",   "dae",  X_full,    X_val_full),
        ("SVDD-Multi",  "svdd", X_full,    X_val_full),
    ]

    all_results = []

    for name, mtype, X_tr, X_va in models_cfg:
        log.info("=" * 55)
        log.info("Evaluating %s across %d seeds", name, len(SEEDS))

        seed_aucs = []

        for seed in SEEDS:
            log.info("  Seed %d...", seed)
            t0 = time.time()

            # Train
            if mtype == "svdd":
                model, c = train_svdd(X_tr, X_va, seed)
                train_s  = svdd_scores(model, c, X_tr[:50000])
            else:
                model = train_ae(X_tr, X_va, seed, noise=(mtype=="dae"))
                train_s = ae_scores(model, X_tr[:50000])

            p1  = float(np.percentile(train_s, 1))
            p99 = float(np.percentile(train_s, 99))

            # Local calibration threshold from ToN benign flows
            if mtype == "svdd":
                cal_s  = norm(svdd_scores(model, c, X_cal), p1, p99)
                test_s = norm(svdd_scores(model, c, X_test), p1, p99)
            else:
                cal_s  = norm(ae_scores(model, X_cal), p1, p99)
                test_s = norm(ae_scores(model, X_test), p1, p99)

            tau = float(np.percentile(cal_s, 95))
            auc = roc_auc_score(y_test, test_s)

            log.info("    AUC=%.4f tau=%.4f (%.1f min)",
                     auc, tau, (time.time()-t0)/60)

            # Tau=1.000 warning
            if tau > 0.99:
                log.warning("    tau=1.000 detected — "
                            "calibration may be broken for this seed")

            seed_aucs.append(auc)
            all_results.append({
                'model': name, 'seed': seed,
                'ToN_AUC': auc, 'tau': tau
            })

        m = np.mean(seed_aucs)
        s = np.std(seed_aucs)
        log.info("%s: mean=%.4f std=%.4f", name, m, s)

    df = pd.DataFrame(all_results)
    out = os.path.join(DEDUP_DIR, "ton_all_models_results.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)

    print("\n" + "="*60)
    print("TON-IOT RESULTS (dedup pipeline, 5 seeds each)")
    print("="*60)
    summary = df.groupby('model')['ToN_AUC'].agg(['mean','std'])
    summary.columns = ['Mean AUC', 'Std']
    print(summary.to_string())

    print("\nKEY: Do multi-env models outperform single-env on ToN?")
    for base in ['AE', 'DAE', 'SVDD']:
        s_auc = df[df['model']==f'{base}-Single']['ToN_AUC'].mean()
        m_auc = df[df['model']==f'{base}-Multi']['ToN_AUC'].mean()
        print(f"  {base}: Single={s_auc:.4f} Multi={m_auc:.4f} "
              f"Gain={m_auc-s_auc:+.4f}")


if __name__ == "__main__":
    run()
