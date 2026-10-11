"""
run_ton_final.py
================
Evaluates all six models on CIC-ToN-IoT
following EXACTLY the same approach as
run_multiseed_v3.py which produced 0.953.

Key fixes from previous broken scripts:
1. IQR scaler uses Q2 (median) not Q1
   z = (x - Q2) / IQR  [not (x - Q1) / IQR]
2. Score normalization uses norm_local()
   which normalizes by each score's OWN
   percentiles, not training percentiles
3. Threshold from ToN benign scores directly

Output:
  artefacts_v2/dedup/ton_final_results.csv

Time: ~12 hours (6 models x 5 seeds)
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


# ── Exact norm_local from run_multiseed_v3.py ─────────────────────────────

def norm_local(s):
    """Normalize scores by their OWN percentiles — not training percentiles."""
    lo = float(np.percentile(s, 1))
    hi = float(np.percentile(s, 99))
    if hi - lo < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ── IQR scaler transform using Q2 (median) ────────────────────────────────

def iqr_transform(X, q2, iqr, clip=10.0):
    """z = (x - Q2) / IQR — same as IQRScaler.transform()"""
    z = (X - q2) / iqr
    return np.clip(z, -clip, clip).astype(np.float32)


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
    best, pat, bs = float('inf'), 0, None
    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            b = X_t[idx[i:i+256]]
            inp = torch.clamp(
                b+torch.randn_like(b)*NOISE_STD,-10,10) if noise else b
            opt.zero_grad(); crit(model(inp),b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            iv = torch.clamp(
                X_v+torch.randn_like(X_v)*NOISE_STD,-10,10) if noise else X_v
            vl = crit(model(iv),X_v).item()
        sched.step(vl)
        if vl < best:
            best=vl; bs={k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=10: log.info("  Early stop ep %d",epoch+1); break
    model.load_state_dict(bs)
    return model


def train_svdd(X_tr, X_va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SVDDNet()
    model.eval()
    with torch.no_grad():
        z=model(torch.FloatTensor(X_tr[:5000]))
        c=z.mean(dim=0); c[(abs(c)<0.01)]=0.01
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-6)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5)
    X_t=torch.FloatTensor(X_tr); X_v=torch.FloatTensor(X_va)
    best,pat,bs=float('inf'),0,None
    for epoch in range(100):
        model.train()
        idx=torch.randperm(len(X_t))
        for i in range(0,len(X_t),256):
            b=X_t[idx[i:i+256]]; opt.zero_grad()
            ((model(b)-c)**2).sum(dim=1).mean().backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl=((model(X_v)-c)**2).sum(dim=1).mean().item()
        sched.step(vl)
        if vl<best:
            best=vl; bs={k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=10: log.info("  Early stop ep %d",epoch+1); break
    model.load_state_dict(bs)
    return model, c


# ── Scoring ────────────────────────────────────────────────────────────────

def ae_scores(model, X):
    model.eval(); out=[]
    with torch.no_grad():
        X_t=torch.FloatTensor(X)
        for i in range(0,len(X_t),10000):
            b=X_t[i:i+10000]
            out.append(((b-model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def svdd_scores(model, c, X):
    model.eval(); out=[]
    with torch.no_grad():
        X_t=torch.FloatTensor(X)
        for i in range(0,len(X_t),10000):
            z=model(X_t[i:i+10000])
            out.append(((z-c)**2).sum(dim=1).numpy())
    return np.concatenate(out)


# ── Load ToN-IoT using training IQR scaler (Q2 version) ───────────────────

def load_ton():
    log.info("Loading CIC-ToN-IoT...")

    # Load training IQR scaler — use Q2 (median) as original script does
    scaler = np.load(os.path.join(ARTEFACT_DIR, "iqr_scaler.npz"))
    q2  = scaler['q2']   # median — this is what IQRScaler.transform uses
    iqr = scaler['iqr']
    iqr[iqr == 0] = 1.0

    frames = []
    total  = 0
    for chunk in pd.read_csv(TON_CSV, chunksize=100000, low_memory=False):
        chunk.columns = (chunk.columns.str.strip().str.lower()
                        .str.replace(r'\s+','_',regex=True)
                        .str.replace(r'[^a-z0-9_]','_',regex=True)
                        .str.strip('_'))
        chunk.rename(columns=TON_RENAME, inplace=True)
        chunk['y'] = (chunk['label'] != 0).astype(int) \
            if 'label' in chunk.columns else 1
        for f in FEATURES:
            if f not in chunk.columns:
                chunk[f] = 0.0
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0.0, inplace=True)
        frames.append(chunk)
        total += len(chunk)
        if total >= 500000:
            break

    df = pd.concat(frames, ignore_index=True).iloc[:500000]
    X_raw = df[FEATURES].values.astype(np.float32)
    X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
    y     = df['y'].values.astype(int)

    # Apply training IQR scaler using Q2 (same as original)
    X = iqr_transform(X_raw, q2, iqr, clip=10.0)

    log.info("ToN: %d rows, %d benign, %d attack",
             len(y), (y==0).sum(), (y==1).sum())
    return X, y


def run():
    log.info("Loading dedup training data...")
    d = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_full   = np.clip(d["X_train"], -10, 10).astype(np.float32)
    X_single = X_full[:len(X_full)//3]

    d = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full   = np.clip(d["X_val"], -10, 10).astype(np.float32)
    X_val_single = X_val_full[:len(X_val_full)//3]

    log.info("Single: %d  Multi: %d", len(X_single), len(X_full))

    # Load ToN
    X_ton, y_ton = load_ton()

    # Fixed calibration split (same rng as original)
    rng        = np.random.default_rng(42)
    benign_idx = np.where(y_ton == 0)[0]
    cal_idx    = rng.choice(benign_idx, min(N_CAL, len(benign_idx)),
                            replace=False)
    X_cal      = X_ton[cal_idx]
    y_cal      = y_ton[cal_idx]
    # Test = everything
    X_test = X_ton
    y_test = y_ton
    log.info("Cal benign: %d  Test: %d", len(X_cal), len(X_test))

    models_cfg = [
        ("AE-Single",   "ae",   X_single, X_val_single),
        ("DAE-Single",  "dae",  X_single, X_val_single),
        ("SVDD-Single", "svdd", X_single, X_val_single),
        ("AE-Multi",    "ae",   X_full,   X_val_full),
        ("DAE-Multi",   "dae",  X_full,   X_val_full),
        ("SVDD-Multi",  "svdd", X_full,   X_val_full),
    ]

    all_results = []

    for name, mtype, X_tr, X_va in models_cfg:
        log.info("="*55)
        log.info("Evaluating %s (%d seeds)", name, len(SEEDS))
        seed_aucs = []

        for seed in SEEDS:
            log.info("  Seed %d...", seed)
            t0 = time.time()

            # Train
            if mtype == "svdd":
                model, c = train_svdd(X_tr, X_va, seed)
                raw_all  = svdd_scores(model, c, X_test)
            else:
                model   = train_ae(X_tr, X_va, seed, noise=(mtype=="dae"))
                raw_all = ae_scores(model, X_test)

            # KEY: normalize by score's OWN percentiles (norm_local)
            norm_all = norm_local(raw_all)

            # Threshold from ToN benign calibration flows
            norm_cal = norm_all[cal_idx]
            tau      = float(np.percentile(norm_cal, 95))

            # AUC on full test set
            auc = roc_auc_score(y_test, norm_all)

            log.info("    tau=%.4f AUC=%.4f (%.1f min)",
                     tau, auc, (time.time()-t0)/60)

            seed_aucs.append(auc)
            all_results.append({
                'model': name, 'seed': seed,
                'ToN_AUC': auc, 'tau': tau,
            })

        m = np.mean(seed_aucs)
        s = np.std(seed_aucs)
        log.info("%s: mean=%.4f std=%.4f", name, m, s)

    df = pd.DataFrame(all_results)
    out = os.path.join(DEDUP_DIR, "ton_final_results.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)

    print("\n" + "="*60)
    print("TON-IOT FINAL RESULTS (norm_local, Q2 scaler, 5 seeds)")
    print("="*60)
    summary = df.groupby('model')['ToN_AUC'].agg(['mean','std'])
    summary.columns = ['Mean AUC', 'Std']
    print(summary.to_string())

    print("\nDiversity effect:")
    for base in ['AE', 'DAE', 'SVDD']:
        s_auc = df[df['model']==f'{base}-Single']['ToN_AUC'].mean()
        m_auc = df[df['model']==f'{base}-Multi']['ToN_AUC'].mean()
        print(f"  {base}: Single={s_auc:.4f} Multi={m_auc:.4f} "
              f"({m_auc-s_auc:+.4f})")

    print("\nDAE-Multi expected: ~0.950")
    dae_multi = df[df['model']=='DAE-Multi']['ToN_AUC']
    print(f"DAE-Multi actual: {dae_multi.mean():.4f} +/- {dae_multi.std():.4f}")


if __name__ == "__main__":
    run()
