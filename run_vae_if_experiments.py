"""
run_vae_if_experiments.py
=========================
Runs VAE and Isolation Forest under the same
experimental conditions as AE, DAE, Deep SVDD:

Conditions:
  - Single-environment: 536,307 CIC17 flows
  - Multi-environment full: 1,608,922 diverse flows
  - 5 seeds: 42, 123, 456, 789, 999
  - Same dedup npz files as other models
  - Same threshold: p95 of validation benign scores
  - Same evaluation: CIC17, CIC18, UNSW AUC + BoT DR

Output:
  artefacts_v2/dedup/vae_results.csv
  artefacts_v2/dedup/if_results.csv

Expected time: ~6 hours
"""
import os, time, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEDUP_DIR = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
SEEDS     = [42, 123, 456, 789, 999]
NOISE_STD = 0.1
BETA      = 1.0  # KL weight


# ── VAE Model ─────────────────────────────────────────────────────────────

class VAEModel(nn.Module):
    def __init__(self, input_dim=18, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, 64),         nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.huber_loss(recon, x, delta=1.0)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + BETA * kl


def train_vae(X_train, X_val, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = VAEModel()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    X_t   = torch.FloatTensor(X_train)
    X_v   = torch.FloatTensor(X_val)
    best, pat, bs = float('inf'), 0, None
    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            b = X_t[idx[i:i+256]]
            opt.zero_grad()
            recon, mu, logvar = model(b)
            vae_loss(recon, b, mu, logvar).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            recon_v, mu_v, lv_v = model(X_v)
            vl = vae_loss(recon_v, X_v, mu_v, lv_v).item()
        sched.step(vl)
        if vl < best:
            best = vl
            bs = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 10:
                log.info("  Early stop ep %d", epoch+1)
                break
    model.load_state_dict(bs)
    return model


def vae_scores(model, X):
    """Anomaly score = reconstruction error (same as AE for comparability)."""
    model.eval(); out = []
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        for i in range(0, len(X_t), 10000):
            b = X_t[i:i+10000]
            recon, mu, logvar = model(b)
            out.append(((b - recon)**2).mean(dim=1).numpy())
    return np.concatenate(out)


# ── Scoring utilities ──────────────────────────────────────────────────────

def norm(s, p1, p99):
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


def evaluate(scores_norm, y, tau):
    auc = roc_auc_score(y, scores_norm)
    dr  = float((scores_norm[y == 1] > tau).mean()) if (y == 1).any() else float('nan')
    return auc, dr


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    log.info("Loading dedup data...")
    d = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_full   = np.clip(d["X_train"], -10, 10).astype(np.float32)
    X_single = X_full[:len(X_full)//3]

    d = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full   = np.clip(d["X_val"], -10, 10).astype(np.float32)
    X_val_single = X_val_full[:len(X_val_full)//3]

    log.info("Single: %d  Multi: %d", len(X_single), len(X_full))

    # Test sets
    test_data = {}
    for name, fname in [
        ('CIC17', 'test_cic17_dedup.npz'),
        ('CIC18', 'test_cic18_dedup.npz'),
        ('UNSW',  'test_unsw_dedup.npz'),
        ('BoT',   'test_botiot_dedup.npz'),
    ]:
        d = np.load(os.path.join(DEDUP_DIR, fname))
        test_data[name] = (
            np.clip(d["X"], -10, 10).astype(np.float32),
            d["y"].astype(int))
        log.info("Loaded %s: %d rows", name, len(test_data[name][0]))

    # ── VAE experiments ────────────────────────────────────────────────────
    vae_results = []

    for condition, X_tr, X_va in [
        ("VAE-Single", X_single,  X_val_single),
        ("VAE-Multi",  X_full,    X_val_full),
    ]:
        log.info("="*55)
        log.info("%s (%d seeds)", condition, len(SEEDS))

        for seed in SEEDS:
            log.info("  Seed %d...", seed)
            t0 = time.time()

            model  = train_vae(X_tr, X_va, seed)
            tr_s   = vae_scores(model, X_tr[:50000])
            p1     = float(np.percentile(tr_s, 1))
            p99    = float(np.percentile(tr_s, 99))
            val_s  = norm(vae_scores(model, X_va), p1, p99)
            tau    = float(np.percentile(val_s, 95))

            row = {'condition': condition, 'seed': seed}
            for name, (X_test, y_test) in test_data.items():
                s = norm(vae_scores(model, X_test), p1, p99)
                if name == 'BoT':
                    row['BoT_DR'] = float((s > tau).mean())
                    log.info("    %s: DR=%.4f", name, row['BoT_DR'])
                else:
                    auc = roc_auc_score(y_test, s)
                    row[f'{name}_AUC'] = auc
                    log.info("    %s: AUC=%.4f", name, auc)

            log.info("  Seed %d done in %.1f min", seed, (time.time()-t0)/60)
            vae_results.append(row)

    df_vae = pd.DataFrame(vae_results)
    out = os.path.join(DEDUP_DIR, "vae_results.csv")
    df_vae.to_csv(out, index=False)
    log.info("Saved -> %s", out)

    # ── IF experiments ─────────────────────────────────────────────────────
    if_results = []

    for condition, X_tr, X_va in [
        ("IF-Single", X_single,  X_val_single),
        ("IF-Multi",  X_full,    X_val_full),
    ]:
        log.info("="*55)
        log.info("%s (%d seeds)", condition, len(SEEDS))

        for seed in SEEDS:
            log.info("  Seed %d...", seed)
            t0 = time.time()

            model = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=seed,
                n_jobs=-1)
            model.fit(X_tr)

            # IF scores: more negative = more anomalous
            # Negate so higher = more anomalous (consistent with other models)
            tr_s  = -model.score_samples(X_tr[:50000])
            p1    = float(np.percentile(tr_s, 1))
            p99   = float(np.percentile(tr_s, 99))
            val_s = norm(-model.score_samples(X_va), p1, p99)
            tau   = float(np.percentile(val_s, 95))

            row = {'condition': condition, 'seed': seed}
            for name, (X_test, y_test) in test_data.items():
                s = norm(-model.score_samples(X_test), p1, p99)
                if name == 'BoT':
                    row['BoT_DR'] = float((s > tau).mean())
                    log.info("    %s: DR=%.4f", name, row['BoT_DR'])
                else:
                    auc = roc_auc_score(y_test, s)
                    row[f'{name}_AUC'] = auc
                    log.info("    %s: AUC=%.4f", name, auc)

            log.info("  Seed %d done in %.1f min", seed, (time.time()-t0)/60)
            if_results.append(row)

    df_if = pd.DataFrame(if_results)
    out = os.path.join(DEDUP_DIR, "if_results.csv")
    df_if.to_csv(out, index=False)
    log.info("Saved -> %s", out)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("VAE AND IF RESULTS (mean ± pop std, 5 seeds)")
    print("="*65)

    for df, name in [(df_vae, "VAE"), (df_if, "IF")]:
        print(f"\n{name}:")
        print(f"  {'Condition':<15} {'CIC17':>8} {'CIC18':>8} "
              f"{'UNSW':>8} {'BoT DR':>8}")
        print(f"  {'-'*55}")
        for cond in [f"{name}-Single", f"{name}-Multi"]:
            sub = df[df['condition']==cond]
            c17 = sub['CIC17_AUC'].mean() if 'CIC17_AUC' in sub else float('nan')
            c18 = sub['CIC18_AUC'].mean() if 'CIC18_AUC' in sub else float('nan')
            uns = sub['UNSW_AUC'].mean()  if 'UNSW_AUC' in sub else float('nan')
            bot = sub['BoT_DR'].mean()    if 'BoT_DR' in sub else float('nan')
            print(f"  {cond:<15} {c17:>8.3f} {c18:>8.3f} "
                  f"{uns:>8.3f} {bot:>8.3f}")

        # Diversity effect
        single = df[df['condition']==f"{name}-Single"]['UNSW_AUC'].mean()
        multi  = df[df['condition']==f"{name}-Multi']['UNSW_AUC'].mean()
        print(f"\n  UNSW diversity effect: {single:.3f} → {multi:.3f} "
              f"({multi-single:+.3f})")


if __name__ == "__main__":
    run()
