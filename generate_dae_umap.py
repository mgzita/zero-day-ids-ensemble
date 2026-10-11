"""
generate_dae_umap.py
=====================
Generates UMAP visualization comparing DAE-Single vs DAE-Multi
latent representations. Shows WHY diversity helps.

Output:
  artefacts_v2/figures/umap_dae_diversity.png

Run from project root:
    python generate_dae_umap.py

Requirements:
    pip install umap-learn

Expected time: ~1 hour
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

SEED         = 42
NOISE_STD    = 0.1
INPUT_DIM    = 18
N_PER_DS     = 2000


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

    def encode(self, x):
        return self.encoder(x)


def train_dae(X_train, X_val, seed, noise_std=NOISE_STD):
    torch.manual_seed(seed)
    np.random.seed(seed)

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
            noisy = torch.clamp(
                batch + torch.randn_like(batch) * noise_std, -10, 10)
            opt.zero_grad()
            crit(model(noisy), batch).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            noisy_v = torch.clamp(
                X_v + torch.randn_like(X_v) * noise_std, -10, 10)
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


def get_latents(model, X):
    model.eval()
    latents = []
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        for i in range(0, len(X_t), 5000):
            z = model.encode(X_t[i:i+5000])
            latents.append(z.numpy())
    return np.concatenate(latents)


def load_sample(fname, n=N_PER_DS, seed=SEED):
    path = os.path.join(DEDUP_DIR, fname)
    d = np.load(path)
    X = np.clip(d["X"], -10, 10).astype(np.float32)
    y = d["y"].astype(int)
    rng = np.random.default_rng(seed)
    b_idx = np.where(y == 0)[0]
    a_idx = np.where(y == 1)[0]
    n_each = min(n // 2, len(b_idx), len(a_idx))
    idx = np.concatenate([
        rng.choice(b_idx, n_each, replace=False),
        rng.choice(a_idx, n_each, replace=False)
    ])
    rng.shuffle(idx)
    return X[idx], y[idx]


def run():
    try:
        import umap
    except ImportError:
        log.error("Install umap-learn: pip install umap-learn")
        sys.exit(1)

    log.info("Loading training data...")
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(
        d_train["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X_train_full) // 3

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full = np.clip(
        d_val["X_val"], -10, 10).astype(np.float32)
    n_val_per_env = len(X_val_full) // 3

    # DAE-Single: CIC17 only
    log.info("Training DAE-Single (CIC17 only, seed %d)...", SEED)
    dae_single = train_dae(
        X_train_full[:n_per_env],
        X_val_full[:n_val_per_env],
        SEED)
    log.info("DAE-Single trained")

    # DAE-Multi: try to load, else train
    multi_path = os.path.join(
        DEDUP_DIR, "seed42_dae", "dae_multi.pt")
    dae_multi = DAEModel(INPUT_DIM)
    if os.path.exists(multi_path):
        log.info("Loading saved DAE-Multi...")
        dae_multi.load_state_dict(
            torch.load(multi_path, map_location='cpu'))
    else:
        log.info("Training DAE-Multi (3 envs, seed %d)...", SEED)
        dae_multi = train_dae(X_train_full, X_val_full, SEED)
        os.makedirs(os.path.dirname(multi_path), exist_ok=True)
        torch.save(dae_multi.state_dict(), multi_path)
    log.info("DAE-Multi ready")

    # Load test samples
    log.info("Loading test samples...")
    datasets = {
        "CIC17": ("test_cic17_dedup.npz", "#1f77b4"),
        "CIC18": ("test_cic18_dedup.npz", "#ff7f0e"),
        "UNSW":  ("test_unsw_dedup.npz",  "#2ca02c"),
    }

    X_all, y_all, ds_labels = [], [], []
    for ds_name, (fname, _) in datasets.items():
        X, y = load_sample(fname)
        X_all.append(X)
        y_all.append(y)
        ds_labels.extend([ds_name] * len(X))
        log.info("  %s: %d samples", ds_name, len(X))

    X_all    = np.vstack(X_all)
    y_all    = np.concatenate(y_all)
    ds_labels = np.array(ds_labels)

    # Extract latents
    log.info("Extracting DAE-Single latents...")
    z_single = get_latents(dae_single, X_all)

    log.info("Extracting DAE-Multi latents...")
    z_multi = get_latents(dae_multi, X_all)

    # UMAP
    log.info("Running UMAP on DAE-Single...")
    r_single = umap.UMAP(
        n_components=2, n_neighbors=30,
        min_dist=0.1, random_state=SEED).fit_transform(z_single)

    log.info("Running UMAP on DAE-Multi...")
    r_multi = umap.UMAP(
        n_components=2, n_neighbors=30,
        min_dist=0.1, random_state=SEED).fit_transform(z_multi)

    # Plot 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "DAE Latent Space: Single- vs Multi-Environment Training\n"
        "(UMAP of encoder bottleneck, seed 42, deduplicated test sets)",
        fontsize=13, fontweight='bold', y=0.98)

    ds_color_map = {
        "CIC17": "#1f77b4",
        "CIC18": "#ff7f0e",
        "UNSW":  "#2ca02c",
    }
    label_colors = {0: "#aaaaaa", 1: "#d62728"}

    configs = [
        (axes[0, 0], "DAE-Single (1 env)\nColored by Dataset",
         "dataset", r_single),
        (axes[0, 1], "DAE-Multi (3 envs)\nColored by Dataset",
         "dataset", r_multi),
        (axes[1, 0], "DAE-Single (1 env)\nColored by Label",
         "label", r_single),
        (axes[1, 1], "DAE-Multi (3 envs)\nColored by Label",
         "label", r_multi),
    ]

    for ax, title, color_by, emb in configs:
        if color_by == "dataset":
            for ds_name, color in ds_color_map.items():
                mask = ds_labels == ds_name
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=color, s=3, alpha=0.4,
                           label=ds_name, rasterized=True)
        else:
            for label, color, lname in [
                (0, "#aaaaaa", "Benign"),
                (1, "#d62728", "Attack")
            ]:
                mask = y_all == label
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=color, s=3, alpha=0.4,
                           label=lname, rasterized=True)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("UMAP-1", fontsize=9)
        ax.set_ylabel("UMAP-2", fontsize=9)
        ax.legend(markerscale=3, loc='best', fontsize=9)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#AAAAAA')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUT_DIR, "umap_dae_diversity.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    log.info("Saved -> %s", out_path)
    log.info("Done.")


if __name__ == "__main__":
    run()
