"""
generate_umap.py
================
Generates UMAP visualization of AE latent representations
comparing single-environment vs multi-environment training.

Shows WHY multi-environment training improves generalization:
- AE-Single: latent space clusters by source domain
- AE-Multi:  latent space mixes environments, separates attack

Run from project root:
    python generate_umap.py

Requirements:
    pip install umap-learn matplotlib

Output:
    artefacts_v2/figures/umap_diversity.png
Expected time: ~1 hour
"""
import sys, os, logging
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.autoencoder import Autoencoder

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

# Sample size for UMAP (keep manageable)
N_PER_DATASET = 2000
SEED = 42


def get_latent(model, X):
    """Extract encoder latent representations."""
    model.model_.eval()
    device = next(model.model_.parameters()).device
    latents = []
    with torch.no_grad():
        for i in range(0, len(X), 5000):
            batch = torch.FloatTensor(X[i:i+5000]).to(device)
            # Get encoder output (bottleneck)
            z = model.model_.encoder(batch)
            latents.append(z.cpu().numpy())
    return np.concatenate(latents)


def load_test_sample(fname, n=N_PER_DATASET, seed=SEED):
    """Load a balanced sample from test set."""
    path = os.path.join(DEDUP_DIR, fname)
    if not os.path.exists(path):
        path = os.path.join(ARTEFACT_DIR, fname.replace('_dedup', ''))
    d = np.load(path)
    X = np.clip(d["X"], -10, 10).astype(np.float32)
    y = d["y"].astype(int)

    # Sample balanced
    rng = np.random.default_rng(seed)
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]

    n_each = min(n // 2, len(benign_idx), len(attack_idx))
    b_idx = rng.choice(benign_idx, n_each, replace=False)
    a_idx = rng.choice(attack_idx, n_each, replace=False)
    idx = np.concatenate([b_idx, a_idx])
    rng.shuffle(idx)

    return X[idx], y[idx]


def run():
    try:
        import umap
    except ImportError:
        log.error("umap-learn not installed. Run: pip install umap-learn")
        sys.exit(1)

    # ── Load AE-Single model (trained on CIC17 only) ──────────────────────
    log.info("Loading AE-Single model...")
    ae_single_path = os.path.join(DEDUP_DIR, "ablation_ae_single_dedup.csv")

    # Retrain AE-Single for latent extraction
    log.info("Training AE-Single (CIC17 only, seed 42)...")
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = d_train["X_train"]
    n = len(X_train_full)
    X_train_single = X_train_full[:n//3]  # CIC17 portion

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    X_val_single = X_val_full[:len(X_val_full)//3]

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ae_single = Autoencoder(input_dim=18)
    ae_single.fit(X_train_single, X_val_single)
    log.info("AE-Single trained")

    # ── Load AE-Multi model (dedup seed42) ────────────────────────────────
    log.info("Loading AE-Multi model (dedup seed42)...")
    ae_multi = Autoencoder(input_dim=18)
    ae_multi.load(os.path.join(DEDUP_DIR, "seed42", "ae.pt"))
    log.info("AE-Multi loaded")

    # ── Load test data samples ─────────────────────────────────────────────
    log.info("Loading test data samples...")
    datasets = {
        "CIC17":  ("test_cic17_dedup.npz",  "#1f77b4"),  # blue
        "CIC18":  ("test_cic18_dedup.npz",  "#ff7f0e"),  # orange
        "UNSW":   ("test_unsw_dedup.npz",   "#2ca02c"),  # green
    }

    X_all, y_all, ds_labels = [], [], []
    for ds_name, (fname, color) in datasets.items():
        X, y = load_test_sample(fname)
        X_all.append(X)
        y_all.append(y)
        ds_labels.extend([ds_name] * len(X))
        log.info("  %s: %d samples (%d benign, %d attack)",
                 ds_name, len(X), (y==0).sum(), (y==1).sum())

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    ds_labels = np.array(ds_labels)

    # ── Extract latent representations ────────────────────────────────────
    log.info("Extracting AE-Single latent representations...")
    z_single = get_latent(ae_single, X_all)

    log.info("Extracting AE-Multi latent representations...")
    z_multi = get_latent(ae_multi, X_all)

    # ── UMAP dimensionality reduction ─────────────────────────────────────
    log.info("Running UMAP on AE-Single latents...")
    reducer_single = umap.UMAP(
        n_components=2, n_neighbors=30,
        min_dist=0.1, random_state=SEED, verbose=False)
    emb_single = reducer_single.fit_transform(z_single)

    log.info("Running UMAP on AE-Multi latents...")
    reducer_multi = umap.UMAP(
        n_components=2, n_neighbors=30,
        min_dist=0.1, random_state=SEED, verbose=False)
    emb_multi = reducer_multi.fit_transform(z_multi)

    # ── Plot ───────────────────────────────────────────────────────────────
    log.info("Plotting...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Latent Space Comparison: Single- vs Multi-Environment AE Training\n"
        "(UMAP of encoder bottleneck representations, seed 42, test sets)",
        fontsize=13, fontweight='bold', y=0.98)

    ds_colors = {"CIC17": "#1f77b4", "CIC17": "#1f77b4",
                 "CIC18": "#ff7f0e", "UNSW": "#2ca02c"}
    label_colors = {0: "#888888", 1: "#d62728"}  # benign=gray, attack=red

    titles = [
        ("AE-Single (1 env)\nColored by Dataset", "dataset", emb_single),
        ("AE-Multi (3 envs)\nColored by Dataset", "dataset", emb_multi),
        ("AE-Single (1 env)\nColored by Label", "label", emb_single),
        ("AE-Multi (3 envs)\nColored by Label", "label", emb_multi),
    ]

    for ax, (title, color_by, emb) in zip(axes.flat, titles):
        if color_by == "dataset":
            for ds_name, color in [("CIC17","#1f77b4"),
                                    ("CIC18","#ff7f0e"),
                                    ("UNSW","#2ca02c")]:
                mask = ds_labels == ds_name
                ax.scatter(emb[mask, 0], emb[mask, 1],
                          c=color, s=3, alpha=0.4,
                          label=ds_name, rasterized=True)
            ax.legend(markerscale=3, loc='best', fontsize=9)
        else:
            for label, color, name in [(0,"#aaaaaa","Benign"),
                                        (1,"#d62728","Attack")]:
                mask = y_all == label
                ax.scatter(emb[mask, 0], emb[mask, 1],
                          c=color, s=3, alpha=0.4,
                          label=name, rasterized=True)
            ax.legend(markerscale=3, loc='best', fontsize=9)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("UMAP-1", fontsize=9)
        ax.set_ylabel("UMAP-2", fontsize=9)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#AAAAAA')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUT_DIR, "umap_diversity.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    log.info("Saved -> %s", out_path)
    log.info("Done.")


if __name__ == "__main__":
    run()
