"""
Pilot Study: DAE, UNSW-NB15, Single vs. Multi (size-matched)
================================================================
Validates the SHAP attribution pipeline on a single, well-understood
model-dataset pair before committing compute to the full confirmatory
attribution matrix (see Paper2_Pilot_Plan.md and Paper2_Methodology.tex).

This script assumes:
  - Paper 1's DAE architecture and preprocessing pipeline (18-feature
    harmonized space, IQR scaling fit on benign training data only).
  - Trained checkpoints for DAE-Single and DAE-Multi-matched (seed 42)
    are available on disk (paths marked TODO below).
  - `shap`, `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn` are installed.

Sections marked TODO must be adapted to your actual project layout
(C:\\MLProject\\zero_day_project\\ conventions) before running.
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score
import shap

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

N_FEATURES = 18
TOLERANCE_AUROC = 0.01          # tolerance for Paper 1 reproduction check
ADDITIVITY_TOL = 1e-3           # numerical tolerance for SHAP additivity
STABLE_GROUP_SIZE = 6           # n: TODO confirm with supervisor/statistician
N_BACKGROUND = 500              # TODO confirm final sample size
N_EVAL = 5000                   # TODO confirm final sample size

# ---------------------------------------------------------------------------
# Checkpointing: every expensive step writes to CACHE_DIR before moving on.
# Re-running main() picks up from the last completed step instead of
# recomputing from scratch. Delete the relevant file(s) in CACHE_DIR to
# force a specific step to rerun.
# ---------------------------------------------------------------------------
CACHE_DIR = "pilot_cache_seed42"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cpath(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def _exists(name: str) -> bool:
    return os.path.exists(_cpath(name))


def _log_skip(step: str):
    print(f"[cache] {step} already completed -> loading from {CACHE_DIR}, skipping recompute")


# ---------------------------------------------------------------------------
# 1. Model definition (DAE / AE architecture, Paper 1 Section 5.1-5.2)
# ---------------------------------------------------------------------------
class Autoencoder(nn.Module):
    """Standard / Denoising Autoencoder, encoder 18->64->32->16->8,
    decoder mirrors the encoder. Huber loss (delta=1.0) used in training;
    not needed at inference time for SHAP (we explain reconstruction error
    directly)."""

    def __init__(self, input_dim: int = N_FEATURES):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class ReconstructionErrorScorer(nn.Module):
    """Wraps an autoencoder so its forward() returns the scalar anomaly
    score (mean squared reconstruction error) rather than the
    reconstruction itself. This is the function SHAP explains."""

    def __init__(self, autoencoder: nn.Module):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x):
        x_hat = self.autoencoder(x)
        # mean squared error per row, output shape (batch, 1)
        return ((x - x_hat) ** 2).mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# 2. Data loading (TODO: adapt to actual project paths)
# ---------------------------------------------------------------------------
def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_unsw_evaluation_data():
    """TODO: replace with actual loader.
    Must return a DataFrame with the 18 harmonized, IQR-scaled features
    (same scaler as fit in Paper 1, applied unchanged -- do not refit)
    plus a binary 'label' column (0 = benign, 1 = attack)."""
    raise NotImplementedError(
        "Plug in the UNSW-NB15 evaluation loader from the Paper 1 pipeline. "
        "Must reuse the frozen IQR scaler fit on benign training data only."
    )


def load_source_environment_data(env_names):
    """TODO: replace with actual loader.
    Returns dict {env_name: DataFrame} of benign, IQR-scaled, deduplicated
    training-partition flows for each source environment. Used for ESS
    (Section 3.2 of Methodology) -- training-source data only, never
    evaluation/target data."""
    raise NotImplementedError(
        "Plug in per-environment benign training-partition loaders."
    )


# ---------------------------------------------------------------------------
# 3. Environment Sensitivity Score (ESS) -- Methodology Section 3.2
# ---------------------------------------------------------------------------
def compute_ess(source_envs: dict, feature_names, n_bins: int = 50) -> pd.Series:
    """
    ESS_j = mean pairwise Jensen-Shannon divergence of feature j's
    histogram across all pairs of source environments.
    Computed from training-source data only, before any SHAP is run.
    """
    env_names = list(source_envs.keys())
    ess = {}

    for feat in feature_names:
        # shared bin edges across all environments for this feature
        all_vals = np.concatenate([source_envs[e][feat].values for e in env_names])
        bin_edges = np.histogram_bin_edges(all_vals, bins=n_bins)

        hists = {}
        for e in env_names:
            hist, _ = np.histogram(source_envs[e][feat].values, bins=bin_edges, density=False)
            hist = hist.astype(float)
            hist_sum = hist.sum()
            hists[e] = hist / hist_sum if hist_sum > 0 else hist

        pairwise = []
        for i in range(len(env_names)):
            for j in range(i + 1, len(env_names)):
                jsd = jensenshannon(hists[env_names[i]], hists[env_names[j]])
                # jensenshannon returns the distance (sqrt of divergence);
                # square it to recover the divergence, bounded [0, 1] for
                # base-2 log (scipy default is natural log -- convert if
                # a [0,1] bound with log2 is required by the frozen protocol)
                pairwise.append(jsd ** 2 if not np.isnan(jsd) else 0.0)

        ess[feat] = float(np.mean(pairwise)) if pairwise else 0.0

    ess_series = pd.Series(ess, name="ESS")
    # normalize to [0, 1] across features for interpretability
    if ess_series.max() > ess_series.min():
        ess_series = (ess_series - ess_series.min()) / (ess_series.max() - ess_series.min())
    return ess_series.sort_values(ascending=True)


def freeze_feature_groups(ess: pd.Series, n_stable: int = STABLE_GROUP_SIZE):
    """Stable = lowest ESS, Sensitive = highest ESS. Frozen BEFORE any
    SHAP computation (Methodology Section 3.2.1)."""
    stable = list(ess.index[:n_stable])
    sensitive = list(ess.index[-n_stable:])
    return stable, sensitive


# ---------------------------------------------------------------------------
# 4. SHAP protocol -- Methodology Section 3.3
# ---------------------------------------------------------------------------
def run_deepshap(scorer: nn.Module, background: np.ndarray, evaluation: np.ndarray):
    """Runs DeepSHAP on the raw (unclipped) anomaly score.
    background: (N_BACKGROUND, 18) array, same partition used to fit model.
    evaluation: (N_EVAL, 18) array, fixed stratified sample.
    Returns: shap_values (N_EVAL, 18) numpy array, and the expected_value
    (baseline) used for the additivity check.
    """
    background_t = torch.tensor(background, dtype=torch.float32)
    evaluation_t = torch.tensor(evaluation, dtype=torch.float32)

    explainer = shap.DeepExplainer(scorer, background_t)
    shap_values = explainer.shap_values(evaluation_t)

    # shap_values may come back as a list (older API) or array depending
    # on shap version / single-output model; normalize to ndarray (N, 18)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values).reshape(evaluation.shape[0], N_FEATURES)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.array(expected_value).flatten()[0])

    return shap_values, expected_value


def validate_additivity(scorer: nn.Module, evaluation: np.ndarray,
                         shap_values: np.ndarray, expected_value: float) -> float:
    """Confirms baseline + sum(SHAP values) reproduces the actual model
    output, per flow. Returns the max absolute reconstruction error across
    the evaluation sample (should be below ADDITIVITY_TOL)."""
    with torch.no_grad():
        actual = scorer(torch.tensor(evaluation, dtype=torch.float32)).numpy().flatten()
    reconstructed = expected_value + shap_values.sum(axis=1)
    max_error = float(np.max(np.abs(actual - reconstructed)))
    return max_error


# ---------------------------------------------------------------------------
# 5. Derived metrics -- Methodology Section 3.4
# ---------------------------------------------------------------------------
def normalized_global_shap(shap_values: np.ndarray, feature_names) -> pd.Series:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total = mean_abs.sum()
    normalized = mean_abs / total if total > 0 else mean_abs
    return pd.Series(normalized, index=feature_names, name="normalized_shap").sort_values(ascending=False)


def compute_esfrp(normalized_shap: pd.Series, stable_features: list) -> float:
    return float(normalized_shap.loc[stable_features].sum())


def compute_asi(normalized_single: pd.Series, normalized_multi: pd.Series) -> float:
    aligned_single, aligned_multi = normalized_single.align(normalized_multi, fill_value=0.0)
    return float(0.5 * np.sum(np.abs(aligned_multi - aligned_single)))


# ---------------------------------------------------------------------------
# 6. Reproduction check -- Formal Protocol Section 2
# ---------------------------------------------------------------------------
def check_reproduction(scorer: nn.Module, eval_df: pd.DataFrame, feature_names,
                        expected_auroc: float, label: str) -> dict:
    with torch.no_grad():
        x = torch.tensor(eval_df[feature_names].values, dtype=torch.float32)
        scores = scorer(x).numpy().flatten()
    auroc = roc_auc_score(eval_df["label"].values, scores)
    passed = abs(auroc - expected_auroc) <= TOLERANCE_AUROC
    print(f"[Reproduction check: {label}] AUROC = {auroc:.4f} "
          f"(expected {expected_auroc:.4f}, tolerance {TOLERANCE_AUROC}) "
          f"-> {'PASS' if passed else 'FAIL'}")
    return {"label": label, "auroc": auroc, "expected": expected_auroc, "passed": passed}


# ---------------------------------------------------------------------------
# 7. Main pilot execution
# ---------------------------------------------------------------------------
def main():
    feature_names = [f"f{i}" for i in range(N_FEATURES)]  # TODO: replace with actual 18 feature names

    # --- Step 1: load checkpoints -----------------------------------------
    dae_single = Autoencoder()
    dae_multi = Autoencoder()
    # TODO: fix paths to actual Paper 1 checkpoints, seed 42
    dae_single = load_checkpoint(dae_single, r"C:\MLProject\zero_day_project\checkpoints\dae_single_seed42.pt")
    dae_multi = load_checkpoint(dae_multi, r"C:\MLProject\zero_day_project\checkpoints\dae_multi_matched_seed42.pt")

    scorer_single = ReconstructionErrorScorer(dae_single).eval()
    scorer_multi = ReconstructionErrorScorer(dae_multi).eval()

    # --- Step 2: reproduction check (cached) --------------------------------
    eval_df = load_unsw_evaluation_data()

    if _exists("reproduction.json"):
        _log_skip("Reproduction check")
        with open(_cpath("reproduction.json")) as f:
            results = json.load(f)
    else:
        results = []
        results.append(check_reproduction(scorer_single, eval_df, feature_names,
                                           expected_auroc=0.499, label="DAE-Single/UNSW"))
        results.append(check_reproduction(scorer_multi, eval_df, feature_names,
                                           expected_auroc=0.934, label="DAE-Multi-matched/UNSW"))
        with open(_cpath("reproduction.json"), "w") as f:
            json.dump(results, f, indent=2)

    if not all(r["passed"] for r in results):
        print("HALT: reproduction check failed. Resolve before proceeding to SHAP. "
              f"Delete {_cpath('reproduction.json')} to retry after fixing.")
        return

    # --- Step 3: ESS and frozen feature groups (cached) ----------------------
    if _exists("ess.json"):
        _log_skip("ESS computation")
        with open(_cpath("ess.json")) as f:
            ess_data = json.load(f)
        ess = pd.Series(ess_data["ess"])
        stable_features = ess_data["stable_features"]
        sensitive_features = ess_data["sensitive_features"]
    else:
        source_envs = load_source_environment_data(["CIC-IDS2017", "CSE-CIC-IDS2018", "UNSW-NB15"])
        ess = compute_ess(source_envs, feature_names)
        stable_features, sensitive_features = freeze_feature_groups(ess)
        with open(_cpath("ess.json"), "w") as f:
            json.dump({
                "ess": ess.to_dict(),
                "stable_features": stable_features,
                "sensitive_features": sensitive_features,
            }, f, indent=2)
    print("ESS (ascending, most stable first):")
    print(ess)
    print(f"Stable group: {stable_features}")
    print(f"Sensitive group: {sensitive_features}")

    # --- Step 4: background and evaluation samples (cached, index-based) ----
    if _exists("sample_indices.json"):
        _log_skip("Background/evaluation sampling")
        with open(_cpath("sample_indices.json")) as f:
            idx_data = json.load(f)
        background_idx = np.array(idx_data["background_idx"])
        eval_idx = np.array(idx_data["eval_idx"])
    else:
        rng = np.random.default_rng(RNG_SEED)
        background_idx = rng.choice(len(eval_df), size=min(N_BACKGROUND, len(eval_df)), replace=False)
        eval_idx = rng.choice(len(eval_df), size=min(N_EVAL, len(eval_df)), replace=False)
        with open(_cpath("sample_indices.json"), "w") as f:
            json.dump({
                "background_idx": background_idx.tolist(),
                "eval_idx": eval_idx.tolist(),
            }, f)
    background = eval_df.iloc[background_idx][feature_names].values
    evaluation = eval_df.iloc[eval_idx][feature_names].values

    # --- Step 5: run DeepSHAP for both conditions (cached, the slow step) ---
    if _exists("shap_single.npz"):
        _log_skip("DeepSHAP (Single)")
        loaded = np.load(_cpath("shap_single.npz"))
        shap_single, ev_single = loaded["shap_values"], float(loaded["expected_value"])
    else:
        shap_single, ev_single = run_deepshap(scorer_single, background, evaluation)
        np.savez(_cpath("shap_single.npz"), shap_values=shap_single, expected_value=ev_single)

    if _exists("shap_multi.npz"):
        _log_skip("DeepSHAP (Multi)")
        loaded = np.load(_cpath("shap_multi.npz"))
        shap_multi, ev_multi = loaded["shap_values"], float(loaded["expected_value"])
    else:
        shap_multi, ev_multi = run_deepshap(scorer_multi, background, evaluation)
        np.savez(_cpath("shap_multi.npz"), shap_values=shap_multi, expected_value=ev_multi)

    # --- Step 6: additivity validation (cheap, always recomputed) -----------
    max_err_single = validate_additivity(scorer_single, evaluation, shap_single, ev_single)
    max_err_multi = validate_additivity(scorer_multi, evaluation, shap_multi, ev_multi)
    print(f"Additivity max error -- Single: {max_err_single:.6f}, Multi: {max_err_multi:.6f} "
          f"(tolerance {ADDITIVITY_TOL})")
    additivity_ok = max_err_single < ADDITIVITY_TOL and max_err_multi < ADDITIVITY_TOL
    if not additivity_ok:
        print("HALT: SHAP additivity did not validate. Debug before proceeding. "
              f"SHAP values are still cached in {CACHE_DIR}/ for inspection -- "
              "delete shap_single.npz / shap_multi.npz only if you need to rerun "
              "DeepSHAP itself (e.g., after a background/eval sample change).")
        return

    # --- Step 7: derived metrics ---------------------------------------------
    norm_single = normalized_global_shap(shap_single, feature_names)
    norm_multi = normalized_global_shap(shap_multi, feature_names)
    esfrp_single = compute_esfrp(norm_single, stable_features)
    esfrp_multi = compute_esfrp(norm_multi, stable_features)
    asi = compute_asi(norm_single, norm_multi)

    print("\nNormalized global SHAP (Single):")
    print(norm_single)
    print("\nNormalized global SHAP (Multi):")
    print(norm_multi)
    print(f"\nESFRP(Single) = {esfrp_single:.4f}")
    print(f"ESFRP(Multi)  = {esfrp_multi:.4f}")
    print(f"ASI           = {asi:.4f}")

    # --- Step 8: pilot outcome -------------------------------------------------
    direction_correct = esfrp_multi > esfrp_single
    outcome = "PASS" if direction_correct else "PROCEED WITH CAUTION"
    print(f"\nPilot outcome (single seed): {outcome}")
    if not direction_correct:
        print("ESFRP did not move in the hypothesized direction on this single "
              "seed. Per the Pilot Plan, expand to all 5 seeds before drawing "
              "any conclusion -- do not treat this as a pipeline failure.")

    # --- Save artifacts ---------------------------------------------------
    summary = {
        "reproduction_checks": results,
        "stable_features": stable_features,
        "sensitive_features": sensitive_features,
        "esfrp_single": esfrp_single,
        "esfrp_multi": esfrp_multi,
        "asi": asi,
        "additivity_max_error_single": max_err_single,
        "additivity_max_error_multi": max_err_multi,
        "outcome": outcome,
    }
    with open("pilot_summary_seed42.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved pilot_summary_seed42.json")


if __name__ == "__main__":
    main()
