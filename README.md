# Learning to Detect the Unseen: Training Diversity for Zero-Day Intrusion Detection

Code and results for:

> M. Zita, M. U. Rehman, M. Kazim, and T. Al Ibaisi. *Learning to Detect the Unseen: Training Diversity for Zero-Day Intrusion Detection.* Submitted to **Computers & Security** (Elsevier), 2026.

The study asks whether training an unsupervised detector on benign traffic from several network environments improves detection in an environment it has never seen, and whether that gain comes from diversity rather than from a larger training set. Three model families are trained only on benign flows: an autoencoder (AE), a denoising autoencoder (DAE), and Deep SVDD.

## Key results (five seeds, deduplicated data)

| Finding | Result |
|---|---|
| UNSW-NB15 ROC-AUC, DAE trained on CIC-IDS2017 only | 0.499 ± 0.002 |
| UNSW-NB15 ROC-AUC, DAE size-matched multi-environment (same 536,307 flows) | 0.934 ± 0.008 |
| UNSW-NB15 ROC-AUC, DAE full multi-environment | 0.943 ± 0.008 |
| Leave-one-dataset-out F1 when CIC-IDS2017 or CSE-CIC-IDS2018 is held out | 0.826 to 0.882 |
| Leave-one-dataset-out F1 when UNSW-NB15 is held out | 0.049 ± 0.000 |
| BoT-IoT detection rate, DAE full multi-environment (fully withheld) | 0.679 ± 0.087 |
| Classical baselines on UNSW-NB15 (Isolation Forest, LOF, OC-SVM, HBOS) | 0.395 to 0.727 |

The size-matched comparison holds training volume fixed and changes only diversity. BoT-IoT is withheld from every stage of training, preprocessing, and calibration.

## Datasets

Download the datasets and place them under `data/raw/` (this folder is not tracked):

- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- BoT-IoT: https://research.unsw.edu.au/projects/bot-iot-dataset

Several scripts set paths near the top of the file (for example `ARTEFACT_DIR = r"C:/MLProject/zero_day_project/artefacts_v2"`). Change these to your own location before running.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on Linux or macOS
pip install -r requirements.txt
```

## Reproducing the paper

Run all commands from the repository root. Times are approximate on a single GPU.

### 1. Preprocessing

| Step | Script | Output |
|---|---|---|
| Feature harmonization (18 features), IQR scaling, splits | `run_preprocessing.py` (uses `preprocessing/` and `preprocessing_v2/`) | `artefacts_v2/*.npz`, `artefacts_v2/selected_features.csv` |
| Removal of exact duplicates and five-seed retraining | `run_dedup_pipeline.py` (about 25 h) | `artefacts_v2/dedup/dedup_results.csv` |
| Duplicate check across datasets | `cross_dataset_duplicate_check.py` | console |
| BoT-IoT category counts | `run_botiot_counts.py` | console |

### 2. Main experiments

| Paper result | Script | Output in `artefacts_v2/dedup/` |
|---|---|---|
| AE single-environment | `run_ae_single.py` | `ae_single_results.csv` |
| DAE single-environment | `run_dae_single.py` (about 8 h) | `dae_single_results.csv` |
| Size-matched diversity control | `run_dae_size_matched.py` | `dae_size_matched_results.csv` |
| Deep SVDD single and multi-environment | `run_svdd_single.py`, `run_svdd_multi.py` | `svdd_single_results.csv`, `svdd_multi_results.csv` |
| Leave-one-dataset-out (LODO) | `run_dae_lodo.py`, `run_lodo_multiseed.py` (about 6 h) | `dae_lodo_results.csv`, `lodo_multiseed_results.csv` |
| Classical baselines | `run_baselines.py`, `run_matched_sample.py` | `../baseline_results_v2.csv`, `../matched_sample_results.csv` |

### 3. Additional analyses

| Paper result | Script | Output |
|---|---|---|
| PR-AUC, threshold sensitivity, distribution distances | `compute_additional_metrics.py` | `dedup/pr_auc_results.csv`, `dedup/threshold_sensitivity.csv`, `dedup/distribution_distances.csv` |
| PR-AUC table | `generate_pr_auc_table.py` | `dedup/pr_auc_table.csv` |
| Per-attack detection | `run_per_attack_unsw.py`, `compute_per_attack_b.py` | `dedup/per_attack_results*.csv` |
| Significance tests (Wilcoxon, Cohen's d, 95% CI) | `run_significance_5seed.py` | `significance_summary.csv` |
| Feature ablations | `run_ablation_no_dst_port.py`, `run_ablation_imputation.py`, `run_dedup_ablations.py` | `dedup/ablation_*.csv` |
| Spearman filter sensitivity | `run_spearman_sensitivity.py` | `spearman_sensitivity.csv` |

### 4. Figures

| Figure | Script | Output in `artefacts_v2/figures/` |
|---|---|---|
| Score distributions | `generate_score_distributions_all.py` | `score_distributions_all.png` |
| Confusion matrices | `run_dedup_seed42_save.py`, `generate_confusion_matrices.py` | `confusion_matrices_dedup.png` |
| ROC and PR curves | `generate_roc_curves.py` | `roc_pr_curves.png` |
| UMAP of learned representations | `generate_dae_umap.py` | `umap_dae_diversity.png` |

## Repository layout

```
models/            AE, Deep SVDD and baseline model definitions (the DAE is defined in the run_dae_*.py scripts)
preprocessing/     loading, cleaning, feature harmonization, splitting
preprocessing_v2/  configuration and IQR scaler used by run_preprocessing.py
artefacts_v2/      fitted scaler, selected features, results, and figures
  dedup/           results on deduplicated data (reported in the paper)
  figures/         generated figures
artefacts_ablation/ feature ablation outputs
```

Scripts named `train_vae*`, `run_vae_if_experiments.py`, `run_weight_search*.py`, and `run_ton_*.py` belong to earlier versions of the study (VAE ensemble and CIC-ToN-IoT evaluation). They are kept for transparency and are not needed to reproduce the submitted paper.

## Random seeds

All reported results use seeds 42, 123, 456, 789, and 999.

## Earlier version

The code for the earlier AE, VAE, and Deep SVDD ensemble version is preserved in the commit history (commit `da1081c`).

## License

MIT License

## Contact

Margaret Zita, De Montfort University, Leicester, United Kingdom
