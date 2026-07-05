# Multi-Environment Training for Cross-Domain Zero-Day Intrusion Detection

## Overview
This repository contains the full implementation of the hybrid unsupervised 
intrusion detection framework proposed in:

Mujeeb Ur Rehman, Margaret Zita, Muhammad Abrar, Muhammad Kazim, Sohail Khalid.
"Multi-Environment Training for Cross-Domain Zero-Day Intrusion Detection 
Using Hybrid Unsupervised Deep Learning"
Results in Physics, 2026.

The framework integrates a Deep Autoencoder (AE), Variational Autoencoder (VAE), 
and Deep SVDD, trained exclusively on benign network traffic from three 
heterogeneous environments to achieve robust zero-day detection.

## Key Results
- Mean AUC: 0.912±0.015 (CIC-IDS2017), 0.959±0.015 (CSE-CIC-IDS2018), 0.884±0.014 (UNSW-NB15)
- BoT-IoT zero-day detection: 82.7%±7.2% mean DR across 5 seeds
- CIC-ToN-IoT zero-day AUC: 0.953±0.006 under local benign calibration
- Multi-environment training improves AUC by +0.295 (CIC18) and +0.391 (UNSW) over single-environment baseline
- Statistical significance confirmed: p < 0.001 (Wilcoxon signed-rank vs LOF)

## Datasets
- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- BoT-IoT: https://research.unsw.edu.au/projects/bot-iot-dataset
- CIC-ToN-IoT: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/

## Repository Structure
- models/ — AE, VAE, Deep SVDD implementations
- preprocessing_v2/ — Feature harmonization and IQR scaler
- artefacts_v2/iqr_scaler.npz — Fitted IQR scaler parameters
- artefacts_v2/selected_features.csv — 18 retained features
- artefacts_v2/multiseed_results.csv — 5-seed stability results
- artefacts_v2/ablation_pairwise_results.csv — All 7 ablation combinations
- artefacts_v2/lodo_results.csv — Leave-one-dataset-out results
- artefacts_v2/baseline_results_v2.csv — HBOS, OC-SVM, LOF results
- artefacts_v2/if_results.csv — Isolation Forest results
- artefacts_v2/ae_single_results.csv — AE-Single baseline results
- artefacts_v2/calibration_sensitivity.csv — Calibration sensitivity
- artefacts_v2/weight_search_results.csv — Fusion weight grid search

## Requirements
- Python >= 3.9
- PyTorch >= 1.12
- numpy, pandas, scipy, scikit-learn, matplotlib

Install: pip install torch numpy pandas scipy scikit-learn matplotlib

## Running the Pipeline

Full pipeline:
python run_full_pipeline.py

Individual steps:
python run_preprocessing_v2.py       # Feature selection and scaling
python run_multiseed_v3.py           # 5-seed stability analysis
python run_ablation_b.py             # Pairwise ablation study
python run_lodo.py                   # Leave-one-dataset-out
python run_baselines_v2.py           # Classical baselines
python run_isolation_forest.py       # Isolation Forest baseline
python run_ae_single.py              # AE-Single baseline
python run_weight_search.py          # Fusion weight grid search
python run_calibration_sensitivity_v2.py  # Calibration sensitivity
python run_inference_throughput.py   # Inference speed benchmark

## Reproducing Figures
python generate_confusion_matrices.py    # Confusion matrices
python generate_score_distributions.py  # Score distributions

## Random Seeds
All results reported with seeds: 42, 123, 456, 789, 999

## IQR Scaler
The fitted scaler can be loaded for new datasets:

from preprocessing_v2.scaler import IQRScaler
scaler = IQRScaler()
scaler.load('artefacts_v2/iqr_scaler.npz')
X_scaled = scaler.transform(X_new)

## License
MIT License

## Contact
Margaret Zita - De Montfort University
