# Zero-Day IDS Ensemble

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

Code repository for the paper:

> **Hybrid Unsupervised Deep Learning Framework for Zero-Day Attack 
> Detection Using Autoencoder–VAE–Deep SVDD Ensemble with 
> Multi-Environment Training**  
> Mujeeb Ur Rehman, Margaret Zita
> De Montfort University / Riphah International University  
> *Submitted to Results in Physics, 2025*

---

## Overview

This repository provides the full implementation of a hybrid 
unsupervised intrusion detection framework that:

- Trains exclusively on **benign network traffic** — no attack 
  labels required at any stage
- Combines three complementary anomaly detectors: **Deep AE**, 
  **VAE**, and **Deep SVDD**
- Uses **multi-environment training** across CIC-IDS2017, 
  CSE-CIC-IDS2018, and UNSW-NB15 to prevent dataset-specific 
  overfitting
- Achieves **99.98% detection rate** on unseen BoT-IoT botnet 
  traffic and **AUC = 0.948** on CIC-ToN-IoT
- Demonstrates that **training diversity** (+0.34 AUC) outweighs 
  model architecture (+0.11 AUC) as a driver of cross-domain 
  generalisation

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
torch>=2.0.0
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
scipy>=1.10.0
matplotlib>=3.7.0

---

## Datasets

Download from their respective institutional repositories:

| Dataset | Source | Role |
|---|---|---|
| CIC-IDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html | Training |
| CSE-CIC-IDS2018 | https://www.unb.ca/cic/datasets/ids-2018.html | Training |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset | Training |
| BoT-IoT | https://research.unsw.edu.au/projects/bot-iot-dataset | Zero-day test |
| CIC-ToN-IoT | https://staff.itee.uq.edu.au/marius/NIDS_datasets/ | Zero-day test |

---

## Repository Structure
zero-day-ids-ensemble/
├── README.md
├── requirements.txt
├── models/
│   ├── autoencoder.py       ← Deep AE architecture
│   ├── vae.py               ← VAE architecture
│   └── deep_svdd.py         ← Deep SVDD architecture
├── preprocessing/
│   ├── scaler.py            ← IQR scaler implementation
│   └── feature_alignment.py ← Cross-dataset feature mapping
├── scripts/
│   ├── train.py             ← Train all three models
│   ├── evaluate.py          ← Evaluate on all 5 datasets
│   ├── run_weight_search.py ← Grid search for fusion weights
│   ├── run_spearman_sensitivity.py  ← Feature threshold analysis
│   ├── run_calibration_sensitivity_v2.py ← Calibration analysis
│   └── run_inference_throughput.py  ← Throughput benchmark
├── artefacts/
│   ├── selected_features.csv ← 18 retained feature names
│   └── scaler.json           ← Fitted IQR scaler parameters
│                               (Q1, Q2, Q3 per feature)
└── 
---

## Reproducing Paper 1 Results

```bash
# Train all three models
python scripts/train.py

# Evaluate on all 5 datasets
python scripts/evaluate.py

# Justify ensemble weights (grid search)
python scripts/run_weight_search.py

# Spearman threshold sensitivity
python scripts/run_spearman_sensitivity.py
```

---

## Scaler Parameters

The fitted IQR scaler parameters are provided in 
`artefacts/scaler.json`. This file contains Q1, Q2, Q3 
and IQR values for all 18 selected features, fitted on 
1,800,000 combined benign samples from the three training 
environments. This enables full reproduction of the 
preprocessing pipeline without access to the original 
training data.

---

## Key Results

| Dataset | AUC | Recall | Notes |
|---|---|---|---|
| CIC-IDS2017 | 0.937 | 59.6% | Training domain |
| CSE-CIC-IDS2018 | 0.956 | 69.0% | Training domain |
| UNSW-NB15 | 0.866 | 37.9% | Training domain |
| BoT-IoT | — | 99.98% | Unseen zero-day |
| CIC-ToN-IoT | 0.948 | 92.2% | Unseen zero-day |

---

## Citation

If you use this code please cite:

```bibtex
@article{rehman2025zerday,
  author  = {Rehman, Mujeeb Ur and Zita, Margaret},
  title   = {Hybrid Unsupervised Deep Learning Framework 
             for Zero-Day Attack Detection Using 
             Autoencoder--VAE--Deep SVDD Ensemble with 
             Multi-Environment Training},
  journal = {Results in Physics},
  year    = {2025}
}
```

---

## License

MIT License — see LICENSE file for details.
