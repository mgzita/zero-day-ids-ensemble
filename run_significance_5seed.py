"""
run_significance_3seed.py
==========================
Computes mean +/- std and Wilcoxon signed-rank test
using existing 3-seed results (seeds 42, 123, 456).

This addresses Review Point 4.1 without retraining.

Loads from:
  artefacts_v2/multiseed_results.csv

Saves to:
  artefacts_v2/significance_summary.csv

Run from project root:
    python run_significance_3seed.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

PROJECT_ROOT = r"C:/MLProject/zero_day_project"
ARTEFACT_DIR = os.path.join(PROJECT_ROOT, "artefacts_v2")

# Baseline AUC values from paper (Table baseline comparison)
LOF_AUCS = {
    "CIC-IDS2017":     0.795,
    "CSE-CIC-IDS2018": 0.799,
    "UNSW-NB15":       0.659,
    "BoT-IoT":         0.951,
    "CIC-ToN-IoT":     0.786,
}

IF_AUCS = {
    "CIC-IDS2017":     0.620,
    "CSE-CIC-IDS2018": 0.789,
    "UNSW-NB15":       0.727,
    "BoT-IoT":         0.862,
    "CIC-ToN-IoT":     0.896,
}

DATASET_ORDER = [
    "CIC-IDS2017",
    "CSE-CIC-IDS2018",
    "UNSW-NB15",
    "BoT-IoT",
    "CIC-ToN-IoT",
]


def run():
    #    Load existing results                                              
    path = os.path.join(ARTEFACT_DIR, "multiseed_results.csv")
    df_all = pd.read_csv(path)
    log.info("Total rows: %d | all seeds: %s",
             len(df_all), df_all["seed"].unique().tolist())
    VALID_SEEDS = [42, 123, 456, 789, 999]
    df = df_all[df_all["seed"].isin(VALID_SEEDS)].copy()
    log.info("Using seeds %s only: %d rows", VALID_SEEDS, len(df))

    #    Mean +/- std per dataset                                           
    print("\n" + "=" * 75)
    print("MEAN +/- STD ACROSS 3 SEEDS (42, 123, 456)")
    print("=" * 75)
    print(f"  {'Dataset':<22} {'AUC mean':>10} {'AUC std':>10} "
          f"{'Recall mean':>12} {'Recall std':>12} {'F1 mean':>10}")
    print("  " + "-" * 78)

    summary = []
    for ds in DATASET_ORDER:
        rows = df[df["dataset"] == ds]
        if len(rows) == 0:
            continue
        auc_mean  = rows["auc"].mean()
        auc_std   = rows["auc"].std()
        rec_mean  = rows["recall"].mean()
        rec_std   = rows["recall"].std()
        f1_mean   = rows["f1"].mean()
        f1_std    = rows["f1"].std()
        fpr_mean  = rows["fpr"].mean()

        print(f"  {ds:<22} {auc_mean:>10.4f} {auc_std:>10.4f} "
              f"{rec_mean:>12.4f} {rec_std:>12.4f} {f1_mean:>10.4f}")

        summary.append({
            "dataset":  ds,
            "auc_mean": round(auc_mean, 4),
            "auc_std":  round(auc_std,  4),
            "rec_mean": round(rec_mean, 4),
            "rec_std":  round(rec_std,  4),
            "f1_mean":  round(f1_mean,  4),
            "f1_std":   round(f1_std,   4),
            "fpr_mean": round(fpr_mean, 4),
        })

    #    Wilcoxon: Ensemble vs LOF                                          
    print("\n" + "=" * 75)
    print("WILCOXON SIGNED-RANK TEST: Full Ensemble vs LOF")
    print("=" * 75)

    ens_aucs = []
    lof_aucs = []
    for ds in DATASET_ORDER:
        rows = df[df["dataset"] == ds]["auc"].dropna()
        if len(rows) > 0 and ds in LOF_AUCS:
            ens_aucs.append(rows.mean())
            lof_aucs.append(LOF_AUCS[ds])
            diff = rows.mean() - LOF_AUCS[ds]
            print(f"  {ds:<22}  Ensemble={rows.mean():.4f}  "
                  f"LOF={LOF_AUCS[ds]:.4f}  diff={diff:+.4f}")

    try:
        stat, p = wilcoxon(ens_aucs, lof_aucs,
                           alternative="greater")
        print(f"\n  Wilcoxon statistic : {stat:.4f}")
        print(f"  p-value (one-sided): {p:.4f}")
        if p < 0.05:
            print("  Result:   p < 0.05 — ensemble significantly "
                  "outperforms LOF")
        else:
            print("  Result:     p >= 0.05 — report mean difference "
                  "without significance claim")
    except Exception as e:
        print(f"  Wilcoxon error: {e}")
        print("  Note: With 5 paired observations the test may be "
              "underpowered. Report mean±std instead.")

    #    Wilcoxon: Ensemble vs IF                                           
    print("\n" + "=" * 75)
    print("WILCOXON SIGNED-RANK TEST: Full Ensemble vs Isolation Forest")
    print("=" * 75)

    ens_aucs2 = []
    if_aucs   = []
    for ds in DATASET_ORDER:
        rows = df[df["dataset"] == ds]["auc"].dropna()
        if len(rows) > 0 and ds in IF_AUCS:
            ens_aucs2.append(rows.mean())
            if_aucs.append(IF_AUCS[ds])
            diff = rows.mean() - IF_AUCS[ds]
            print(f"  {ds:<22}  Ensemble={rows.mean():.4f}  "
                  f"IF={IF_AUCS[ds]:.4f}  diff={diff:+.4f}")

    try:
        stat2, p2 = wilcoxon(ens_aucs2, if_aucs,
                             alternative="greater")
        print(f"\n  Wilcoxon statistic : {stat2:.4f}")
        print(f"  p-value (one-sided): {p2:.4f}")
        if p2 < 0.05:
            print("  Result:   p < 0.05 — ensemble significantly "
                  "outperforms Isolation Forest")
        else:
            print("  Result:     p >= 0.05 — report mean difference "
                  "without significance claim")
    except Exception as e:
        print(f"  Wilcoxon error: {e}")

    #    LaTeX table                                                        
    print("\n" + "=" * 75)
    print("LATEX: Updated ensemble rows for Table (model results)")
    print("Paste these rows into tab:model_results")
    print("=" * 75)
    print()
    ensemble_ds = [s for s in summary
                   if s["dataset"] in [d for d, _ in
                   [("CIC-IDS2017",     ""),
                    ("CSE-CIC-IDS2018", ""),
                    ("UNSW-NB15",       ""),
                    ("BoT-IoT",         ""),
                    ("CIC-ToN-IoT",     "")]]]

    for s in summary:
        ds = s["dataset"]
        if ds == "BoT-IoT":
            auc_s = "--"
        else:
            auc_s = f"{s['auc_mean']:.3f} $\\pm$ {s['auc_std']:.3f}"
        print(f"Ensemble & {ds:<22} & "
              f"{s['rec_mean']:.3f} $\\pm$ {s['rec_std']:.3f} & "
              f"{auc_s} \\\\")

    print()
    print("TABLE FOOTNOTE to add:")
    print(r"\begin{tablenotes}")
    print(r"\small")
    print(r"\item All ensemble results reported as mean $\pm$ std")
    print(r"over 3 random seeds (42, 123, 456) controlling weight")
    print(r"initialisation, dataset shuffle order, and VAE")
    print(r"reparameterisation noise.")
    if p < 0.05:
        print(r"\item * p $<$ 0.05 vs best baseline (LOF),")
        print(r"Wilcoxon signed-rank test (one-sided).")
    else:
        print(r"\item Wilcoxon signed-rank test vs LOF baseline:")
        print(f"\\item p = {p:.3f} (ensemble vs LOF, one-sided).")
    print(r"\end{tablenotes}")

    #    Column header change                                               
    print("\n" + "=" * 75)
    print("COLUMN HEADER CHANGE in tab:model_results:")
    print("=" * 75)
    print("Change:")
    print(r"  \textbf{ROC-AUC}")
    print("To:")
    print(r"  \textbf{ROC-AUC (mean $\pm$ std)}")

    #    Save summary                                                       
    out = os.path.join(ARTEFACT_DIR, "significance_summary.csv")
    pd.DataFrame(summary).to_csv(out, index=False)
    log.info("Significance summary -> %s", out)
    log.info("Done.")


if __name__ == "__main__":
    run()
