"""
run_full_pipeline.py
====================
Complete pipeline with checkpoint resuming.
Saves to artefacts_v2/ (back this up first).

Before running:
  xcopy artefacts_v2 artefacts_v2_backup /E /I /H

Run from project root:
    python run_full_pipeline.py

To resume after a failure:
    python run_full_pipeline.py
    (completed steps are skipped automatically)

To force redo a step:
    del artefacts_v2\checkpoints\step03_ablation.done
    python run_full_pipeline.py
"""

import subprocess
import sys
import os
import time
from pathlib import Path

PROJECT_ROOT   = r"C:/MLProject/zero_day_project"
ARTEFACT_DIR   = os.path.join(PROJECT_ROOT, "artefacts_v2")
CHECKPOINT_DIR = os.path.join(ARTEFACT_DIR, "checkpoints")
PYTHON         = sys.executable

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def checkpoint_path(step_name):
    return os.path.join(CHECKPOINT_DIR, f"{step_name}.done")

def is_done(step_name):
    return os.path.exists(checkpoint_path(step_name))

def mark_done(step_name):
    Path(checkpoint_path(step_name)).touch()

def run(script, step_name, description, optional=False):
    path = os.path.join(PROJECT_ROOT, script)

    print(f"\n{'='*70}")
    print(f">>> {description}")

    if is_done(step_name):
        print(f"    SKIPPED — already completed")
        print(f"    (delete checkpoints/{step_name}.done to rerun)")
        return 0

    print(f"    Script : {script}")
    print(f"{'='*70}")

    if not os.path.exists(path):
        if optional:
            print(f"    SKIP — script not found (optional)")
            mark_done(step_name)
            return 0
        else:
            print(f"\n    ERROR — script not found: {path}")
            sys.exit(1)

    t0     = time.time()
    result = subprocess.run([PYTHON, path], cwd=PROJECT_ROOT)
    elapsed = (time.time() - t0) / 60

    if result.returncode != 0:
        print(f"\n    FAILED after {elapsed:.1f} min")
        print(f"    Fix the error then rerun:")
        print(f"    python run_full_pipeline.py")
        sys.exit(1)

    mark_done(step_name)
    print(f"\n    DONE in {elapsed:.1f} min")
    return elapsed

#    Pipeline                                                                  

print(f"\n{'='*70}")
print(f"FULL PIPELINE — Zero-Day IDS Ensemble")
print(f"Output : {ARTEFACT_DIR}")
print(f"{'='*70}")
print(f"NOTE: Back up artefacts_v2 before running:")
print(f"  xcopy artefacts_v2 artefacts_v2_backup /E /I /H")
print()

total = 0

total += run("run_preprocessing_v2.py",          "step01_preprocess",
    "STEP  1/12 — Preprocessing: feature selection, IQR scaling, NPZ export")

total += run("run_multiseed_v3.py",                  "step02_multiseed",
    "STEP  2/12 — Multi-seed training + evaluation (seeds 42, 123, 456)")

total += run("run_ablation_pairwise_v3.py",          "step03_ablation",
    "STEP  3/12 — Pairwise ablation: all 7 model combinations")

total += run("run_lodo.py",                       "step04_lodo",
    "STEP  4/12 — LODO: 3 leave-one-dataset-out experiments")

total += run("run_baselines_v2.py",               "step05_baselines",
    "STEP  5/12 — Baselines: HBOS, OC-SVM, LOF, IF, AE-Single")

total += run("run_weight_search_v3.py",              "step06_weights",
    "STEP  6/12 — Grid search: ensemble fusion weights")

total += run("run_spearman_sensitivity.py",       "step07_spearman",
    "STEP  7/12 — Spearman threshold sensitivity (0.80, 0.85, 0.90)")

total += run("run_calibration_sensitivity_v2.py", "step08_calibration",
    "STEP  8/12 — Calibration sensitivity (1k, 5k, 10k flows)")

total += run("run_inference_throughput.py",       "step09_throughput",
    "STEP  9/12 — Inference throughput benchmark")

total += run("run_significance_5seed.py",         "step10_significance",
    "STEP 10/12 — Statistical significance (3 seeds, Wilcoxon test)")

total += run("export_scaler_json.py",             "step11_scaler",
    "STEP 11/12 — Export scaler.json")

total += run("generate_confusion_matrices.py",    "step12a_confusion",
    "STEP 12a — Confusion matrix figures", optional=True)

total += run("generate_roc_curves.py",            "step12b_roc",
    "STEP 12b — ROC and PR curve figures", optional=True)

total += run("generate_score_distributions.py",   "step12c_scores",
    "STEP 12c — Score distribution figures", optional=True)

#    Summary                                                                   

print(f"\n{'='*70}")
print(f"ALL STEPS COMPLETE — Total: {total:.0f} min ({total/60:.1f} hrs)")
print(f"{'='*70}")
print(f"""
Paste these files here for verification:
  artefacts_v2/multiseed_results.csv
  artefacts_v2/ablation_results.csv
  artefacts_v2/lodo_results.csv
  artefacts_v2/baseline_results.csv
  artefacts_v2/weight_search_results.csv
  artefacts_v2/spearman_sensitivity.csv
  artefacts_v2/calibration_sensitivity.csv
""")
