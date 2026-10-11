import sys, os, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.autoencoder import Autoencoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

ARTEFACT_DIR = r'C:/MLProject/zero_day_project/artefacts_v2'
DEDUP_DIR    = os.path.join(ARTEFACT_DIR, 'dedup')
SEEDS        = [42, 123, 456, 789, 999]

def norm_scores(scores, p1, p99):
    eps = 1e-8
    return np.clip((scores - p1) / (p99 - p1 + eps), 0, 1)

def run():
    log.info('Loading data...')

    d_train = np.load(os.path.join(DEDUP_DIR, 'train_dedup.npz'))
    X_train_full = np.clip(d_train['X_train'], -10, 10).astype(np.float32)
    n_per_env = len(X_train_full) // 3
    X_train_single = X_train_full[:n_per_env]
    log.info('CIC17-only training: %d flows', len(X_train_single))

    d_val = np.load(os.path.join(DEDUP_DIR, 'val_dedup.npz'))
    X_val_full = np.clip(d_val['X_val'], -10, 10).astype(np.float32)
    n_val_per_env = len(X_val_full) // 3
    X_val_single = X_val_full[:n_val_per_env]

    test_sets = {}
    for name, fname in [
        ('CIC17', 'test_cic17_dedup.npz'),
        ('CIC18', 'test_cic18_dedup.npz'),
        ('UNSW',  'test_unsw_dedup.npz'),
        ('BoT',   'test_botiot_dedup.npz'),
    ]:
        path = os.path.join(DEDUP_DIR, fname)
        if os.path.exists(path):
            d = np.load(path)
            test_sets[name] = (
                np.clip(d['X'], -10, 10).astype(np.float32),
                d['y'].astype(int)
            )
            log.info('Loaded %s: %d rows', name, len(test_sets[name][0]))

    results = []

    for seed in SEEDS:
        log.info('='*50)
        log.info('SEED %d', seed)
        t0 = time.time()

        np.random.seed(seed)
        torch.manual_seed(seed)

        log.info('Training AE-Single (CIC17 only)...')
        ae = Autoencoder(input_dim=18)
        ae.fit(X_train_single, X_val_single)

        train_scores = ae.anomaly_scores(X_train_single)
        p1  = float(np.percentile(train_scores, 1))
        p99 = float(np.percentile(train_scores, 99))

        val_scores = ae.anomaly_scores(X_val_single)
        val_norm = norm_scores(val_scores, p1, p99)
        tau = float(np.percentile(val_norm, 95))
        log.info('  tau=%.4f', tau)

        row = {'seed': seed, 'tau': tau}

        for name, (X_test, y_test) in test_sets.items():
            scores = ae.anomaly_scores(X_test)
            scores_norm = norm_scores(scores, p1, p99)

            if name == 'BoT':
                dr = float((scores_norm > tau).mean())
                row['BoT-IoT_DR'] = dr
                log.info('  BoT DR=%.4f', dr)
            else:
                if len(np.unique(y_test)) < 2:
                    continue
                auc = roc_auc_score(y_test, scores_norm)
                col = {
                    'CIC17': 'CIC-IDS2017_AUC',
                    'CIC18': 'CSE-CIC-IDS2018_AUC',
                    'UNSW':  'UNSW-NB15_AUC',
                }[name]
                row[col] = auc
                log.info('  %s AUC=%.4f', name, auc)

        elapsed = time.time() - t0
        log.info('Seed %d done in %.1f min', seed, elapsed/60)
        results.append(row)

    df = pd.DataFrame(results)
    out_path = os.path.join(DEDUP_DIR, 'ae_single_results.csv')
    df.to_csv(out_path, index=False)
    log.info('Saved -> %s', out_path)

    print('\nAE-SINGLE RESULTS:')
    print(df.to_string(index=False))
    print('\nMEAN +/- STD:')
    cols = {
        'CIC17': 'CIC-IDS2017_AUC',
        'CIC18': 'CSE-CIC-IDS2018_AUC',
        'UNSW':  'UNSW-NB15_AUC',
        'BoT':   'BoT-IoT_DR',
    }
    ae_multi = {
        'CIC17': 0.8299, 'CIC18': 0.9859,
        'UNSW':  0.7403, 'BoT':   0.5802,
    }
    print(f'{"Dataset":<10} {"Single":>10} {"Multi":>10} {"Gain":>8}')
    print('-'*42)
    for name, col in cols.items():
        if col in df.columns:
            m = df[col].mean()
            s = df[col].std()
            gain = ae_multi[name] - m
            print(f'{name:<10} {m:>6.4f}+/-{s:.4f} {ae_multi[name]:>10.4f} {gain:>+8.4f}')

if __name__ == '__main__':
    run()
