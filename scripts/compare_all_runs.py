#!/usr/bin/env python3
"""Reconstruct SemCons-IDS v8 splits and audit 12 saved XGB-vs-final runs.
Run from the GitHub package root on Narval; trusted notebook and joblib only.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--notebook', default='notebooks/core/SemCons_IDS_MiniLM_ModernBERT_Comparative_Final_v8.ipynb')
    p.add_argument('--data', default='/project/def-missro01/malj30/datasets/CICIDS2017')
    p.add_argument('--models', default='/project/def-missro01/malj30/semcons_ids/models')
    p.add_argument('--output', default='reports/crisis_analysis')
    p.add_argument('--batch-size', type=int, default=20000)
    p.add_argument('--tolerance', type=float, default=5e-6)
    return p.parse_args()


def load_original_functions(nb_path):
    nb = json.loads(nb_path.read_text())
    src6 = ''.join(nb['cells'][6]['source'])
    src10 = ''.join(nb['cells'][10]['source'])
    scope = {'pd': pd, 'np': np, 're': re, 'Path': Path, 'List': List,
             'Dict': Dict, 'math': math, 'train_test_split': train_test_split}
    exec(src6.split('files = find_data_files')[0], scope)
    exec(src10.split('def capped_indices_per_class')[0], scope)
    return scope


def load_clean_dataset(scope, data_path):
    paths = scope['find_data_files'](str(data_path))
    if len(paths) != 8:
        raise ValueError(f'Expected eight original CSVs; found {len(paths)}: {[p.name for p in paths]}')
    print('Original files:', *[p.name for p in paths], sep='\n  ', flush=True)
    df = pd.concat([scope['read_one_file'](p) for p in paths], ignore_index=True)
    label_candidates = [c for c in df if c == 'label' or 'label' in c or 'class' in c]
    if not label_candidates:
        raise ValueError('No label column')
    label = label_candidates[0]
    df[label] = df[label].map(scope['normalize_label'])
    df = df.dropna(subset=[label]).copy()
    dedup_columns = [c for c in df if c not in {'__source_file__', '__row_in_file__'}]
    df = df.drop_duplicates(subset=dedup_columns).reset_index(drop=True)
    features = [c for c in df if c not in {label, '__source_file__', '__row_in_file__'}]
    for c in features:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    counts = df[label].value_counts()
    df = df[df[label].isin(counts[counts >= 200].index)].reset_index(drop=True)
    if len(df) != 2520172 or len(features) != 78:
        raise ValueError(f'Unexpected cleaned dataset: rows={len(df)}, features={len(features)}')
    print(f'Clean dataset validated: {len(df):,} rows / {len(features)} features', flush=True)
    return df, label


def metric_reference(run, name):
    tab = pd.read_csv(run / 'branch_summary.csv')
    if 'Unnamed: 0' in tab:
        key = 'Unnamed: 0'
    else:
        key = tab.columns[0]
    subset = tab.loc[tab[key].astype(str).str.strip().str.lower() == name.lower()]
    if len(subset) != 1:
        raise ValueError(f'Missing or ambiguous {name} row in branch_summary.csv')
    return subset.iloc[0]


def check_reference(row, accuracy, macro_present, macro_all, tolerance, tag):
    for key, observed in [('accuracy', accuracy), ('macro_f1_present', macro_present), ('macro_f1_all', macro_all)]:
        expected = float(row[key])
        if not np.isclose(expected, observed, atol=tolerance, rtol=0):
            raise ValueError(f'{tag}: {key} observed={observed:.9f} ref={expected:.9f}')


def per_run(run, test, label_col, output, batch_size, tolerance):
    sys.path.insert(0, str(Path('src').resolve()))
    from semcons_corrected_pipeline_utils import load_run_artifacts, transform_with_saved_preprocessor
    art = load_run_artifacts(run)
    if len(test) != 252018:
        raise ValueError(f'Incorrect test length: {len(test)}')
    saved = pd.read_csv(run / 'test_predictions.csv', usecols=['true_label', 'predicted_label'])
    y = saved['true_label'].astype(str).to_numpy()
    yf = saved['predicted_label'].astype(str).to_numpy()
    actual = test[label_col].astype(str).to_numpy()
    if len(y) != len(actual) or not np.array_equal(y, actual):
        raise ValueError('Row-by-row true-label alignment failed')
    pred = []
    for i in range(0, len(test), batch_size):
        batch = test.iloc[i:i+batch_size]
        matrices = transform_with_saved_preprocessor(batch, art)
        p = np.asarray(art.xgb_model.predict_proba(matrices['raw_numeric']))
        pred.append(p.argmax(axis=1))
    xp = art.label_encoder.inverse_transform(np.concatenate(pred)).astype(str)
    classes = list(map(str, art.label_encoder.classes_))
    present = [c for c in classes if np.any(y == c)]
    expected_present = 5 if run.name.startswith('temporal_within_file') else 10
    if len(present) != expected_present:
        raise ValueError(f'Expected {expected_present} represented classes, found {present}')
    xacc = accuracy_score(y, xp)
    facc = accuracy_score(y, yf)
    xpresent = f1_score(y, xp, labels=present, average='macro', zero_division=0)
    fpresent = f1_score(y, yf, labels=present, average='macro', zero_division=0)
    xall = f1_score(y, xp, labels=classes, average='macro', zero_division=0)
    fall = f1_score(y, yf, labels=classes, average='macro', zero_division=0)
    check_reference(metric_reference(run, 'XGBoost'), xacc, xpresent, xall, tolerance, 'XGBoost')
    check_reference(metric_reference(run, 'Final'), facc, fpresent, fall, tolerance, 'Final')
    parts = run.name.split('__')
    protocol, backbone, seed = parts[0], parts[1], int(parts[2].replace('seed_', ''))
    corrected = (xp != y) & (yf == y)
    introduced = (xp == y) & (yf != y)
    common = (xp != y) & (yf != y)
    rows = []
    for cls in classes:
        mask = y == cls
        px, rx, fx, _ = precision_recall_fscore_support(y, xp, labels=[cls], zero_division=0)
        pf, rf, ff, _ = precision_recall_fscore_support(y, yf, labels=[cls], zero_division=0)
        rows.append({'protocol': protocol, 'backbone': backbone, 'seed': seed, 'class': cls,
                     'support': int(mask.sum()), 'xgb_precision': float(px[0]), 'xgb_recall': float(rx[0]),
                     'xgb_f1': float(fx[0]), 'final_precision': float(pf[0]),
                     'final_recall': float(rf[0]), 'final_f1': float(ff[0]),
                     'delta_f1': float(ff[0]-fx[0]), 'xgb_errors': int((mask & (xp != y)).sum()),
                     'final_errors': int((mask & (yf != y)).sum()),
                     'corrected': int((mask & corrected).sum()), 'introduced': int((mask & introduced).sum()),
                     'common_errors': int((mask & common).sum())})
    summary = {'protocol': protocol, 'backbone': backbone, 'seed': seed, 'n_test': len(y),
               'present_classes': len(present), 'xgb_accuracy': xacc, 'final_accuracy': facc,
               'xgb_macro_f1_present': xpresent, 'final_macro_f1_present': fpresent,
               'delta_macro_f1_present': fpresent-xpresent,
               'xgb_macro_f1_all': xall, 'final_macro_f1_all': fall,
               'corrected': int(corrected.sum()), 'introduced': int(introduced.sum()),
               'common_errors': int(common.sum()), 'net_corrected': int(corrected.sum()-introduced.sum()),
               'validation': 'PASS: row labels + xgb and final original metrics'}
    print(f"PASS {run.name}: Macro-F1 {xpresent:.6f} -> {fpresent:.6f}, corrected {summary['corrected']}, introduced {summary['introduced']}", flush=True)
    return rows, summary


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scope = load_original_functions(Path(args.notebook))
    df, label = load_clean_dataset(scope, Path(args.data))
    models = Path(args.models)
    all_rows, all_summary, fails = [], [], []
    for protocol in ['temporal_within_file', 'iid_stratified']:
        for seed in [42, 123, 2026]:
            print(f'\nReconstructing {protocol} / seed {seed}...', flush=True)
            splits = scope['make_splits'](df, protocol, seed, label, .70, .10, .10, .10)
            test = splits['test']
            del splits
            for backbone in ['minilm', 'modernbert']:
                run = models / f'{protocol}__{backbone}__seed_{seed}'
                try:
                    rows, summary = per_run(run, test, label, output, args.batch_size, args.tolerance)
                    all_rows.extend(rows)
                    all_summary.append(summary)
                except Exception as exc:
                    print(f'FAIL {run.name}: {type(exc).__name__}: {exc}', flush=True)
                    fails.append({'run': run.name, 'reason': str(exc)})
    pd.DataFrame(all_rows).to_csv(output/'validated_xgb_fusion_classwise_12runs.csv', index=False)
    pd.DataFrame(all_summary).to_csv(output/'validated_xgb_fusion_run_summary.csv', index=False)
    pd.DataFrame(fails, columns=['run', 'reason']).to_csv(output/'failed_xgb_fusion_runs.csv', index=False)
    if len(all_summary) != 12:
        raise SystemExit(f'INCOMPLETE: validated {len(all_summary)}/12 runs; see failed_xgb_fusion_runs.csv')
    class_df = pd.DataFrame(all_rows)
    grouped = class_df.groupby(['protocol', 'backbone', 'class'], as_index=False).agg(
        seeds=('seed', 'nunique'), support_per_seed=('support', 'mean'),
        xgb_f1_mean=('xgb_f1', 'mean'), xgb_f1_std=('xgb_f1', 'std'),
        final_f1_mean=('final_f1', 'mean'), final_f1_std=('final_f1', 'std'),
        delta_f1_mean=('delta_f1', 'mean'), delta_f1_std=('delta_f1', 'std'),
        corrected_sum=('corrected', 'sum'), introduced_sum=('introduced', 'sum'),
        xgb_errors_sum=('xgb_errors', 'sum'), final_errors_sum=('final_errors', 'sum'))
    grouped.to_csv(output/'validated_xgb_fusion_classwise_mean_std.csv', index=False)
    run_df = pd.DataFrame(all_summary)
    run_df.groupby(['protocol', 'backbone'], as_index=False).agg(
        n_seeds=('seed', 'nunique'), xgb_mean=('xgb_macro_f1_present', 'mean'),
        final_mean=('final_macro_f1_present', 'mean'), delta_mean=('delta_macro_f1_present', 'mean'),
        delta_sd=('delta_macro_f1_present', 'std'), corrected_sum=('corrected', 'sum'),
        introduced_sum=('introduced', 'sum'), net_corrected_sum=('net_corrected', 'sum'))\
        .to_csv(output/'validated_xgb_fusion_macro_mean_std.csv', index=False)
    print('\nALL 12 RUNS VALIDATED; CSVs written to', output, flush=True)


if __name__ == '__main__':
    main()
