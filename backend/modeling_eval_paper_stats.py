#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELING_DIR = REPO_ROOT / 'backend' / 'experiments' / 'modeling_eval'
OUTPUT_PATH = REPO_ROOT / 'backend' / 'figures' / 'modeling_eval_paper_stats.json'

TARGET_METHOD = 'request_bayes_linear_gaussian'
BASELINE_METHODS = [
    'clip_text_similarity',
    'prompt_ensemble',
    'label_only_linear',
    'knn_label_propagation',
]
PAPER_METHODS = BASELINE_METHODS + [TARGET_METHOD]


def latest_run_id(modeling_dir: Path) -> str:
    runs = pd.read_csv(modeling_dir / 'runs.csv')
    if runs.empty:
        raise RuntimeError(f'No runs found in {modeling_dir}')
    return str(runs['run_id'].iloc[-1])


def load_frame(modeling_dir: Path, name: str, run_id: str) -> pd.DataFrame:
    path = modeling_dir / f'{name}.csv'
    frame = pd.read_csv(path)
    return frame[frame['run_id'].astype(str) == str(run_id)].copy()


def mean_series(frame: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> List[Dict[str, Any]]:
    grouped = frame.groupby(group_cols)[value_cols].mean(numeric_only=True).reset_index()
    grouped = grouped.sort_values(group_cols, kind='mergesort')
    rows: List[Dict[str, Any]] = []
    for row in grouped.to_dict(orient='records'):
        rows.append({key: (None if pd.isna(value) else float(value) if isinstance(value, (np.floating, float)) else value) for key, value in row.items()})
    return rows


def task_level_auc(frame: pd.DataFrame, value_col: str) -> Dict[str, float]:
    scores: Dict[str, List[float]] = {}
    for method, method_frame in frame.groupby('method'):
        values: List[float] = []
        for _, task_frame in method_frame.groupby(['dataset', 'task_id'], dropna=False):
            task_frame = task_frame.sort_values('interaction_count', kind='mergesort')
            xs = task_frame['interaction_count'].to_numpy(dtype=float)
            ys = task_frame[value_col].to_numpy(dtype=float)
            mask = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[mask]
            ys = ys[mask]
            if xs.size == 0:
                continue
            if xs.size == 1 or float(xs.max() - xs.min()) <= 1e-8:
                values.append(float(ys[-1]))
            else:
                values.append(float(np.trapezoid(ys, xs) / float(xs.max() - xs.min())))
        if values:
            scores[str(method)] = float(np.mean(values))
    return scores


def build_report(modeling_dir: Path, run_id: str) -> Dict[str, Any]:
    tasks = load_frame(modeling_dir, 'tasks', run_id)
    prior = load_frame(modeling_dir, 'prior', run_id)
    refinement = load_frame(modeling_dir, 'refinement', run_id)
    uncertainty = load_frame(modeling_dir, 'uncertainty', run_id)
    query_policy = load_frame(modeling_dir, 'query_policy', run_id)

    prior = prior[prior['method'].isin(['clip_text_similarity', 'prompt_ensemble', TARGET_METHOD])].copy()
    prior_summary = mean_series(prior, ['method'], ['spearman', 'auroc_binary', 'kendall_tau', 'pairwise_acc'])

    prior_pivot = prior.pivot_table(index=['dataset', 'task_id'], columns='method', values=['spearman', 'auroc_binary'], aggfunc='first')
    prior_equivalence: Dict[str, Dict[str, float]] = {}
    if ('spearman', TARGET_METHOD) in prior_pivot.columns and ('spearman', 'prompt_ensemble') in prior_pivot.columns:
        diff = (prior_pivot[('spearman', TARGET_METHOD)] - prior_pivot[('spearman', 'prompt_ensemble')]).abs()
        prior_equivalence['spearman_vs_prompt_ensemble'] = {
            'mean_abs_diff': float(diff.mean()),
            'max_abs_diff': float(diff.max()),
        }
    if ('auroc_binary', TARGET_METHOD) in prior_pivot.columns and ('auroc_binary', 'prompt_ensemble') in prior_pivot.columns:
        diff = (prior_pivot[('auroc_binary', TARGET_METHOD)] - prior_pivot[('auroc_binary', 'prompt_ensemble')]).abs()
        prior_equivalence['auroc_vs_prompt_ensemble'] = {
            'mean_abs_diff': float(diff.mean()),
            'max_abs_diff': float(diff.max()),
        }

    steps = refinement[
        (refinement['row_type'].astype(str) == 'step')
        & (refinement['variant'].astype(str) == 'main')
        & (refinement['policy'].astype(str) == 'hybrid')
        & (~refinement['allow_undefined'].astype(bool))
        & (refinement['method'].isin(PAPER_METHODS))
    ].copy()
    for col in ['interaction_count', 'spearman_all', 'auroc_binary_all', 'kendall_tau_all', 'pairwise_acc_all']:
        steps[col] = pd.to_numeric(steps[col], errors='coerce')
    curve_summary = mean_series(steps, ['method', 'interaction_count'], ['spearman_all', 'auroc_binary_all', 'kendall_tau_all', 'pairwise_acc_all'])

    final_budget = int(pd.to_numeric(steps['interaction_count'], errors='coerce').max())
    final_steps = steps[steps['interaction_count'] == final_budget].copy()
    final_summary = mean_series(final_steps, ['method'], ['spearman_all', 'auroc_binary_all', 'kendall_tau_all', 'pairwise_acc_all'])

    aulc_spearman = task_level_auc(steps, 'spearman_all')
    aulc_auroc = task_level_auc(steps.dropna(subset=['auroc_binary_all']).copy(), 'auroc_binary_all')

    gain_rows: List[Dict[str, Any]] = []
    for interaction_count, frame in steps.groupby('interaction_count', dropna=False):
        pivot = frame.pivot_table(index=['dataset', 'task_id'], columns='method', values='spearman_all', aggfunc='first')
        baseline_cols = [method for method in BASELINE_METHODS if method in pivot.columns]
        if TARGET_METHOD not in pivot.columns or not baseline_cols:
            continue
        diff = pivot[TARGET_METHOD] - pivot[baseline_cols].max(axis=1)
        gain_rows.append(
            {
                'interaction_count': int(interaction_count),
                'mean_gain_vs_best_baseline': float(diff.mean()),
                'median_gain_vs_best_baseline': float(diff.median()),
                'win_rate_vs_best_baseline': float((diff > 0.0).mean()),
            }
        )

    uncertainty_rows = uncertainty[uncertainty['method'].astype(str) == TARGET_METHOD].copy()
    uncertainty_summary: Dict[str, Any] = {}
    if not uncertainty_rows.empty:
        err = uncertainty_rows[uncertainty_rows['row_type'].astype(str) == 'error_prediction'].copy()
        if not err.empty:
            uncertainty_summary['error_prediction_mean'] = {
                'corr_uncert_abs_error': float(pd.to_numeric(err['corr_uncert_abs_error'], errors='coerce').mean()),
                'auroc_future_correction': float(pd.to_numeric(err['auroc_future_correction'], errors='coerce').mean()),
            }

    qp = query_policy[
        (query_policy['variant'].astype(str) == 'query_policy')
        & (query_policy['row_type'].astype(str) == 'step')
        & (query_policy['method'].astype(str) == TARGET_METHOD)
    ].copy()
    for col in ['interaction_count', 'spearman_all', 'auroc_binary_all']:
        qp[col] = pd.to_numeric(qp[col], errors='coerce')
    query_summary = mean_series(qp, ['policy', 'interaction_count'], ['spearman_all', 'auroc_binary_all'])

    report: Dict[str, Any] = {
        'run_id': run_id,
        'n_tasks': int(len(tasks)),
        'tasks_per_dataset': {str(key): int(value) for key, value in tasks.groupby('dataset').size().to_dict().items()},
        'prior_summary': prior_summary,
        'prior_equivalence': prior_equivalence,
        'interaction_curve_summary': curve_summary,
        'final_budget': final_budget,
        'final_summary': final_summary,
        'aulc_summary': [
            {
                'method': method,
                'spearman_aulc': float(aulc_spearman.get(method, float('nan'))),
                'auroc_aulc': float(aulc_auroc.get(method, float('nan'))),
            }
            for method in PAPER_METHODS
            if method in aulc_spearman or method in aulc_auroc
        ],
        'gain_vs_best_baseline': gain_rows,
        'uncertainty_summary': uncertainty_summary,
        'query_policy_summary': query_summary,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize modeling-study results for the paper.')
    parser.add_argument('--run-id', default='', help='Optional run id. Defaults to the latest run in backend/experiments/modeling_eval.')
    parser.add_argument('--output', default=str(OUTPUT_PATH), help='Where to write the JSON summary.')
    args = parser.parse_args()

    run_id = str(args.run_id).strip() or latest_run_id(MODELING_DIR)
    report = build_report(MODELING_DIR, run_id)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
