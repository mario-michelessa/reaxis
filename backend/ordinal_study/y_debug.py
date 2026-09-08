from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import rank_percentile_01, spearman_corr


DEBUG_Y_TRACE_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'task', 'axis_name', 'method', 'budget',
    'target_mode', 'image_id', 'selected_index', 'raw_ground_truth_y',
    'normalized_ground_truth_y', 'prior_score', 'prior_percentile',
    'feedback_value_before_transform', 'bayes_y_used',
    'modeling_study_y_used', 'current_ordinal_y_used', 'bin_index',
    'bin_center', 'warnings',
]

DEBUG_Y_COMPARISON_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'task', 'method', 'budget', 'target_mode',
    'image_id', 'selected_index', 'modeling_study_y_used',
    'current_ordinal_y_used', 'bayes_y_used', 'abs_modeling_vs_ordinal',
    'abs_modeling_vs_bayes', 'abs_ordinal_vs_bayes',
]


def _is_close(a: float, b: float, tol: float = 1e-5) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    return abs(float(a) - float(b)) <= tol


def _prior_scores(session, task) -> np.ndarray:
    if hasattr(session, 'debug_prior_scores'):
        scores = np.asarray(session.debug_prior_scores(), dtype=np.float32).reshape(-1)
        if scores.size == len(task.ids):
            return scores
    return np.zeros((len(task.ids),), dtype=np.float32)


def _bayes_y_used(session, fallback: float) -> float:
    if hasattr(session, 'debug_bayes_y_used'):
        value = float(session.debug_bayes_y_used())
        if np.isfinite(value):
            return value
    return float(fallback)


def make_debug_y_rows(
    *,
    ts_utc: str,
    run_id: str,
    task,
    method: str,
    budget: int,
    target_mode: str,
    selected_index: int,
    image_id: str,
    feedback_before_transform: float,
    session,
) -> tuple[dict[str, object], dict[str, object]]:
    idx = int(selected_index)
    if idx < 0 or idx >= len(task.ids):
        raise IndexError(f'selected_index out of range: {idx}')

    prior = _prior_scores(session, task)
    prior_pct = rank_percentile_01(prior)
    raw_y = float(task.y_raw[idx])
    norm_y = float(task.y01[idx])
    bayes_y = _bayes_y_used(session, feedback_before_transform)
    modeling_y = norm_y
    ordinal_y = float(feedback_before_transform)
    bin_idx = int(min(9, max(0, np.floor(norm_y * 10.0))))
    bin_center = float((bin_idx + 0.5) / 10.0)

    warnings: list[str] = []
    if str(task.ids[idx]) != str(image_id):
        warnings.append('image_id_index_mismatch')
    if task.id_to_index.get(str(image_id)) != idx:
        warnings.append('image_id_reverse_lookup_mismatch')
    if not np.isfinite(bayes_y):
        warnings.append('bayes_y_nonfinite')
    if _is_close(bayes_y, float(prior[idx])):
        warnings.append('bayes_y_equals_prior_score')
    if _is_close(bayes_y, float(prior_pct[idx])):
        warnings.append('bayes_y_equals_prior_percentile')
    if _is_close(bayes_y, norm_y):
        warnings.append('bayes_y_equals_normalized_label')
    if _is_close(bayes_y, float(bin_idx)):
        warnings.append('bayes_y_equals_bin_index')
    if _is_close(bayes_y, bin_center):
        warnings.append('bayes_y_equals_bin_center')

    trace = {
        'ts_utc': ts_utc,
        'run_id': run_id,
        'dataset': task.dataset,
        'task': task.axis_field,
        'axis_name': task.axis_name,
        'method': method,
        'budget': int(budget),
        'target_mode': target_mode,
        'image_id': image_id,
        'selected_index': idx,
        'raw_ground_truth_y': raw_y,
        'normalized_ground_truth_y': norm_y,
        'prior_score': float(prior[idx]),
        'prior_percentile': float(prior_pct[idx]),
        'feedback_value_before_transform': float(feedback_before_transform),
        'bayes_y_used': bayes_y,
        'modeling_study_y_used': modeling_y,
        'current_ordinal_y_used': ordinal_y,
        'bin_index': bin_idx,
        'bin_center': bin_center,
        'warnings': ';'.join(warnings),
    }
    comparison = {
        'ts_utc': ts_utc,
        'run_id': run_id,
        'dataset': task.dataset,
        'task': task.axis_field,
        'method': method,
        'budget': int(budget),
        'target_mode': target_mode,
        'image_id': image_id,
        'selected_index': idx,
        'modeling_study_y_used': modeling_y,
        'current_ordinal_y_used': ordinal_y,
        'bayes_y_used': bayes_y,
        'abs_modeling_vs_ordinal': abs(modeling_y - ordinal_y),
        'abs_modeling_vs_bayes': abs(modeling_y - bayes_y) if np.isfinite(bayes_y) else float('nan'),
        'abs_ordinal_vs_bayes': abs(ordinal_y - bayes_y) if np.isfinite(bayes_y) else float('nan'),
    }
    return trace, comparison


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xx) & np.isfinite(yy)
    if int(np.sum(mask)) < 2:
        return float('nan')
    xx = xx[mask]
    yy = yy[mask]
    if float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
        return float('nan')
    return float(np.corrcoef(xx, yy)[0, 1])


def export_y_debug_reports(output_dir: Path, run_id: str) -> list[str]:
    root = Path(output_dir)
    trace_path = root / 'debug_y_trace.csv'
    compare_path = root / 'debug_y_modeling_vs_ordinal.csv'
    lines: list[str] = []

    if trace_path.exists() and trace_path.stat().st_size > 0:
        trace = pd.read_csv(trace_path)
        trace = trace[trace['run_id'].astype(str) == str(run_id)].copy()
        for col in [
            'budget', 'selected_index', 'raw_ground_truth_y',
            'normalized_ground_truth_y', 'prior_score', 'prior_percentile',
            'feedback_value_before_transform', 'bayes_y_used',
            'modeling_study_y_used', 'current_ordinal_y_used', 'bin_index',
            'bin_center',
        ]:
            if col in trace.columns:
                trace[col] = pd.to_numeric(trace[col], errors='coerce')
        row_warnings = trace['warnings'].fillna('').astype(str)
        warned = int(np.sum(row_warnings.str.len() > 0))
        lines.append(f'[debug-y] trace rows={len(trace)} row_warning_rows={warned}')
        for keys, grp in trace.groupby(['dataset', 'task', 'method', 'target_mode'], dropna=False):
            if grp.empty:
                continue
            distinct = int(np.unique(np.round(grp['normalized_ground_truth_y'].dropna().to_numpy(dtype=float), 8)).size)
            corr = _pearson(grp['bayes_y_used'].to_numpy(dtype=float), grp['normalized_ground_truth_y'].to_numpy(dtype=float))
            rho = spearman_corr(grp['bayes_y_used'].to_numpy(dtype=float), grp['normalized_ground_truth_y'].to_numpy(dtype=float))
            prior_std = float(np.nanstd(grp['prior_score'].to_numpy(dtype=float)))
            bayes_std = float(np.nanstd(grp['bayes_y_used'].to_numpy(dtype=float)))
            ratio = bayes_std / prior_std if prior_std > 1e-12 else float('nan')
            group_warnings: list[str] = []
            if distinct >= 3 and np.isfinite(corr) and corr <= 0.0:
                group_warnings.append(f'nonpositive_corr={corr:.3f}')
            if np.isfinite(ratio) and (ratio > 20.0 or ratio < 0.05):
                group_warnings.append(f'std_ratio={ratio:.3f}')
            if np.isfinite(rho) and rho < -0.5:
                group_warnings.append(f'high_labels_low_targets_spearman={rho:.3f}')
            if group_warnings:
                lines.append(f'[debug-y][warn] {keys}: ' + ', '.join(group_warnings))

    if compare_path.exists() and compare_path.stat().st_size > 0:
        comp = pd.read_csv(compare_path)
        comp = comp[comp['run_id'].astype(str) == str(run_id)].copy()
        for col in [
            'modeling_study_y_used', 'current_ordinal_y_used', 'bayes_y_used',
            'abs_modeling_vs_ordinal', 'abs_modeling_vs_bayes',
            'abs_ordinal_vs_bayes',
        ]:
            if col in comp.columns:
                comp[col] = pd.to_numeric(comp[col], errors='coerce')
        mae_mo = float(comp['abs_modeling_vs_ordinal'].mean()) if not comp.empty else float('nan')
        rho_mo = spearman_corr(comp['modeling_study_y_used'], comp['current_ordinal_y_used']) if not comp.empty else float('nan')
        mae_mb = float(comp['abs_modeling_vs_bayes'].mean()) if not comp.empty else float('nan')
        rho_mb = spearman_corr(comp['modeling_study_y_used'], comp['bayes_y_used']) if not comp.empty else float('nan')
        lines.append(f'[debug-y] modeling_vs_ordinal mean_abs_diff={mae_mo:.6g} spearman={rho_mo:.6g}')
        lines.append(f'[debug-y] modeling_vs_bayes mean_abs_diff={mae_mb:.6g} spearman={rho_mb:.6g}')
        if mae_mo <= 1e-8 and np.isfinite(rho_mo) and rho_mo > 0.999:
            lines.append('[debug-y] ordinal y matches modeling-study y for selected examples.')
        else:
            lines.append('[debug-y][warn] ordinal y does not exactly match modeling-study y.')
        mismatches = comp[
            (comp['abs_modeling_vs_ordinal'] > 1e-6)
            | (comp['abs_modeling_vs_bayes'] > 1e-6)
        ].head(20)
        if not mismatches.empty:
            lines.append('[debug-y] first 20 mismatches:')
            keep = [
                'dataset', 'task', 'method', 'budget', 'target_mode', 'image_id',
                'modeling_study_y_used', 'current_ordinal_y_used', 'bayes_y_used',
                'abs_modeling_vs_ordinal', 'abs_modeling_vs_bayes',
            ]
            lines.extend(mismatches[keep].to_string(index=False).splitlines())
    for line in lines:
        print(line)
    return lines
