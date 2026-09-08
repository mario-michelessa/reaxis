from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import cohen_kappa_score

from .config import AXIS_BIN_COUNT, MAX_PAIRWISE_METRIC_PAIRS, QWK_CLASS_COUNT


def rank_percentile_01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    n = int(arr.size)
    if n <= 1:
        return np.full((n,), 0.5, dtype=np.float32)
    order = np.argsort(arr, kind='mergesort')
    sorted_vals = arr[order]
    out = np.zeros_like(arr, dtype=np.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(float(sorted_vals[j]) - float(sorted_vals[i])) <= 1e-12:
            j += 1
        out[order[i:j]] = float(i + (j - i - 1) * 0.5) / float(n - 1)
        i = j
    return out


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[finite]
    ya = ya[finite]
    if xa.size < 2:
        return float('nan')
    xa = xa - float(np.mean(xa))
    ya = ya - float(np.mean(ya))
    denom = float(np.linalg.norm(xa) * np.linalg.norm(ya))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xa, ya) / denom)


def spearman_corr(pred: np.ndarray, ref: np.ndarray) -> float:
    return pearson_corr(rank_percentile_01(pred), rank_percentile_01(ref))


def sampled_kendall_tau(pred: np.ndarray, ref: np.ndarray, rng: np.random.Generator) -> float:
    p = np.asarray(pred, dtype=np.float32).reshape(-1)
    r = np.asarray(ref, dtype=np.float32).reshape(-1)
    n = int(p.size)
    if n < 2:
        return float('nan')
    total_pairs = (n * (n - 1)) // 2
    if total_pairs <= MAX_PAIRWISE_METRIC_PAIRS:
        ii, jj = np.triu_indices(n, k=1)
    else:
        ii = rng.integers(0, n, size=MAX_PAIRWISE_METRIC_PAIRS, endpoint=False)
        jj = rng.integers(0, n - 1, size=MAX_PAIRWISE_METRIC_PAIRS, endpoint=False)
        jj = np.where(jj >= ii, jj + 1, jj)
    ref_diff = r[ii] - r[jj]
    pred_diff = p[ii] - p[jj]
    valid = np.abs(ref_diff) > 1e-8
    if not np.any(valid):
        return float('nan')
    signed = pred_diff[valid] * ref_diff[valid]
    concordant = float(np.sum(signed > 0.0))
    discordant = float(np.sum(signed < 0.0))
    return float((concordant - discordant) / max(1.0, float(np.sum(valid))))


def isotonic_calibrated(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    x = rank_percentile_01(pred)
    y = np.asarray(ref, dtype=np.float32).reshape(-1)
    if y.size < 2:
        return np.full_like(y, float(np.mean(y)) if y.size else 0.5)
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    return np.asarray(model.fit_transform(x, y), dtype=np.float32)


def discretize01(values: np.ndarray, n_classes: int = QWK_CLASS_COUNT) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    bins = np.floor(np.clip(arr, 0.0, 1.0 - 1e-8) * int(n_classes)).astype(np.int32)
    return np.clip(bins, 0, int(n_classes) - 1)


def quadratic_weighted_kappa(pred_calibrated: np.ndarray, ref: np.ndarray) -> float:
    y_true = discretize01(ref)
    y_pred = discretize01(pred_calibrated)
    if int(np.min(y_true)) == int(np.max(y_true)):
        return float('nan')
    return float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))


def axis_gradient(pred: np.ndarray, ref: np.ndarray, n_bins: int = AXIS_BIN_COUNT) -> Tuple[Dict[str, float], List[dict]]:
    p = np.asarray(pred, dtype=np.float32).reshape(-1)
    r = np.asarray(ref, dtype=np.float32).reshape(-1)
    finite = np.isfinite(p) & np.isfinite(r)
    p = p[finite]
    r = r[finite]
    if p.size < n_bins:
        return {'axis_r2': float('nan'), 'monotonicity_violations': float('nan')}, []
    order = np.argsort(p, kind='mergesort')
    chunks = np.array_split(order, int(n_bins))
    rows = []
    means = []
    for bin_idx, idx in enumerate(chunks):
        mean_ref = float(np.mean(r[idx])) if idx.size else float('nan')
        means.append(mean_ref)
        rows.append({
            'bin_index': int(bin_idx),
            'bin_count': int(idx.size),
            'score_min': float(np.min(p[idx])) if idx.size else float('nan'),
            'score_max': float(np.max(p[idx])) if idx.size else float('nan'),
            'mean_reference': mean_ref,
        })
    y = np.asarray(means, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    if np.all(np.isfinite(y)) and y.size >= 2:
        coef = np.polyfit(x, y, deg=1)
        fit = (coef[0] * x) + coef[1]
        ss_res = float(np.sum((y - fit) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
        violations = int(np.sum(np.diff(y) < -1e-6))
    else:
        r2 = float('nan')
        violations = float('nan')
    return {'axis_r2': float(r2), 'monotonicity_violations': float(violations)}, rows


def compute_metrics(pred: np.ndarray, ref: np.ndarray, rng: np.random.Generator) -> tuple[dict, list[dict]]:
    calibrated = isotonic_calibrated(pred, ref)
    gradient_metrics, bin_rows = axis_gradient(pred, ref)
    mae = float(np.mean(np.abs(calibrated - np.asarray(ref, dtype=np.float32).reshape(-1))))
    metrics = {
        'spearman': spearman_corr(pred, ref),
        'kendall_tau': sampled_kendall_tau(pred, ref, rng),
        'mae_isotonic': mae,
        'qwk': quadratic_weighted_kappa(calibrated, ref),
        **gradient_metrics,
    }
    return metrics, bin_rows
