from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from .data import OrdinalTask
from .metrics import rank_percentile_01, spearman_corr


DIAGNOSTIC_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'axis_field', 'axis_name', 'method',
    'policy', 'replicate', 'budget', 'label_count', 'prior_s0_mean',
    'prior_s0_std', 'prior_s0_min', 'prior_s0_max', 'current_score_mean',
    'current_score_std', 'current_score_min', 'current_score_max', 'y_mean',
    'y_std', 'y_min', 'y_max', 'prior_rank_mean', 'prior_rank_std',
    'labeled_y_mean', 'labeled_y_std', 'labeled_y_min', 'labeled_y_max',
    'residual_raw_mean', 'residual_raw_std', 'residual_raw_abs_mean',
    'residual_raw_mean_abs_over_std', 'residual_rank_mean',
    'residual_rank_std', 'residual_rank_abs_mean', 'mu_d0_cos',
    'mu_d0_rel_change', 'mu_norm', 'd0_norm', 'mu_xbar_cos',
    'score_preservation_spearman', 'global_spearman',
    'labeled_mae_rank', 'labeled_mae_raw', 'labeled_prior_rank_mae',
    'centered_ridge_spearman', 'intercept_ridge_spearman',
    'reaxis_sigma2_over_alpha', 'prior_blend_best_lambda',
    'prior_blend_best_spearman', 'prior_blend_lambda_0_spearman',
    'prior_blend_lambda_1_spearman', 'prior_blend_lambda_10_spearman',
    'prior_blend_lambda_100_spearman', 'selected_y_bin_0',
    'selected_y_bin_1', 'selected_y_bin_2', 'selected_y_bin_3',
    'selected_y_bin_4', 'uncertainty_mean', 'uncertainty_std',
    'uncertainty_error_corr', 'uncertainty_error_auroc',
    'pair_count', 'move_count',
]

_PRIOR_BLEND_LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def is_reaxis_session(session: object) -> bool:
    return hasattr(session, 'state')


def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f'{prefix}_mean': float('nan'),
            f'{prefix}_std': float('nan'),
            f'{prefix}_min': float('nan'),
            f'{prefix}_max': float('nan'),
        }
    return {
        f'{prefix}_mean': float(np.mean(arr)),
        f'{prefix}_std': float(np.std(arr)),
        f'{prefix}_min': float(np.min(arr)),
        f'{prefix}_max': float(np.max(arr)),
    }


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size != bb.size or aa.size == 0:
        return float('nan')
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return float('nan')
    return float(np.dot(aa, bb) / denom)


def _selected_indices(task: OrdinalTask, state: object) -> np.ndarray:
    mapping = task.id_to_index
    out = [
        int(mapping[str(image_id)])
        for image_id in getattr(state, 'move_order', [])
        if str(image_id) in mapping
    ]
    return np.asarray(out, dtype=np.int64)


def _selected_histogram(y: np.ndarray, selected: np.ndarray) -> dict[str, int]:
    out = {f'selected_y_bin_{idx}': 0 for idx in range(5)}
    if selected.size == 0:
        return out
    bins = np.floor(np.clip(y[selected], 0.0, 1.0 - 1e-8) * 5.0).astype(np.int64)
    counts = np.bincount(np.clip(bins, 0, 4), minlength=5)
    for idx, count in enumerate(counts[:5]):
        out[f'selected_y_bin_{idx}'] = int(count)
    return out


def _ridge_ablations(X: np.ndarray, y: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    if selected.size < 2:
        return {
            'centered_ridge_spearman': float('nan'),
            'intercept_ridge_spearman': float('nan'),
        }

    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X[selected], y[selected])
    intercept_pred = np.asarray(model.predict(X), dtype=np.float32)

    x_mean = np.mean(X, axis=0, keepdims=True)
    y_mean = float(np.mean(y))
    x_centered = np.asarray(X - x_mean, dtype=np.float32)
    y_centered = np.asarray(y - y_mean, dtype=np.float32)
    centered = Ridge(alpha=1.0, fit_intercept=False)
    centered.fit(x_centered[selected], y_centered[selected])
    centered_pred = np.asarray(centered.predict(x_centered) + y_mean, dtype=np.float32)

    return {
        'centered_ridge_spearman': spearman_corr(centered_pred, y),
        'intercept_ridge_spearman': spearman_corr(intercept_pred, y),
    }


def _prior_blend_sweep(X: np.ndarray, y: np.ndarray, selected: np.ndarray, s0: np.ndarray) -> dict[str, float]:
    empty = {
        'prior_blend_best_lambda': float('nan'),
        'prior_blend_best_spearman': float('nan'),
        'prior_blend_lambda_0_spearman': float('nan'),
        'prior_blend_lambda_1_spearman': float('nan'),
        'prior_blend_lambda_10_spearman': float('nan'),
        'prior_blend_lambda_100_spearman': float('nan'),
    }
    if selected.size < 2:
        return empty

    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X[selected], y[selected])
    label_pred = np.asarray(model.predict(X), dtype=np.float32)
    prior_rank = rank_percentile_01(s0)
    best_lambda = float('nan')
    best_spearman = -np.inf
    by_lambda: dict[float, float] = {}
    for lam in _PRIOR_BLEND_LAMBDAS:
        blended = ((float(lam) * prior_rank) + label_pred) / (float(lam) + 1.0)
        rho = spearman_corr(blended, y)
        by_lambda[float(lam)] = float(rho)
        if np.isfinite(rho) and float(rho) > best_spearman:
            best_spearman = float(rho)
            best_lambda = float(lam)
    return {
        'prior_blend_best_lambda': best_lambda,
        'prior_blend_best_spearman': best_spearman if np.isfinite(best_spearman) else float('nan'),
        'prior_blend_lambda_0_spearman': by_lambda.get(0.0, float('nan')),
        'prior_blend_lambda_1_spearman': by_lambda.get(1.0, float('nan')),
        'prior_blend_lambda_10_spearman': by_lambda.get(10.0, float('nan')),
        'prior_blend_lambda_100_spearman': by_lambda.get(100.0, float('nan')),
    }


def _uncertainty_quality(uncertainty: np.ndarray, scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    u = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    score_rank = rank_percentile_01(scores)
    err = np.abs(np.asarray(score_rank, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    finite = np.isfinite(u) & np.isfinite(err)
    if np.sum(finite) < 3:
        return {
            'uncertainty_mean': float('nan'),
            'uncertainty_std': float('nan'),
            'uncertainty_error_corr': float('nan'),
            'uncertainty_error_auroc': float('nan'),
        }
    uf = u[finite]
    ef = err[finite]
    if float(np.std(uf)) <= 1e-12 or float(np.std(ef)) <= 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(uf, ef)[0, 1])
    threshold = float(np.quantile(ef, 0.75))
    high_error = (ef >= threshold).astype(np.int32)
    if int(np.min(high_error)) == int(np.max(high_error)):
        auc = float('nan')
    else:
        auc = float(roc_auc_score(high_error, uf))
    return {
        'uncertainty_mean': float(np.mean(uf)),
        'uncertainty_std': float(np.std(uf)),
        'uncertainty_error_corr': corr,
        'uncertainty_error_auroc': auc,
    }


def compute_reaxis_diagnostics(
    *,
    task: OrdinalTask,
    session: object,
    scores: np.ndarray,
    global_spearman: float,
) -> Mapping[str, object]:
    state = session.state
    X = np.asarray(task.X, dtype=np.float32)
    y = np.asarray(task.y01, dtype=np.float32).reshape(-1)
    s0 = np.asarray(getattr(state, 'z0_all'), dtype=np.float32).reshape(-1)
    mu = np.asarray(getattr(state, 'mu'), dtype=np.float32).reshape(-1)
    d0 = np.asarray(getattr(state, 'w0'), dtype=np.float32).reshape(-1)
    current = np.asarray(scores, dtype=np.float32).reshape(-1)
    selected = _selected_indices(task, state)
    prior_rank = rank_percentile_01(s0)
    current_rank = rank_percentile_01(current)

    if selected.size > 0:
        residual_raw = y[selected] - s0[selected]
        residual_rank = y[selected] - prior_rank[selected]
        labeled_mae_rank = float(np.mean(np.abs(current_rank[selected] - y[selected])))
        labeled_mae_raw = float(np.mean(np.abs(current[selected] - y[selected])))
        labeled_prior_mae = float(np.mean(np.abs(prior_rank[selected] - y[selected])))
        labeled_stats = _safe_stats(y[selected], 'labeled_y')
    else:
        residual_raw = np.zeros((0,), dtype=np.float32)
        residual_rank = np.zeros((0,), dtype=np.float32)
        labeled_mae_rank = float('nan')
        labeled_mae_raw = float('nan')
        labeled_prior_mae = float('nan')
        labeled_stats = _safe_stats(residual_raw, 'labeled_y')

    residual_raw_std = float(np.std(residual_raw)) if residual_raw.size else float('nan')
    residual_ratio = (
        abs(float(np.mean(residual_raw))) / max(residual_raw_std, 1e-12)
        if residual_raw.size else float('nan')
    )
    d0_norm = float(np.linalg.norm(d0))
    mu_norm = float(np.linalg.norm(mu))
    rel_change = (
        float(np.linalg.norm(mu - d0) / max(d0_norm, 1e-12))
        if mu.size == d0.size and mu.size > 0 else float('nan')
    )

    row: dict[str, object] = {}
    row.update(_safe_stats(s0, 'prior_s0'))
    row.update(_safe_stats(current, 'current_score'))
    row.update(_safe_stats(y, 'y'))
    row.update({
        'prior_rank_mean': float(np.mean(prior_rank)) if prior_rank.size else float('nan'),
        'prior_rank_std': float(np.std(prior_rank)) if prior_rank.size else float('nan'),
    })
    row.update(labeled_stats)
    row.update({
        'residual_raw_mean': float(np.mean(residual_raw)) if residual_raw.size else float('nan'),
        'residual_raw_std': residual_raw_std,
        'residual_raw_abs_mean': float(np.mean(np.abs(residual_raw))) if residual_raw.size else float('nan'),
        'residual_raw_mean_abs_over_std': residual_ratio,
        'residual_rank_mean': float(np.mean(residual_rank)) if residual_rank.size else float('nan'),
        'residual_rank_std': float(np.std(residual_rank)) if residual_rank.size else float('nan'),
        'residual_rank_abs_mean': float(np.mean(np.abs(residual_rank))) if residual_rank.size else float('nan'),
        'mu_d0_cos': _safe_cosine(mu, d0),
        'mu_d0_rel_change': rel_change,
        'mu_norm': mu_norm,
        'd0_norm': d0_norm,
        'mu_xbar_cos': _safe_cosine(mu, np.mean(X, axis=0)),
        'score_preservation_spearman': spearman_corr(s0, current),
        'global_spearman': float(global_spearman),
        'labeled_mae_rank': labeled_mae_rank,
        'labeled_mae_raw': labeled_mae_raw,
        'labeled_prior_rank_mae': labeled_prior_mae,
        'reaxis_sigma2_over_alpha': float(getattr(state, 'sigma2')) / max(float(getattr(state, 'alpha')), 1e-12),
        'pair_count': int(len(getattr(state, 'pair_i', []))),
        'move_count': int(len(getattr(state, 'move_order', []))),
    })
    row.update(_ridge_ablations(X, y, selected))
    row.update(_prior_blend_sweep(X, y, selected, s0))
    row.update(_selected_histogram(y, selected))
    row.update(_uncertainty_quality(np.asarray(session.uncertainty(), dtype=np.float32), current, y))
    return row
