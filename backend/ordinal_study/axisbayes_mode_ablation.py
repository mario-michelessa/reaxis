from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import mode_label_for_method
from .fixed_reaxis import safe_cosine
from .metrics import spearman_corr


AXISBAYES_MODE_DEBUG_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'axis_field', 'axis_name', 'method',
    'mode', 'policy', 'replicate', 'budget', 'label_count', 'target_p01',
    'gaussian_scalar_target', 'scalar_observation_count',
    'rank_constraint_count', 'mu_d0_cos', 'score_preservation_spearman',
]

METRIC_COLS = [
    'spearman', 'kendall_tau', 'mae_isotonic', 'qwk',
    'axis_r2', 'monotonicity_violations',
]


def make_axisbayes_mode_debug_row(
    *,
    ts_utc: str,
    run_id: str,
    task,
    method: str,
    policy: str,
    replicate: int,
    budget: int,
    label_count: int,
    session,
    scores: np.ndarray,
) -> dict[str, object]:
    state = session.state
    mode = str(getattr(session, 'axisbayes_mode', getattr(state, 'mode', '')) or '')
    return {
        'ts_utc': ts_utc,
        'run_id': run_id,
        'dataset': task.dataset,
        'axis_field': task.axis_field,
        'axis_name': task.axis_name,
        'method': method,
        'mode': mode,
        'policy': policy,
        'replicate': int(replicate),
        'budget': int(budget),
        'label_count': int(label_count),
        'target_p01': float(getattr(session, 'last_target_p01', np.nan)),
        'gaussian_scalar_target': float(getattr(session, 'last_gaussian_scalar_target', np.nan)),
        'scalar_observation_count': int(np.asarray(getattr(state, 'y', []), dtype=np.float32).reshape(-1).size),
        'rank_constraint_count': int(len(getattr(state, 'pair_i', []))),
        'mu_d0_cos': safe_cosine(
            np.asarray(getattr(state, 'mu', []), dtype=np.float32),
            np.asarray(getattr(state, 'w0', []), dtype=np.float32),
        ),
        'score_preservation_spearman': spearman_corr(
            np.asarray(getattr(state, 'z0_all', []), dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
        ),
    }


def export_axisbayes_mode_ablation_summary(
    output_dir: Path,
    *,
    run_id: str,
    default_axisbayes_mode: str = 'gaussian',
) -> str:
    root = Path(output_dir)
    metrics_path = root / 'metrics.csv'
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return ''
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics['run_id'].astype(str) == str(run_id)].copy()
    if metrics.empty:
        return ''
    numeric_cols = ['budget', 'replicate', 'label_count', *METRIC_COLS]
    for col in numeric_cols:
        metrics[col] = pd.to_numeric(metrics[col], errors='coerce')
    metrics['mode'] = [
        mode_label_for_method(method, default_axisbayes_mode)
        for method in metrics['method'].astype(str).tolist()
    ]
    summary = (
        metrics.groupby(['method', 'mode', 'budget'], dropna=False)[METRIC_COLS]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(['method', 'mode', 'budget'])
    )
    path = root / 'ordinal_axisbayes_mode_ablation_summary.csv'
    summary.to_csv(path, index=False)
    _validate_acceptance(root, run_id=run_id, summary=summary)
    return _markdown_table(summary)


def _validate_acceptance(root: Path, *, run_id: str, summary: pd.DataFrame) -> None:
    debug_path = root / 'axisbayes_mode_debug.csv'
    if debug_path.exists() and debug_path.stat().st_size > 0:
        debug = pd.read_csv(debug_path)
        debug = debug[debug['run_id'].astype(str) == str(run_id)].copy()
        for col in ['budget', 'scalar_observation_count', 'rank_constraint_count']:
            debug[col] = pd.to_numeric(debug[col], errors='coerce')
        gaussian = debug[debug['mode'].astype(str) == 'gaussian']
        if not gaussian.empty:
            bad_rank = gaussian[gaussian['rank_constraint_count'] != 0]
            if not bad_rank.empty:
                raise RuntimeError('Gaussian AxisBayes mode produced rank constraints')
            bad_scalar = gaussian[gaussian['scalar_observation_count'] != gaussian['budget']]
            if not bad_scalar.empty:
                raise RuntimeError('Gaussian AxisBayes scalar observation count did not equal budget')

    random_rows = summary[summary['method'].isin(['reaxis_random_gaussian', 'reaxis_random_rank'])]
    if {'reaxis_random_gaussian', 'reaxis_random_rank'}.issubset(set(random_rows['method'])):
        pivot = random_rows.pivot_table(index='budget', columns='method', values='spearman')
        pivot = pivot[pivot.index > 0].dropna()
        if not pivot.empty and np.allclose(
            pivot['reaxis_random_gaussian'].to_numpy(dtype=float),
            pivot['reaxis_random_rank'].to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-8,
        ):
            raise RuntimeError('reaxis_random_gaussian and reaxis_random_rank produced identical b>0 Spearman values')


def _fmt(value: object) -> str:
    try:
        val = float(value)
    except Exception:
        return ''
    if pd.isna(val):
        return ''
    return f'{val:.3f}'


def _markdown_table(summary: pd.DataFrame) -> str:
    headers = ['method', 'mode', 'budget', *METRIC_COLS]
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in summary.itertuples(index=False):
        values = [str(row.method), str(row.mode), str(int(row.budget))]
        values.extend(_fmt(getattr(row, col)) for col in METRIC_COLS)
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)
