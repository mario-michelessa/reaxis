from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def write_gaussian_sweep_summaries(
    root: Path,
    run_id: str,
    comparison_dirs: Sequence[Path],
    metric_cols: Sequence[str],
) -> None:
    path = Path(root) / 'ordinal_gaussian_sweep_results.csv'
    if not path.exists() or path.stat().st_size == 0:
        return
    rows = pd.read_csv(path)
    rows = rows[rows['run_id'].astype(str) == str(run_id)].copy()
    if rows.empty:
        return
    for col in ['replicate', 'budget', 'label_count', 'alpha', 'sigma2', 'bias_alpha', *metric_cols]:
        rows[col] = pd.to_numeric(rows[col], errors='coerce')
    summary = (
        rows.groupby(['policy', 'alpha', 'sigma2', 'bias_alpha', 'score_mode', 'budget'], dropna=False)[list(metric_cols)]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    summary.columns = ['_'.join(str(part) for part in col if str(part)) for col in summary.columns.to_flat_index()]
    summary.to_csv(Path(root) / 'ordinal_gaussian_sweep_summary.csv', index=False)
    best = _best_progression(rows)
    best.to_csv(Path(root) / 'ordinal_gaussian_sweep_best_progression.csv', index=False)
    comparison = _compare_to_envelope(rows, comparison_dirs)
    comparison.to_csv(Path(root) / 'ordinal_gaussian_sweep_vs_label_envelope.csv', index=False)
    _write_report(Path(root) / 'ordinal_gaussian_sweep_report.md', best, comparison, rows)


def _best_progression(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(['policy', 'alpha', 'sigma2', 'bias_alpha', 'score_mode', 'budget'], dropna=False)['spearman'].mean().reset_index()
    budgets = sorted(int(v) for v in grouped['budget'].dropna().unique())
    final_budget = max(budgets)
    out = []
    for key, group in grouped.groupby(['policy', 'alpha', 'sigma2', 'bias_alpha', 'score_mode'], dropna=False):
        curve = group.set_index('budget')['spearman'].to_dict()
        values = np.asarray([float(curve.get(b, np.nan)) for b in budgets], dtype=float)
        finite_values = values[np.isfinite(values)]
        adjacent = np.diff(values) if values.size >= 2 and np.all(np.isfinite(values)) else np.asarray([], dtype=float)
        out.append({
            'policy': key[0],
            'alpha': float(key[1]),
            'sigma2': float(key[2]),
            'bias_alpha': float(key[3]),
            'score_mode': key[4],
            'spearman_b0': float(curve.get(0, np.nan)),
            f'spearman_b{final_budget}': float(curve.get(final_budget, np.nan)),
            'gain_final_minus_b0': float(curve.get(final_budget, np.nan) - curve.get(0, np.nan)),
            'mean_spearman': float(np.mean(finite_values)) if finite_values.size else float('nan'),
            'min_adjacent_gain': float(np.min(adjacent)) if adjacent.size else float('nan'),
            'curve': ', '.join(f'b{b}={curve.get(b, float("nan")):.3f}' for b in budgets),
        })
    return pd.DataFrame(out).sort_values(
        ['gain_final_minus_b0', f'spearman_b{final_budget}', 'min_adjacent_gain'],
        ascending=[False, False, False],
    )


def _compare_to_envelope(rows: pd.DataFrame, comparison_dirs: Sequence[Path]) -> pd.DataFrame:
    budgets = sorted(int(v) for v in rows['budget'].dropna().unique())
    final_budget = max(budgets)
    best = _best_progression(rows)
    top_keys = best.head(20)[['policy', 'alpha', 'sigma2', 'bias_alpha', 'score_mode']]
    merged = rows.merge(top_keys, on=['policy', 'alpha', 'sigma2', 'bias_alpha', 'score_mode'], how='inner')
    reaxis = (
        merged.groupby(['policy', 'alpha', 'sigma2', 'bias_alpha', 'score_mode', 'budget'], dropna=False)['spearman']
        .mean()
        .reset_index(name='reaxis_spearman')
    )
    envelope = _load_label_only_envelope(comparison_dirs)
    if envelope.empty:
        reaxis['best_label_only_spearman'] = np.nan
        reaxis['text_prior_spearman'] = reaxis.groupby(['policy', 'alpha', 'sigma2', 'bias_alpha', 'score_mode'])['reaxis_spearman'].transform('first')
    else:
        reaxis = reaxis.merge(envelope, on='budget', how='left')
        fallback_prior = reaxis.groupby(['policy', 'alpha', 'sigma2', 'bias_alpha', 'score_mode'])['reaxis_spearman'].transform('first')
        reaxis['text_prior_spearman'] = reaxis['text_prior_spearman'].fillna(fallback_prior)
    reaxis['prior_or_label_envelope'] = np.nanmax(
        reaxis[['text_prior_spearman', 'best_label_only_spearman']].to_numpy(dtype=float),
        axis=1,
    )
    reaxis['margin_vs_envelope'] = reaxis['reaxis_spearman'] - reaxis['prior_or_label_envelope']
    reaxis['final_budget'] = int(final_budget)
    return reaxis.sort_values(['budget', 'margin_vs_envelope'], ascending=[True, False])


def _load_label_only_envelope(comparison_dirs: Sequence[Path]) -> pd.DataFrame:
    frames = []
    prior_frames = []
    label_only = {
        'label_mean', 'ols_linear', 'bayesian_ridge', 'elastic_net', 'pcr_ridge',
        'ordinal_ridge', 'rank_svm', 'knn_ordinal', 'kernel_ridge', 'linear_svr',
    }
    for root in comparison_dirs:
        path = Path(root) / 'metrics.csv'
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if {'method', 'budget', 'spearman'}.issubset(frame.columns):
            prior_frames.append(frame[frame['method'].astype(str) == 'text_prior'].copy())
            frames.append(frame[frame['method'].astype(str).isin(label_only)].copy())
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame['budget'] = pd.to_numeric(frame['budget'], errors='coerce')
    frame['spearman'] = pd.to_numeric(frame['spearman'], errors='coerce')
    by_method = frame.groupby(['method', 'budget'], dropna=False)['spearman'].mean().reset_index()
    best = by_method.sort_values('spearman', ascending=False).groupby('budget', as_index=False).first()
    prior = _load_text_prior(prior_frames)
    out = best[['budget', 'method', 'spearman']].rename(
        columns={'method': 'best_label_only_method', 'spearman': 'best_label_only_spearman'},
    )
    return out.merge(prior, on='budget', how='left')


def _load_text_prior(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frame_list = [frame for frame in frames if not frame.empty]
    if not frame_list:
        return pd.DataFrame({'budget': [], 'text_prior_spearman': []})
    all_rows = pd.concat(frame_list, ignore_index=True)
    prior = all_rows[all_rows['method'].astype(str) == 'text_prior'].copy()
    if prior.empty:
        return pd.DataFrame({'budget': [], 'text_prior_spearman': []})
    prior['budget'] = pd.to_numeric(prior['budget'], errors='coerce')
    prior['spearman'] = pd.to_numeric(prior['spearman'], errors='coerce')
    return prior.groupby('budget', as_index=False)['spearman'].mean().rename(columns={'spearman': 'text_prior_spearman'})


def _fmt(value: object, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ''
    if not np.isfinite(val):
        return ''
    return f'{val:.{digits}f}'


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    out.extend('| ' + ' | '.join(str(item) for item in row) + ' |' for row in rows)
    return '\n'.join(out)


def _write_report(path: Path, best: pd.DataFrame, comparison: pd.DataFrame, rows: pd.DataFrame) -> None:
    budgets = sorted(int(v) for v in rows['budget'].dropna().unique())
    final_budget = max(budgets)
    top_rows = []
    for row in best.head(12).itertuples(index=False):
        top_rows.append([
            row.policy,
            _fmt(row.alpha, 5),
            _fmt(row.sigma2, 5),
            _fmt(row.bias_alpha, 5),
            row.score_mode,
            _fmt(row.spearman_b0),
            _fmt(getattr(row, f'spearman_b{final_budget}')),
            _fmt(row.gain_final_minus_b0),
            _fmt(row.min_adjacent_gain),
            row.curve,
        ])
    margin_rows = []
    if not comparison.empty:
        final = comparison[comparison['budget'] == final_budget].sort_values('margin_vs_envelope', ascending=False).head(8)
        for row in final.itertuples(index=False):
            margin_rows.append([
                row.policy,
                _fmt(row.alpha, 5),
                _fmt(row.sigma2, 5),
                _fmt(row.bias_alpha, 5),
                row.score_mode,
                _fmt(row.reaxis_spearman),
                getattr(row, 'best_label_only_method', ''),
                _fmt(getattr(row, 'best_label_only_spearman', float('nan'))),
                _fmt(row.margin_vs_envelope),
            ])
    lines = [
        '# Ordinal Gaussian Reaxis Hyperparameter Sweep',
        '',
        'This sweep keeps the AxisBayes Gaussian scalar target path: ordinal feedback values in `[0,1]` are mapped to prior-score quantiles before the Bayesian linear update. It varies only `alpha`, `sigma2`, `bias_alpha`, selection policy, and whether the posterior is ranked by the current UI-like cosine score or by the raw Gaussian linear score.',
        '',
        '## Best B0 To Final-Budget Progressions',
        '',
        _markdown_table(
            top_rows,
            ['policy', 'alpha', 'sigma2', 'bias_alpha', 'score', 'b0 rho', f'b{final_budget} rho', 'gain', 'min step', 'curve'],
        ),
        '',
        '## Best Final-Budget Margins Against Prior/Label-Only Envelope',
        '',
        _markdown_table(
            margin_rows,
            ['policy', 'alpha', 'sigma2', 'bias_alpha', 'score', f'b{final_budget} rho', 'best label-only', 'label rho', 'margin'],
        ) if margin_rows else 'No comparison envelope was available.',
        '',
        '## Interpretation Notes',
        '',
        '- Smaller `sigma2` means each feedback placement is trusted more strongly; larger `sigma2` keeps the posterior closer to the text direction.',
        '- Smaller `alpha` means weaker prior precision on direction movement; larger `alpha` makes the direction harder to move.',
        '- `bias_alpha` controls the intercept freedom in the Gaussian update. It does not directly change rank by itself, but it changes how much residual error is absorbed by the intercept versus by the direction.',
        '- If `linear` score mode wins while `cosine` does not, the update is learning a useful scalar Gaussian predictor whose magnitude is being discarded by the cosine projection used in the UI path.',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
