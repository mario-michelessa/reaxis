from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = (
    'spearman',
    'kendall_tau',
    'mae_isotonic',
    'qwk',
    'axis_r2',
    'monotonicity_violations',
)


def _aggregate_results(output_dir: Path, run_id: str) -> pd.DataFrame:
    results_path = output_dir / 'low_level_ordinal_results.csv'
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    results = pd.read_csv(results_path)
    results = results[results['run_id'].astype(str) == str(run_id)].copy()
    if results.empty:
        raise RuntimeError(f'No low-level ordinal result rows for run_id={run_id}')

    rows: list[dict[str, object]] = []
    for (method, budget), frame in results.groupby(['method', 'budget'], dropna=False):
        row: dict[str, object] = {'run_id': run_id, 'method': method, 'budget': int(budget), 'n': int(len(frame))}
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(frame[metric], errors='coerce')
            row[f'{metric}_mean'] = float(values.mean())
            row[f'{metric}_sem'] = float(values.sem()) if int(values.count()) > 1 else float('nan')
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(['method', 'budget']).reset_index(drop=True)
    summary.to_csv(output_dir / 'low_level_ordinal_summary.csv', index=False)

    by_axis = (
        results
        .groupby(['dataset', 'axis_field', 'axis_name', 'method', 'budget'], dropna=False)[list(METRIC_COLUMNS)]
        .mean()
        .reset_index()
        .sort_values(['dataset', 'axis_field', 'method', 'budget'])
    )
    by_axis.to_csv(output_dir / 'low_level_ordinal_by_axis_summary.csv', index=False)
    return summary


def markdown_metric_table(summary: pd.DataFrame) -> str:
    display = summary.copy()
    display = display[['method', 'budget', *[f'{metric}_mean' for metric in METRIC_COLUMNS]]]
    display.columns = ['method', 'budget', *METRIC_COLUMNS]
    for metric in METRIC_COLUMNS:
        display[metric] = display[metric].map(lambda value: '' if pd.isna(value) else f'{float(value):.3f}')
    return display.to_markdown(index=False)


def _axis_outcome_table(results: pd.DataFrame, final_budget: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = results.groupby(['dataset', 'axis_field', 'axis_name'], dropna=False)
    for (dataset, _, axis_name), frame in grouped:
        text0 = frame[(frame['method'] == 'text_prior') & (frame['budget'] == 0)]['spearman'].mean()
        centered20 = frame[(frame['method'] == 'reaxis_centered') & (frame['budget'] == final_budget)]['spearman'].mean()
        residual20 = frame[
            (frame['method'] == 'reaxis_calibrated_residual') & (frame['budget'] == final_budget)
        ]['spearman'].mean()
        best = (
            frame[frame['budget'] == final_budget]
            .groupby('method')['spearman']
            .mean()
            .sort_values(ascending=False)
        )
        rows.append({
            'dataset': dataset,
            'axis': axis_name,
            'text_b0': text0,
            'reaxis_centered_b20': centered20,
            'reaxis_residual_b20': residual20,
            'best_b20_method': best.index[0] if not best.empty else '',
            'best_b20_spearman': float(best.iloc[0]) if not best.empty else float('nan'),
        })
    return pd.DataFrame(rows).sort_values(['dataset', 'axis']).reset_index(drop=True)


def _parameter_table(results: pd.DataFrame, final_budget: int) -> pd.DataFrame:
    bfinal = results[results['budget'] == final_budget]
    return (
        bfinal[bfinal['method'].isin(['reaxis_centered', 'reaxis_calibrated_residual'])]
        .groupby('method', dropna=False)[
            ['selected_lambda', 'selected_beta', 'score_preservation_spearman', 'prior_spearman_b0']
        ]
        .mean()
        .reset_index()
    )


def _append_interpretation(lines: list[str], results: pd.DataFrame, axis_table: pd.DataFrame, param_table: pd.DataFrame) -> None:
    final_budget = int(results['budget'].max())
    b0 = results[results['budget'] == 0]
    bfinal = results[results['budget'] == final_budget]
    best_final = (
        bfinal.groupby('method')['spearman']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    if not best_final.empty:
        top = best_final.iloc[0]
        lines.append(f'At budget `{final_budget}`, the highest aggregate Spearman is `{float(top.spearman):.3f}` from `{top.method}`.')
    text_prior_b0 = b0[b0['method'] == 'text_prior']['spearman'].mean()
    for method in ('reaxis_centered', 'reaxis_calibrated_residual'):
        method_b0 = b0[b0['method'] == method]['spearman'].mean()
        method_final = bfinal[bfinal['method'] == method]['spearman'].mean()
        if np.isfinite(method_b0) and np.isfinite(method_final):
            lines.append(f'`{method}` changes from Spearman `{method_b0:.3f}` at b=0 to `{method_final:.3f}` at b={final_budget}.')
    if np.isfinite(text_prior_b0):
        lines.append(f'The zero-label text initialization has aggregate Spearman `{text_prior_b0:.3f}`.')
    _append_parameter_interpretation(lines, param_table, final_budget)
    if not axis_table.empty:
        strongest = axis_table.sort_values('best_b20_spearman', ascending=False).iloc[0]
        weakest = axis_table.sort_values('best_b20_spearman', ascending=True).iloc[0]
        lines.append(
            f'The strongest low-level targets are `{strongest.axis}` (`{strongest.best_b20_method}`, Spearman `{float(strongest.best_b20_spearman):.3f}`) and the weakest is `{weakest.axis}` (`{weakest.best_b20_method}`, Spearman `{float(weakest.best_b20_spearman):.3f}`).'
        )


def _append_parameter_interpretation(lines: list[str], param_table: pd.DataFrame, final_budget: int) -> None:
    if param_table.empty:
        return
    centered_row = param_table[param_table['method'] == 'reaxis_centered']
    residual_row = param_table[param_table['method'] == 'reaxis_calibrated_residual']
    if not centered_row.empty:
        lam = float(centered_row.iloc[0]['selected_lambda'])
        preservation = float(centered_row.iloc[0]['score_preservation_spearman'])
        lines.append(
            f'For `reaxis_centered`, the mean selected lambda at b={final_budget} is `{lam:.3f}` and score-preservation Spearman is `{preservation:.3f}`, so the update is conservative and mostly preserves the text-prior ordering.'
        )
    if not residual_row.empty:
        lam = float(residual_row.iloc[0]['selected_lambda'])
        beta = float(residual_row.iloc[0]['selected_beta'])
        preservation = float(residual_row.iloc[0]['score_preservation_spearman'])
        lines.append(
            f'For `reaxis_calibrated_residual`, the mean selected lambda is `{lam:.3f}`, mean beta is `{beta:.3f}`, and score-preservation Spearman is `{preservation:.3f}`, indicating a moderate residual correction blended with the prior.'
        )


def _write_report(output_dir: Path, run_id: str, summary: pd.DataFrame) -> Path:
    stats = pd.read_csv(output_dir / 'low_level_feature_stats.csv')
    stats = stats[stats['run_id'].astype(str) == str(run_id)].copy()
    results = pd.read_csv(output_dir / 'low_level_ordinal_results.csv')
    results = results[results['run_id'].astype(str) == str(run_id)].copy()
    final_budget = int(results['budget'].max())
    axis_table = _axis_outcome_table(results, final_budget)
    param_table = _parameter_table(results, final_budget)

    lines = [
        '# Low-Level Ordinal Feature Study',
        '',
        f'Run id: `{run_id}`.',
        '',
        'This study adds deterministic visual-processing labels to prepared datasets and evaluates whether CLIP-text axes and sparse-feedback axis updates recover those labels as ordinal targets.',
        '',
        '## Ground Truths',
        '',
        stats[['dataset', 'axis_name', 'label_source', 'raw_min', 'raw_median', 'raw_max', 'raw_std']]
        .to_markdown(index=False, floatfmt='.4f'),
        '',
        'All targets are min-max normalized within each dataset and feature before fitting or scoring. Rank metrics are therefore determined by the raw visual measurement order.',
        '',
        '## Methods',
        '',
        '- `text_prior`: the zero-feedback CLIP contrastive direction from the feature low/high prompts.',
        '- `prompt_ladder`: zero-shot ordered text prompts from very low to very high feature strength.',
        '- `ordinal_ridge`: ridge regression from the sampled labels only.',
        '- `rank_svm`: pairwise linear ranking constraints from the sampled labels only.',
        '- `knn_ordinal`: nearest-neighbor label propagation in CLIP embedding space.',
        '- `kernel_ridge`: nonlinear RBF kernel ridge regression from sampled labels only.',
        '- `reaxis_centered`: centered Bayesian linear update initialized from the text prior.',
        '- `reaxis_calibrated_residual`: affine-calibrated text prior plus conservative residual update.',
        '',
        'Rank-mode Reaxis is not included.',
        '',
        '## Aggregate Metrics',
        '',
        markdown_metric_table(summary),
        '',
        '## Per-Feature Outcome',
        '',
        axis_table.to_markdown(index=False, floatfmt='.3f'),
        '',
        '## Fixed-Reaxis Parameters at Final Budget',
        '',
        param_table.to_markdown(index=False, floatfmt='.3f'),
        '',
        '## Interpretation',
        '',
    ]
    _append_interpretation(lines, results, axis_table, param_table)
    lines.append('')
    lines.append('Per-axis summaries are written to `low_level_ordinal_by_axis_summary.csv`.')

    report_path = output_dir / 'low_level_ordinal_report.md'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def export_low_level_reports(output_dir: Path, run_id: str) -> tuple[pd.DataFrame, Path, str]:
    summary = _aggregate_results(Path(output_dir), run_id)
    report_path = _write_report(Path(output_dir), run_id, summary)
    return summary, report_path, markdown_metric_table(summary)
