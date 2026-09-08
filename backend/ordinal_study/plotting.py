from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_SPECS = [
    ('spearman', "Spearman's rho"),
    ('kendall_tau', "Kendall's tau"),
    ('mae_isotonic', 'Isotonic MAE'),
    ('qwk', 'Quadratic weighted kappa'),
    ('axis_r2', 'Axis-gradient R2'),
    ('monotonicity_violations', 'Monotonicity violations'),
]
RANK_REAXIS_METHODS = {'reaxis_random_rank', 'reaxis_active_rank'}


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def export_reports(output_dir: Path, run_id: str | None = None) -> None:
    root = Path(output_dir)
    metrics_path = root / 'metrics.csv'
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return
    if run_id is not None and 'run_id' in metrics.columns:
        metrics = metrics[metrics['run_id'].astype(str) == str(run_id)].copy()
        if metrics.empty:
            return
    value_cols = ['budget', 'replicate', 'label_count', 'spearman', 'kendall_tau', 'mae_isotonic', 'qwk', 'axis_r2', 'monotonicity_violations']
    metrics = _numeric(metrics, value_cols)
    summary = (
        metrics.groupby(['dataset', 'axis_field', 'axis_name', 'method', 'budget'], dropna=False)
        [['spearman', 'kendall_tau', 'mae_isotonic', 'qwk', 'axis_r2', 'monotonicity_violations']]
        .agg(['mean', 'std', 'count'])
    )
    summary.columns = ['_'.join(str(part) for part in col if part) for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(root / 'summary_metrics.csv', index=False)

    final_budget = int(metrics['budget'].max())
    main_table = (
        metrics[metrics['budget'] == final_budget]
        .groupby(['method'], dropna=False)
        [['spearman', 'kendall_tau', 'mae_isotonic', 'qwk', 'axis_r2', 'monotonicity_violations']]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values('spearman', ascending=False)
    )
    main_table.to_csv(root / 'main_table.csv', index=False)
    plot_metrics = _filter_plot_methods(metrics)
    for metric, ylabel in METRIC_SPECS:
        _plot_metric_curves(plot_metrics, root, metric=metric, ylabel=ylabel)
    _plot_all_metric_curves(plot_metrics, root)
    _plot_dataset_metric_curves(plot_metrics, root)
    _export_diagnostic_reports(root, run_id=run_id)


def _filter_plot_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    if 'method' not in metrics.columns:
        return metrics.copy()
    mask = ~metrics['method'].astype(str).isin(RANK_REAXIS_METHODS)
    return metrics[mask].copy()


def _aggregate_metric_curve(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        metrics.dropna(subset=['budget', metric])
        .groupby(['method', 'budget'], dropna=False)[metric]
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .sort_values(['method', 'budget'])
    )


def _plot_metric_curves(
    metrics: pd.DataFrame,
    root: Path,
    *,
    metric: str,
    ylabel: str,
    filename: str | None = None,
    title: str | None = None,
) -> None:
    if metric not in metrics.columns:
        return
    grouped = _aggregate_metric_curve(metrics, metric)
    if grouped.empty:
        return
    root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for method, grp in grouped.groupby('method', dropna=False):
        yerr = _sem(grp['std'], grp['count'])
        ax.errorbar(
            grp['budget'],
            grp['mean'],
            yerr=yerr,
            marker='o',
            linewidth=1.8,
            capsize=2.5,
            elinewidth=0.9,
            label=str(method),
        )
    ax.set_xlabel('Labeled examples')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(root / (filename or f'{metric}_curves.pdf'))
    plt.close(fig)


def _plot_all_metric_curves(
    metrics: pd.DataFrame,
    root: Path,
    *,
    filename: str = 'all_metrics_curves.pdf',
    title: str | None = None,
) -> None:
    available = [(metric, ylabel) for metric, ylabel in METRIC_SPECS if metric in metrics.columns]
    if not available:
        return
    root.mkdir(parents=True, exist_ok=True)
    n_cols = 3
    n_rows = int(np.ceil(len(available) / n_cols))
    top_pad = 0.94 if title else 1.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.0, 3.4 * n_rows), squeeze=False)
    handles = []
    labels = []
    for ax, (metric, ylabel) in zip(axes.flat, available):
        grouped = _aggregate_metric_curve(metrics, metric)
        for method, grp in grouped.groupby('method', dropna=False):
            container = ax.errorbar(
                grp['budget'],
                grp['mean'],
                yerr=_sem(grp['std'], grp['count']),
                marker='o',
                linewidth=1.5,
                capsize=2.0,
                elinewidth=0.8,
                label=str(method),
            )
            if str(method) not in labels:
                handles.append(container)
                labels.append(str(method))
        ax.set_title(ylabel)
        ax.set_xlabel('Labeled examples')
        ax.grid(True, alpha=0.25)
    for ax in axes.flat[len(available):]:
        ax.axis('off')
    if title:
        fig.suptitle(title, y=0.995, fontsize=12)
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, top_pad))
    fig.savefig(root / filename)
    plt.close(fig)


def _plot_dataset_metric_curves(metrics: pd.DataFrame, root: Path) -> None:
    if 'dataset' not in metrics.columns:
        return
    by_dataset = root / 'figures_by_dataset'
    for dataset, dataset_metrics in metrics.groupby('dataset', dropna=False):
        if dataset_metrics.empty:
            continue
        dataset_name = str(dataset)
        slug = _safe_slug(dataset_name)
        _plot_all_metric_curves(
            dataset_metrics,
            by_dataset,
            filename=f'{slug}_all_metrics_curves.pdf',
            title=dataset_name,
        )
        for metric, ylabel in METRIC_SPECS:
            _plot_metric_curves(
                dataset_metrics,
                by_dataset,
                metric=metric,
                ylabel=ylabel,
                filename=f'{slug}_{metric}_curves.pdf',
                title=dataset_name,
            )


def _safe_slug(value: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value).strip())
    slug = re.sub(r'_+', '_', slug).strip('._-')
    if not slug:
        raise ValueError('Cannot create figure slug from empty dataset name')
    return slug


def _sem(std: pd.Series, count: pd.Series) -> np.ndarray:
    std_arr = pd.to_numeric(std, errors='coerce').to_numpy(dtype=float)
    count_arr = pd.to_numeric(count, errors='coerce').to_numpy(dtype=float)
    denom = np.sqrt(np.maximum(count_arr, 1.0))
    sem = std_arr / denom
    return np.nan_to_num(sem, nan=0.0)


def _export_diagnostic_reports(root: Path, run_id: str | None = None) -> None:
    diagnostics_path = root / 'diagnostics.csv'
    if not diagnostics_path.exists() or diagnostics_path.stat().st_size == 0:
        return
    diagnostics = pd.read_csv(diagnostics_path)
    if diagnostics.empty:
        return
    if run_id is not None and 'run_id' in diagnostics.columns:
        diagnostics = diagnostics[diagnostics['run_id'].astype(str) == str(run_id)].copy()
        if diagnostics.empty:
            return

    id_cols = {
        'ts_utc', 'run_id', 'dataset', 'axis_field', 'axis_name', 'method',
        'policy',
    }
    numeric_cols = [col for col in diagnostics.columns if col not in id_cols]
    diagnostics = _numeric(diagnostics, numeric_cols)
    metric_cols = [
        col for col in numeric_cols
        if col not in {'replicate', 'budget', 'label_count'}
    ]
    summary = (
        diagnostics.groupby(['method', 'budget'], dropna=False)[metric_cols]
        .agg(['mean', 'std', 'count'])
    )
    summary.columns = ['_'.join(str(part) for part in col if part) for col in summary.columns]
    summary.reset_index().to_csv(root / 'diagnostics_summary.csv', index=False)

    bin_cols = [f'selected_y_bin_{idx}' for idx in range(5) if f'selected_y_bin_{idx}' in diagnostics.columns]
    audit_cols = [
        'labeled_y_mean', 'labeled_y_std', 'labeled_y_min', 'labeled_y_max',
        'uncertainty_error_corr', 'uncertainty_error_auroc',
    ]
    available_audit = [col for col in audit_cols if col in diagnostics.columns]
    grouped = diagnostics.groupby(['method', 'budget'], dropna=False)
    audit = grouped[available_audit].mean(numeric_only=True).reset_index() if available_audit else grouped.size().reset_index(name='count')
    if bin_cols:
        bins = grouped[bin_cols].sum(numeric_only=True).reset_index()
        audit = audit.merge(bins, on=['method', 'budget'], how='left')
    audit.to_csv(root / 'active_sample_audit.csv', index=False)


def export_fixed_reports(output_dir: Path, run_id: str | None = None) -> str:
    root = Path(output_dir)
    results_path = root / 'ordinal_eval_results_fixed.csv'
    if not results_path.exists() or results_path.stat().st_size == 0:
        return ''
    results = pd.read_csv(results_path)
    if results.empty:
        return ''
    if run_id is not None and 'run_id' in results.columns:
        results = results[results['run_id'].astype(str) == str(run_id)].copy()
        if results.empty:
            return ''

    metric_cols = [
        'spearman', 'kendall_tau', 'mae_isotonic', 'qwk',
        'axis_r2', 'monotonicity_violations',
    ]
    diag_cols = [
        'selected_lambda', 'selected_beta', 'score_preservation_spearman',
        'mu_d0_cos', 'direction_relative_change', 'labeled_pairwise_agreement',
        'labeled_mae', 'prior_spearman_b0',
    ]
    numeric_cols = ['budget', 'replicate', 'label_count', *metric_cols, *diag_cols]
    results = _numeric(results, numeric_cols)
    summary = (
        results.groupby(['method', 'budget'], dropna=False)[metric_cols + diag_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    counts = results.groupby(['method', 'budget'], dropna=False).size().reset_index(name='count')
    summary = summary.merge(counts, on=['method', 'budget'], how='left')
    summary = summary.sort_values(['budget', 'method']).reset_index(drop=True)
    summary.to_csv(root / 'ordinal_eval_summary_fixed.csv', index=False)

    markdown_table = _fixed_markdown_table(summary, metric_cols)
    report = _fixed_markdown_report(results, summary, markdown_table, metric_cols)
    (root / 'diagnostic_metrics_digest.md').write_text(report, encoding='utf-8')
    return markdown_table


def _fmt(value: object, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return ''
    if pd.isna(val):
        return ''
    return f'{val:.{digits}f}'


def _fixed_markdown_table(summary: pd.DataFrame, metric_cols: list[str]) -> str:
    headers = ['method', 'budget', *metric_cols]
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in summary.itertuples(index=False):
        payload = [str(getattr(row, 'method')), str(int(getattr(row, 'budget')))]
        payload.extend(_fmt(getattr(row, col)) for col in metric_cols)
        lines.append('| ' + ' | '.join(payload) + ' |')
    return '\n'.join(lines)


def _fixed_markdown_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    markdown_table: str,
    metric_cols: list[str],
) -> str:
    final_budget = int(summary['budget'].max()) if not summary.empty else 0
    final = summary[summary['budget'] == final_budget].copy()
    old = final[final['method'].isin(['reaxis_random', 'reaxis_active'])]
    new = final[final['method'].isin(['reaxis_centered', 'reaxis_calibrated_residual'])]
    label_only = final[final['method'].isin(['ordinal_ridge', 'rank_svm', 'knn_ordinal', 'kernel_ridge'])]

    lines = [
        '# Ordinal Reaxis Diagnostics Digest',
        '',
        '## Aggregate Metric Table',
        '',
        markdown_table,
        '',
        '## Interpretation',
        '',
    ]
    if not old.empty and not new.empty:
        old_best = old.sort_values('spearman', ascending=False).iloc[0]
        new_best = new.sort_values('spearman', ascending=False).iloc[0]
        lines.append(
            f'- At budget {final_budget}, the best fixed Reaxis variant is '
            f'`{new_best.method}` with Spearman {_fmt(new_best.spearman)}, compared with '
            f'the best existing Reaxis variant `{old_best.method}` at {_fmt(old_best.spearman)}.'
        )
    if not new.empty and not label_only.empty:
        label_best = label_only.sort_values('spearman', ascending=False).iloc[0]
        new_best = new.sort_values('spearman', ascending=False).iloc[0]
        verdict = 'matches or beats' if float(new_best.spearman) >= float(label_best.spearman) else 'does not match'
        lines.append(
            f'- At budget {final_budget}, `{new_best.method}` {verdict} the best label-only baseline '
            f'(`{label_best.method}` Spearman {_fmt(label_best.spearman)}).'
        )

    for method in ['reaxis_centered', 'reaxis_calibrated_residual']:
        rows = summary[(summary['method'] == method) & (summary['budget'] > 0)]
        if rows.empty:
            continue
        lambdas = ', '.join(
            f"b={int(row.budget)}: {_fmt(row.selected_lambda, 2)}"
            for row in rows.itertuples(index=False)
            if not pd.isna(row.selected_lambda)
        )
        betas = ', '.join(
            f"b={int(row.budget)}: {_fmt(row.selected_beta, 2)}"
            for row in rows.itertuples(index=False)
            if not pd.isna(row.selected_beta)
        )
        lines.append(f'- Selected lambda for `{method}` by budget: {lambdas or "none"}.')
        if betas:
            lines.append(f'- Selected beta for `{method}` by budget: {betas}.')

    lines.extend(['', '## Failure Cases At Final Budget', ''])
    task_rows = (
        results[results['budget'] == final_budget]
        .groupby(['dataset', 'axis_field', 'method'], dropna=False)['spearman']
        .mean()
        .reset_index()
    )
    if task_rows.empty:
        lines.append('- No final-budget task rows were available.')
    else:
        pivot = task_rows.pivot_table(index=['dataset', 'axis_field'], columns='method', values='spearman')
        interesting = [col for col in ['reaxis_centered', 'reaxis_calibrated_residual', 'ordinal_ridge', 'rank_svm', 'knn_ordinal'] if col in pivot.columns]
        for idx, row in pivot.iterrows():
            values = ', '.join(f'{col}={_fmt(row[col])}' for col in interesting if col in row and not pd.isna(row[col]))
            lines.append(f'- `{idx[0]} / {idx[1]}`: {values}.')

    lines.extend(['', '## Diagnostics Columns', ''])
    lines.append(
        '- `score_preservation_spearman` measures how much the final score preserves the initial text-prior ordering. '
        '`mu_d0_cos` and `direction_relative_change` measure direction drift. '
        '`labeled_pairwise_agreement` and `labeled_mae` are computed only on revealed labels. '
        '`selected_lambda` and `selected_beta` are selected only from labeled examples.'
    )
    _ = metric_cols
    return '\n'.join(lines) + '\n'
