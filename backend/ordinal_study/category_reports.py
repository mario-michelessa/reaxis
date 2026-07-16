from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATASET_SPECS, REAXIS_AXISBAYES_PARAMS
from .plotting import METRIC_SPECS, _plot_all_metric_curves, _plot_metric_curves, _safe_slug


METRIC_COLS = [metric for metric, _ in METRIC_SPECS]
BASELINE_DESCRIPTIONS = {
    'text_prior': 'CLIP text direction from the low/high anchor prompts, with no feedback.',
    'prompt_ladder': 'Zero-shot ordinal VLM baseline using five ordered text prompts and an expected level score.',
    'label_mean': 'Constant baseline that predicts the mean of the revealed ordinal labels for every image.',
    'ols_linear': 'Ordinary least-squares linear regression from revealed ordinal labels only.',
    'bayesian_ridge': 'Bayesian ridge linear regression from revealed ordinal labels only, with automatic shrinkage.',
    'elastic_net': 'Sparse pointwise linear regression from revealed ordinal labels only.',
    'pcr_ridge': 'Principal-component ridge regression: unsupervised PCA of CLIP features followed by ridge on revealed labels.',
    'ordinal_ridge': 'Linear ridge regression from the revealed ordinal labels only.',
    'prior_affine': 'Affine calibration of the zero-label text-prior score using the revealed ordinal labels.',
    'rank_svm': 'Pairwise ranking SVM built from label order comparisons among revealed examples.',
    'linear_svr': 'Pointwise linear support-vector regression from the revealed ordinal labels only.',
    'knn_ordinal': 'Nearest-neighbor ordinal label propagation in CLIP embedding space.',
    'kernel_ridge': 'RBF kernel ridge regression from the revealed ordinal labels only.',
    'reaxis_random': 'Gaussian AxisBayes Reaxis update with random feedback examples.',
    'reaxis_active': 'Gaussian AxisBayes Reaxis update with uncertainty-selected feedback examples.',
    'reaxis_quantile_gaussian': 'Explicit UI-like Gaussian AxisBayes Reaxis update using ordinal values as placement quantiles.',
    'reaxis_log': 'Gaussian AxisBayes Reaxis update with random feedback after log-normalizing the raw ordinal target.',
    'reaxis_pairwise': 'Prior-regularized pairwise Reaxis ablation using the text direction as the prior mean and labeled pair order as supervision.',
}


def _category_lookup() -> pd.DataFrame:
    rows = []
    for dataset, spec in DATASET_SPECS.items():
        for axis in spec.axes:
            rows.append({
                'dataset': dataset,
                'axis_field': axis.field,
                'axis_name_config': axis.name,
                'axis_category': axis.category,
                'domain_config': spec.domain,
                'label_source_config': axis.label_source,
            })
    return pd.DataFrame(rows)


def _read_metrics(output_dir: Path, run_id: str | None) -> pd.DataFrame:
    path = Path(output_dir) / 'metrics.csv'
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    metrics = pd.read_csv(path)
    if run_id is not None and 'run_id' in metrics.columns:
        metrics = metrics[metrics['run_id'].astype(str) == str(run_id)].copy()
    if metrics.empty:
        return metrics
    for col in ['budget', 'replicate', 'label_count', *METRIC_COLS]:
        if col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors='coerce')
    lookup = _category_lookup()
    metrics = metrics.merge(lookup, on=['dataset', 'axis_field'], how='left')
    metrics['axis_category'] = metrics['axis_category'].fillna('uncategorized')
    return metrics


def export_category_reports(output_dir: Path, run_id: str | None = None) -> Path | None:
    root = Path(output_dir)
    metrics = _read_metrics(root, run_id)
    if metrics.empty:
        return None

    summary = (
        metrics.groupby(['axis_category', 'method', 'budget'], dropna=False)[METRIC_COLS]
        .agg(['mean', 'std', 'count'])
    )
    summary.columns = ['_'.join(str(part) for part in col if part) for col in summary.columns]
    summary = summary.reset_index().sort_values(['axis_category', 'budget', 'method'])
    summary.to_csv(root / 'category_summary_metrics.csv', index=False)

    by_task = (
        metrics.groupby(['axis_category', 'dataset', 'axis_field', 'axis_name', 'method', 'budget'], dropna=False)[METRIC_COLS]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(['axis_category', 'dataset', 'axis_field', 'budget', 'method'])
    )
    by_task.to_csv(root / 'category_task_summary.csv', index=False)

    figure_dir = root / 'figures_by_category'
    for category, frame in metrics.groupby('axis_category', dropna=False):
        if frame.empty:
            continue
        slug = _safe_slug(str(category))
        _plot_all_metric_curves(
            frame,
            figure_dir,
            filename=f'{slug}_all_metrics_curves.pdf',
            title=f'{category} axes',
        )
        for metric, ylabel in METRIC_SPECS:
            _plot_metric_curves(
                frame,
                figure_dir,
                metric=metric,
                ylabel=ylabel,
                filename=f'{slug}_{metric}_curves.pdf',
                title=f'{category} axes',
            )

    report_path = root / 'ordinal_modeling_expanded_report.md'
    report_path.write_text(_build_markdown_report(metrics, summary), encoding='utf-8')
    return report_path


def _fmt(value: object, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ''
    if pd.isna(val):
        return ''
    return f'{val:.{digits}f}'


def _table(rows: list[list[object]], headers: list[str]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(str(item) for item in row) + ' |')
    return '\n'.join(out)


def _axis_inventory(metrics: pd.DataFrame) -> str:
    rows = []
    tasks = metrics[['dataset', 'domain_config', 'axis_category', 'axis_name', 'label_source_config']].drop_duplicates()
    for dataset, group in tasks.sort_values(['dataset', 'axis_category', 'axis_name']).groupby('dataset'):
        categories = []
        for category, cgroup in group.groupby('axis_category'):
            categories.append(f'{category}: ' + ', '.join(cgroup['axis_name'].astype(str).tolist()))
        rows.append([dataset, group['domain_config'].dropna().iloc[0] if group['domain_config'].notna().any() else '', '; '.join(categories)])
    return _table(rows, ['dataset', 'domain', 'axes'])


def _category_result_table(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return ''
    final_budget = int(metrics['budget'].max())
    rows = []
    for category, group in metrics.groupby('axis_category'):
        b0 = group[group['budget'] == 0].groupby('method')['spearman'].mean(numeric_only=True)
        bf = group[group['budget'] == final_budget].groupby('method')['spearman'].mean(numeric_only=True)
        best = bf.sort_values(ascending=False)
        reaxis = bf[[m for m in ['reaxis_random', 'reaxis_active'] if m in bf.index]]
        best_reaxis = reaxis.sort_values(ascending=False).head(1)
        rows.append([
            category,
            _fmt(b0.max() if not b0.empty else float('nan')),
            best.index[0] if not best.empty else '',
            _fmt(best.iloc[0] if not best.empty else float('nan')),
            best_reaxis.index[0] if not best_reaxis.empty else '',
            _fmt(best_reaxis.iloc[0] if not best_reaxis.empty else float('nan')),
        ])
    return _table(rows, ['axis category', 'best b=0 rho', f'best b={final_budget} method', f'best b={final_budget} rho', 'best Reaxis', 'best Reaxis rho'])


def _global_progression_table(metrics: pd.DataFrame) -> str:
    final_budget = int(metrics['budget'].max())
    rows = []
    for method in BASELINE_DESCRIPTIONS:
        group = metrics[metrics['method'] == method]
        if group.empty:
            continue
        b0 = group[group['budget'] == 0]['spearman'].mean()
        bf = group[group['budget'] == final_budget]
        delta = float('nan')
        if not bf.empty and pd.notna(b0):
            delta = bf['spearman'].mean() - b0
        rows.append([
            method,
            _fmt(b0),
            _fmt(bf['spearman'].mean() if not bf.empty else float('nan')),
            _fmt(delta),
            _fmt(bf['kendall_tau'].mean() if not bf.empty else float('nan')),
            _fmt(bf['qwk'].mean() if not bf.empty else float('nan')),
            _fmt(bf['axis_r2'].mean() if not bf.empty else float('nan')),
            _fmt(bf['monotonicity_violations'].mean() if not bf.empty else float('nan')),
        ])
    return _table(
        rows,
        ['method', 'b=0 rho', f'b={final_budget} rho', 'rho gain', f'b={final_budget} tau', f'b={final_budget} QWK', f'b={final_budget} axis R2', f'b={final_budget} violations'],
    )


def _dataset_result_table(metrics: pd.DataFrame) -> str:
    final_budget = int(metrics['budget'].max())
    rows = []
    for dataset, group in metrics.groupby('dataset'):
        prior = group[(group['method'] == 'text_prior') & (group['budget'] == 0)]['spearman'].mean()
        final = group[group['budget'] == final_budget].groupby('method')['spearman'].mean(numeric_only=True)
        best = final.sort_values(ascending=False)
        rows.append([
            dataset,
            _fmt(prior),
            best.index[0] if not best.empty else '',
            _fmt(best.iloc[0] if not best.empty else float('nan')),
            _fmt(final.get('reaxis_random', float('nan'))),
            _fmt(final.get('reaxis_active', float('nan'))),
        ])
    return _table(rows, ['dataset', 'b=0 text rho', f'best b={final_budget} method', f'best b={final_budget} rho', 'reaxis_random rho', 'reaxis_active rho'])


def _failure_case_table(metrics: pd.DataFrame) -> str:
    final_budget = int(metrics['budget'].max())
    final = metrics[metrics['budget'] == final_budget]
    rows = []
    for (category, dataset, axis_field, axis_name), group in final.groupby(['axis_category', 'dataset', 'axis_field', 'axis_name']):
        by_method = group.groupby('method')['spearman'].mean(numeric_only=True).sort_values(ascending=False)
        if by_method.empty or by_method.iloc[0] >= 0.10:
            continue
        rows.append([
            category,
            dataset,
            axis_name,
            by_method.index[0],
            _fmt(by_method.iloc[0]),
            _fmt(by_method.get('reaxis_random', float('nan'))),
            _fmt(by_method.get('reaxis_active', float('nan'))),
        ])
    if not rows:
        return 'No task had best-method Spearman below 0.10 at the final budget.'
    return _table(rows, ['category', 'dataset', 'axis', 'best method', f'best b={final_budget} rho', 'reaxis_random rho', 'reaxis_active rho'])


def _build_markdown_report(metrics: pd.DataFrame, summary: pd.DataFrame) -> str:
    final_budget = int(metrics['budget'].max()) if not metrics.empty else 0
    methods = [m for m in BASELINE_DESCRIPTIONS if m in set(metrics['method'].astype(str))]
    method_rows = [[method, BASELINE_DESCRIPTIONS[method]] for method in methods]
    lines = [
        '# Expanded Ordinal Modeling Study',
        '',
        'This study evaluates whether image-axis methods recover ordered visual targets rather than only binary separations. The expanded dataset set adds facial attractiveness, dermoscopy geometry, diabetic-retinopathy severity, mammography assessment, photographic aesthetics, memorability, emotion ratings, and house-price targets.',
        '',
        '## Dataset And Axis Inventory',
        '',
        _axis_inventory(metrics),
        '',
        'Axes are grouped into `low_level` visual measurements, `medium_level` geometry or image-composition attributes, `high_level` clinical or affective labels, and `abstract` judgments such as aesthetics, memorability, attractiveness, or price.',
        '',
        '## Baselines',
        '',
        _table(method_rows, ['method', 'definition']),
        '',
        'All methods use the same fixed CLIP image embeddings and the same simulated feedback indices per task, method policy, replicate, and budget. The Reaxis variants here use Gaussian AxisBayes scalar feedback; Reaxis rank-mode, centered Reaxis, and calibrated-residual Reaxis are intentionally excluded from this expanded report.',
        '',
        '## Parameters',
        '',
        f'Gaussian Reaxis uses the fixed AxisBayes settings `alpha={_fmt(REAXIS_AXISBAYES_PARAMS["alpha"], 4)}`, `sigma2={_fmt(REAXIS_AXISBAYES_PARAMS["sigma2"], 4)}`, and `bias_alpha={_fmt(REAXIS_AXISBAYES_PARAMS["bias_alpha"], 2)}` from the prior ordinal diagnostics. The earlier sweeps are used only to keep the Gaussian update stable; this run does not tune hyperparameters on the evaluation labels.',
        '',
        '## Aggregate Results By Axis Category',
        '',
        _category_result_table(metrics),
        '',
        f'## Global Results From b=0 To b={final_budget}',
        '',
        _global_progression_table(metrics),
        '',
        f'## Dataset-Level Results At b={final_budget}',
        '',
        _dataset_result_table(metrics),
        '',
        f'## Failure Cases At b={final_budget}',
        '',
        _failure_case_table(metrics),
        '',
        'Full category summaries are in `category_summary_metrics.csv`, per-task summaries are in `category_task_summary.csv`, and category-specific metric curves are in `figures_by_category/`.',
        '',
        '## Interpretation',
        '',
        f'The main comparison is the progression from zero labels to budget {final_budget}. A useful ordinal update should improve rank metrics over the text-prior initialization while keeping monotonicity violations low. Label-only baselines test whether sparse ordinal labels alone are enough; Reaxis tests whether sparse labels can refine a text-initialized axis.',
        '',
        'For low-level axes, improvements are expected when CLIP embeddings preserve basic color, luminance, edge, and texture cues. Medium-level axes test whether geometry and image-composition metadata are visible in the image. High-level axes are harder because clinical or affective labels may depend on subtle domain-specific cues. Abstract axes are subjective or market-derived, so they often have a useful prior but noisier feedback gains.',
        '',
    ]
    _ = summary
    return '\n'.join(lines) + '\n'
