from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ['spearman', 'kendall_tau', 'mae_isotonic', 'qwk', 'axis_r2', 'monotonicity_violations']
BASELINE_RUN_ID = 'axisbayes_mode_v1'
LEGACY_RUN_ID = 'ordinal_v2'
PROMPT_RUN_ID = 'prompt_init_targeted_v1'
BUDGETS = (0, 1, 3, 5, 10, 20)

AXISBAYES_BASELINES = (
    'ordinal_ridge',
    'rank_svm',
    'knn_ordinal',
    'reaxis_random_gaussian',
    'reaxis_active_gaussian',
    'reaxis_centered',
    'reaxis_calibrated_residual',
)
LEGACY_BASELINES = ('text_prior', 'prompt_ladder', 'kernel_ridge')
REFERENCE_BASELINES = {'text_prior', 'prompt_ladder'}

METHOD_LABELS = {
    'text_prior': 'Text prior',
    'prompt_ladder': 'Prompt ladder',
    'ordinal_ridge': 'Ordinal ridge',
    'rank_svm': 'RankSVM',
    'knn_ordinal': 'kNN ordinal',
    'kernel_ridge': 'Kernel ridge',
    'reaxis_random_gaussian': 'Reaxis Gaussian random',
    'reaxis_active_gaussian': 'Reaxis Gaussian active',
    'reaxis_centered': 'Reaxis centered',
    'reaxis_calibrated_residual': 'Reaxis calibrated residual',
}
PROMPT_LABELS = {
    'manual_anchors': 'Prompt: manual anchors',
    'ui_fixed_template': 'Prompt: UI fixed template',
    'simple_query': 'Prompt: high/low query',
    'more_less_query': 'Prompt: more/less query',
    'evidence_query': 'Prompt: evidence query',
    'axis_name_query': 'Prompt: axis-name query',
}


def export_prompt_baseline_plot(
    *,
    diagnostics_dir: Path,
    baseline_dir: Path,
    output_dir: Path | None = None,
    baseline_run_id: str = BASELINE_RUN_ID,
    legacy_run_id: str = LEGACY_RUN_ID,
    prompt_run_id: str = PROMPT_RUN_ID,
) -> Path:
    diagnostics_dir = Path(diagnostics_dir)
    baseline_dir = Path(baseline_dir)
    output_dir = Path(output_dir or diagnostics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        [
            _load_axisbayes_baselines(diagnostics_dir, baseline_run_id),
            _load_legacy_baselines(baseline_dir, legacy_run_id),
            _load_prompt_variants(diagnostics_dir, prompt_run_id),
        ],
        ignore_index=True,
    )
    if combined.empty:
        raise RuntimeError('No rows available for combined ordinal prompt/baseline plot')
    _write_summary(combined, output_dir)
    return _plot_spearman(combined, output_dir)


def _load_axisbayes_baselines(root: Path, run_id: str) -> pd.DataFrame:
    rows = pd.read_csv(root / 'metrics.csv')
    rows = rows[
        (rows['run_id'].astype(str) == str(run_id))
        & rows['method'].astype(str).isin(AXISBAYES_BASELINES)
    ].copy()
    rows['display_method'] = rows['method'].map(METHOD_LABELS)
    rows['curve_group'] = 'baseline'
    rows['source_run_id'] = str(run_id)
    rows['source_file'] = str(root / 'metrics.csv')
    return _curve_columns(rows)


def _load_legacy_baselines(root: Path, run_id: str) -> pd.DataFrame:
    rows = pd.read_csv(root / 'metrics.csv')
    rows = rows[
        (rows['run_id'].astype(str) == str(run_id))
        & rows['method'].astype(str).isin(LEGACY_BASELINES)
    ].copy()
    rows['display_method'] = rows['method'].map(METHOD_LABELS)
    rows['curve_group'] = 'baseline'
    rows['source_run_id'] = str(run_id)
    rows['source_file'] = str(root / 'metrics.csv')
    return _curve_columns(_expand_reference_baselines(rows))


def _load_prompt_variants(root: Path, run_id: str) -> pd.DataFrame:
    rows = pd.read_csv(root / 'ordinal_prompt_initialization_sweep_results.csv')
    rows = rows[rows['run_id'].astype(str) == str(run_id)].copy()
    if rows.empty:
        raise RuntimeError(f'No prompt-sweep rows found for run_id={run_id}')
    settings = _select_prompt_settings(root, run_id)
    selected = []
    for setting in settings.itertuples(index=False):
        variant_rows = rows[
            (rows['prompt_variant'].astype(str) == str(setting.prompt_variant))
            & (pd.to_numeric(rows['lambda_value'], errors='coerce') == float(setting.lambda_value))
            & (pd.to_numeric(rows['beta'], errors='coerce') == float(setting.beta))
        ].copy()
        variant_rows['method'] = 'prompt_' + variant_rows['prompt_variant'].astype(str)
        variant_rows['display_method'] = variant_rows['prompt_variant'].map(PROMPT_LABELS)
        variant_rows['curve_group'] = 'prompt_variant'
        variant_rows['source_run_id'] = str(run_id)
        variant_rows['source_file'] = str(root / 'ordinal_prompt_initialization_sweep_results.csv')
        variant_rows['selected_lambda'] = float(setting.lambda_value)
        variant_rows['selected_beta'] = float(setting.beta)
        selected.append(variant_rows)
    if not selected:
        raise RuntimeError('Prompt setting selection returned no variants')
    return _curve_columns(pd.concat(selected, ignore_index=True))


def _select_prompt_settings(root: Path, run_id: str) -> pd.DataFrame:
    progression = pd.read_csv(root / 'ordinal_prompt_initialization_sweep_progression.csv')
    if 'run_id' in progression.columns:
        progression = progression[progression['run_id'].astype(str) == str(run_id)].copy()
    if progression.empty:
        raise RuntimeError('Prompt progression table is empty')
    progression['budget_decrease_count'] = pd.to_numeric(progression['budget_decrease_count'], errors='coerce')
    progression['delta_b20_b0_mean'] = pd.to_numeric(progression['delta_b20_b0_mean'], errors='coerce')
    progression['spearman_b20_mean'] = pd.to_numeric(progression['spearman_b20_mean'], errors='coerce')
    chosen = []
    for variant, group in progression.groupby('prompt_variant', dropna=False):
        monotone = group[group['budget_decrease_count'] == 0]
        pool = monotone if not monotone.empty else group
        best = pool.sort_values(
            ['delta_b20_b0_mean', 'spearman_b20_mean'],
            ascending=[False, False],
        ).iloc[0]
        chosen.append(best)
    return pd.DataFrame(chosen)


def _expand_reference_baselines(rows: pd.DataFrame) -> pd.DataFrame:
    expanded = []
    for row in rows.itertuples(index=False):
        if str(row.method) not in REFERENCE_BASELINES:
            expanded.append(row._asdict())
            continue
        payload = row._asdict()
        for budget in BUDGETS:
            copied = dict(payload)
            copied['budget'] = int(budget)
            expanded.append(copied)
    return pd.DataFrame(expanded)


def _curve_columns(rows: pd.DataFrame) -> pd.DataFrame:
    keep = [
        'dataset', 'axis_field', 'axis_name', 'method', 'display_method',
        'curve_group', 'source_run_id', 'source_file', 'replicate', 'budget',
        'label_count', 'selected_lambda', 'selected_beta', *METRICS,
    ]
    for col in keep:
        if col not in rows.columns:
            rows[col] = np.nan
    for col in ['replicate', 'budget', 'label_count', 'selected_lambda', 'selected_beta', *METRICS]:
        rows[col] = pd.to_numeric(rows[col], errors='coerce')
    return rows[keep].copy()


def _write_summary(rows: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        rows.groupby(['display_method', 'method', 'curve_group', 'budget'], dropna=False)[METRICS]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    grouped.columns = [
        '_'.join(str(part) for part in col if str(part))
        for col in grouped.columns.to_flat_index()
    ]
    grouped.to_csv(output_dir / 'ordinal_prompt_baseline_combined_summary.csv', index=False)

    sources = (
        rows[['display_method', 'method', 'curve_group', 'source_run_id', 'source_file', 'selected_lambda', 'selected_beta']]
        .drop_duplicates()
        .sort_values(['curve_group', 'display_method'])
    )
    sources.to_csv(output_dir / 'ordinal_prompt_baseline_combined_sources.csv', index=False)


def _plot_spearman(rows: pd.DataFrame, output_dir: Path) -> Path:
    grouped = (
        rows.dropna(subset=['budget', 'spearman'])
        .groupby(['display_method', 'curve_group', 'budget'], dropna=False)['spearman']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .sort_values(['curve_group', 'display_method', 'budget'])
    )
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    for (display_method, curve_group), group in grouped.groupby(['display_method', 'curve_group'], dropna=False):
        yerr = _sem(group['std'], group['count'])
        style = _style_for(str(display_method), str(curve_group))
        ax.errorbar(
            group['budget'],
            group['mean'],
            yerr=yerr,
            marker=style['marker'],
            linestyle=style['linestyle'],
            linewidth=style['linewidth'],
            capsize=2.2,
            elinewidth=0.8,
            alpha=style['alpha'],
            color=style['color'],
            label=str(display_method),
        )
    ax.set_xlabel('Labeled examples')
    ax.set_ylabel("Spearman's rho")
    ax.set_title('Ordinal axis quality: prompt initializations and baselines')
    ax.set_xticks(list(BUDGETS))
    ax.grid(True, alpha=0.25)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, ncol=1, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    pdf_path = output_dir / 'ordinal_prompt_baseline_spearman_curves.pdf'
    png_path = output_dir / 'ordinal_prompt_baseline_spearman_curves.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return pdf_path


def _style_for(display_method: str, curve_group: str) -> dict[str, object]:
    prompt_colors = {
        'Prompt: manual anchors': '#1f77b4',
        'Prompt: UI fixed template': '#ff7f0e',
        'Prompt: high/low query': '#2ca02c',
        'Prompt: more/less query': '#9467bd',
        'Prompt: evidence query': '#d62728',
        'Prompt: axis-name query': '#8c564b',
    }
    baseline_colors = {
        'Text prior': '#5f6368',
        'Prompt ladder': '#7f7f7f',
        'Ordinal ridge': '#17becf',
        'RankSVM': '#bcbd22',
        'kNN ordinal': '#e377c2',
        'Kernel ridge': '#aec7e8',
        'Reaxis Gaussian random': '#98df8a',
        'Reaxis Gaussian active': '#ffbb78',
        'Reaxis centered': '#c5b0d5',
        'Reaxis calibrated residual': '#c49c94',
    }
    if curve_group == 'prompt_variant':
        return {
            'color': prompt_colors.get(display_method, '#1f77b4'),
            'linestyle': '-',
            'linewidth': 2.2,
            'marker': 'o',
            'alpha': 0.95,
        }
    return {
        'color': baseline_colors.get(display_method, '#6b7280'),
        'linestyle': '--' if display_method in {'Text prior', 'Prompt ladder'} else ':',
        'linewidth': 1.5,
        'marker': 's',
        'alpha': 0.85,
    }


def _sem(std: pd.Series, count: pd.Series) -> np.ndarray:
    std_arr = pd.to_numeric(std, errors='coerce').to_numpy(dtype=float)
    count_arr = pd.to_numeric(count, errors='coerce').to_numpy(dtype=float)
    denom = np.sqrt(np.maximum(count_arr, 1.0))
    return np.nan_to_num(std_arr / denom, nan=0.0)
