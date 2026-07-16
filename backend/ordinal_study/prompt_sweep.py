from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from .baselines import ClipTextEmbedder
from .centered_sweep import (
    METRIC_COLS,
    SWEEP_BETAS,
    SWEEP_LAMBDAS,
    _centered_scores,
    _random_feedback_indices,
    _stable_seed,
)
from .config import ACTIVE_DATASETS, BUDGETS, GLOBAL_SEED, OUTPUT_DIR, REPETITIONS
from .data import OrdinalTask, load_tasks
from .fixed_reaxis import safe_cosine
from .io import CSVAppender, utc_now_iso, write_json
from .metrics import compute_metrics, spearman_corr


@dataclass(frozen=True)
class PromptVariant:
    name: str
    description: str
    build: Callable[[OrdinalTask, ClipTextEmbedder], tuple[list[str], list[str]]]


def _manual_anchors(task: OrdinalTask, _: ClipTextEmbedder) -> tuple[list[str], list[str]]:
    return [task.high_text], [task.low_text]


def _ui_fixed_template(task: OrdinalTask, embedder: ClipTextEmbedder) -> tuple[list[str], list[str]]:
    return embedder.engine._build_fixed_prompt_ensemble(task.query)


def _simple_query(task: OrdinalTask, _: ClipTextEmbedder) -> tuple[list[str], list[str]]:
    return [f'high {task.query}'], [f'low {task.query}']


def _more_less_query(task: OrdinalTask, _: ClipTextEmbedder) -> tuple[list[str], list[str]]:
    return [f'more {task.query}'], [f'less {task.query}']


def _evidence_query(task: OrdinalTask, _: ClipTextEmbedder) -> tuple[list[str], list[str]]:
    return [f'an image with strong evidence of {task.query}'], [f'an image with weak evidence of {task.query}']


def _axis_name_query(task: OrdinalTask, _: ClipTextEmbedder) -> tuple[list[str], list[str]]:
    axis = task.axis_name.lower()
    return [f'high {axis}'], [f'low {axis}']


PROMPT_VARIANTS = (
    PromptVariant('manual_anchors', 'Current explicit task low/high ordinal anchors.', _manual_anchors),
    PromptVariant('ui_fixed_template', 'AxisBayes fixed template built from the task query.', _ui_fixed_template),
    PromptVariant('simple_query', 'Single high/low prompt around the task query.', _simple_query),
    PromptVariant('more_less_query', 'Single more/less prompt around the task query.', _more_less_query),
    PromptVariant('evidence_query', 'Strong/weak visual evidence prompt around the task query.', _evidence_query),
    PromptVariant('axis_name_query', 'Single high/low prompt around the human axis name.', _axis_name_query),
)

PROMPT_SWEEP_FIELDS = [
    'ts_utc', 'run_id', 'prompt_variant', 'dataset', 'axis_field', 'axis_name',
    'method', 'policy', 'replicate', 'budget', 'label_count', 'lambda_value',
    'beta', 'prior_spearman_b0', 'score_preservation_spearman',
    'mu_d0_cos', 'direction_relative_change', 'pos_prompts_json',
    'neg_prompts_json', *METRIC_COLS,
]


def _variant_by_name(names: Sequence[str]) -> list[PromptVariant]:
    lookup = {variant.name: variant for variant in PROMPT_VARIANTS}
    selected = []
    for name in names:
        if name not in lookup:
            raise ValueError(f'Unknown prompt variant {name!r}; expected one of {sorted(lookup)}')
        selected.append(lookup[name])
    return selected


def _direction_from_prompts(
    embedder: ClipTextEmbedder,
    pos_prompts: Sequence[str],
    neg_prompts: Sequence[str],
) -> np.ndarray:
    w0, _, _, _ = embedder.engine._embed_prompt_lists(
        list(pos_prompts),
        list(neg_prompts),
        semantic_method='clip',
        norm=True,
        source='ordinal_prompt_sweep',
        provider='manual',
    )
    return np.asarray(w0, dtype=np.float32)


def _read_completed(path: Path, run_id: str) -> set[tuple[str, str, str, int, int, float, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(
        path,
        usecols=['run_id', 'prompt_variant', 'dataset', 'axis_field', 'replicate', 'budget', 'lambda_value', 'beta'],
    )
    frame = frame[frame['run_id'].astype(str) == str(run_id)]
    completed: set[tuple[str, str, str, int, int, float, float]] = set()
    for row in frame.itertuples(index=False):
        completed.add((
            str(row.prompt_variant),
            str(row.dataset),
            str(row.axis_field),
            int(row.replicate),
            int(row.budget),
            float(row.lambda_value),
            float(row.beta),
        ))
    return completed


def run_prompt_sweep(
    *,
    run_id: str,
    output_dir: Path = OUTPUT_DIR,
    datasets: Sequence[str] = ACTIVE_DATASETS,
    budgets: Sequence[int] = BUDGETS,
    repetitions: int = REPETITIONS,
    lambdas: Sequence[float] = SWEEP_LAMBDAS,
    betas: Sequence[float] = SWEEP_BETAS,
    prompt_variants: Sequence[str] = tuple(variant.name for variant in PROMPT_VARIANTS),
    max_tasks: int = 0,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    budgets = sorted({int(v) for v in budgets})
    lambda_values = tuple(float(v) for v in lambdas)
    beta_values = tuple(float(v) for v in betas)
    variants = _variant_by_name(prompt_variants)
    tasks = load_tasks(datasets)
    if int(max_tasks) > 0:
        tasks = tasks[: int(max_tasks)]
    if not tasks:
        raise RuntimeError('No ordinal tasks selected for prompt sweep')

    write_json(root / 'ordinal_prompt_initialization_sweep_manifest.json', {
        'run_id': run_id,
        'created_utc': utc_now_iso(),
        'datasets': list(datasets),
        'budgets': budgets,
        'repetitions': int(repetitions),
        'lambdas': list(lambda_values),
        'betas': list(beta_values),
        'prompt_variants': [
            {'name': variant.name, 'description': variant.description}
            for variant in variants
        ],
        'max_tasks': int(max_tasks),
        'method': 'reaxis_gaussian_centered_prompt_sweep',
    })

    results_path = root / 'ordinal_prompt_initialization_sweep_results.csv'
    logger = CSVAppender(results_path, PROMPT_SWEEP_FIELDS)
    completed = _read_completed(results_path, run_id)
    embedder = ClipTextEmbedder()
    max_budget = max(budgets)

    for task in tqdm(tasks, desc='prompt sweep tasks'):
        for variant in tqdm(variants, desc=task.task_id, leave=False):
            pos_prompts, neg_prompts = variant.build(task, embedder)
            d0 = _direction_from_prompts(embedder, pos_prompts, neg_prompts)
            prior_scores = np.asarray(task.X @ d0, dtype=np.float32)
            prior_spearman = spearman_corr(prior_scores, task.y01)
            pos_json = json.dumps(list(pos_prompts), ensure_ascii=True)
            neg_json = json.dumps(list(neg_prompts), ensure_ascii=True)
            for replicate in range(int(repetitions)):
                selected_idx = _random_feedback_indices(task, replicate, max_budget)
                for lambda_value in lambda_values:
                    for beta in beta_values:
                        for budget in budgets:
                            key = (
                                variant.name,
                                task.dataset,
                                task.axis_field,
                                int(replicate),
                                int(budget),
                                float(lambda_value),
                                float(beta),
                            )
                            if key in completed:
                                continue
                            scores, mu = _centered_scores(
                                task=task,
                                d0=d0,
                                selected_idx=selected_idx,
                                budget=budget,
                                lambda_value=lambda_value,
                                beta=beta,
                            )
                            rng = np.random.default_rng(
                                GLOBAL_SEED + _stable_seed(
                                    task.task_id, 'prompt_sweep', variant.name,
                                    replicate, budget, lambda_value, beta,
                                )
                            )
                            metrics, _ = compute_metrics(scores, task.y01, rng)
                            logger.append({
                                'ts_utc': utc_now_iso(),
                                'run_id': run_id,
                                'prompt_variant': variant.name,
                                'dataset': task.dataset,
                                'axis_field': task.axis_field,
                                'axis_name': task.axis_name,
                                'method': 'reaxis_gaussian_centered_prompt_sweep',
                                'policy': 'random',
                                'replicate': int(replicate),
                                'budget': int(budget),
                                'label_count': int(budget),
                                'lambda_value': float(lambda_value),
                                'beta': float(beta),
                                'prior_spearman_b0': float(prior_spearman),
                                'score_preservation_spearman': spearman_corr(prior_scores, scores),
                                'mu_d0_cos': safe_cosine(mu, d0),
                                'direction_relative_change': float(
                                    np.linalg.norm(mu - d0) / max(float(np.linalg.norm(d0)), 1e-12)
                                ),
                                'pos_prompts_json': pos_json,
                                'neg_prompts_json': neg_json,
                                **metrics,
                            })
                            completed.add(key)

    _write_prompt_sweep_reports(root, run_id)
    return root / 'ordinal_prompt_initialization_sweep_progression.csv'


def _write_prompt_sweep_reports(root: Path, run_id: str) -> None:
    results_path = root / 'ordinal_prompt_initialization_sweep_results.csv'
    rows = pd.read_csv(results_path)
    rows = rows[rows['run_id'].astype(str) == str(run_id)].copy()
    if rows.empty:
        return
    numeric = ['replicate', 'budget', 'label_count', 'lambda_value', 'beta', *METRIC_COLS]
    for col in numeric:
        rows[col] = pd.to_numeric(rows[col], errors='coerce')
    summary = (
        rows.groupby(['prompt_variant', 'lambda_value', 'beta', 'budget'], dropna=False)[METRIC_COLS]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    summary.columns = [
        '_'.join(str(part) for part in col if str(part))
        for col in summary.columns.to_flat_index()
    ]
    summary.to_csv(root / 'ordinal_prompt_initialization_sweep_summary.csv', index=False)
    progression = _build_prompt_progression(rows)
    progression.to_csv(root / 'ordinal_prompt_initialization_sweep_progression.csv', index=False)
    _write_prompt_best_markdown(root / 'ordinal_prompt_initialization_sweep_best.md', progression, rows)


def _build_prompt_progression(rows: pd.DataFrame) -> pd.DataFrame:
    budget_means = (
        rows.groupby(['prompt_variant', 'lambda_value', 'beta', 'budget'], dropna=False)['spearman']
        .mean()
        .reset_index()
    )
    final_budget = int(rows['budget'].max())
    b0 = rows[rows['budget'] == 0][
        ['prompt_variant', 'lambda_value', 'beta', 'dataset', 'axis_field', 'replicate', 'spearman']
    ].rename(columns={'spearman': 'spearman_b0'})
    b20 = rows[rows['budget'] == final_budget][
        ['prompt_variant', 'lambda_value', 'beta', 'dataset', 'axis_field', 'replicate', 'spearman']
    ].rename(columns={'spearman': 'spearman_b20'})
    paired = b20.merge(
        b0,
        on=['prompt_variant', 'lambda_value', 'beta', 'dataset', 'axis_field', 'replicate'],
        how='inner',
    )
    paired['delta_b20_b0'] = paired['spearman_b20'] - paired['spearman_b0']
    out = []
    group_cols = ['prompt_variant', 'lambda_value', 'beta']
    for (variant, lambda_value, beta), group in paired.groupby(group_cols, dropna=False):
        mask = (
            (budget_means['prompt_variant'] == str(variant))
            & (budget_means['lambda_value'] == float(lambda_value))
            & (budget_means['beta'] == float(beta))
        )
        means = budget_means[mask].sort_values('budget')['spearman'].to_numpy(dtype=float)
        adjacent = np.diff(means) if means.size >= 2 else np.asarray([], dtype=float)
        diff = group['delta_b20_b0'].to_numpy(dtype=float)
        p_value = float('nan')
        if diff.size >= 2 and float(np.std(diff)) > 1e-12:
            p_value = float(stats.ttest_1samp(diff, 0.0, alternative='greater').pvalue)
        out.append({
            'prompt_variant': str(variant),
            'lambda_value': float(lambda_value),
            'beta': float(beta),
            'final_budget': final_budget,
            'spearman_b0_mean': float(group['spearman_b0'].mean()),
            'spearman_b20_mean': float(group['spearman_b20'].mean()),
            'delta_b20_b0_mean': float(np.mean(diff)),
            'delta_b20_b0_std': float(np.std(diff, ddof=1)) if diff.size > 1 else float('nan'),
            'delta_b20_b0_sem': float(stats.sem(diff)) if diff.size > 1 else float('nan'),
            'p_one_sided_t': p_value,
            'paired_count': int(diff.size),
            'budget_decrease_count': int(np.sum(adjacent < -1e-8)),
            'min_adjacent_delta': float(np.min(adjacent)) if adjacent.size else float('nan'),
        })
    return pd.DataFrame(out).sort_values(
        ['delta_b20_b0_mean', 'spearman_b20_mean', 'budget_decrease_count'],
        ascending=[False, False, True],
    )


def _write_prompt_best_markdown(path: Path, progression: pd.DataFrame, rows: pd.DataFrame) -> None:
    lines = [
        '# Ordinal Prompt Initialization Sweep',
        '',
        'Rows rank prompt/lambda/beta settings by B20-B0 Spearman gain. '
        'A large gain can mean the initial prompt was weak, so B20 should be read with the gain.',
        '',
        '| prompt_variant | lambda | beta | b0 | b20 | delta | decreases | p_one_sided |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in progression.head(20).itertuples(index=False):
        lines.append(
            f'| {row.prompt_variant} | {row.lambda_value:g} | {row.beta:g} | '
            f'{row.spearman_b0_mean:.3f} | {row.spearman_b20_mean:.3f} | '
            f'{row.delta_b20_b0_mean:.3f} | {int(row.budget_decrease_count)} | '
            f'{row.p_one_sided_t:.3g} |'
        )
    best_by_variant = (
        progression.sort_values(['delta_b20_b0_mean', 'spearman_b20_mean'], ascending=[False, False])
        .groupby('prompt_variant', dropna=False)
        .head(1)
        .sort_values('delta_b20_b0_mean', ascending=False)
    )
    lines.extend([
        '',
        '## Best Setting Per Prompt Variant',
        '',
        '| prompt_variant | lambda | beta | b0 | b20 | delta | decreases |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ])
    for row in best_by_variant.itertuples(index=False):
        lines.append(
            f'| {row.prompt_variant} | {row.lambda_value:g} | {row.beta:g} | '
            f'{row.spearman_b0_mean:.3f} | {row.spearman_b20_mean:.3f} | '
            f'{row.delta_b20_b0_mean:.3f} | {int(row.budget_decrease_count)} |'
        )
    lines.extend(_task_delta_lines(rows, best_by_variant))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _task_delta_lines(rows: pd.DataFrame, settings: pd.DataFrame) -> list[str]:
    lines = ['', '## Task Deltas For Best Variant Settings', '']
    final_budget = int(rows['budget'].max())
    for setting in settings.itertuples(index=False):
        subset = rows[
            (rows['prompt_variant'] == str(setting.prompt_variant))
            & (rows['lambda_value'] == float(setting.lambda_value))
            & (rows['beta'] == float(setting.beta))
        ]
        task_curve = (
            subset.groupby(['dataset', 'axis_field', 'budget'], dropna=False)['spearman']
            .mean()
            .reset_index()
            .pivot_table(index=['dataset', 'axis_field'], columns='budget', values='spearman')
            .reset_index()
        )
        lines.extend([
            f'### {setting.prompt_variant}',
            '',
            '| dataset | axis | b0 | b20 | delta |',
            '| --- | --- | --- | --- | --- |',
        ])
        for _, row in task_curve.iterrows():
            b0 = float(row.get(0, float('nan')))
            b20 = float(row.get(final_budget, float('nan')))
            lines.append(
                f'| {row["dataset"]} | {row["axis_field"]} | {b0:.3f} | '
                f'{b20:.3f} | {b20 - b0:.3f} |'
            )
        lines.append('')
    return lines
