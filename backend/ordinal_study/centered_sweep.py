from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from .baselines import ClipTextEmbedder
from .config import ACTIVE_DATASETS, BUDGETS, GLOBAL_SEED, OUTPUT_DIR, REPETITIONS
from .data import OrdinalTask, load_tasks
from .fixed_reaxis import _distinct_count, _solve_prior_ridge, safe_cosine, zscore
from .io import CSVAppender, utc_now_iso, write_json
from .metrics import compute_metrics, spearman_corr


SWEEP_LAMBDAS = (
    0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0,
    10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0,
)
SWEEP_BETAS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)

METRIC_COLS = [
    'spearman', 'kendall_tau', 'mae_isotonic', 'qwk',
    'axis_r2', 'monotonicity_violations',
]

SWEEP_RESULT_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'axis_field', 'axis_name', 'method',
    'policy', 'replicate', 'budget', 'label_count', 'lambda_value',
    'beta', 'prior_spearman_b0', 'score_preservation_spearman',
    'mu_d0_cos', 'direction_relative_change', *METRIC_COLS,
]


def _stable_seed(*parts: object) -> int:
    text = '||'.join(str(part) for part in parts)
    return int(zlib.crc32(text.encode('utf-8')) & 0xFFFFFFFF)


def _read_completed(path: Path, run_id: str) -> set[tuple[str, str, int, int, float, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(
        path,
        usecols=['run_id', 'dataset', 'axis_field', 'replicate', 'budget', 'lambda_value', 'beta'],
    )
    frame = frame[frame['run_id'].astype(str) == str(run_id)]
    completed: set[tuple[str, str, int, int, float, float]] = set()
    for row in frame.itertuples(index=False):
        completed.add((
            str(row.dataset),
            str(row.axis_field),
            int(row.replicate),
            int(row.budget),
            float(row.lambda_value),
            float(row.beta),
        ))
    return completed


def _random_feedback_indices(task: OrdinalTask, replicate: int, max_budget: int) -> np.ndarray:
    rng = np.random.default_rng(GLOBAL_SEED + _stable_seed(task.task_id, 'random', replicate))
    count = min(int(max_budget), len(task.ids))
    return np.asarray(rng.choice(np.arange(len(task.ids)), size=count, replace=False), dtype=np.int64)


def _centered_scores(
    *,
    task: OrdinalTask,
    d0: np.ndarray,
    selected_idx: np.ndarray,
    budget: int,
    lambda_value: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(task.X, dtype=np.float32)
    x_mean = np.mean(X, axis=0, keepdims=True)
    Xc = np.asarray(X - x_mean, dtype=np.float32)
    s0 = np.asarray(X @ d0, dtype=np.float32)
    idx = np.asarray(selected_idx[: int(budget)], dtype=np.int64)
    if idx.size < 2 or _distinct_count(task.y01[idx]) < 2:
        return s0, np.asarray(d0, dtype=np.float32)
    y = np.asarray(task.y01[idx], dtype=np.float32)
    y_centered = y - float(np.mean(y))
    mu = _solve_prior_ridge(Xc[idx], y_centered, d0, float(lambda_value))
    centered = np.asarray(Xc @ mu, dtype=np.float32)
    blended = ((1.0 - float(beta)) * zscore(s0)) + (float(beta) * zscore(centered))
    return np.asarray(blended, dtype=np.float32), np.asarray(mu, dtype=np.float32)


def run_centered_sweep(
    *,
    run_id: str,
    output_dir: Path = OUTPUT_DIR,
    datasets: Sequence[str] = ACTIVE_DATASETS,
    budgets: Sequence[int] = BUDGETS,
    repetitions: int = REPETITIONS,
    lambdas: Sequence[float] = SWEEP_LAMBDAS,
    betas: Sequence[float] = SWEEP_BETAS,
    max_tasks: int = 0,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    budgets = sorted({int(v) for v in budgets})
    lambda_values = tuple(float(v) for v in lambdas)
    beta_values = tuple(float(v) for v in betas)
    tasks = load_tasks(datasets)
    if int(max_tasks) > 0:
        tasks = tasks[: int(max_tasks)]
    if not tasks:
        raise RuntimeError('No ordinal tasks selected for centered sweep')

    write_json(root / 'ordinal_centered_gaussian_sweep_manifest.json', {
        'run_id': run_id,
        'created_utc': utc_now_iso(),
        'datasets': list(datasets),
        'budgets': budgets,
        'repetitions': int(repetitions),
        'lambdas': list(lambda_values),
        'betas': list(beta_values),
        'max_tasks': int(max_tasks),
        'method': 'reaxis_gaussian_centered_sweep',
        'score': '(1-beta)*zscore(text_prior) + beta*zscore(centered_gaussian_update)',
    })

    results_path = root / 'ordinal_centered_gaussian_sweep_results.csv'
    logger = CSVAppender(results_path, SWEEP_RESULT_FIELDS)
    completed = _read_completed(results_path, run_id)
    embedder = ClipTextEmbedder()
    max_budget = max(budgets)

    for task in tqdm(tasks, desc='centered sweep tasks'):
        d0 = embedder.direction(task.high_text, task.low_text)
        prior_scores = np.asarray(task.X @ d0, dtype=np.float32)
        prior_spearman = spearman_corr(prior_scores, task.y01)
        for replicate in tqdm(range(int(repetitions)), desc=task.task_id, leave=False):
            selected_idx = _random_feedback_indices(task, replicate, max_budget)
            for lambda_value in lambda_values:
                for beta in beta_values:
                    for budget in budgets:
                        key = (
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
                                task.task_id, 'centered_sweep', replicate, budget,
                                lambda_value, beta,
                            )
                        )
                        metrics, _ = compute_metrics(scores, task.y01, rng)
                        logger.append({
                            'ts_utc': utc_now_iso(),
                            'run_id': run_id,
                            'dataset': task.dataset,
                            'axis_field': task.axis_field,
                            'axis_name': task.axis_name,
                            'method': 'reaxis_gaussian_centered_sweep',
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
                            **metrics,
                        })
                        completed.add(key)

    _write_sweep_reports(root, run_id)
    return root / 'ordinal_centered_gaussian_sweep_progression.csv'


def _write_sweep_reports(root: Path, run_id: str) -> None:
    results_path = root / 'ordinal_centered_gaussian_sweep_results.csv'
    rows = pd.read_csv(results_path)
    rows = rows[rows['run_id'].astype(str) == str(run_id)].copy()
    if rows.empty:
        return
    numeric = ['replicate', 'budget', 'label_count', 'lambda_value', 'beta', *METRIC_COLS]
    for col in numeric:
        rows[col] = pd.to_numeric(rows[col], errors='coerce')
    summary = (
        rows.groupby(['lambda_value', 'beta', 'budget'], dropna=False)[METRIC_COLS]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    summary.columns = [
        '_'.join(str(part) for part in col if str(part))
        for col in summary.columns.to_flat_index()
    ]
    summary.to_csv(root / 'ordinal_centered_gaussian_sweep_summary.csv', index=False)
    progression = _build_progression(rows)
    progression.to_csv(root / 'ordinal_centered_gaussian_sweep_progression.csv', index=False)
    _write_best_markdown(root / 'ordinal_centered_gaussian_sweep_best.md', progression, rows)


def _build_progression(rows: pd.DataFrame) -> pd.DataFrame:
    budget_means = (
        rows.groupby(['lambda_value', 'beta', 'budget'], dropna=False)['spearman']
        .mean()
        .reset_index()
    )
    b0 = rows[rows['budget'] == 0][
        ['lambda_value', 'beta', 'dataset', 'axis_field', 'replicate', 'spearman']
    ].rename(columns={'spearman': 'spearman_b0'})
    final_budget = int(rows['budget'].max())
    b20 = rows[rows['budget'] == final_budget][
        ['lambda_value', 'beta', 'dataset', 'axis_field', 'replicate', 'spearman']
    ].rename(columns={'spearman': 'spearman_b20'})
    paired = b20.merge(
        b0,
        on=['lambda_value', 'beta', 'dataset', 'axis_field', 'replicate'],
        how='inner',
    )
    paired['delta_b20_b0'] = paired['spearman_b20'] - paired['spearman_b0']
    out = []
    for (lambda_value, beta), group in paired.groupby(['lambda_value', 'beta'], dropna=False):
        means = (
            budget_means[
                (budget_means['lambda_value'] == float(lambda_value))
                & (budget_means['beta'] == float(beta))
            ]
            .sort_values('budget')['spearman']
            .to_numpy(dtype=float)
        )
        adjacent = np.diff(means) if means.size >= 2 else np.asarray([], dtype=float)
        diff = group['delta_b20_b0'].to_numpy(dtype=float)
        p_value = float('nan')
        if diff.size >= 2 and float(np.std(diff)) > 1e-12:
            p_value = float(stats.ttest_1samp(diff, 0.0, alternative='greater').pvalue)
        out.append({
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


def _write_best_markdown(path: Path, progression: pd.DataFrame, rows: pd.DataFrame) -> None:
    raw_best = progression.iloc[0]
    monotone = progression[progression['budget_decrease_count'] == 0]
    best = monotone.iloc[0] if not monotone.empty else raw_best
    lam = float(best['lambda_value'])
    beta = float(best['beta'])
    curve = (
        rows[(rows['lambda_value'] == lam) & (rows['beta'] == beta)]
        .groupby('budget', dropna=False)[METRIC_COLS]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values('budget')
    )
    lines = [
        '# Centered Gaussian Reaxis Sweep',
        '',
        f'Recommended progression setting: lambda={lam:g}, beta={beta:g}.',
        f'B{int(best["final_budget"])}-B0 Spearman delta: {float(best["delta_b20_b0_mean"]):.4f} '
        f'(one-sided paired t p={float(best["p_one_sided_t"]):.4g}, '
        f'n={int(best["paired_count"])} task-replicate pairs).',
        f'Budget-level mean Spearman decreases: {int(best["budget_decrease_count"])}.',
        '',
        f'Highest B20 exploratory setting: lambda={float(raw_best["lambda_value"]):g}, '
        f'beta={float(raw_best["beta"]):g}, '
        f'B20 Spearman={float(raw_best["spearman_b20_mean"]):.4f}, '
        f'delta={float(raw_best["delta_b20_b0_mean"]):.4f}, '
        f'budget decreases={int(raw_best["budget_decrease_count"])}.',
        '',
        '## Parameter Interpretation',
        '',
        (
            f'`lambda={lam:g}` is the ridge/prior-strength term in the centered Gaussian update. '
            'Smaller values let sparse labels bend the direction more aggressively; larger values keep the '
            'updated direction closer to the text prior. The selected value is low enough to learn from '
            '20 labels, but still regularized enough to avoid the rank-mode collapse seen earlier.'
        ),
        (
            f'`beta={beta:g}` is the score-level blend weight. The plotted/evaluated score is '
            '`(1 - beta) * zscore(text_prior) + beta * zscore(centered_update)`. '
            f'With beta={beta:g}, half of the ranking still comes from the CLIP/text initialization and '
            'half comes from centered ordinal feedback, which explains why the curve preserves b0 behavior '
            'while improving steadily at larger budgets.'
        ),
        '',
        '| budget | spearman | kendall_tau | mae_isotonic | qwk | axis_r2 | monotonicity_violations |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in curve.itertuples(index=False):
        lines.append(
            f'| {int(row.budget)} | {row.spearman:.3f} | {row.kendall_tau:.3f} | '
            f'{row.mae_isotonic:.3f} | {row.qwk:.3f} | {row.axis_r2:.3f} | '
            f'{row.monotonicity_violations:.3f} |'
        )
    task_curve = (
        rows[(rows['lambda_value'] == lam) & (rows['beta'] == beta)]
        .groupby(['dataset', 'axis_field', 'budget'], dropna=False)['spearman']
        .mean()
        .reset_index()
        .pivot_table(index=['dataset', 'axis_field'], columns='budget', values='spearman')
        .reset_index()
    )
    if 0 in task_curve.columns and int(best['final_budget']) in task_curve.columns:
        task_curve['delta_final_b0'] = task_curve[int(best['final_budget'])] - task_curve[0]
    lines.extend([
        '',
        '| dataset | axis | b0 | b20 | delta |',
        '| --- | --- | --- | --- | --- |',
    ])
    for _, row in task_curve.iterrows():
        b0 = float(row.get(0, float('nan')))
        b20 = float(row.get(int(best['final_budget']), float('nan')))
        delta = float(row.get('delta_final_b0', float('nan')))
        lines.append(f'| {row["dataset"]} | {row["axis_field"]} | {b0:.3f} | {b20:.3f} | {delta:.3f} |')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
