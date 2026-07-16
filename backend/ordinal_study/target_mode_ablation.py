from __future__ import annotations

import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .baselines import ClipTextEmbedder, make_session
from .config import GLOBAL_SEED
from .data import load_tasks
from .fixed_reaxis import TARGET_MODE_GROUND_TRUTH, TARGET_MODE_PRIOR_QUANTILE, validate_target_mode
from .io import CSVAppender, utc_now_iso
from .metrics import compute_metrics


TARGET_MODE_RESULT_FIELDS = [
    'ts_utc', 'run_id', 'target_mode', 'dataset', 'axis_field', 'axis_name',
    'method', 'replicate', 'budget', 'label_count', 'spearman',
    'kendall_tau', 'mae_isotonic', 'qwk', 'axis_r2',
    'monotonicity_violations',
]


def _stable_seed(*parts: object) -> int:
    text = '||'.join(str(part) for part in parts)
    return int(zlib.crc32(text.encode('utf-8')) & 0xFFFFFFFF)


def _select_random_index(queried: np.ndarray, rng: np.random.Generator) -> int | None:
    available = np.flatnonzero(~queried)
    if available.size == 0:
        return None
    return int(rng.choice(available))


def _read_completed(path: Path, run_id: str) -> set[tuple[str, str, str, int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=['run_id', 'dataset', 'axis_field', 'method', 'target_mode', 'replicate', 'budget'])
    frame = frame[frame['run_id'].astype(str) == str(run_id)]
    out: set[tuple[str, str, str, int, int]] = set()
    for row in frame.itertuples(index=False):
        out.add((
            str(row.dataset),
            str(row.axis_field),
            f'{row.method}:{row.target_mode}',
            int(row.replicate),
            int(row.budget),
        ))
    return out


def run_target_mode_ablation(
    *,
    run_id: str,
    output_dir: Path,
    datasets: Sequence[str],
    budgets: Sequence[int],
    repetitions: int,
    max_tasks: int = 0,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    modes = (TARGET_MODE_GROUND_TRUTH, TARGET_MODE_PRIOR_QUANTILE)
    methods = ('reaxis_random', 'reaxis_centered')
    tasks = load_tasks(datasets)
    if int(max_tasks) > 0:
        tasks = tasks[: int(max_tasks)]
    budgets = sorted({int(v) for v in budgets})
    results_path = root / 'ordinal_target_mode_results.csv'
    results_logger = CSVAppender(results_path, TARGET_MODE_RESULT_FIELDS)
    completed = _read_completed(results_path, run_id)
    embedder = ClipTextEmbedder()

    for task in tqdm(tasks, desc='target-mode tasks'):
        for mode in modes:
            target_mode = validate_target_mode(mode)
            for method in methods:
                for replicate in range(int(repetitions)):
                    needed = [
                        (task.dataset, task.axis_field, f'{method}:{target_mode}', replicate, budget)
                        for budget in budgets
                    ]
                    if all(key in completed for key in needed):
                        continue
                    rng = np.random.default_rng(GLOBAL_SEED + _stable_seed(task.task_id, 'target_mode', replicate))
                    session = make_session(method, task, embedder, target_mode=target_mode)
                    queried = np.zeros((len(task.ids),), dtype=bool)
                    label_count = 0
                    max_budget = max(budgets)
                    for budget in budgets:
                        if budget == 0:
                            _append_result(
                                results_logger, completed, run_id, task, method,
                                target_mode, replicate, budget, label_count, session, rng,
                            )
                    for step in range(1, max_budget + 1):
                        idx = _select_random_index(queried, rng)
                        if idx is None:
                            break
                        queried[int(idx)] = True
                        session.observe(int(idx), float(task.y01[int(idx)]))
                        label_count += 1
                        if step in budgets:
                            _append_result(
                                results_logger, completed, run_id, task, method,
                                target_mode, replicate, step, label_count, session, rng,
                            )

    _write_summary(root, run_id)
    return root / 'ordinal_target_mode_summary.csv'


def _append_result(
    logger: CSVAppender,
    completed: set[tuple[str, str, str, int, int]],
    run_id: str,
    task,
    method: str,
    target_mode: str,
    replicate: int,
    budget: int,
    label_count: int,
    session,
    rng: np.random.Generator,
) -> None:
    key = (task.dataset, task.axis_field, f'{method}:{target_mode}', int(replicate), int(budget))
    if key in completed:
        return
    metrics, _ = compute_metrics(np.asarray(session.scores(), dtype=np.float32), task.y01, rng)
    logger.append({
        'ts_utc': utc_now_iso(),
        'run_id': run_id,
        'target_mode': target_mode,
        'dataset': task.dataset,
        'axis_field': task.axis_field,
        'axis_name': task.axis_name,
        'method': method,
        'replicate': int(replicate),
        'budget': int(budget),
        'label_count': int(label_count),
        **metrics,
    })
    completed.add(key)


def _write_summary(root: Path, run_id: str) -> None:
    results_path = root / 'ordinal_target_mode_results.csv'
    if not results_path.exists() or results_path.stat().st_size == 0:
        return
    rows = pd.read_csv(results_path)
    rows = rows[rows['run_id'].astype(str) == str(run_id)].copy()
    if rows.empty:
        return
    metric_cols = ['spearman', 'kendall_tau', 'mae_isotonic', 'qwk', 'axis_r2', 'monotonicity_violations']
    for col in ['budget', 'replicate', 'label_count', *metric_cols]:
        rows[col] = pd.to_numeric(rows[col], errors='coerce')
    summary = (
        rows.groupby(['method', 'target_mode', 'budget'], dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(['method', 'target_mode', 'budget'])
    )
    counts = rows.groupby(['method', 'target_mode', 'budget'], dropna=False).size().reset_index(name='count')
    summary.merge(counts, on=['method', 'target_mode', 'budget'], how='left').to_csv(
        root / 'ordinal_target_mode_summary.csv',
        index=False,
    )
