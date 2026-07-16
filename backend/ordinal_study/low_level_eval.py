from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ..gallery_backend import ImageGalleryEngine
except ImportError:
    from gallery_backend import ImageGalleryEngine

from .baselines import ClipTextEmbedder, METHOD_SUPPORTS_FEEDBACK, make_session
from .config import BUDGETS, DATASETS_ROOT, GLOBAL_SEED, REPETITIONS, REPO_ROOT
from .data import OrdinalTask, _load_clip_embeddings, _load_metadata
from .fixed_reaxis import labeled_rank_mae, pairwise_agreement, validate_target_mode
from .io import CSVAppender, read_completed_metric_keys, utc_now_iso, write_json
from .low_level_features import LOW_LEVEL_FEATURES, compute_dataset_low_level_features
from .low_level_reports import export_low_level_reports
from .metrics import compute_metrics, spearman_corr


LOW_LEVEL_OUTPUT_DIR = REPO_ROOT / 'backend' / 'experiments' / 'low_level_ordinal'
LOW_LEVEL_METHODS = (
    'text_prior',
    'prompt_ladder',
    'ordinal_ridge',
    'rank_svm',
    'knn_ordinal',
    'kernel_ridge',
    'reaxis_centered',
    'reaxis_calibrated_residual',
)
RESULT_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'axis_field', 'axis_name', 'method',
    'replicate', 'budget', 'label_count', 'spearman', 'kendall_tau',
    'mae_isotonic', 'qwk', 'axis_r2', 'monotonicity_violations',
    'selected_lambda', 'selected_beta', 'score_preservation_spearman',
    'mu_d0_cos', 'direction_relative_change', 'labeled_pairwise_agreement',
    'labeled_mae', 'prior_spearman_b0',
]
TASK_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'domain', 'axis_field', 'axis_name',
    'query', 'label_source', 'image_count', 'raw_min', 'raw_max',
    'low_text', 'high_text', 'ladder_json',
]
STAT_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'axis_field', 'axis_name', 'label_source',
    'count', 'raw_min', 'raw_p05', 'raw_mean', 'raw_median', 'raw_p95',
    'raw_max', 'raw_std',
]


@dataclass(frozen=True)
class LowLevelStudyOptions:
    run_id: str
    output_dir: Path = LOW_LEVEL_OUTPUT_DIR
    datasets: Sequence[str] = ('EmoSet',)
    methods: Sequence[str] = LOW_LEVEL_METHODS
    budgets: Sequence[int] = BUDGETS
    repetitions: int = REPETITIONS
    max_tasks: int = 0
    max_edge: int = 512
    overwrite_features: bool = False
    target_mode: str = 'ground_truth_normalized'


def stable_seed(*parts: object) -> int:
    text = '||'.join(str(part) for part in parts)
    return int(zlib.crc32(text.encode('utf-8')) & 0xFFFFFFFF)


def validate_low_level_methods(methods: Sequence[str]) -> list[str]:
    out = [str(method).strip() for method in methods if str(method).strip()]
    unknown = sorted(set(out) - set(METHOD_SUPPORTS_FEEDBACK))
    if unknown:
        raise RuntimeError(f'Unknown low-level ordinal methods: {unknown}')
    rank_methods = [method for method in out if method.endswith('_rank')]
    if rank_methods:
        raise RuntimeError(f'Rank Reaxis methods are intentionally excluded from low-level ordinal study: {rank_methods}')
    return out


def dataset_roots(dataset_names: Sequence[str]) -> list[Path]:
    roots: list[Path] = []
    for name in dataset_names:
        clean = str(name).strip()
        if not clean:
            continue
        root = Path(clean)
        if not root.is_absolute():
            root = DATASETS_ROOT / clean
        if not root.exists():
            raise FileNotFoundError(f'Prepared dataset not found: {root}')
        roots.append(root)
    return roots


def all_prepared_dataset_names() -> tuple[str, ...]:
    names = [
        path.name
        for path in DATASETS_ROOT.iterdir()
        if path.is_dir() and (path / 'metadata.csv').exists() and (path / 'metadata.csv').stat().st_size > 0
    ]
    return tuple(sorted(names))


def _normalize_dynamic(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise RuntimeError(f'Cannot normalize constant or invalid low-level labels: min={lo}, max={hi}')
    return np.asarray((arr - lo) / (hi - lo), dtype=np.float32), lo, hi


def load_low_level_tasks(dataset_root: Path) -> list[OrdinalTask]:
    dataset_root = Path(dataset_root)
    engine = ImageGalleryEngine(str(dataset_root))
    entries = engine.list_images()
    if not entries:
        raise RuntimeError(f'No images found in prepared dataset: {dataset_root}')
    ids = [str(entry.id) for entry in entries]
    paths = [str(entry.path) for entry in entries]
    metadata = _load_metadata(dataset_root, ids)
    X = _load_clip_embeddings(dataset_root, engine, entries)
    tasks: list[OrdinalTask] = []
    for spec in LOW_LEVEL_FEATURES:
        if spec.column not in metadata.columns:
            raise RuntimeError(f'Missing low-level feature column {spec.column!r} in {dataset_root / "metadata.csv"}')
        values = pd.to_numeric(metadata[spec.column], errors='coerce')
        keep = values.notna().to_numpy(dtype=bool)
        if int(np.sum(keep)) < 3:
            raise RuntimeError(f'Feature {spec.name} has fewer than 3 numeric labels in {dataset_root}')
        y_raw = values.to_numpy(dtype=np.float32)[keep]
        y01, _, _ = _normalize_dynamic(y_raw)
        tasks.append(OrdinalTask(
            dataset=dataset_root.name,
            domain='Low-level visual features',
            axis_name=spec.name,
            axis_field=spec.column,
            query=spec.query,
            low_text=spec.low_text,
            high_text=spec.high_text,
            ladder=tuple(spec.ladder),
            label_source=spec.label_source,
            dataset_root=dataset_root,
            ids=[ids[idx] for idx, flag in enumerate(keep) if bool(flag)],
            paths=[paths[idx] for idx, flag in enumerate(keep) if bool(flag)],
            X=np.asarray(X[keep], dtype=np.float32),
            y_raw=np.asarray(y_raw, dtype=np.float32),
            y01=np.asarray(y01, dtype=np.float32),
        ))
    return tasks


def _append_task_and_stats(run_id: str, output_dir: Path, tasks: Sequence[OrdinalTask]) -> None:
    task_logger = CSVAppender(output_dir / 'low_level_tasks.csv', TASK_FIELDS)
    stats_logger = CSVAppender(output_dir / 'low_level_feature_stats.csv', STAT_FIELDS)
    for task in tasks:
        raw = np.asarray(task.y_raw, dtype=np.float64)
        task_logger.append({
            'ts_utc': utc_now_iso(),
            'run_id': run_id,
            'dataset': task.dataset,
            'domain': task.domain,
            'axis_field': task.axis_field,
            'axis_name': task.axis_name,
            'query': task.query,
            'label_source': task.label_source,
            'image_count': len(task.ids),
            'raw_min': float(np.min(raw)),
            'raw_max': float(np.max(raw)),
            'low_text': task.low_text,
            'high_text': task.high_text,
            'ladder_json': json.dumps(list(task.ladder), separators=(',', ':')),
        })
        stats_logger.append({
            'ts_utc': utc_now_iso(),
            'run_id': run_id,
            'dataset': task.dataset,
            'axis_field': task.axis_field,
            'axis_name': task.axis_name,
            'label_source': task.label_source,
            'count': int(raw.size),
            'raw_min': float(np.min(raw)),
            'raw_p05': float(np.quantile(raw, 0.05)),
            'raw_mean': float(np.mean(raw)),
            'raw_median': float(np.median(raw)),
            'raw_p95': float(np.quantile(raw, 0.95)),
            'raw_max': float(np.max(raw)),
            'raw_std': float(np.std(raw)),
        })


def _selection_order(task: OrdinalTask, replicate: int, max_budget: int) -> list[int]:
    if int(max_budget) <= 0:
        return []
    rng = np.random.default_rng(GLOBAL_SEED + stable_seed('low_level_selection', task.task_id, replicate))
    size = min(int(max_budget), len(task.ids))
    return [int(idx) for idx in rng.choice(np.arange(len(task.ids)), size=size, replace=False).tolist()]


def _fixed_diagnostics(task: OrdinalTask, session, scores: np.ndarray) -> dict[str, float]:
    if hasattr(session, 'fixed_diagnostics'):
        return dict(session.fixed_diagnostics(task.y01))
    labeled_idx = getattr(session, 'labeled_idx', [])
    labeled_scores = np.asarray(scores, dtype=np.float32)[labeled_idx] if labeled_idx else np.zeros((0,), dtype=np.float32)
    labeled_y = np.asarray(task.y01, dtype=np.float32)[labeled_idx] if labeled_idx else np.zeros((0,), dtype=np.float32)
    prior = session.debug_prior_scores() if hasattr(session, 'debug_prior_scores') else np.zeros_like(scores)
    return {
        'selected_lambda': float('nan'),
        'selected_beta': float('nan'),
        'score_preservation_spearman': spearman_corr(np.asarray(prior, dtype=np.float32), scores),
        'mu_d0_cos': float('nan'),
        'direction_relative_change': float('nan'),
        'labeled_pairwise_agreement': pairwise_agreement(labeled_scores, labeled_y),
        'labeled_mae': labeled_rank_mae(scores, task.y01, list(labeled_idx)),
        'prior_spearman_b0': spearman_corr(np.asarray(prior, dtype=np.float32), task.y01),
    }


def _append_result(
    logger: CSVAppender,
    *,
    run_id: str,
    task: OrdinalTask,
    method: str,
    replicate: int,
    budget: int,
    label_count: int,
    session,
) -> None:
    rng = np.random.default_rng(GLOBAL_SEED + stable_seed('low_level_metric', task.task_id, method, replicate, budget))
    scores = np.asarray(session.scores(), dtype=np.float32)
    metrics, _ = compute_metrics(scores, task.y01, rng)
    logger.append({
        'ts_utc': utc_now_iso(),
        'run_id': run_id,
        'dataset': task.dataset,
        'axis_field': task.axis_field,
        'axis_name': task.axis_name,
        'method': method,
        'replicate': int(replicate),
        'budget': int(budget),
        'label_count': int(label_count),
        **metrics,
        **_fixed_diagnostics(task, session, scores),
    })


def run_low_level_study(options: LowLevelStudyOptions) -> Path:
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_mode = validate_target_mode(options.target_mode)
    methods = validate_low_level_methods(options.methods)
    budgets = sorted({int(budget) for budget in options.budgets})
    roots = dataset_roots(options.datasets)

    for root in roots:
        compute_dataset_low_level_features(
            root,
            overwrite=bool(options.overwrite_features),
            max_edge=int(options.max_edge),
        )

    tasks: list[OrdinalTask] = []
    for root in roots:
        tasks.extend(load_low_level_tasks(root))
    if int(options.max_tasks) > 0:
        tasks = tasks[: int(options.max_tasks)]
    if not tasks:
        raise RuntimeError('No low-level ordinal tasks selected')

    write_json(output_dir / 'run_manifest.json', {
        'run_id': options.run_id,
        'created_utc': utc_now_iso(),
        'datasets': [root.name for root in roots],
        'methods': methods,
        'budgets': budgets,
        'repetitions': int(options.repetitions),
        'max_tasks': int(options.max_tasks),
        'max_edge': int(options.max_edge),
        'overwrite_features': bool(options.overwrite_features),
        'target_mode': target_mode,
        'features': [spec.column for spec in LOW_LEVEL_FEATURES],
    })

    _append_task_and_stats(options.run_id, output_dir, tasks)
    result_logger = CSVAppender(output_dir / 'low_level_ordinal_results.csv', RESULT_FIELDS)
    completed = read_completed_metric_keys(output_dir / 'low_level_ordinal_results.csv', options.run_id)
    embedder = ClipTextEmbedder()

    for task in tqdm(tasks, desc='low-level tasks'):
        max_budget = max(budgets)
        for method in tqdm(methods, desc=task.task_id, leave=False):
            reps = int(options.repetitions) if METHOD_SUPPORTS_FEEDBACK[method] else 1
            method_budgets = budgets if METHOD_SUPPORTS_FEEDBACK[method] else [0]
            for replicate in range(reps):
                needed = [(task.dataset, task.axis_field, method, replicate, budget) for budget in method_budgets]
                if all(key in completed for key in needed):
                    continue
                session = make_session(method, task, embedder, target_mode=target_mode, axisbayes_mode='gaussian')
                order = _selection_order(task, replicate, max_budget)
                label_count = 0
                if 0 in method_budgets:
                    key = (task.dataset, task.axis_field, method, replicate, 0)
                    if key not in completed:
                        _append_result(
                            result_logger,
                            run_id=options.run_id,
                            task=task,
                            method=method,
                            replicate=replicate,
                            budget=0,
                            label_count=label_count,
                            session=session,
                        )
                        completed.add(key)
                for step, idx in enumerate(order, start=1):
                    session.observe(int(idx), float(task.y01[int(idx)]))
                    label_count += 1
                    if step not in method_budgets:
                        continue
                    key = (task.dataset, task.axis_field, method, replicate, step)
                    if key in completed:
                        continue
                    _append_result(
                        result_logger,
                        run_id=options.run_id,
                        task=task,
                        method=method,
                        replicate=replicate,
                        budget=step,
                        label_count=label_count,
                        session=session,
                    )
                    completed.add(key)

    _, report_path, markdown_table = export_low_level_reports(output_dir, options.run_id)
    print(markdown_table)
    print(f'[low-level-ordinal] wrote report {report_path}')
    return output_dir
