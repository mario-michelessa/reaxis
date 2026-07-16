from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from tqdm import tqdm

from .axisbayes_mode_ablation import (
    AXISBAYES_MODE_DEBUG_FIELDS,
    export_axisbayes_mode_ablation_summary,
    make_axisbayes_mode_debug_row,
)
from .baselines import (
    METHOD_SUPPORTS_FEEDBACK,
    ClipTextEmbedder,
    axisbayes_mode_for_method,
    base_reaxis_method,
    is_axisbayes_reaxis_method,
    make_session,
    validate_axisbayes_mode,
)
from .category_reports import export_category_reports
from .config import ACTIVE_DATASETS, ACTIVE_METHODS, BUDGETS, GLOBAL_SEED, OUTPUT_DIR, REPETITIONS
from .data import OrdinalTask, load_tasks
from .diagnostics import DIAGNOSTIC_FIELDS, compute_reaxis_diagnostics, is_reaxis_session
from .fixed_reaxis import TARGET_MODE_GROUND_TRUTH, labeled_rank_mae, pairwise_agreement, validate_target_mode
from .io import CSVAppender, read_completed_metric_keys, utc_now_iso, write_json
from .metrics import compute_metrics, spearman_corr
from .plotting import export_fixed_reports, export_reports
from .target_mode_ablation import run_target_mode_ablation
from .y_debug import (
    DEBUG_Y_COMPARISON_FIELDS,
    DEBUG_Y_TRACE_FIELDS,
    export_y_debug_reports,
    make_debug_y_rows,
)


TASK_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'domain', 'axis_field', 'axis_name', 'query',
    'label_source', 'image_count', 'low_text', 'high_text', 'ladder_json',
]
METRIC_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'axis_field', 'axis_name', 'method',
    'policy', 'replicate', 'budget', 'label_count', 'spearman', 'kendall_tau',
    'mae_isotonic', 'qwk', 'axis_r2', 'monotonicity_violations',
]
BIN_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'axis_field', 'method', 'replicate',
    'budget', 'bin_index', 'bin_count', 'score_min', 'score_max',
    'mean_reference',
]
PRED_FIELDS = [
    'ts_utc', 'run_id', 'dataset', 'axis_field', 'method', 'replicate',
    'budget', 'image_id', 'image_path', 'reference01', 'prediction',
]
FIXED_RESULT_FIELDS = [
    *METRIC_FIELDS,
    'selected_lambda', 'selected_beta', 'score_preservation_spearman',
    'mu_d0_cos', 'direction_relative_change', 'labeled_pairwise_agreement',
    'labeled_mae', 'prior_spearman_b0',
]


@dataclass(frozen=True)
class StudyOptions:
    run_id: str
    output_dir: Path = OUTPUT_DIR
    datasets: Sequence[str] = ACTIVE_DATASETS
    methods: Sequence[str] = ACTIVE_METHODS
    budgets: Sequence[int] = BUDGETS
    repetitions: int = REPETITIONS
    max_tasks: int = 0
    write_predictions: bool = False
    prediction_limit: int = 0
    debug_y: bool = False
    target_mode: str = TARGET_MODE_GROUND_TRUTH
    axisbayes_mode: str = 'gaussian'
    target_mode_ablation: bool = False


def stable_seed(*parts: object) -> int:
    text = '||'.join(str(part) for part in parts)
    return int(zlib.crc32(text.encode('utf-8')) & 0xFFFFFFFF)


def _policy_for_method(method: str) -> str:
    if base_reaxis_method(method) == 'reaxis_active':
        return 'uncertainty'
    if base_reaxis_method(method) in {'reaxis_random', 'reaxis_log', 'reaxis_pairwise', 'reaxis_centered', 'reaxis_calibrated_residual'}:
        return 'random'
    return 'random'


def _selection_seed_part(method: str) -> str:
    return 'uncertainty' if base_reaxis_method(method) == 'reaxis_active' else 'random'


def _select_next_index(method: str, session, queried: np.ndarray, rng: np.random.Generator) -> int | None:
    available = np.flatnonzero(~queried)
    if available.size == 0:
        return None
    if base_reaxis_method(method) == 'reaxis_active':
        uncertainty = np.asarray(session.uncertainty(), dtype=np.float32)
        if uncertainty.shape[0] == queried.shape[0] and np.any(np.isfinite(uncertainty[available])):
            score = np.where(np.isfinite(uncertainty), uncertainty, -np.inf)
            score[queried] = -np.inf
            best = int(np.argmax(score))
            if np.isfinite(score[best]):
                return best
    return int(rng.choice(available))


def _append_predictions(
    logger: CSVAppender,
    *,
    run_id: str,
    task: OrdinalTask,
    method: str,
    replicate: int,
    budget: int,
    scores: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> None:
    indices = np.arange(len(task.ids), dtype=np.int64)
    if int(limit) > 0 and len(indices) > int(limit):
        indices = np.asarray(sorted(rng.choice(indices, size=int(limit), replace=False).tolist()), dtype=np.int64)
    for idx in indices.tolist():
        logger.append({
            'ts_utc': utc_now_iso(),
            'run_id': run_id,
            'dataset': task.dataset,
            'axis_field': task.axis_field,
            'method': method,
            'replicate': int(replicate),
            'budget': int(budget),
            'image_id': task.ids[int(idx)],
            'image_path': task.paths[int(idx)],
            'reference01': float(task.y01[int(idx)]),
            'prediction': float(scores[int(idx)]),
        })


def _emit_step(
    *,
    options: StudyOptions,
    task: OrdinalTask,
    method: str,
    replicate: int,
    budget: int,
    label_count: int,
    session,
    metrics_logger: CSVAppender,
    bins_logger: CSVAppender,
    diagnostics_logger: CSVAppender,
    fixed_results_logger: CSVAppender,
    predictions_logger: CSVAppender | None,
    axisbayes_mode_debug_logger: CSVAppender | None,
    completed_metrics: set[tuple[str, str, str, int, int]],
    completed_diagnostics: set[tuple[str, str, str, int, int]],
    completed_fixed: set[tuple[str, str, str, int, int]],
) -> None:
    key = (task.dataset, task.axis_field, method, int(replicate), int(budget))
    needs_metrics = key not in completed_metrics
    needs_diagnostics = is_reaxis_session(session) and key not in completed_diagnostics
    needs_fixed = key not in completed_fixed
    if not needs_metrics and not needs_diagnostics and not needs_fixed:
        return
    rng = np.random.default_rng(GLOBAL_SEED + stable_seed(task.task_id, method, replicate, budget))
    scores = np.asarray(session.scores(), dtype=np.float32)
    metrics, bin_rows = compute_metrics(scores, task.y01, rng)
    row_base = {
        'ts_utc': utc_now_iso(),
        'run_id': options.run_id,
        'dataset': task.dataset,
        'axis_field': task.axis_field,
        'axis_name': task.axis_name,
        'method': method,
        'policy': _policy_for_method(method),
        'replicate': int(replicate),
        'budget': int(budget),
        'label_count': int(label_count),
    }
    if needs_metrics:
        metrics_logger.append({**row_base, **metrics})
        for bin_row in bin_rows:
            bins_logger.append({
                'ts_utc': utc_now_iso(),
                'run_id': options.run_id,
                'dataset': task.dataset,
                'axis_field': task.axis_field,
                'method': method,
                'replicate': int(replicate),
                'budget': int(budget),
                **bin_row,
            })
    if needs_diagnostics:
        diagnostics_logger.append({
            **row_base,
            **compute_reaxis_diagnostics(
                task=task,
                session=session,
                scores=scores,
                global_spearman=float(metrics['spearman']),
            ),
        })
    if needs_fixed:
        fixed_results_logger.append({
            **row_base,
            **metrics,
            **_fixed_result_diagnostics(task=task, session=session, scores=scores, global_spearman=float(metrics['spearman'])),
        })
    if predictions_logger is not None and needs_metrics:
        _append_predictions(
            predictions_logger,
            run_id=options.run_id,
            task=task,
            method=method,
            replicate=replicate,
            budget=budget,
            scores=scores,
            limit=int(options.prediction_limit),
            rng=rng,
        )
    if axisbayes_mode_debug_logger is not None and is_axisbayes_reaxis_method(method):
        axisbayes_mode_debug_logger.append(make_axisbayes_mode_debug_row(
            ts_utc=utc_now_iso(),
            run_id=options.run_id,
            task=task,
            method=method,
            policy=_policy_for_method(method),
            replicate=replicate,
            budget=budget,
            label_count=label_count,
            session=session,
            scores=scores,
        ))
    if needs_metrics:
        completed_metrics.add(key)
    if needs_diagnostics:
        completed_diagnostics.add(key)
    if needs_fixed:
        completed_fixed.add(key)


def _fixed_result_diagnostics(*, task: OrdinalTask, session, scores: np.ndarray, global_spearman: float) -> dict[str, float]:
    if hasattr(session, 'fixed_diagnostics'):
        return dict(session.fixed_diagnostics(task.y01))
    if is_reaxis_session(session):
        diag = compute_reaxis_diagnostics(
            task=task,
            session=session,
            scores=scores,
            global_spearman=global_spearman,
        )
        state = session.state
        idx = [
            task.id_to_index[str(image_id)]
            for image_id in getattr(state, 'move_order', [])
            if str(image_id) in task.id_to_index
        ]
        labeled_scores = np.asarray(scores, dtype=np.float32)[idx] if idx else np.zeros((0,), dtype=np.float32)
        labeled_y = np.asarray(task.y01, dtype=np.float32)[idx] if idx else np.zeros((0,), dtype=np.float32)
        return {
            'selected_lambda': float('nan'),
            'selected_beta': float('nan'),
            'score_preservation_spearman': float(diag['score_preservation_spearman']),
            'mu_d0_cos': float(diag['mu_d0_cos']),
            'direction_relative_change': float(diag['mu_d0_rel_change']),
            'labeled_pairwise_agreement': pairwise_agreement(labeled_scores, labeled_y),
            'labeled_mae': labeled_rank_mae(scores, task.y01, idx),
            'prior_spearman_b0': spearman_corr(np.asarray(state.z0_all, dtype=np.float32), task.y01),
        }
    return {
        'selected_lambda': float('nan'),
        'selected_beta': float('nan'),
        'score_preservation_spearman': float('nan'),
        'mu_d0_cos': float('nan'),
        'direction_relative_change': float('nan'),
        'labeled_pairwise_agreement': float('nan'),
        'labeled_mae': float('nan'),
        'prior_spearman_b0': float('nan'),
    }


def _append_task_rows(task_logger: CSVAppender, run_id: str, tasks: Sequence[OrdinalTask]) -> None:
    for task in tasks:
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
            'low_text': task.low_text,
            'high_text': task.high_text,
            'ladder_json': json.dumps(list(task.ladder), separators=(',', ':')),
        })


def validate_methods(methods: Sequence[str]) -> List[str]:
    out = [str(method).strip() for method in methods if str(method).strip()]
    unknown = sorted(set(out) - set(METHOD_SUPPORTS_FEEDBACK))
    if unknown:
        raise RuntimeError(f'Unknown ordinal methods: {unknown}')
    return out


def run_study(options: StudyOptions) -> Path:
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = validate_methods(options.methods)
    budgets = sorted({int(b) for b in options.budgets})
    tasks = load_tasks(options.datasets)
    if int(options.max_tasks) > 0:
        tasks = tasks[: int(options.max_tasks)]
    if not tasks:
        raise RuntimeError('No ordinal tasks selected')

    write_json(output_dir / 'run_manifest.json', {
        'run_id': options.run_id,
        'created_utc': utc_now_iso(),
        'datasets': list(options.datasets),
        'methods': methods,
        'budgets': budgets,
        'repetitions': int(options.repetitions),
        'max_tasks': int(options.max_tasks),
        'write_predictions': bool(options.write_predictions),
        'prediction_limit': int(options.prediction_limit),
        'debug_y': bool(options.debug_y),
        'target_mode': validate_target_mode(options.target_mode),
        'axisbayes_mode': validate_axisbayes_mode(options.axisbayes_mode),
        'target_mode_ablation': bool(options.target_mode_ablation),
    })

    task_logger = CSVAppender(output_dir / 'tasks.csv', TASK_FIELDS)
    metrics_logger = CSVAppender(output_dir / 'metrics.csv', METRIC_FIELDS)
    bins_logger = CSVAppender(output_dir / 'axis_bins.csv', BIN_FIELDS)
    diagnostics_logger = CSVAppender(output_dir / 'diagnostics.csv', DIAGNOSTIC_FIELDS)
    fixed_results_logger = CSVAppender(output_dir / 'ordinal_eval_results_fixed.csv', FIXED_RESULT_FIELDS)
    predictions_logger = CSVAppender(output_dir / 'member_predictions.csv', PRED_FIELDS) if options.write_predictions else None
    axisbayes_mode_debug_logger = CSVAppender(output_dir / 'axisbayes_mode_debug.csv', AXISBAYES_MODE_DEBUG_FIELDS)
    debug_y_trace_logger = CSVAppender(output_dir / 'debug_y_trace.csv', DEBUG_Y_TRACE_FIELDS) if options.debug_y else None
    debug_y_compare_logger = CSVAppender(output_dir / 'debug_y_modeling_vs_ordinal.csv', DEBUG_Y_COMPARISON_FIELDS) if options.debug_y else None
    completed_metrics = read_completed_metric_keys(output_dir / 'metrics.csv', options.run_id)
    completed_diagnostics = read_completed_metric_keys(output_dir / 'diagnostics.csv', options.run_id)
    completed_fixed = read_completed_metric_keys(output_dir / 'ordinal_eval_results_fixed.csv', options.run_id)
    _append_task_rows(task_logger, options.run_id, tasks)

    embedder = ClipTextEmbedder()
    total = len(tasks) * len(methods)
    for task in tqdm(tasks, desc='ordinal tasks'):
        for method in tqdm(methods, desc=f'{task.task_id}', leave=False):
            reps = int(options.repetitions) if METHOD_SUPPORTS_FEEDBACK[method] else 1
            method_budgets = budgets if METHOD_SUPPORTS_FEEDBACK[method] else [0]
            for replicate in range(reps):
                needed = [
                    (task.dataset, task.axis_field, method, replicate, budget)
                    for budget in method_budgets
                ]
                if all(
                    key in completed_metrics
                    and (not is_axisbayes_reaxis_method(method) or key in completed_diagnostics)
                    and key in completed_fixed
                    for key in needed
                ):
                    continue
                seed = GLOBAL_SEED + stable_seed(task.task_id, _selection_seed_part(method), replicate)
                rng = np.random.default_rng(seed)
                session = make_session(
                    method,
                    task,
                    embedder,
                    target_mode=validate_target_mode(options.target_mode),
                    axisbayes_mode=axisbayes_mode_for_method(method, options.axisbayes_mode),
                )
                queried = np.zeros((len(task.ids),), dtype=bool)
                label_count = 0
                max_budget = max(method_budgets)
                if 0 in method_budgets:
                    _emit_step(
                        options=options,
                        task=task,
                        method=method,
                        replicate=replicate,
                        budget=0,
                        label_count=label_count,
                        session=session,
                        metrics_logger=metrics_logger,
                        bins_logger=bins_logger,
                        diagnostics_logger=diagnostics_logger,
                        fixed_results_logger=fixed_results_logger,
                        predictions_logger=predictions_logger,
                        axisbayes_mode_debug_logger=axisbayes_mode_debug_logger,
                        completed_metrics=completed_metrics,
                        completed_diagnostics=completed_diagnostics,
                        completed_fixed=completed_fixed,
                    )
                for step in range(1, max_budget + 1):
                    idx = _select_next_index(method, session, queried, rng)
                    if idx is None:
                        break
                    queried[int(idx)] = True
                    target01 = float(task.y01[int(idx)])
                    session.observe(int(idx), target01)
                    if debug_y_trace_logger is not None and debug_y_compare_logger is not None:
                        ts = utc_now_iso()
                        image_id = task.ids[int(idx)]
                        trace_row, compare_row = make_debug_y_rows(
                            ts_utc=ts,
                            run_id=options.run_id,
                            task=task,
                            method=method,
                            budget=step,
                            target_mode=validate_target_mode(options.target_mode),
                            selected_index=int(idx),
                            image_id=image_id,
                            feedback_before_transform=target01,
                            session=session,
                        )
                        debug_y_trace_logger.append(trace_row)
                        debug_y_compare_logger.append(compare_row)
                    label_count += 1
                    if step in method_budgets:
                        _emit_step(
                            options=options,
                            task=task,
                            method=method,
                            replicate=replicate,
                            budget=step,
                            label_count=label_count,
                            session=session,
                            metrics_logger=metrics_logger,
                            bins_logger=bins_logger,
                            diagnostics_logger=diagnostics_logger,
                            fixed_results_logger=fixed_results_logger,
                            predictions_logger=predictions_logger,
                            axisbayes_mode_debug_logger=axisbayes_mode_debug_logger,
                            completed_metrics=completed_metrics,
                            completed_diagnostics=completed_diagnostics,
                            completed_fixed=completed_fixed,
                        )
    export_reports(output_dir, run_id=options.run_id)
    category_report = export_category_reports(output_dir, run_id=options.run_id)
    if category_report is not None:
        print(f'[ordinal-study] wrote category report {category_report}')
    markdown_table = export_fixed_reports(output_dir, run_id=options.run_id)
    if markdown_table:
        print(markdown_table)
    if options.debug_y:
        export_y_debug_reports(output_dir, options.run_id)
    if options.target_mode_ablation:
        path = run_target_mode_ablation(
            run_id=options.run_id,
            output_dir=output_dir,
            datasets=options.datasets,
            budgets=budgets,
            repetitions=int(options.repetitions),
            max_tasks=int(options.max_tasks),
        )
        print(f'[target-mode] wrote {path}')
    axisbayes_table = export_axisbayes_mode_ablation_summary(
        output_dir,
        run_id=options.run_id,
        default_axisbayes_mode=validate_axisbayes_mode(options.axisbayes_mode),
    )
    if axisbayes_table:
        print(axisbayes_table)
    _ = total
    return output_dir
