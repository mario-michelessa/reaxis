#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

try:
    from . import modeling_evaluation as me
except ImportError:
    import modeling_evaluation as me


DEFAULT_SECTIONS = ('prior', 'refinement', 'uncertainty', 'representative_rankings')
SECTION_ALIASES = {
    'representative': 'representative_rankings',
    'representative_rankings': 'representative_rankings',
    'prior': 'prior',
    'refinement': 'refinement',
    'uncertainty': 'uncertainty',
    'query_policy': 'query_policy',
    'undefined': 'undefined',
    'ablation': 'ablation',
}


class RowCollector:
    def __init__(self, fieldnames: Sequence[str]):
        self.fieldnames = list(fieldnames)
        self.rows: List[Dict[str, Any]] = []

    def append(self, row: Mapping[str, Any]) -> None:
        self.rows.append({key: row.get(key, '') for key in self.fieldnames})


def parse_json_list(raw: Any) -> List[Any]:
    text = str(raw or '').strip()
    if not text:
        return []
    value = json.loads(text)
    return list(value) if isinstance(value, list) else []


def parse_json_dict(raw: Any) -> Dict[str, Any]:
    text = str(raw or '').strip()
    if not text:
        return {}
    value = json.loads(text)
    return dict(value) if isinstance(value, dict) else {}


def normalize_sections(raw_sections: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in raw_sections:
        for token in str(raw or '').split(','):
            key = SECTION_ALIASES.get(str(token).strip().lower())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out or list(DEFAULT_SECTIONS)


def load_run_row(output_dir: Path, run_id: str) -> Dict[str, str]:
    path = Path(output_dir) / 'runs.csv'
    if not path.exists():
        raise FileNotFoundError(f'No runs.csv found at {path}')
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get('run_id', '')).strip() == str(run_id).strip():
                return {str(key): str(value or '') for key, value in row.items()}
    raise KeyError(f'Run id not found in {path}: {run_id}')


def resolve_method_specs(names: Sequence[str], payload_json: str, registry: Mapping[str, me.MethodSpec]) -> List[me.MethodSpec]:
    payloads = parse_json_dict(payload_json)
    specs: List[me.MethodSpec] = []
    for name in names:
        if name not in registry:
            raise KeyError(f'Unknown method in run config: {name}')
        base = registry[name]
        override = dict(payloads.get(name) or {})
        specs.append(
            replace(
                base,
                name=str(override.get('name') or base.name),
                kind=str(override.get('kind') or base.kind),
                model_type=str(override.get('model_type') or base.model_type),
                mode=str(override.get('mode') or base.mode),
                feature_space=str(override.get('feature_space') or base.feature_space),
                clip_weight=float(override.get('clip_weight', base.clip_weight)),
                dino_weight=float(override.get('dino_weight', base.dino_weight)),
                params=dict(override.get('params') or base.params),
            )
        )
    return specs


def load_dataset_tasks(dataset_names: Sequence[str]) -> List[me.EvaluationTask]:
    tasks: List[me.EvaluationTask] = []
    for dataset_name in dataset_names:
        if dataset_name not in me.DATASET_REGISTRY:
            raise KeyError(f'Unknown dataset: {dataset_name}')
        spec = me.DATASET_REGISTRY[dataset_name]
        loaded = me.load_dataset(spec, require_dino=False)
        tasks.extend(me.build_tasks_for_dataset(loaded, spec))
    if not tasks:
        raise RuntimeError(f'No evaluation tasks built for datasets={list(dataset_names)}')
    return tasks


def apply_prompt_cache(tasks: Sequence[me.EvaluationTask]) -> None:
    prompt_records = me.hydrate_prompt_cache(
        tasks,
        cache_path=me.CONTRASTIVE_PROMPT_CACHE_PATH,
        source=me.CONTRASTIVE_PROMPT_SOURCE,
        n_prompts=me.CONTRASTIVE_PROMPT_COUNT,
        regenerate=me.CONTRASTIVE_PROMPT_REGENERATE,
        max_retries=me.CONTRASTIVE_PROMPT_MAX_RETRIES,
        retry_wait_sec=me.CONTRASTIVE_PROMPT_RETRY_WAIT_SEC,
        logger=print,
    )
    me.apply_prompt_cache_to_tasks(tasks, prompt_records)


def collect_rows(
    *,
    run_id: str,
    tasks: Sequence[me.EvaluationTask],
    method_specs: Sequence[me.MethodSpec],
    ablation_specs: Sequence[me.MethodSpec],
    budgets: Sequence[int],
    main_policy: str,
    sections: Sequence[str],
    representative_tasks_per_dataset: int,
    representative_images_per_task: int,
) -> Dict[str, List[Dict[str, Any]]]:
    selected_sections = set(sections)
    collectors = {
        name: RowCollector(me.TABLE_FIELDNAMES[name])
        for name in ['tasks', *selected_sections]
    }
    for task in tasks:
        me.append_task_row(collectors['tasks'], run_id, task)

    context = me.EvaluationContext(tasks)
    if 'prior' in selected_sections:
        me.evaluate_prior_quality(run_id, tasks, method_specs, context, collectors['prior'])

    if 'refinement' in selected_sections:
        representative_task_id_set = set(me.select_representative_task_ids(tasks, representative_tasks_per_dataset))
        for task in tasks:
            representative_indices: Optional[Sequence[int]] = None
            if 'representative_rankings' in selected_sections and task.task_id in representative_task_id_set:
                representative_indices = me.select_representative_example_indices(task, representative_images_per_task)
            for method in method_specs:
                me.run_refinement_loop(
                    run_id=run_id,
                    task=task,
                    method=method,
                    context=context,
                    logger=collectors['refinement'],
                    uncertainty_logger=collectors.get('uncertainty'),
                    representative_logger=collectors.get('representative_rankings') if representative_indices is not None else None,
                    budgets=budgets,
                    policy=main_policy,
                    allow_undefined=False,
                    row_prefix='main',
                    representative_example_indices=representative_indices,
                )

    if 'query_policy' in selected_sections:
        me.evaluate_query_policies(run_id, tasks, method_specs, context, collectors['query_policy'])
    if 'undefined' in selected_sections:
        me.evaluate_undefined(run_id, tasks, method_specs, context, collectors['undefined'])
    if 'ablation' in selected_sections and ablation_specs:
        me.evaluate_ablations(run_id, tasks, ablation_specs, context, collectors['ablation'])

    return {name: collector.rows for name, collector in collectors.items()}


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return [{str(key): str(value or '') for key, value in row.items()} for row in reader]


def write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def merge_dataset_rows(
    output_dir: Path,
    table_name: str,
    *,
    run_id: str,
    dataset_names: Sequence[str],
    new_rows: Sequence[Mapping[str, Any]],
) -> None:
    path = Path(output_dir) / f'{table_name}.csv'
    fieldnames = me.TABLE_FIELDNAMES[table_name]
    dataset_keys = {me.normalize_dataset_name(name).lower() for name in dataset_names}
    kept_rows: List[Mapping[str, Any]] = []
    for row in read_rows(path):
        row_run_id = str(row.get('run_id', '')).strip()
        row_dataset = me.normalize_dataset_name(row.get('dataset', '')).lower()
        if row_run_id == str(run_id).strip() and row_dataset in dataset_keys:
            continue
        kept_rows.append(row)
    merged = [*kept_rows, *new_rows]
    write_rows(path, fieldnames, merged)


def update_run_metadata(
    output_dir: Path,
    *,
    run_id: str,
    dataset_names: Sequence[str],
    new_representative_task_ids: Sequence[str],
) -> None:
    path = Path(output_dir) / 'runs.csv'
    rows = read_rows(path)
    if not rows:
        raise RuntimeError(f'No run rows found in {path}')
    updated: List[Dict[str, Any]] = []
    found = False
    for row in rows:
        if str(row.get('run_id', '')).strip() != str(run_id).strip():
            updated.append(row)
            continue
        found = True
        active_datasets = parse_json_list(row.get('active_datasets_json', ''))
        active_set = list(dict.fromkeys([*active_datasets, *dataset_names]))
        representative_ids = parse_json_list(row.get('representative_task_ids_json', ''))
        representative_set = list(dict.fromkeys([*representative_ids, *new_representative_task_ids]))
        next_row = dict(row)
        next_row['active_datasets_json'] = json.dumps(active_set, separators=(',', ':'))
        next_row['representative_task_ids_json'] = json.dumps(representative_set, separators=(',', ':'))
        updated.append(next_row)
    if not found:
        raise KeyError(f'Run id not found in {path}: {run_id}')
    write_rows(path, me.TABLE_FIELDNAMES['runs'], updated)


def modeling_process_active() -> bool:
    result = subprocess.run(
        ['ps', '-eo', 'cmd'],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    lines = [str(line).strip() for line in result.stdout.splitlines()]
    return any('backend/modeling_evaluation.py' in line for line in lines)


def guard_live_run(output_dir: Path, run_id: str, force: bool) -> None:
    default_dir = Path(me.OUTPUT_DIR).resolve()
    target_dir = Path(output_dir).resolve()
    if force or target_dir != default_dir:
        return
    latest_run = me.latest_modeling_run_id(output_dir=target_dir)
    if str(latest_run).strip() != str(run_id).strip():
        return
    if modeling_process_active():
        raise RuntimeError(
            f'Run {run_id} appears to be live in {target_dir}. '
            'Refusing to rewrite active modeling outputs. Re-run after it finishes or pass --force-live-run.'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Add one or more datasets to an existing modeling-eval run and regenerate figures.')
    parser.add_argument('--run-id', default='', help='Target run id. Defaults to the latest run in the output directory.')
    parser.add_argument('--datasets', nargs='+', default=['HAM10000'], help='Datasets to recompute and merge.')
    parser.add_argument(
        '--sections',
        nargs='*',
        default=list(DEFAULT_SECTIONS),
        help='Subset of result tables to recompute. Default: prior refinement uncertainty representative_rankings',
    )
    parser.add_argument('--output-dir', default=str(me.OUTPUT_DIR), help='Modeling output directory to patch.')
    parser.add_argument('--representative-tasks-per-dataset', type=int, default=me.REPRESENTATIVE_TASKS_PER_DATASET)
    parser.add_argument('--representative-images-per-task', type=int, default=me.REPRESENTATIVE_IMAGES_PER_TASK)
    parser.add_argument('--skip-figures', action='store_true', help='Skip figure regeneration after merging rows.')
    parser.add_argument('--force-live-run', action='store_true', help='Allow rewriting the latest run even if modeling_evaluation.py is currently running.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_id = str(args.run_id).strip() or me.latest_modeling_run_id(output_dir=output_dir)
    sections = normalize_sections(args.sections)
    datasets = [str(name).strip() for name in args.datasets if str(name).strip()]
    if not datasets:
        raise RuntimeError('Provide at least one dataset name.')
    guard_live_run(output_dir, run_id, bool(args.force_live_run))

    run_row = load_run_row(output_dir, run_id)
    active_method_names = parse_json_list(run_row.get('active_methods_json', ''))
    active_ablation_names = parse_json_list(run_row.get('active_ablations_json', ''))
    method_specs = resolve_method_specs(active_method_names, run_row.get('method_specs_json', ''), me.METHOD_REGISTRY)
    ablation_specs = resolve_method_specs(active_ablation_names, run_row.get('ablation_specs_json', ''), me.ABLATION_REGISTRY)
    dino_methods = [spec.name for spec in [*method_specs, *ablation_specs] if me.method_uses_dino(spec)]
    if dino_methods:
        raise RuntimeError(f'Incremental dataset recompute expects CLIP-only configs, but these methods still require DINO: {sorted(dino_methods)}')

    budgets = [int(value) for value in parse_json_list(run_row.get('budgets_json', ''))] or list(me.REFINEMENT_BUDGETS)
    main_policy = str(run_row.get('main_policy') or me.MAIN_QUERY_POLICY).strip() or me.MAIN_QUERY_POLICY

    tasks = load_dataset_tasks(datasets)
    apply_prompt_cache(tasks)
    rows_by_table = collect_rows(
        run_id=run_id,
        tasks=tasks,
        method_specs=method_specs,
        ablation_specs=ablation_specs,
        budgets=budgets,
        main_policy=main_policy,
        sections=sections,
        representative_tasks_per_dataset=int(args.representative_tasks_per_dataset),
        representative_images_per_task=int(args.representative_images_per_task),
    )

    merge_dataset_rows(output_dir, 'tasks', run_id=run_id, dataset_names=datasets, new_rows=rows_by_table['tasks'])
    for table_name in sections:
        merge_dataset_rows(output_dir, table_name, run_id=run_id, dataset_names=datasets, new_rows=rows_by_table.get(table_name, []))

    new_representative_ids = me.select_representative_task_ids(tasks, int(args.representative_tasks_per_dataset))
    update_run_metadata(output_dir, run_id=run_id, dataset_names=datasets, new_representative_task_ids=new_representative_ids)

    figure_dir = None
    if not args.skip_figures:
        figure_dir = me.export_modeling_eval_figures(run_id, output_dir=output_dir)

    merged_counts = {name: len(rows) for name, rows in rows_by_table.items()}
    payload = {
        'run_id': run_id,
        'datasets': datasets,
        'sections': sections,
        'output_dir': str(output_dir),
        'figure_dir': str(figure_dir) if figure_dir is not None else '',
        'merged_counts': merged_counts,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
