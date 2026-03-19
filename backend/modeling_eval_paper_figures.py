#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELING_DIR = REPO_ROOT / 'backend' / 'experiments' / 'modeling_eval'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'backend' / 'figures' / 'modeling_eval_paper'
DEFAULT_BUMP_CONFIG = REPO_ROOT / 'backend' / 'modeling_eval_paper_bump_config.json'

METHOD_ORDER = [
    'clip_text_similarity',
    'prompt_ensemble',
    'label_only_linear',
    'knn_label_propagation',
    'request_bayes_linear_gaussian',
]

METHOD_LABELS = {
    'clip_text_similarity': 'CLIP text',
    'prompt_ensemble': 'Prompt ensemble',
    'label_only_linear': 'Label-only linear',
    'knn_label_propagation': 'kNN propagation',
    'request_bayes_linear_gaussian': 'ReQuest Bayesian Linear',
}

METHOD_COLORS = {
    'clip_text_similarity': '#4c78a8',
    'prompt_ensemble': '#72b7b2',
    'label_only_linear': '#f58518',
    'knn_label_propagation': '#54a24b',
    'request_bayes_linear_gaussian': '#e45756',
}

STATIC_METHODS = {
    'clip_text_similarity',
    'prompt_ensemble',
}

LINE_METRICS = [
    ('spearman_all', 'Mean Spearman'),
    ('auroc_binary_all', 'Mean Binary AUROC'),
]


def latest_run_id(modeling_dir: Path) -> str:
    runs = pd.read_csv(modeling_dir / 'runs.csv')
    if runs.empty:
        raise RuntimeError(f'No runs found in {modeling_dir}')
    if 'ts_utc' in runs.columns:
        runs = runs.sort_values('ts_utc', kind='mergesort')
    return str(runs.iloc[-1]['run_id'])


def load_frame(modeling_dir: Path, name: str, run_id: str) -> pd.DataFrame:
    frame = pd.read_csv(modeling_dir / f'{name}.csv')
    if 'run_id' in frame.columns:
        frame = frame[frame['run_id'].astype(str) == str(run_id)].copy()
    return frame


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    values = series.astype(str).str.strip().str.lower()
    return values.isin({'1', 'true', 't', 'yes', 'y'})


def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(text).strip().lower()).strip('_')


def save_figure(fig: Any, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'{name}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def load_steps(run_id: str) -> pd.DataFrame:
    steps = load_frame(MODELING_DIR, 'refinement', run_id)
    steps = steps[
        (steps['row_type'].astype(str) == 'step')
        & (steps['variant'].astype(str) == 'main')
        & (steps['policy'].astype(str) == 'hybrid')
        & (~coerce_bool_series(steps['allow_undefined']))
        & (steps['method'].isin(METHOD_ORDER))
    ].copy()
    for col, _ in LINE_METRICS:
        steps[col] = pd.to_numeric(steps[col], errors='coerce')
    steps['interaction_count'] = pd.to_numeric(steps['interaction_count'], errors='coerce')
    return steps


def plot_interaction_curves(run_id: str, output_dir: Path) -> None:
    steps = load_steps(run_id)
    if steps.empty:
        raise RuntimeError(f'No refinement data found for run {run_id}')

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharex=True)
    full_interactions = sorted(steps['interaction_count'].dropna().astype(int).unique().tolist())

    for ax, (metric_col, metric_label) in zip(axes, LINE_METRICS):
        metric_steps = steps.dropna(subset=[metric_col]).copy()
        for method in METHOD_ORDER:
            method_frame = metric_steps[metric_steps['method'].astype(str) == method].copy()
            if method_frame.empty:
                continue
            grouped = (
                method_frame.groupby('interaction_count', dropna=False)[metric_col]
                .agg(['mean', 'std', 'count'])
                .reset_index()
                .sort_values('interaction_count', kind='mergesort')
            )
            if method in STATIC_METHODS and grouped.shape[0] == 1 and full_interactions:
                row = grouped.iloc[0].to_dict()
                grouped = pd.DataFrame(
                    [
                        {
                            'interaction_count': float(interaction),
                            'mean': row['mean'],
                            'std': row['std'],
                            'count': row['count'],
                        }
                        for interaction in full_interactions
                    ]
                )
            grouped['sem'] = grouped['std'].fillna(0.0) / np.sqrt(grouped['count'].clip(lower=1))
            xs = grouped['interaction_count'].to_numpy(dtype=float)
            ys = grouped['mean'].to_numpy(dtype=float)
            ci = 1.96 * grouped['sem'].to_numpy(dtype=float)
            color = METHOD_COLORS[method]
            ax.plot(
                xs,
                ys,
                color=color,
                marker='o',
                linewidth=2.3,
                markersize=4.5,
                linestyle='--' if method in STATIC_METHODS else '-',
                label=METHOD_LABELS[method],
            )
            if np.isfinite(ci).any():
                ax.fill_between(xs, ys - ci, ys + ci, color=color, alpha=0.12)
        ax.set_title(metric_label)
        ax.set_xlabel('Interaction count')
        ax.set_ylabel(metric_label)
        ax.grid(axis='y', alpha=0.25)
        ax.set_xticks(full_interactions)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=min(3, len(handles)), frameon=False, bbox_to_anchor=(0.5, 1.06))
    save_figure(fig, output_dir, 'interaction_curves')


def resolve_latest_interaction(frame: pd.DataFrame, task_id: str, method: str, requested: Any) -> int:
    sub = frame[
        (frame['task_id'].astype(str) == str(task_id))
        & (frame['method'].astype(str) == str(method))
    ].copy()
    if sub.empty:
        raise KeyError(f'No representative rows for task={task_id} method={method}')
    sub['interaction_count'] = pd.to_numeric(sub['interaction_count'], errors='coerce')
    available = sorted(int(v) for v in sub['interaction_count'].dropna().unique().tolist())
    if not available:
        raise RuntimeError(f'No interaction counts available for task={task_id} method={method}')
    if str(requested).strip().lower() == 'latest':
        return available[-1]
    value = int(requested)
    if value in available:
        return value
    lower = [item for item in available if item <= value]
    if lower:
        return lower[-1]
    return available[0]


def placeholder_thumbnail(size: int = 64) -> np.ndarray:
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    canvas[[0, -1], :, :] = 210
    canvas[:, [0, -1], :] = 210
    return canvas


def load_thumbnail(path_str: str, size: int = 64) -> np.ndarray:
    path = Path(str(path_str))
    if not path.exists():
        return placeholder_thumbnail(size)
    try:
        image = Image.open(path).convert('RGB')
        image.thumbnail((size, size))
        canvas = Image.new('RGB', (size, size), color=(248, 248, 248))
        ox = int((size - image.width) * 0.5)
        oy = int((size - image.height) * 0.5)
        canvas.paste(image, (ox, oy))
        return np.asarray(canvas)
    except Exception:
        return placeholder_thumbnail(size)


def prepare_panel_records(
    representative: pd.DataFrame,
    panel_cfg: Mapping[str, Any],
    method_cfgs: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    task_id = str(panel_cfg['task_id'])
    selection_method = str(panel_cfg.get('selection_method', 'request_bayes_linear_gaussian'))
    selection_interaction = panel_cfg.get('selection_interaction_count', 'latest')
    selection_count = resolve_latest_interaction(representative, task_id, selection_method, selection_interaction)
    source = representative[
        (representative['task_id'].astype(str) == task_id)
        & (representative['method'].astype(str) == selection_method)
        & (pd.to_numeric(representative['interaction_count'], errors='coerce') == float(selection_count))
    ].copy()
    if source.empty:
        raise RuntimeError(f'No source rows found for task={task_id} selection_method={selection_method}')

    source['example_slot'] = pd.to_numeric(source['example_slot'], errors='coerce')
    source['reference_rank_sample'] = pd.to_numeric(source['reference_rank_sample'], errors='coerce')
    source = source.sort_values(['example_slot', 'image_id'], kind='mergesort')

    if panel_cfg.get('image_ids'):
        image_order = [str(item) for item in panel_cfg['image_ids']]
    else:
        slot_order = [int(item) for item in panel_cfg.get('example_slots', list(range(1, 11)))]
        slot_map = {slot: idx for idx, slot in enumerate(slot_order)}
        picked = source[source['example_slot'].astype(int).isin(slot_order)].copy()
        picked['slot_order'] = picked['example_slot'].astype(int).map(slot_map)
        picked = picked.sort_values(['slot_order', 'example_slot'], kind='mergesort')
        image_order = picked['image_id'].astype(str).tolist()
    image_order = list(dict.fromkeys(image_order))
    if not image_order:
        raise RuntimeError(f'No images selected for task={task_id}')

    representative = representative.copy()
    representative['pred_rank_sample'] = pd.to_numeric(representative['pred_rank_sample'], errors='coerce')
    representative['reference_rank_sample'] = pd.to_numeric(representative['reference_rank_sample'], errors='coerce')

    image_meta: Dict[str, Dict[str, Any]] = {}
    for image_id in image_order:
        row = source[source['image_id'].astype(str) == str(image_id)]
        if row.empty:
            continue
        first = row.iloc[0]
        image_meta[str(image_id)] = {
            'image_id': str(image_id),
            'image_path': str(first['image_path']),
            'reference_rank_sample': float(first['reference_rank_sample']),
        }

    resolved_methods: List[Dict[str, Any]] = []
    for item in method_cfgs:
        method = str(item['name'])
        resolved_count = resolve_latest_interaction(representative, task_id, method, item.get('interaction_count', 'latest'))
        resolved_methods.append(
            {
                'name': method,
                'label': str(item.get('label', METHOD_LABELS.get(method, method))),
                'interaction_count': resolved_count,
            }
        )

    rows_by_method: Dict[str, pd.DataFrame] = {}
    for method_spec in resolved_methods:
        method = method_spec['name']
        count = method_spec['interaction_count']
        sub = representative[
            (representative['task_id'].astype(str) == task_id)
            & (representative['method'].astype(str) == method)
            & (pd.to_numeric(representative['interaction_count'], errors='coerce') == float(count))
            & (representative['image_id'].astype(str).isin(image_order))
        ].copy()
        rows_by_method[method] = sub

    records: List[Dict[str, Any]] = []
    for image_id in image_order:
        if image_id not in image_meta:
            continue
        record = dict(image_meta[image_id])
        method_ranks: Dict[str, float] = {}
        missing = False
        for method_spec in resolved_methods:
            sub = rows_by_method[method_spec['name']]
            row = sub[sub['image_id'].astype(str) == image_id]
            if row.empty or not np.isfinite(float(row.iloc[0]['pred_rank_sample'])):
                missing = True
                break
            method_ranks[method_spec['name']] = float(row.iloc[0]['pred_rank_sample'])
        if missing:
            continue
        record['method_ranks'] = method_ranks
        records.append(record)

    records.sort(key=lambda item: float(item['reference_rank_sample']))
    return records, resolved_methods


def plot_bump_charts(run_id: str, output_dir: Path, config_path: Path) -> None:
    representative = load_frame(MODELING_DIR, 'representative_rankings', run_id)
    if representative.empty:
        raise RuntimeError(f'No representative rankings found for run {run_id}')

    config = json.loads(config_path.read_text(encoding='utf-8'))
    method_cfgs = list(config.get('methods', []))
    panels = list(config.get('panels', []))
    if not method_cfgs or not panels:
        raise RuntimeError(f'Invalid bump config at {config_path}')

    n_panels = len(panels)
    n_cols = min(3, n_panels)
    n_rows = int(math.ceil(n_panels / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 7.2 * n_rows), sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for panel_index, panel_cfg in enumerate(panels):
        ax = axes[panel_index // n_cols][panel_index % n_cols]
        ax.set_visible(True)
        records, resolved_methods = prepare_panel_records(representative, panel_cfg, method_cfgs)
        x_positions = np.arange(len(resolved_methods), dtype=float)
        line_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(10, len(records))))[: len(records)]

        for idx, record in enumerate(records):
            ys = [record['method_ranks'][spec['name']] for spec in resolved_methods]
            ax.plot(x_positions, ys, color=line_colors[idx], linewidth=2.0, marker='o', markersize=4.5, alpha=0.82, zorder=2)
            thumb = load_thumbnail(record['image_path'], size=72)
            artist = AnnotationBbox(
                OffsetImage(thumb, zoom=0.46),
                (-0.72, float(record['reference_rank_sample'])),
                frameon=False,
                box_alignment=(1.0, 0.5),
                zorder=3,
            )
            ax.add_artist(artist)

        ax.set_title(str(panel_cfg.get('title', panel_cfg['task_id'])))
        ax.set_xlim(-0.95, len(resolved_methods) - 0.15)
        max_rank = max(float(record['reference_rank_sample']) for record in records) if records else 20.0
        ax.set_ylim(max_rank + 0.75, 0.25)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [f"{spec['label']}\n({spec['interaction_count']})" for spec in resolved_methods],
            rotation=20,
            ha='right',
        )
        ax.set_yticks(sorted({1, 5, 10, 15, 20} | {int(round(float(item['reference_rank_sample']))) for item in records}))
        ax.set_ylabel('Sample rank (1 = most positive)')
        ax.grid(axis='y', alpha=0.22)
        ax.axvline(-0.46, color='#cbd5e1', linewidth=1.0, linestyle='--', zorder=1)
        ax.text(-0.78, 0.7, 'Reference', fontsize=9, rotation=90, va='bottom', ha='center')

    save_figure(fig, output_dir, 'qualitative_bump_charts')


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate paper-specific modeling-study figures.')
    parser.add_argument('--run-id', default='', help='Optional modeling study run id. Defaults to the latest run.')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Where to write figure files.')
    parser.add_argument('--bump-config', default=str(DEFAULT_BUMP_CONFIG), help='Path to the qualitative bump-chart config JSON.')
    parser.add_argument(
        '--figure',
        action='append',
        choices=['curves', 'bump', 'all'],
        help='Which figure(s) to generate. Defaults to all.',
    )
    args = parser.parse_args()

    requested = args.figure or ['all']
    figure_kinds = {'curves', 'bump'} if 'all' in requested else set(requested)
    run_id = str(args.run_id).strip() or latest_run_id(MODELING_DIR)
    output_dir = Path(args.output_dir)
    bump_config = Path(args.bump_config)

    if 'curves' in figure_kinds:
        plot_interaction_curves(run_id, output_dir)
    if 'bump' in figure_kinds:
        plot_bump_charts(run_id, output_dir, bump_config)

    print(json.dumps({'run_id': run_id, 'output_dir': str(output_dir), 'figures': sorted(figure_kinds)}, indent=2))


if __name__ == '__main__':
    main()
