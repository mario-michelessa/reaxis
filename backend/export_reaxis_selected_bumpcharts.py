#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from . import modeling_evaluation as me
except ImportError:
    import modeling_evaluation as me


DEFAULT_METHOD_NAME = 'request_bayes_linear_gaussian'
DEFAULT_POLICY = me.MAIN_QUERY_POLICY
DEFAULT_BUDGETS = (0, 5, 20)
DEFAULT_SAMPLE_SIZE = 20
EXPORT_IMAGE_SIZE_PX = 256


@dataclass(frozen=True)
class TargetSpec:
    dataset_name: str
    task_id: str
    display_name: str
    slug: str
    note: str = ''


DEFAULT_TARGETS: Sequence[TargetSpec] = (
    TargetSpec(
        dataset_name='paintings_wikiart',
        task_id='paintings_wikiart__genre__abstract_painting',
        display_name='Abstract painting',
        slug='paintings_abstract',
    ),
    TargetSpec(
        dataset_name='HAM10000',
        task_id='ham10000__dx__mel',
        display_name='Melanoma',
        slug='ham10000_melanoma',
    ),
    TargetSpec(
        dataset_name='celeba_dataset',
        task_id='celeba_dataset__binary_prefix__attr_bangs',
        display_name='Bangs',
        slug='celeba_bangs_proxy',
        note='Proxy for hair length; CelebA does not include a direct hair-length attribute.',
    ),
)


class MemoryAppender:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def append(self, row: Mapping[str, Any]) -> None:
        self.rows.append(dict(row))


def load_target_tasks(targets: Sequence[TargetSpec]) -> List[me.EvaluationTask]:
    by_dataset: Dict[str, List[TargetSpec]] = {}
    for target in targets:
        by_dataset.setdefault(target.dataset_name, []).append(target)
    task_ids = {target.task_id for target in targets}
    selected: List[me.EvaluationTask] = []
    for dataset_name in sorted(by_dataset):
        if dataset_name not in me.DATASET_REGISTRY:
            raise KeyError(f'Unknown dataset: {dataset_name}')
        spec = me.DATASET_REGISTRY[dataset_name]
        loaded = me.load_dataset(spec, require_dino=False)
        tasks = me.build_tasks_for_dataset(loaded, spec)
        selected.extend(task for task in tasks if task.task_id in task_ids)
    if len(selected) != len(task_ids):
        found = {task.task_id for task in selected}
        missing = sorted(task_ids.difference(found))
        raise RuntimeError(f'Missing requested tasks: {missing}')
    prompt_records = me.hydrate_prompt_cache(
        selected,
        cache_path=me.CONTRASTIVE_PROMPT_CACHE_PATH,
        source=me.CONTRASTIVE_PROMPT_SOURCE,
        n_prompts=me.CONTRASTIVE_PROMPT_COUNT,
        regenerate=False,
        max_retries=me.CONTRASTIVE_PROMPT_MAX_RETRIES,
        retry_wait_sec=me.CONTRASTIVE_PROMPT_RETRY_WAIT_SEC,
        logger=print,
    )
    me.apply_prompt_cache_to_tasks(selected, prompt_records)
    return selected


def sample_example_indices(
    task: me.EvaluationTask,
    sample_size: int,
    *,
    sample_variant: int = 0,
) -> np.ndarray:
    count = min(int(sample_size), int(len(task.ids)))
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    seed = me.GLOBAL_SEED + (
        me.stable_seed(task.task_id, 'random_sample', count, int(sample_variant)) % 1_000_000
    )
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(task.ids), size=count, replace=False)
    return np.asarray(np.sort(picks), dtype=np.int64)


def collect_reaxis_rows(
    *,
    task: me.EvaluationTask,
    method: me.MethodSpec,
    context: me.EvaluationContext,
    budgets: Sequence[int],
    policy: str,
    sample_indices: Sequence[int],
) -> pd.DataFrame:
    budget_set = {int(value) for value in budgets}
    max_budget = max(budget_set) if budget_set else 0
    seed = me.GLOBAL_SEED + (me.stable_seed(task.task_id, method.name, policy, 'selected_bumpchart') % 1_000_000)
    rng = np.random.default_rng(seed)
    session = context.create_session(method, task, seed)
    collector = MemoryAppender()

    def capture(interaction_count: int) -> None:
        me.append_representative_ranking_rows(
            collector,
            run_id='selected_bumpchart',
            task=task,
            method=method,
            session=session,
            policy=policy,
            variant='selected_bumpchart',
            interaction_count=int(interaction_count),
            example_indices=sample_indices,
        )

    if 0 in budget_set:
        capture(0)
    for interaction_count in range(1, max_budget + 1):
        query_idx = me.select_query_index(policy, session, task, rng)
        if query_idx is None:
            break
        query_id = str(task.ids[int(query_idx)])
        session.observe(
            query_id,
            float(task.reference_scores01[int(query_idx)]),
            move_type='score',
        )
        if interaction_count in budget_set:
            capture(interaction_count)
    return pd.DataFrame(collector.rows)


def render_thumbnail(image_path: str, cell_w: int, cell_h: int) -> Image.Image:
    thumb = Image.new('RGB', (cell_w, cell_h), color=(245, 245, 245))
    try:
        with Image.open(str(image_path)) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
            fitted = ImageOps.pad(
                img,
                (cell_w, cell_h),
                method=Image.Resampling.LANCZOS,
                color=(245, 245, 245),
            )
            thumb.paste(fitted, (0, 0))
    except Exception:
        pass
    return thumb


def save_contact_sheet(frame: pd.DataFrame, output_pdf: Path, title: str) -> None:
    rows = frame.sort_values('example_slot', kind='mergesort')
    count = int(len(rows))
    cols = min(6, max(1, count))
    rows_n = int(math.ceil(count / float(cols)))
    cell_w = EXPORT_IMAGE_SIZE_PX
    cell_h = EXPORT_IMAGE_SIZE_PX
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.55, rows_n * 2.95), squeeze=False)
    flat_axes = axes.flatten()
    for ax in flat_axes:
        ax.axis('off')
    for row_idx, (_, item) in enumerate(rows.iterrows()):
        ax = flat_axes[row_idx]
        thumb = render_thumbnail(str(item['image_path']), cell_w, cell_h)
        ax.imshow(np.asarray(thumb))
        ax.set_title(
            f"{int(item['example_slot'])}: {Path(str(item['image_id'])).name}"[:28],
            fontsize=7,
            pad=4,
        )
        ax.axis('off')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)


def save_bump_chart(
    frame: pd.DataFrame,
    *,
    output_pdf: Path,
    title: str,
    subtitle: str,
    budgets: Sequence[int],
) -> None:
    plot_df = frame.copy()
    plot_df['interaction_count'] = pd.to_numeric(plot_df['interaction_count'], errors='coerce')
    plot_df['example_slot'] = pd.to_numeric(plot_df['example_slot'], errors='coerce')
    plot_df['pred_rank_sample'] = pd.to_numeric(plot_df['pred_rank_sample'], errors='coerce')
    plot_df['reference_score'] = pd.to_numeric(plot_df['reference_score'], errors='coerce')
    budget_list = [int(value) for value in budgets]
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    slots = sorted(plot_df['example_slot'].dropna().astype(int).unique().tolist())
    x_positions = list(range(len(budget_list)))
    for slot in slots:
        slot_rows = (
            plot_df[plot_df['example_slot'].astype(int) == int(slot)]
            .drop_duplicates(subset=['interaction_count'], keep='first')
            .set_index('interaction_count')
            .reindex(budget_list)
            .reset_index()
        )
        if slot_rows['pred_rank_sample'].isna().any():
            continue
        ref_score = float(slot_rows['reference_score'].dropna().iloc[0]) if slot_rows['reference_score'].notna().any() else 0.5
        color = cmap(norm(ref_score))
        ax.plot(
            x_positions,
            slot_rows['pred_rank_sample'].to_numpy(dtype=float),
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )
        ax.text(
            -0.08,
            float(slot_rows['pred_rank_sample'].iloc[0]),
            str(int(slot)),
            fontsize=6,
            ha='right',
            va='center',
            color=color,
        )
    ax.set_title(f'{title}\n{subtitle}', fontsize=12)
    ax.set_xlabel('Interaction count')
    ax.set_ylabel(f'Rank within {len(slots)} sampled images')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(value) for value in budget_list])
    ax.invert_yaxis()
    ax.grid(axis='y', alpha=0.18)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label('Reference score', rotation=90)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)


def save_rank_grid(
    frame: pd.DataFrame,
    *,
    output_pdf: Path,
    title: str,
    subtitle: str,
    budgets: Sequence[int],
) -> None:
    rank_df = frame.copy()
    rank_df['interaction_count'] = pd.to_numeric(rank_df['interaction_count'], errors='coerce')
    rank_df['pred_rank_sample'] = pd.to_numeric(rank_df['pred_rank_sample'], errors='coerce')
    budget_list = [int(value) for value in budgets]
    col_count = max(1, int(rank_df['example_slot'].dropna().shape[0] / max(1, len(budget_list))))
    cell_w = EXPORT_IMAGE_SIZE_PX
    cell_h = EXPORT_IMAGE_SIZE_PX
    fig, axes = plt.subplots(
        len(budget_list),
        col_count,
        figsize=(col_count * 1.32, len(budget_list) * 1.55 + 1.0),
        squeeze=False,
    )
    fig.suptitle(f'{title}\n{subtitle}', fontsize=10)
    for row_idx, budget in enumerate(budget_list):
        row_rows = (
            rank_df[rank_df['interaction_count'].astype(int) == int(budget)]
            .sort_values(['pred_rank_sample', 'example_slot'], kind='mergesort')
            .reset_index(drop=True)
        )
        for col_idx in range(col_count):
            ax = axes[row_idx][col_idx]
            ax.axis('off')
            if col_idx >= len(row_rows):
                continue
            item = row_rows.iloc[col_idx]
            thumb = render_thumbnail(str(item['image_path']), cell_w, cell_h)
            ax.imshow(np.asarray(thumb))
            if row_idx == 0:
                ax.set_title(str(col_idx + 1), fontsize=6, pad=2)
            if col_idx == 0:
                ax.text(
                    -0.18,
                    0.5,
                    str(budget),
                    fontsize=7,
                    ha='right',
                    va='center',
                    transform=ax.transAxes,
                )
            ax.axis('off')
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.9], w_pad=0.1, h_pad=0.35)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)


def build_manifest_payload(
    *,
    run_id: str,
    method_name: str,
    policy: str,
    budgets: Sequence[int],
    sample_size: int,
    sample_variant: int,
    rows_by_target: Mapping[str, pd.DataFrame],
    files: Sequence[Path],
    targets: Sequence[TargetSpec],
) -> Dict[str, Any]:
    target_payloads: List[Dict[str, Any]] = []
    target_meta = {target.task_id: target for target in targets}
    for task_id, frame in rows_by_target.items():
        target = target_meta[task_id]
        sample_rows = (
            frame[frame['interaction_count'].astype(int) == int(min(budgets))]
            .sort_values('example_slot', kind='mergesort')
        )
        target_payloads.append(
            {
                'task_id': task_id,
                'dataset_name': target.dataset_name,
                'display_name': target.display_name,
                'note': target.note,
                'sample_image_ids': [str(value) for value in sample_rows['image_id'].tolist()],
                'sample_slots': [int(value) for value in sample_rows['example_slot'].tolist()],
            }
        )
    return {
        'run_id': run_id,
        'method_name': method_name,
        'policy': policy,
        'budgets': [int(value) for value in budgets],
        'sample_size': int(sample_size),
        'sample_variant': int(sample_variant),
        'files': [str(path.name) for path in files],
        'targets': target_payloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Export targeted Reaxis bump charts at 0/5/20 interactions.')
    parser.add_argument('--run-id', default='', help='Run id used only for output naming. Defaults to latest run in modeling outputs.')
    parser.add_argument('--output-dir', default='', help='Directory where the PDFs should be written. Defaults to figures/<run_id>/selected_reaxis_bumpcharts')
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE, help='Random sample size per target task.')
    parser.add_argument('--replicates', type=int, default=1, help='Number of distinct deterministic random subsets to export.')
    args = parser.parse_args()

    run_dir = Path(me.OUTPUT_DIR)
    run_id = str(args.run_id).strip() or me.latest_modeling_run_id(output_dir=run_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / 'figures' / run_id / 'selected_reaxis_bumpcharts')
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_target_tasks(DEFAULT_TARGETS)
    task_by_id = {task.task_id: task for task in tasks}
    context = me.EvaluationContext(tasks)
    method = me.METHOD_REGISTRY[DEFAULT_METHOD_NAME]

    replicate_count = max(1, int(args.replicates))
    top_level_written_files: List[Path] = []
    summary_payload: Dict[str, Any] = {
        'run_id': run_id,
        'replicates': [],
    }
    for sample_variant in range(replicate_count):
        replicate_dir = output_dir / f'replicate_{sample_variant + 1}'
        replicate_dir.mkdir(parents=True, exist_ok=True)
        rows_by_target: Dict[str, pd.DataFrame] = {}
        written_files: List[Path] = []
        for target in DEFAULT_TARGETS:
            task = task_by_id[target.task_id]
            sample_indices = sample_example_indices(
                task,
                int(args.sample_size),
                sample_variant=sample_variant,
            )
            rows = collect_reaxis_rows(
                task=task,
                method=method,
                context=context,
                budgets=DEFAULT_BUDGETS,
                policy=DEFAULT_POLICY,
                sample_indices=sample_indices,
            )
            if rows.empty:
                raise RuntimeError(f'No ranking rows produced for task_id={task.task_id}')
            rows_by_target[target.task_id] = rows
            bump_pdf = replicate_dir / f'reaxis_bump_{target.slug}.pdf'
            contact_pdf = replicate_dir / f'reaxis_sample_sheet_{target.slug}.pdf'
            rank_grid_pdf = replicate_dir / f'reaxis_rank_grid_{target.slug}.pdf'
            note_suffix = f' | {target.note}' if target.note else ''
            save_bump_chart(
                rows,
                output_pdf=bump_pdf,
                title=f'Reaxis bump chart: {target.display_name}',
                subtitle=(
                    f'{task.dataset_name} | policy={DEFAULT_POLICY} | budgets={list(DEFAULT_BUDGETS)} '
                    f'| n={len(sample_indices)} | subset={sample_variant + 1}{note_suffix}'
                ),
                budgets=DEFAULT_BUDGETS,
            )
            save_contact_sheet(
                rows[rows['interaction_count'].astype(int) == int(min(DEFAULT_BUDGETS))].copy(),
                output_pdf=contact_pdf,
                title=f'{target.display_name} | sampled images ({len(sample_indices)}) | subset {sample_variant + 1}',
            )
            save_rank_grid(
                rows,
                output_pdf=rank_grid_pdf,
                title=f'Reaxis ranked images: {target.display_name}',
                subtitle=(
                    f'{task.dataset_name} | subset={sample_variant + 1} | rows show rank order at '
                    f'interactions {list(DEFAULT_BUDGETS)}{note_suffix}'
                ),
                budgets=DEFAULT_BUDGETS,
            )
            written_files.extend([bump_pdf, contact_pdf, rank_grid_pdf])

        manifest = build_manifest_payload(
            run_id=run_id,
            method_name=method.name,
            policy=DEFAULT_POLICY,
            budgets=DEFAULT_BUDGETS,
            sample_size=int(args.sample_size),
            sample_variant=sample_variant + 1,
            rows_by_target=rows_by_target,
            files=written_files,
            targets=DEFAULT_TARGETS,
        )
        manifest_path = replicate_dir / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        written_files.append(manifest_path)
        top_level_written_files.extend(written_files)
        summary_payload['replicates'].append(
            {
                'sample_variant': sample_variant + 1,
                'directory': str(replicate_dir.name),
                'files': [str(path.name) for path in written_files],
            }
        )

    summary_manifest_path = output_dir / 'manifest.json'
    summary_manifest_path.write_text(json.dumps(summary_payload, indent=2), encoding='utf-8')
    top_level_written_files.append(summary_manifest_path)
    print(
        json.dumps(
            {
                'output_dir': str(output_dir),
                'replicates': summary_payload['replicates'],
                'files': [str(path.relative_to(output_dir)) for path in top_level_written_files],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
