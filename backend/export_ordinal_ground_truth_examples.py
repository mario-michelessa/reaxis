#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image, ImageOps

try:
    from .ordinal_study.config import EXPANDED_DATASETS, OUTPUT_DIR
    from .ordinal_study.data import OrdinalTask, load_tasks
except ImportError:
    from ordinal_study.config import EXPANDED_DATASETS, OUTPUT_DIR
    from ordinal_study.data import OrdinalTask, load_tasks


POSITIONS = ('low', 'median', 'high')


def _slug(text: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9]+', '_', str(text).strip()).strip('_').lower()
    if not slug:
        raise RuntimeError(f'Cannot create filename slug from {text!r}')
    return slug


def _selected_indices(y: np.ndarray, examples_per_position: int) -> Dict[str, List[int]]:
    values = np.asarray(y, dtype=np.float32).reshape(-1)
    if values.size < examples_per_position * len(POSITIONS):
        raise RuntimeError(
            f'Need at least {examples_per_position * len(POSITIONS)} labels, got {values.size}'
        )

    order = np.argsort(values, kind='mergesort')
    low = [int(idx) for idx in order[:examples_per_position]]
    high = [int(idx) for idx in order[-examples_per_position:][::-1]]

    center = (values.size - 1) / 2.0
    center_order_positions = np.argsort(np.abs(np.arange(values.size) - center), kind='mergesort')
    median: List[int] = []
    for order_pos in center_order_positions:
        idx = int(order[int(order_pos)])
        if idx not in low and idx not in high:
            median.append(idx)
        if len(median) == examples_per_position:
            break
    if len(median) != examples_per_position:
        raise RuntimeError('Could not select distinct median examples')

    return {'low': low, 'median': median, 'high': high}


def _save_jpeg(src: Path, dst: Path) -> None:
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')
        image.save(dst, format='JPEG', quality=95, optimize=True)


def _clean_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob('*.jpg'):
        path.unlink()
    manifest_path = output_dir / 'manifest.csv'
    if manifest_path.exists():
        manifest_path.unlink()


def _task_axis_slug(task: OrdinalTask) -> str:
    return _slug(f'{task.dataset}_{task.axis_field}')


def export_examples(
    tasks: Iterable[OrdinalTask],
    output_dir: Path,
    examples_per_position: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.csv'
    rows = []

    for task in tasks:
        selections = _selected_indices(task.y01, examples_per_position)
        axis_slug = _task_axis_slug(task)
        for position in POSITIONS:
            for rank, selected_index in enumerate(selections[position], start=1):
                src = Path(task.paths[selected_index])
                filename = f'{axis_slug}_{position}_{rank}.jpg'
                dst = output_dir / filename
                _save_jpeg(src, dst)
                rows.append({
                    'dataset': task.dataset,
                    'axis_field': task.axis_field,
                    'axis_name': task.axis_name,
                    'axis_category': task.axis_category,
                    'position': position,
                    'position_rank': rank,
                    'image_id': task.ids[selected_index],
                    'selected_index': selected_index,
                    'source_path': str(src),
                    'output_file': filename,
                    'y_raw': float(task.y_raw[selected_index]),
                    'y01': float(task.y01[selected_index]),
                })

    with manifest_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export low/median/high ground-truth examples for ordinal modeling-study axes.'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=list(EXPANDED_DATASETS),
        help='Ordinal dataset names to export. Defaults to EXPANDED_DATASETS.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR / 'ground_truth_axis_examples',
        help='Flat output folder for JPG examples and manifest.csv.',
    )
    parser.add_argument(
        '--examples-per-position',
        type=int,
        default=2,
        help='Number of examples to copy for each of low, median, and high.',
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Remove existing JPGs and manifest.csv in the output folder before exporting.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.examples_per_position < 1:
        raise RuntimeError('--examples-per-position must be positive')
    if args.clean:
        _clean_output(args.output_dir)
    tasks = load_tasks(args.datasets)
    manifest_path = export_examples(tasks, args.output_dir, args.examples_per_position)
    expected_images = len(tasks) * len(POSITIONS) * args.examples_per_position
    print(f'Exported {expected_images} images for {len(tasks)} axes')
    print(f'Output folder: {args.output_dir}')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
