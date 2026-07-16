#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

try:
    from .constants import DATASET_LLM_CONTEXT
    from .import_curated_dataset import DATASETS_ROOT
except ImportError:
    from constants import DATASET_LLM_CONTEXT
    from import_curated_dataset import DATASETS_ROOT


def _fallback_description(name: str) -> str:
    clean = name.replace('_', ' ').replace('-', ' ').strip()
    return f'Prepared UI image dataset `{clean}` with metadata available in its `metadata.csv` file.'


def write_dataset_inventory(output_path: Path | None = None) -> Path:
    output_path = output_path or (DATASETS_ROOT.parent / 'dataset.md')
    rows: list[tuple[str, int, str]] = []
    for root in sorted(DATASETS_ROOT.iterdir(), key=lambda p: p.name.lower()):
        metadata_path = root / 'metadata.csv'
        if not root.is_dir() or not metadata_path.exists() or metadata_path.stat().st_size == 0:
            continue
        image_count = sum(1 for path in root.iterdir() if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
        rows.append((root.name, image_count, DATASET_LLM_CONTEXT.get(root.name, _fallback_description(root.name))))

    lines = [
        '# UI Dataset Inventory',
        '',
        'Each row is one prepared dataset currently visible to the local UI.',
        '',
        '| dataset | images | description |',
        '| --- | ---: | --- |',
    ]
    for name, image_count, description in rows:
        lines.append(f'| `{name}` | {image_count} | {description} |')
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_path


if __name__ == '__main__':
    path = write_dataset_inventory()
    print(f'[dataset-inventory] wrote {path}')
