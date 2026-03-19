#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from .import_curated_dataset import (
        DATASETS_ROOT,
        DEFAULT_LOCAL_ROOT,
        PRESETS,
        ensure_empty_dataset_dir,
        import_local_preset,
        log,
        parse_methods,
        precompute_dataset,
    )
except ImportError:
    from import_curated_dataset import (
        DATASETS_ROOT,
        DEFAULT_LOCAL_ROOT,
        PRESETS,
        ensure_empty_dataset_dir,
        import_local_preset,
        log,
        parse_methods,
        precompute_dataset,
    )


@dataclass(frozen=True)
class BatchItem:
    preset: str
    output_name: str
    local_root: Path | None = None
    limit: int = 0
    overwrite: bool = False
    max_source_pixels: int = 0


# Local source root containing the raw curated datasets.
LOCAL_ROOT = Path(DEFAULT_LOCAL_ROOT)
DATASYNTH_LOCAL_ROOT = Path('/home/mario/codes/datasynth-xai/data')

# Output datasets created under data/datasets/.
BATCH_DATASETS = [
    BatchItem('ancient-tamil-stone-inscriptions', 'ancient_tamil_inscriptions'),
    BatchItem('archaeomind-images', 'archaeomind_images'),
    BatchItem('brain-mri-images-for-brain-tumor-detection', 'brain_mri_images'),
    BatchItem('celeba-dataset', 'celeba_dataset'),
    BatchItem('chest-xray-pneumonia-balanced-dataset', 'chest_xray_pneumonia'),
    BatchItem('egyptian-archaeological-stone-statue-dataset', 'egyptian_stone_statues'),
    BatchItem('metal-albums-artwork', 'metal_albums_artwork'),
    BatchItem('qajar-dynasty-iranian-carpet-dataset', 'qajar_carpets'),
    BatchItem('sinhala-early-brahmi-inscription-dataset', 'sinhala_brahmi_inscriptions'),
    BatchItem('broden1_224', 'broden1_224', local_root=DATASYNTH_LOCAL_ROOT, limit=1500, overwrite=True),
    BatchItem('cub_local', 'CUB', local_root=DATASYNTH_LOCAL_ROOT, limit=1500, overwrite=True),
    BatchItem('ham10000_local', 'HAM10000', local_root=DATASYNTH_LOCAL_ROOT, limit=1500, overwrite=True),
    BatchItem('imagenet_n02958343_local', 'ImageNet_n02958343', local_root=DATASYNTH_LOCAL_ROOT, limit=1500, overwrite=True),
    BatchItem('imagenet_r_local', 'ImageNet_R', local_root=DATASYNTH_LOCAL_ROOT, limit=1500, overwrite=True),
    BatchItem('imagenette_local', 'Imagenette1500', local_root=DATASYNTH_LOCAL_ROOT, limit=1500, overwrite=True),
    BatchItem('inat2021birds_local', 'inat2021birds', limit=1500, overwrite=True),
    BatchItem('mapillary_vistas_local', 'MapillaryVistas', limit=1500, overwrite=True, max_source_pixels=20_000_000),
]

# Import settings.
LIMIT = 2000
MAX_EDGE = 512
SEED = 7
MAX_SOURCE_PIXELS = 5_000_000
OVERWRITE = False

# Precompute settings.
SKIP_PRECOMPUTE = False
METHODS = 'color_rgb,siglip2,clip,dino'
REDUCTION = 'pca'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Batch-import local curated datasets into data/datasets/.')
    parser.add_argument(
        '--only',
        nargs='*',
        default=[],
        help='Optional preset names or output names to import. Defaults to all batch items.',
    )
    parser.add_argument('--skip-precompute', action='store_true', help='Skip embedding and PCA precompute.')
    parser.add_argument('--methods', default=METHODS, help='Comma-separated embedding methods to precompute.')
    parser.add_argument('--reduction', default=REDUCTION, help='2D reduction method to precompute.')
    return parser


def select_batch_items(only: list[str]) -> list[BatchItem]:
    wanted = {str(item).strip() for item in only if str(item).strip()}
    if not wanted:
        return list(BATCH_DATASETS)
    selected: list[BatchItem] = []
    matched: set[str] = set()
    for item in BATCH_DATASETS:
        if item.preset in wanted or item.output_name in wanted:
            selected.append(item)
            matched.add(item.preset)
            matched.add(item.output_name)
    missing = sorted(wanted - matched)
    if missing:
        raise RuntimeError(f'Unknown batch dataset(s): {", ".join(missing)}')
    return selected


def import_one(item: BatchItem, *, skip_precompute: bool, methods: list[str], reduction: str) -> tuple[str, str, int]:
    preset = PRESETS[item.preset]
    if preset.kind != 'local':
        raise RuntimeError(f'Batch preset is not local: {item.preset}')
    local_root = item.local_root or LOCAL_ROOT
    limit = item.limit if item.limit > 0 else LIMIT
    overwrite = item.overwrite or OVERWRITE
    max_source_pixels = item.max_source_pixels if item.max_source_pixels > 0 else MAX_SOURCE_PIXELS
    dataset_root = DATASETS_ROOT / item.output_name
    reused_existing, existing_count = ensure_empty_dataset_dir(dataset_root, overwrite=overwrite)
    if reused_existing:
        count = existing_count
        log(f'[{item.preset}] reusing existing dataset with {count} images: {dataset_root}')
    else:
        log(
            f'[{item.preset}] importing into {dataset_root} '
            f'from local_root={local_root} limit={limit} overwrite={overwrite} '
            f'max_source_pixels={max_source_pixels}'
        )
        count = import_local_preset(
            preset=preset,
            dataset_root=dataset_root,
            local_root=local_root,
            limit=limit,
            seed=SEED,
            max_edge=MAX_EDGE,
            max_source_pixels=max_source_pixels,
        )
        log(f'[{item.preset}] imported {count} images')
    if not skip_precompute:
        precompute_dataset(dataset_root=dataset_root, methods=methods, reduction=reduction)
        log(f'[{item.preset}] precomputed methods={methods} reduction={reduction}')
    return item.preset, item.output_name, count


def main() -> int:
    args = build_arg_parser().parse_args()
    methods = parse_methods(args.methods)
    selected_items = select_batch_items(args.only)
    skip_precompute = bool(SKIP_PRECOMPUTE or args.skip_precompute)
    log(f'batch start datasets={len(selected_items)} default_local_root={LOCAL_ROOT}')
    log(
        'settings '
        f'limit={LIMIT} max_edge={MAX_EDGE} seed={SEED} '
        f'methods={methods} reduction={args.reduction} overwrite={OVERWRITE}'
    )
    results: list[tuple[str, str, int]] = []
    failures: list[tuple[str, str]] = []
    for item in selected_items:
        try:
            results.append(import_one(item, skip_precompute=skip_precompute, methods=methods, reduction=args.reduction))
        except Exception as exc:
            failures.append((item.preset, str(exc)))
            log(f'[{item.preset}] FAILED: {exc}')
    log('batch summary')
    for preset_name, output_name, count in results:
        log(f'  ok preset={preset_name} output={output_name} images={count}')
    if failures:
        for preset_name, message in failures:
            log(f'  failed preset={preset_name} reason={message}')
        return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        raise SystemExit(1)
