#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .ordinal_dataset_import import (
        import_ordinal_dataset,
        log,
        precompute_ordinal_dataset,
        selected_specs,
    )
    from .runtime_config import RAW_DATASETS_ROOT
except ImportError:
    from ordinal_dataset_import import (
        import_ordinal_dataset,
        log,
        precompute_ordinal_dataset,
        selected_specs,
    )
    from runtime_config import RAW_DATASETS_ROOT


DEFAULT_METHODS = 'clip'
DEFAULT_REDUCTION = 'pca,umap'
DEFAULT_LIMIT = 2000
DEFAULT_SEED = 7
DEFAULT_MAX_EDGE = 512
DEFAULT_MAX_SOURCE_PIXELS = 80_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Import local ordinal datasets into data/datasets/ for the Reaxis UI and ordinal study.'
    )
    parser.add_argument('--only', nargs='*', default=[], help='Optional dataset keys: utkface, affectnet, koniq10k.')
    parser.add_argument('--local-root', default=str(RAW_DATASETS_ROOT), help='Root containing raw dataset folders.')
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help='Maximum images per imported dataset; <=0 keeps all.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Sampling seed.')
    parser.add_argument('--max-edge', type=int, default=DEFAULT_MAX_EDGE, help='Prepared image maximum edge length.')
    parser.add_argument(
        '--max-source-pixels',
        type=int,
        default=DEFAULT_MAX_SOURCE_PIXELS,
        help='Resize sources above this pixel count before final preparation.',
    )
    parser.add_argument('--overwrite', action='store_true', help='Replace existing prepared dataset folders.')
    parser.add_argument('--skip-precompute', action='store_true', help='Skip embedding/layout precompute.')
    parser.add_argument('--methods', default=DEFAULT_METHODS, help='Comma-separated embedding methods for precompute.')
    parser.add_argument('--reduction', default=DEFAULT_REDUCTION, help='Comma-separated reductions: pca, umap, tsne, or all.')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    specs = selected_specs(args.only)
    results: list[tuple[str, Path, int, bool]] = []
    for spec in specs:
        log(f'[{spec.name}] import start output={spec.output_name}')
        dataset_root, count, reused = import_ordinal_dataset(
            spec,
            local_root=Path(args.local_root),
            limit=int(args.limit),
            seed=int(args.seed),
            max_edge=int(args.max_edge),
            max_source_pixels=int(args.max_source_pixels),
            overwrite=bool(args.overwrite),
        )
        results.append((spec.name, dataset_root, count, reused))
        action = 'reused' if reused else 'imported'
        log(f'[{spec.name}] {action} images={count} root={dataset_root}')
        if not args.skip_precompute:
            precompute_ordinal_dataset(dataset_root, methods_csv=args.methods, reduction=args.reduction)
            log(f'[{spec.name}] precomputed methods={args.methods} reduction={args.reduction}')
    log('ordinal import summary')
    for name, root, count, reused in results:
        log(f'  ok name={name} images={count} reused={reused} root={root}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        raise SystemExit(1)
