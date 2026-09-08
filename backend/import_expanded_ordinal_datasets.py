#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .ordinal_study.expanded_dataset_import import (
        import_expanded_dataset,
        precompute_expanded_dataset,
        selected_specs,
    )
    from .import_curated_dataset import log
    from .runtime_config import RAW_DATASETS_ROOT
except ImportError:
    from ordinal_study.expanded_dataset_import import (
        import_expanded_dataset,
        precompute_expanded_dataset,
        selected_specs,
    )
    from import_curated_dataset import log
    from runtime_config import RAW_DATASETS_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Import expanded ordinal datasets for UI and modeling.')
    parser.add_argument('--only', nargs='*', default=[], help='Optional keys: scut isic2024 messidor2 vindr aadb lamem oasis house.')
    parser.add_argument('--local-root', default=str(RAW_DATASETS_ROOT), help='Root containing raw dataset folders.')
    parser.add_argument('--limit', type=int, default=1500, help='Maximum images per dataset; <=0 keeps all.')
    parser.add_argument('--seed', type=int, default=7, help='Deterministic sampling seed.')
    parser.add_argument('--max-edge', type=int, default=512, help='Prepared image maximum edge length.')
    parser.add_argument('--max-source-pixels', type=int, default=80_000_000, help='Resize sources above this pixel count.')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing prepared dataset folders.')
    parser.add_argument('--overwrite-low-level', action='store_true', help='Recompute low-level feature columns after import.')
    parser.add_argument('--skip-precompute', action='store_true', help='Skip embedding/layout precompute.')
    parser.add_argument('--methods', default='clip', help='Comma-separated embedding methods for precompute.')
    parser.add_argument('--reduction', default='pca,umap', help='Comma-separated reductions to precompute.')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results: list[tuple[str, Path, int, bool]] = []
    for spec in selected_specs(args.only):
        log(f'[{spec.name}] expanded import start output={spec.output_name}')
        root, count, reused = import_expanded_dataset(
            spec,
            local_root=Path(args.local_root),
            limit=int(args.limit),
            seed=int(args.seed),
            max_edge=int(args.max_edge),
            max_source_pixels=int(args.max_source_pixels),
            overwrite=bool(args.overwrite),
            overwrite_low_level=bool(args.overwrite_low_level),
        )
        results.append((spec.name, root, count, reused))
        log(f'[{spec.name}] {"reused" if reused else "imported"} images={count} root={root}')
        if not args.skip_precompute:
            precompute_expanded_dataset(root, methods_csv=args.methods, reduction=args.reduction)
            log(f'[{spec.name}] precomputed methods={args.methods} reduction={args.reduction}')
    log('expanded ordinal import summary')
    for name, root, count, reused in results:
        log(f'  ok name={name} images={count} reused={reused} root={root}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        raise SystemExit(1)
