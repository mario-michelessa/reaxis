#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

try:
    from .ordinal_study.config import BUDGETS, REPETITIONS
    from .ordinal_study.low_level_eval import (
        LOW_LEVEL_METHODS,
        LOW_LEVEL_OUTPUT_DIR,
        LowLevelStudyOptions,
        all_prepared_dataset_names,
        dataset_roots,
        run_low_level_study,
    )
    from .ordinal_study.low_level_features import compute_many_dataset_low_level_features
except ImportError:
    from ordinal_study.config import BUDGETS, REPETITIONS
    from ordinal_study.low_level_eval import (
        LOW_LEVEL_METHODS,
        LOW_LEVEL_OUTPUT_DIR,
        LowLevelStudyOptions,
        all_prepared_dataset_names,
        dataset_roots,
        run_low_level_study,
    )
    from ordinal_study.low_level_features import compute_many_dataset_low_level_features


def _csv_tuple(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text or '').split(',') if part.strip())


def _int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(text or '').split(',') if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compute and evaluate low-level ordinal visual features.')
    parser.add_argument('--run-id', default='', help='Optional run id. Reusing one resumes completed metric rows.')
    parser.add_argument('--output-dir', default=str(LOW_LEVEL_OUTPUT_DIR), help='Directory for study artifacts.')
    parser.add_argument('--datasets', default='EmoSet', help='Comma-separated prepared dataset names or paths.')
    parser.add_argument('--all-datasets', action='store_true', help='Use every prepared dataset with metadata.csv.')
    parser.add_argument('--methods', default=','.join(LOW_LEVEL_METHODS), help='Comma-separated method names.')
    parser.add_argument('--budgets', default=','.join(str(v) for v in BUDGETS), help='Comma-separated feedback budgets.')
    parser.add_argument('--repetitions', type=int, default=REPETITIONS, help='Simulation repetitions for feedback methods.')
    parser.add_argument('--max-tasks', type=int, default=0, help='Optional task cap for smoke tests.')
    parser.add_argument('--max-edge', type=int, default=512, help='Maximum image edge used for feature extraction.')
    parser.add_argument('--overwrite-features', action='store_true', help='Recompute low-level metadata columns.')
    parser.add_argument(
        '--compute-only',
        action='store_true',
        help='Only compute/update low-level metadata columns; do not run modeling evaluation.',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    datasets = all_prepared_dataset_names() if args.all_datasets else _csv_tuple(args.datasets)
    if not datasets:
        raise RuntimeError('No datasets selected')

    if args.compute_only:
        roots = dataset_roots(datasets)
        paths = compute_many_dataset_low_level_features(
            roots,
            overwrite=bool(args.overwrite_features),
            max_edge=int(args.max_edge),
        )
        for path in paths:
            print(f'[low-level-features] updated {path}')
        return 0

    run_id = args.run_id.strip() or uuid.uuid4().hex[:12]
    output_dir = run_low_level_study(LowLevelStudyOptions(
        run_id=run_id,
        output_dir=Path(args.output_dir),
        datasets=datasets,
        methods=_csv_tuple(args.methods),
        budgets=_int_tuple(args.budgets),
        repetitions=int(args.repetitions),
        max_tasks=int(args.max_tasks),
        max_edge=int(args.max_edge),
        overwrite_features=bool(args.overwrite_features),
    ))
    print(f'[low-level-ordinal] finished run_id={run_id} output_dir={output_dir}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        raise SystemExit(1)
