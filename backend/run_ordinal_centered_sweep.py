#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

try:
    from .ordinal_study.centered_sweep import SWEEP_BETAS, SWEEP_LAMBDAS, run_centered_sweep
    from .ordinal_study.config import ACTIVE_DATASETS, BUDGETS, OUTPUT_DIR, REPETITIONS
except ImportError:
    from ordinal_study.centered_sweep import SWEEP_BETAS, SWEEP_LAMBDAS, run_centered_sweep
    from ordinal_study.config import ACTIVE_DATASETS, BUDGETS, OUTPUT_DIR, REPETITIONS


def _csv_tuple(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text or '').split(',') if part.strip())


def _float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in str(text or '').split(',') if part.strip())


def _int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(text or '').split(',') if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sweep centered Gaussian Reaxis update/blend parameters.')
    parser.add_argument('--run-id', default='', help='Optional run id. Reusing one resumes completed sweep rows.')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR), help='Directory for ordinal study artifacts.')
    parser.add_argument('--datasets', default=','.join(ACTIVE_DATASETS), help='Comma-separated prepared dataset names.')
    parser.add_argument('--budgets', default=','.join(str(v) for v in BUDGETS), help='Comma-separated label budgets.')
    parser.add_argument('--repetitions', type=int, default=REPETITIONS, help='Simulation repetitions.')
    parser.add_argument('--lambdas', default=','.join(str(v) for v in SWEEP_LAMBDAS), help='Comma-separated ridge lambdas.')
    parser.add_argument('--betas', default=','.join(str(v) for v in SWEEP_BETAS), help='Comma-separated prior/update blend weights.')
    parser.add_argument('--max-tasks', type=int, default=0, help='Optional task cap for smoke tests.')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id.strip() or uuid.uuid4().hex[:12]
    path = run_centered_sweep(
        run_id=run_id,
        output_dir=Path(args.output_dir),
        datasets=_csv_tuple(args.datasets),
        budgets=_int_tuple(args.budgets),
        repetitions=int(args.repetitions),
        lambdas=_float_tuple(args.lambdas),
        betas=_float_tuple(args.betas),
        max_tasks=int(args.max_tasks),
    )
    print(f'[centered-sweep] finished run_id={run_id} progression={path}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        raise SystemExit(1)
