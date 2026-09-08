#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

try:
    from .ordinal_study.config import BUDGETS, EXPANDED_DATASETS, OUTPUT_DIR, REPETITIONS
    from .ordinal_study.gaussian_sweep import (
        DEFAULT_ALPHAS,
        DEFAULT_BIAS_ALPHAS,
        DEFAULT_POLICIES,
        DEFAULT_SCORE_MODES,
        DEFAULT_SIGMA2S,
        run_gaussian_sweep,
    )
except ImportError:
    from ordinal_study.config import BUDGETS, EXPANDED_DATASETS, OUTPUT_DIR, REPETITIONS
    from ordinal_study.gaussian_sweep import (
        DEFAULT_ALPHAS,
        DEFAULT_BIAS_ALPHAS,
        DEFAULT_POLICIES,
        DEFAULT_SCORE_MODES,
        DEFAULT_SIGMA2S,
        run_gaussian_sweep,
    )


def _csv_tuple(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text or '').split(',') if part.strip())


def _float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in str(text or '').split(',') if part.strip())


def _int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(text or '').split(',') if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sweep Gaussian AxisBayes hyperparameters for ordinal Reaxis.')
    parser.add_argument('--run-id', default='', help='Optional run id. Reusing one resumes completed rows.')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR / 'gaussian_sweep'), help='Directory for sweep artifacts.')
    parser.add_argument('--datasets', default=','.join(EXPANDED_DATASETS), help='Comma-separated prepared dataset names.')
    parser.add_argument('--budgets', default=','.join(str(v) for v in BUDGETS), help='Comma-separated feedback budgets.')
    parser.add_argument('--repetitions', type=int, default=REPETITIONS, help='Simulation repetitions.')
    parser.add_argument('--alphas', default=','.join(str(v) for v in DEFAULT_ALPHAS), help='Comma-separated Gaussian direction prior precisions.')
    parser.add_argument('--sigma2s', default=','.join(str(v) for v in DEFAULT_SIGMA2S), help='Comma-separated Gaussian observation variances.')
    parser.add_argument('--bias-alphas', default=','.join(str(v) for v in DEFAULT_BIAS_ALPHAS), help='Comma-separated Gaussian intercept precisions.')
    parser.add_argument('--score-modes', default=','.join(DEFAULT_SCORE_MODES), help='Comma-separated score readouts: cosine,linear.')
    parser.add_argument('--policies', default=','.join(DEFAULT_POLICIES), help='Comma-separated selection policies: random,active.')
    parser.add_argument('--max-tasks', type=int, default=0, help='Optional task cap for smoke tests.')
    parser.add_argument('--metric-mode', default='full', choices=('full', 'spearman'), help='Use spearman for broad search, full for final metric rows.')
    parser.add_argument(
        '--comparison-dirs',
        default=str(OUTPUT_DIR.parent / 'ordinal_label_only_more'),
        help='Comma-separated experiment dirs containing metrics.csv for label-only envelope comparison.',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id.strip() or uuid.uuid4().hex[:12]
    report = run_gaussian_sweep(
        run_id=run_id,
        output_dir=Path(args.output_dir),
        datasets=_csv_tuple(args.datasets),
        budgets=_int_tuple(args.budgets),
        repetitions=int(args.repetitions),
        alphas=_float_tuple(args.alphas),
        sigma2s=_float_tuple(args.sigma2s),
        bias_alphas=_float_tuple(args.bias_alphas),
        score_modes=_csv_tuple(args.score_modes),
        policies=_csv_tuple(args.policies),
        max_tasks=int(args.max_tasks),
        comparison_dirs=tuple(Path(part) for part in _csv_tuple(args.comparison_dirs)),
        metric_mode=str(args.metric_mode),
    )
    print(f'[gaussian-sweep] finished run_id={run_id} report={report}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        raise SystemExit(1)
