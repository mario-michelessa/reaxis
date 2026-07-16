#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

try:
    from .ordinal_study.config import ACTIVE_DATASETS, ACTIVE_METHODS, BUDGETS, OUTPUT_DIR, REPETITIONS
    from .ordinal_study.runner import StudyOptions, run_study
except ImportError:
    from ordinal_study.config import ACTIVE_DATASETS, ACTIVE_METHODS, BUDGETS, OUTPUT_DIR, REPETITIONS
    from ordinal_study.runner import StudyOptions, run_study


def _csv_tuple(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text or '').split(',') if part.strip())


def _int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(text or '').split(',') if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the ordinal-axis modeling study.')
    parser.add_argument('--run-id', default='', help='Optional run id. Reusing one resumes completed metric rows.')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR), help='Directory for ordinal study artifacts.')
    parser.add_argument('--datasets', default=','.join(ACTIVE_DATASETS), help='Comma-separated prepared dataset names.')
    parser.add_argument('--methods', default=','.join(ACTIVE_METHODS), help='Comma-separated method names.')
    parser.add_argument('--budgets', default=','.join(str(v) for v in BUDGETS), help='Comma-separated label budgets.')
    parser.add_argument('--repetitions', type=int, default=REPETITIONS, help='Simulation repetitions for feedback methods.')
    parser.add_argument('--max-tasks', type=int, default=0, help='Optional task cap for smoke tests.')
    parser.add_argument('--write-predictions', action='store_true', help='Write member_predictions.csv rows.')
    parser.add_argument('--prediction-limit', type=int, default=0, help='Optional per-step prediction row cap; 0 writes all.')
    parser.add_argument(
        '--debug-y',
        action='store_true',
        help='Write debug_y_trace.csv and debug_y_modeling_vs_ordinal.csv for selected feedback examples.',
    )
    parser.add_argument(
        '--target-mode',
        default='ground_truth_normalized',
        choices=('ground_truth_normalized', 'prior_quantile'),
        help='Feedback target transform for Reaxis debug/ablation runs.',
    )
    parser.add_argument(
        '--axisbayes-mode',
        default='gaussian',
        choices=('gaussian', 'rank'),
        help='AxisBayes mode for unsuffixed reaxis_random/reaxis_active methods.',
    )
    parser.add_argument(
        '--target-mode-ablation',
        action='store_true',
        help='Run original Reaxis and centered Reaxis over ground_truth_normalized and prior_quantile target modes.',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id.strip() or uuid.uuid4().hex[:12]
    output_dir = run_study(StudyOptions(
        run_id=run_id,
        output_dir=Path(args.output_dir),
        datasets=_csv_tuple(args.datasets),
        methods=_csv_tuple(args.methods),
        budgets=_int_tuple(args.budgets),
        repetitions=int(args.repetitions),
        max_tasks=int(args.max_tasks),
        write_predictions=bool(args.write_predictions),
        prediction_limit=int(args.prediction_limit),
        debug_y=bool(args.debug_y),
        target_mode=str(args.target_mode),
        axisbayes_mode=str(args.axisbayes_mode),
        target_mode_ablation=bool(args.target_mode_ablation),
    ))
    print(f'[ordinal-study] finished run_id={run_id} output_dir={output_dir}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        raise SystemExit(1)
