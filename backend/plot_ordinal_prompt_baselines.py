#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .ordinal_study.combined_prompt_plot import export_prompt_baseline_plot
    from .ordinal_study.config import OUTPUT_DIR, REPO_ROOT
except ImportError:
    from ordinal_study.combined_prompt_plot import export_prompt_baseline_plot
    from ordinal_study.config import OUTPUT_DIR, REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    diagnostics_dir = REPO_ROOT / 'backend' / 'experiments' / 'ordinal_modeling_diagnostics'
    parser = argparse.ArgumentParser(description='Plot ordinal prompt-initialization variants with baselines.')
    parser.add_argument('--diagnostics-dir', default=str(diagnostics_dir))
    parser.add_argument('--baseline-dir', default=str(OUTPUT_DIR))
    parser.add_argument('--output-dir', default=str(diagnostics_dir))
    parser.add_argument('--baseline-run-id', default='axisbayes_mode_v1')
    parser.add_argument('--legacy-run-id', default='ordinal_v2')
    parser.add_argument('--prompt-run-id', default='prompt_init_targeted_v1')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = export_prompt_baseline_plot(
        diagnostics_dir=Path(args.diagnostics_dir),
        baseline_dir=Path(args.baseline_dir),
        output_dir=Path(args.output_dir),
        baseline_run_id=args.baseline_run_id,
        legacy_run_id=args.legacy_run_id,
        prompt_run_id=args.prompt_run_id,
    )
    print(f'[ordinal-combined-plot] wrote {path}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        raise SystemExit(1)
