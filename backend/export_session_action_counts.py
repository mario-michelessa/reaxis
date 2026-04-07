#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from .plot_session_timelines import ACTION_STYLES, parse_log_line
    from .session_store import ACTIVITY_LOG_FILENAME, normalize_session_name
except ImportError:
    from plot_session_timelines import ACTION_STYLES, parse_log_line
    from session_store import ACTIVITY_LOG_FILENAME, normalize_session_name


def discover_session_logs(root: Path, session_names: Iterable[str] | None = None) -> list[tuple[str, Path]]:
    if session_names:
        requested = [normalize_session_name(name) for name in session_names]
        return [
            (session_name, (root / session_name / ACTIVITY_LOG_FILENAME).resolve())
            for session_name in requested
        ]
    logs: list[tuple[str, Path]] = []
    for path in sorted(root.glob(f'*/{ACTIVITY_LOG_FILENAME}')):
        logs.append((path.parent.name, path.resolve()))
    return logs


def count_actions(log_path: Path) -> Counter[str]:
    if not log_path.exists():
        raise FileNotFoundError(f'Activity log not found: {log_path}')
    counts: Counter[str] = Counter()
    with log_path.open('r', encoding='utf-8') as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            _, action, _ = parse_log_line(raw_line, line_number=line_number)
            counts[action] += 1
    return counts


def ordered_actions(counters: Iterable[Counter[str]]) -> list[str]:
    seen = {action for counter in counters for action in counter}
    known = [action for action in ACTION_STYLES if action in seen]
    unknown = sorted(action for action in seen if action not in ACTION_STYLES)
    return known + unknown


def write_counts_csv(
    *,
    session_logs: list[tuple[str, Path]],
    output_path: Path,
) -> Path:
    counts_by_session: list[tuple[str, Path, Counter[str]]] = []
    for session_name, log_path in session_logs:
        counts_by_session.append((session_name, log_path, count_actions(log_path)))

    action_columns = ordered_actions(counter for _, _, counter in counts_by_session)
    fieldnames = ['session_name', 'log_path', 'total_events', *action_columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for session_name, log_path, counts in counts_by_session:
            row = {
                'session_name': session_name,
                'log_path': str(log_path),
                'total_events': sum(counts.values()),
            }
            for action in action_columns:
                row[action] = counts.get(action, 0)
            writer.writerow(row)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Count session activity actions per session and write a CSV.')
    parser.add_argument(
        'sessions',
        nargs='*',
        help='Optional session names. Defaults to every session with an activity log under the root.',
    )
    parser.add_argument(
        '--root',
        default='data/sessions',
        help='Directory containing session folders.',
    )
    parser.add_argument(
        '--output',
        default='data/sessions/session_action_counts.csv',
        help='CSV output path.',
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    session_logs = discover_session_logs(root, args.sessions)
    if not session_logs:
        raise FileNotFoundError(f'No activity logs found under {root}')
    written = write_counts_csv(session_logs=session_logs, output_path=output_path)
    print(str(written))


if __name__ == '__main__':
    main()
