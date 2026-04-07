#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .session_store import ACTIVITY_LOG_FILENAME, normalize_session_name
except ImportError:
    from session_store import ACTIVITY_LOG_FILENAME, normalize_session_name


LOG_LINE_RE = re.compile(r'^\[(?P<timestamp>[^\]]+)\]\s+\[(?P<action>[^\]]*)\]\s+\[(?P<detail>[^\]]*)\]\s*$')
# Explicit per-action style map, grouped by coherent color families.
ACTION_STYLES: dict[str, dict[str, str]] = {
    # 1. Axis definition: blue family
    'suggest axes': {'marker': 'o', 'color': '#1d4ed8'},
    'create ax': {'marker': 's', 'color': '#2563eb'},
    'delete ax': {'marker': 'X', 'color': '#60a5fa'},
    # 2. Image browsing: teal family
    'local gridding': {'marker': 'D', 'color': '#0f766e'},
    'change view': {'marker': '^', 'color': '#14b8a6'},
    # 3. Reuse: green family
    'enter session': {'marker': 'P', 'color': "#FFFFFF"},
    'save ax': {'marker': 'h', 'color': "#ffffff"},
    'save visualization': {'marker': '*', 'color': "#ffffff"},
    'change dataset': {'marker': '8', 'color': "#ffffff"},
    # 4. Refinement: amber family
    'change anchor': {'marker': 'v', 'color': '#b45309'},
    'reorder': {'marker': '<', 'color': '#d97706'},
    'slider change': {'marker': '>', 'color': '#f59e0b'},
    'grabbing': {'marker': 'p', 'color': '#fbbf24'},
    # 5. Selection: rose family
    'lasso selection': {'marker': 'o', 'color': '#be123c'},
    'ax slicing': {'marker': 'd', 'color': '#e11d48'},
    'clear subset': {'marker': 'x', 'color': '#fb7185'},
}
DEFAULT_ACTION_STYLE = {'marker': 'o', 'color': '#64748b'}
UNFILLED_MARKERS = frozenset({'x', '+', '1', '2', '3', '4', '|', '_'})
JITTER_Y_AMPLITUDE = 0.075
MAX_MINUTES = 60.0
AXIS_ADD_ACTION = 'create ax'
IMAGE_EXPLORATION_ACTIONS = frozenset({'local gridding', 'change view'})
AXIS_REFINEMENT_ACTIONS = frozenset({'change anchor', 'reorder', 'slider change', 'grabbing'})
REUSE_ACTIONS = frozenset({'enter session', 'save ax', 'save visualization', 'change dataset'})
MIDDLE_RIBBON_ACTIONS = frozenset({
    'suggest axes',
    'create ax',
    'delete ax',
    'lasso selection',
    'ax slicing',
    'clear subset',
})
SUPPRESSED_MARKER_ACTIONS = frozenset(
    set(IMAGE_EXPLORATION_ACTIONS)
    | set(REUSE_ACTIONS)
    | {'reorder', 'slider change', 'grabbing'}
)
TOP_RIBBON_COLOR = '#22c55e'
TOP_RIBBON_ALPHA = 0.22
TOP_RIBBON_HEIGHT = 0.18
TOP_RIBBON_OFFSET = 0.19
MIDDLE_RIBBON_COLOR = '#ef4444'
MIDDLE_RIBBON_ALPHA = 0.16
MIDDLE_RIBBON_HEIGHT = 0.22
MIDDLE_RIBBON_OFFSET = 0.0
BOTTOM_RIBBON_COLOR = '#eab308'
BOTTOM_RIBBON_ALPHA = 0.22
BOTTOM_RIBBON_HEIGHT = 0.18
BOTTOM_RIBBON_OFFSET = -0.19
AXIS_ADD_LINE_HALF_HEIGHT = 0.26
AXIS_ADD_LINE_ALPHA = 0.78
AXIS_ADD_LINE_WIDTH = 1.3


@dataclass(frozen=True)
class SessionEvent:
    timestamp: datetime
    action: str
    detail: str
    minutes_from_start: float
    line_number: int


@dataclass(frozen=True)
class SessionTimeline:
    session_name: str
    log_path: Path
    start_timestamp: datetime
    events: tuple[SessionEvent, ...]


def parse_log_line(line: str, *, line_number: int) -> tuple[datetime, str, str]:
    match = LOG_LINE_RE.match(str(line or '').strip())
    if match is None:
        raise ValueError(f'Invalid log line at {line_number}: {line.rstrip()}')
    timestamp = datetime.fromisoformat(match.group('timestamp'))
    action = str(match.group('action') or '').strip()
    detail = str(match.group('detail') or '').strip()
    if not action:
        raise ValueError(f'Missing action at line {line_number}')
    return timestamp, action, detail


def load_session_timeline(root: Path, session_name: str, *, max_minutes: float | None = MAX_MINUTES) -> SessionTimeline:
    normalized = normalize_session_name(session_name)
    log_path = (Path(root) / normalized / ACTIVITY_LOG_FILENAME).resolve()
    if not log_path.exists():
        raise FileNotFoundError(f'Activity log not found for session `{normalized}` at {log_path}')

    parsed_rows: list[tuple[datetime, str, str, int]] = []
    with log_path.open('r', encoding='utf-8') as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            timestamp, action, detail = parse_log_line(raw_line, line_number=line_number)
            parsed_rows.append((timestamp, action, detail, line_number))

    if not parsed_rows:
        raise ValueError(f'Activity log is empty for session `{normalized}` at {log_path}')

    start_timestamp = parsed_rows[0][0]
    max_minutes_value = None if max_minutes is None else float(max_minutes)
    events_list: list[SessionEvent] = []
    for timestamp, action, detail, line_number in parsed_rows:
        minutes_from_start = ((timestamp - start_timestamp).total_seconds() / 60.0)
        if max_minutes_value is not None and max_minutes_value >= 0.0 and minutes_from_start > max_minutes_value:
            continue
        events_list.append(
            SessionEvent(
                timestamp=timestamp,
                action=action,
                detail=detail,
                minutes_from_start=minutes_from_start,
                line_number=line_number,
            )
        )
    events = tuple(events_list)
    return SessionTimeline(
        session_name=normalized,
        log_path=log_path,
        start_timestamp=start_timestamp,
        events=events,
    )


def ordered_actions(timelines: Sequence[SessionTimeline]) -> list[str]:
    seen: set[str] = set()
    observed_order: list[str] = []
    for timeline in timelines:
        for event in timeline.events:
            if event.action in seen:
                continue
            seen.add(event.action)
            observed_order.append(event.action)

    known_actions = [action for action in ACTION_STYLES if action in seen]
    unknown_actions = [action for action in observed_order if action not in ACTION_STYLES]
    return known_actions + unknown_actions


def action_style(action: str) -> dict[str, str]:
    return ACTION_STYLES.get(action, DEFAULT_ACTION_STYLE)


def marker_facecolor(style: dict[str, str]) -> str:
    marker = style.get('marker', 'o')
    color = style.get('color', DEFAULT_ACTION_STYLE['color'])
    return 'none' if marker in UNFILLED_MARKERS else color


def marker_edgecolor(style: dict[str, str]) -> str:
    marker = style.get('marker', 'o')
    color = style.get('color', DEFAULT_ACTION_STYLE['color'])
    return color if marker in UNFILLED_MARKERS else '#ffffff'


def jittered_y(base_y: float, timeline: SessionTimeline, event: SessionEvent) -> float:
    raw_key = (
        f'{timeline.session_name}|{event.action}|'
        f'{event.minutes_from_start:.4f}|{event.line_number}'
    ).encode('utf-8')
    digest = hashlib.sha256(raw_key).digest()
    unit = int.from_bytes(digest[:8], byteorder='big') / float((1 << 64) - 1)
    return base_y + ((unit * 2.0) - 1.0) * JITTER_Y_AMPLITUDE


def minute_windows_for_actions(
    timeline: SessionTimeline,
    actions: frozenset[str] | set[str],
    *,
    min_count: int = 1,
) -> list[tuple[float, float, int]]:
    counts_by_minute: dict[int, int] = {}
    for event in timeline.events:
        if event.action not in actions:
            continue
        minute_bucket = int(event.minutes_from_start)
        counts_by_minute[minute_bucket] = counts_by_minute.get(minute_bucket, 0) + 1
    return [
        (float(minute_bucket), float(minute_bucket + 1), count)
        for minute_bucket, count in sorted(counts_by_minute.items())
        if count >= min_count
    ]


def plot_timelines(
    timelines: Sequence[SessionTimeline],
    *,
    output_path: Path,
    title: str,
    dpi: int,
    max_minutes: float | None = MAX_MINUTES,
) -> Path:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    if len(timelines) == 0:
        raise ValueError('No session timelines to plot')

    actions = ordered_actions(timelines)
    if len(actions) == 0:
        raise ValueError('No actions found in the provided session logs')

    style_by_action = {action: action_style(action) for action in actions}

    observed_max_minutes = max(
        max((event.minutes_from_start for event in timeline.events), default=0.0)
        for timeline in timelines
    )
    max_minutes_value = None if max_minutes is None else float(max_minutes)
    if max_minutes_value is not None and max_minutes_value >= 0.0:
        x_max = max(1.0, max_minutes_value)
    else:
        x_max = max(1.0, observed_max_minutes + max(0.25, observed_max_minutes * 0.04))
    y_positions = {
        timeline.session_name: (len(timelines) - 1 - idx)
        for idx, timeline in enumerate(timelines)
    }

    fig_width = max(10.0, min(18.0, 8.0 + (x_max * 0.22)))
    fig_height = max(3.0, 1.15 + (1.05 * len(timelines)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    for timeline in timelines:
        y = y_positions[timeline.session_name]
        ax.hlines(y, 0, x_max, color='#e2e8f0', linewidth=1.0, zorder=0)
        ribbon_specs = (
            (
                minute_windows_for_actions(timeline, IMAGE_EXPLORATION_ACTIONS, min_count=1),
                TOP_RIBBON_OFFSET,
                TOP_RIBBON_HEIGHT,
                TOP_RIBBON_COLOR,
                TOP_RIBBON_ALPHA,
            ),
            (
                minute_windows_for_actions(timeline, MIDDLE_RIBBON_ACTIONS, min_count=1),
                MIDDLE_RIBBON_OFFSET,
                MIDDLE_RIBBON_HEIGHT,
                MIDDLE_RIBBON_COLOR,
                MIDDLE_RIBBON_ALPHA,
            ),
            (
                minute_windows_for_actions(timeline, AXIS_REFINEMENT_ACTIONS, min_count=1),
                BOTTOM_RIBBON_OFFSET,
                BOTTOM_RIBBON_HEIGHT,
                BOTTOM_RIBBON_COLOR,
                BOTTOM_RIBBON_ALPHA,
            ),
        )
        for windows, offset, height, color, alpha in ribbon_specs:
            for x0, x1, _count in windows:
                if x0 >= x_max:
                    continue
                clipped_x1 = min(x1, x_max)
                if clipped_x1 <= x0:
                    continue
                ax.add_patch(
                    Rectangle(
                        (x0, y + offset - (height * 0.5)),
                        clipped_x1 - x0,
                        height,
                        facecolor=color,
                        edgecolor='none',
                        alpha=alpha,
                        zorder=1,
                    )
                )

    create_ax_style = action_style(AXIS_ADD_ACTION)
    for timeline in timelines:
        y = y_positions[timeline.session_name]
        axis_add_xs = [
            event.minutes_from_start
            for event in timeline.events
            if event.action == AXIS_ADD_ACTION and event.minutes_from_start <= x_max
        ]
        if not axis_add_xs:
            continue
        ax.vlines(
            axis_add_xs,
            y - AXIS_ADD_LINE_HALF_HEIGHT,
            y + AXIS_ADD_LINE_HALF_HEIGHT,
            colors=create_ax_style['color'],
            linewidth=AXIS_ADD_LINE_WIDTH,
            alpha=AXIS_ADD_LINE_ALPHA,
            zorder=2,
        )

    for action in actions:
        if action in SUPPRESSED_MARKER_ACTIONS:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for timeline in timelines:
            y = y_positions[timeline.session_name]
            for event in timeline.events:
                if event.action != action:
                    continue
                xs.append(event.minutes_from_start)
                ys.append(jittered_y(y, timeline, event))
        if not xs:
            continue
        style = style_by_action[action]
        scatter_kwargs = dict(
            s=70,
            marker=style['marker'],
            color=style['color'],
            linewidths=0.8,
            alpha=0.96,
            label=action,
            zorder=3,
        )
        if style['marker'] in UNFILLED_MARKERS:
            ax.scatter(xs, ys, **scatter_kwargs)
        else:
            ax.scatter(
                xs,
                ys,
                facecolors=marker_facecolor(style),
                edgecolors=marker_edgecolor(style),
                **scatter_kwargs,
            )

    ax.set_xlim(-0.05, x_max)
    ax.set_ylim(-0.6, len(timelines) - 0.4)
    ax.set_xlabel('Time Since Session Start (min.)')
    ax.set_ylabel('Session')
    ax.set_title(title)
    ax.set_yticks([y_positions[timeline.session_name] for timeline in timelines])
    ax.set_yticklabels([timeline.session_name for timeline in timelines])
    ax.grid(axis='x', color='#cbd5e1', linewidth=0.8, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_handles = [
        Patch(
            facecolor=TOP_RIBBON_COLOR,
            edgecolor='none',
            alpha=TOP_RIBBON_ALPHA,
            label='image exploration',
        ),
        Patch(
            facecolor=MIDDLE_RIBBON_COLOR,
            edgecolor='none',
            alpha=MIDDLE_RIBBON_ALPHA,
            label='axis / selection activity',
        ),
        Patch(
            facecolor=BOTTOM_RIBBON_COLOR,
            edgecolor='none',
            alpha=BOTTOM_RIBBON_ALPHA,
            label='axis refinement',
        ),
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker=style_by_action[action]['marker'],
                color='none',
                markerfacecolor=marker_facecolor(style_by_action[action]),
                markeredgecolor=marker_edgecolor(style_by_action[action]),
                markeredgewidth=0.8,
                markersize=9,
                linestyle='None',
                label=action,
            )
            for action in actions
            if action not in SUPPRESSED_MARKER_ACTIONS
        ]
    )
    ax.legend(
        handles=legend_handles,
        title='Action',
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(max(72, dpi)), bbox_inches='tight')
    plt.close(fig)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot a multi-session activity timeline from session activity logs.',
    )
    parser.add_argument(
        'sessions',
        nargs='+',
        help='Session names to plot. Each session is resolved under data/sessions/<session>/activity.txt.',
    )
    parser.add_argument(
        '--root',
        default='data/sessions',
        help='Session root directory. Defaults to data/sessions.',
    )
    parser.add_argument(
        '-o',
        '--output',
        default='session_timeline.pdf',
        help='Output image path. Defaults to ./session_timeline.pdf.',
    )
    parser.add_argument(
        '--title',
        default='Session Activity Timeline',
        help='Figure title.',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=180,
        help='Output DPI. Defaults to 180.',
    )
    parser.add_argument(
        '--max-minutes',
        type=float,
        default=MAX_MINUTES,
        help='Crop out events after this many minutes from each session start. Set a negative value to disable cropping. Defaults to 60.',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    max_minutes = None if float(args.max_minutes) < 0.0 else float(args.max_minutes)
    timelines = [load_session_timeline(root, session_name, max_minutes=max_minutes) for session_name in args.sessions]
    output_path = plot_timelines(
        timelines,
        output_path=Path(args.output),
        title=str(args.title or 'Session Activity Timeline').strip() or 'Session Activity Timeline',
        dpi=int(args.dpi),
        max_minutes=max_minutes,
    )
    print(output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
