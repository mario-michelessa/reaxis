from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_SESSION_NAME = 'default'
LEGACY_SESSION_NAME = 'P0'
AXES_FILENAME = 'axes.json'
VISUALIZATIONS_FILENAME = 'visualizations.json'
SUBSETS_FILENAME = 'subsets.json'
ACTIVITY_LOG_FILENAME = 'activity.txt'
AXIS_PROJECTION_SNAPSHOTS_FILENAME = 'axis_projection_snapshots.jsonl'
MAX_SESSION_NAME_LENGTH = 80

_INVALID_SESSION_CHARS = re.compile(r'[^A-Za-z0-9._ -]+')
_WHITESPACE = re.compile(r'\s+')


def normalize_session_name(value: object, default: str = DEFAULT_SESSION_NAME) -> str:
    raw = str(value or '').strip()
    if not raw:
        return default
    raw = raw.replace('/', ' ').replace('\\', ' ')
    raw = _INVALID_SESSION_CHARS.sub(' ', raw)
    raw = _WHITESPACE.sub('_', raw).strip('._- ')
    if not raw:
        return default
    return raw[:MAX_SESSION_NAME_LENGTH]


def session_dir(root: Path, session_name: object, *, create: bool = True) -> Path:
    root_resolved = root.resolve()
    root_resolved.mkdir(parents=True, exist_ok=True)
    normalized = normalize_session_name(session_name)
    candidate = (root_resolved / normalized).resolve()
    candidate.relative_to(root_resolved)
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def session_axes_path(root: Path, session_name: object) -> Path:
    return session_dir(root, session_name) / AXES_FILENAME


def session_visualizations_path(root: Path, session_name: object) -> Path:
    return session_dir(root, session_name) / VISUALIZATIONS_FILENAME


def session_subsets_path(root: Path, session_name: object) -> Path:
    return session_dir(root, session_name) / SUBSETS_FILENAME


def session_activity_log_path(root: Path, session_name: object) -> Path:
    return session_dir(root, session_name) / ACTIVITY_LOG_FILENAME


def session_axis_projection_snapshots_path(root: Path, session_name: object) -> Path:
    return session_dir(root, session_name) / AXIS_PROJECTION_SNAPSHOTS_FILENAME


def iter_session_names(root: Path, *, include_default: bool = True) -> Iterable[str]:
    root_resolved = root.resolve()
    names = set()
    if root_resolved.exists():
        for child in root_resolved.iterdir():
            try:
                if child.is_dir() and not child.name.startswith('.'):
                    names.add(normalize_session_name(child.name))
            except Exception:
                continue
    if include_default:
        names.add(DEFAULT_SESSION_NAME)
    names.discard('')
    return sorted(names, key=lambda value: (value != DEFAULT_SESSION_NAME, value.lower()))


def sanitize_log_field(value: object, fallback: str = 'none') -> str:
    text = str(value or '').strip()
    if not text:
        return fallback
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\\n')
    text = re.sub(r'\s+', ' ', text).strip()
    return text or fallback


def append_session_log(root: Path, session_name: object, action: object, detail: object) -> Path:
    path = session_activity_log_path(root, session_name)
    timestamp = datetime.now(timezone.utc).isoformat()
    action_text = sanitize_log_field(action).lower()
    detail_text = sanitize_log_field(detail)
    line = f'[{timestamp}] [{action_text}] [{detail_text}]\n'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(line)
    return path


def session_summary(root: Path, session_name: object, *, axes_count: int, visualizations_count: int) -> dict:
    normalized = normalize_session_name(session_name)
    session_dir(root, normalized, create=True)
    return {
        'id': normalized,
        'label': normalized,
        'axes_count': int(axes_count),
        'visualizations_count': int(visualizations_count),
    }
