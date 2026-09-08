from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .session_store import normalize_session_name, session_axis_projection_snapshots_path
except ImportError:
    from session_store import normalize_session_name, session_axis_projection_snapshots_path


SNAPSHOT_SCHEMA_VERSION = 1
_WRITE_LOCK = threading.Lock()


def _finite_vector(raw: Any, *, field: str, expected_size: int | None = None) -> list[float]:
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if expected_size is not None and values.size != expected_size:
        raise ValueError(f'{field} has {values.size} values; expected {expected_size}')
    if not np.isfinite(values).all():
        raise ValueError(f'{field} contains non-finite values')
    return values.astype(float).tolist()


def _read_axis_rows(path: Path, axis_id: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'Invalid snapshot row at {path}:{line_number}')
            if str(row.get('axis_id') or '') == axis_id:
                rows.append(row)
    return rows


def has_axis_snapshot(sessions_root: Path, session_name: str, axis_id: str) -> bool:
    path = session_axis_projection_snapshots_path(sessions_root, session_name)
    return bool(_read_axis_rows(path, axis_id))


def _direction_fields(state: Any) -> dict[str, Any]:
    prior = np.asarray(getattr(state, 'w0', []), dtype=np.float64).reshape(-1)
    posterior = np.asarray(getattr(state, 'mu', []), dtype=np.float64).reshape(-1)
    model_type = str(getattr(state, 'model_type', '') or '')
    mode = str(getattr(state, 'mode', '') or '')
    available = (
        model_type == 'bayes_linear'
        and mode != 'graph'
        and prior.size > 0
        and posterior.shape == prior.shape
        and np.isfinite(prior).all()
        and np.isfinite(posterior).all()
    )
    if not available:
        return {
            'direction_available': False,
            'direction_unavailable_reason': f'no_single_linear_direction:model_type={model_type},mode={mode}',
            'prior_direction': [],
            'posterior_direction': [],
        }
    return {
        'direction_available': True,
        'direction_unavailable_reason': '',
        'prior_direction': prior.astype(float).tolist(),
        'posterior_direction': posterior.astype(float).tolist(),
    }


def append_axis_projection_snapshot(
    *,
    sessions_root: Path,
    session_name: str,
    state: Any,
    payload: dict[str, Any],
    event_type: str,
    baseline_kind: str,
    image_id: str = '',
    move_type: str = '',
    target_score_0_100: float | None = None,
) -> Path:
    session = normalize_session_name(session_name)
    axis_id = str(payload.get('axis_id') or '').strip()
    if not axis_id:
        raise ValueError('Snapshot payload is missing axis_id')
    ids = [str(value) for value in payload.get('ids') or []]
    if not ids:
        raise ValueError(f'Snapshot payload for {axis_id} has no image IDs')
    projection_values = _finite_vector(
        payload.get('projection_values'), field='projection_values', expected_size=len(ids)
    )
    scores = _finite_vector(payload.get('scores'), field='scores', expected_size=len(ids))
    uncertainty = _finite_vector(payload.get('std'), field='std', expected_size=len(ids))
    feedback_selected_index = ids.index(image_id) if image_id else None
    path = session_axis_projection_snapshots_path(sessions_root, session)

    with _WRITE_LOCK:
        previous = _read_axis_rows(path, axis_id)
        if event_type == 'create':
            if previous:
                raise ValueError(f'Axis {axis_id} already has snapshots in {path}')
            prior_revision = 0
            interaction_index = 0
        elif event_type == 'prompt_update':
            if not previous:
                raise ValueError(f'Axis {axis_id} has no baseline snapshot before prompt update')
            prior_revision = int(previous[-1]['prior_revision']) + 1
            interaction_index = 0
        else:
            if not previous:
                raise ValueError(f'Axis {axis_id} has no baseline snapshot before {event_type}')
            prior_revision = int(previous[-1]['prior_revision'])
            interaction_index = int(previous[-1]['b_interaction']) + 1

        row = {
            'schema_version': SNAPSHOT_SCHEMA_VERSION,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'session': session,
            'event_type': str(event_type),
            'baseline_kind': str(baseline_kind),
            'axis_id': axis_id,
            'axis_name': str(payload.get('axis', {}).get('name') or payload.get('q') or axis_id),
            'dataset': Path(str(payload.get('dataset_root') or '')).name,
            'dataset_root': Path(str(payload.get('dataset_root') or '')).name,
            'model_type': str(payload.get('model_type') or ''),
            'mode': str(payload.get('mode') or ''),
            'semantic_method': str(payload.get('w0_summary', {}).get('semantic_method') or ''),
            'norm': bool(payload.get('norm')),
            'prior_revision': prior_revision,
            'b_interaction': interaction_index,
            'b_active_feedback': int(payload.get('move_count') or 0),
            'image_id': str(image_id or ''),
            'feedback_selected_index': feedback_selected_index,
            'move_type': str(move_type or ''),
            'target_score_0_100': (
                float(target_score_0_100) if target_score_0_100 is not None else None
            ),
            'ids': ids,
            'projection_values': projection_values,
            'percentile_scores_0_100': scores,
            'uncertainty_std': uncertainty,
            **_direction_fields(state),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, separators=(',', ':')) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
    return path


def ensure_axis_projection_baseline(
    *,
    sessions_root: Path,
    session_name: str,
    state: Any,
    payload: dict[str, Any],
) -> Path | None:
    axis_id = str(payload.get('axis_id') or '').strip()
    if has_axis_snapshot(sessions_root, session_name, axis_id):
        return None
    return append_axis_projection_snapshot(
        sessions_root=sessions_root,
        session_name=session_name,
        state=state,
        payload=payload,
        event_type='create',
        baseline_kind='attached_current_state',
    )
