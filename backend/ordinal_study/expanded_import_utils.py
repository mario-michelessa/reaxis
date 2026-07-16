from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from ..import_curated_dataset import LocalImportItem, Preset, materialize_local_items, sample_local_items
except ImportError:
    from import_curated_dataset import LocalImportItem, Preset, materialize_local_items, sample_local_items

from .low_level_features import compute_dataset_low_level_features, feature_columns


def clean_text(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and not np.isfinite(value):
        return ''
    return str(value).strip()


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def fmt_float(value: object, digits: int = 6) -> str:
    out = finite_float(value)
    return '' if out is None else f'{out:.{digits}f}'


def quantile_edges(values: Iterable[object], n_bins: int = 5) -> list[float]:
    arr = np.asarray([v for v in (finite_float(value) for value in values) if v is not None], dtype=np.float64)
    if arr.size == 0:
        return []
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)[1:-1]
    return sorted(set(float(v) for v in np.quantile(arr, qs).tolist()))


def quantile_bin(value: object, edges: Sequence[float]) -> str:
    val = finite_float(value)
    if val is None:
        return 'missing'
    return f'bin_{int(np.searchsorted(np.asarray(edges, dtype=np.float64), val, side="right")):02d}'


def add_minmax_low_level_columns(dataset_root: Path, *, max_edge: int = 512, overwrite: bool = False) -> None:
    compute_dataset_low_level_features(dataset_root, overwrite=overwrite, max_edge=max_edge)
    metadata_path = Path(dataset_root) / 'metadata.csv'
    metadata = pd.read_csv(metadata_path)
    skipped: list[str] = []
    for column in feature_columns():
        values = pd.to_numeric(metadata[column], errors='coerce')
        lo = float(values.min())
        hi = float(values.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            skipped.append(f'{column}: min={lo}, max={hi}')
            continue
        metadata[f'{column}_01'] = ((values - lo) / (hi - lo)).clip(0.0, 1.0)
    metadata.to_csv(metadata_path, index=False)
    if skipped:
        (Path(dataset_root) / 'skipped_low_level_axes.txt').write_text('\n'.join(skipped) + '\n', encoding='utf-8')


def materialize_expanded_items(
    *,
    name: str,
    output_name: str,
    description: str,
    source_relpath: str,
    dataset_root: Path,
    items: Sequence[LocalImportItem],
    limit: int,
    seed: int,
    balance_key: str,
    max_edge: int,
    max_source_pixels: int,
    overwrite_low_level: bool,
) -> int:
    selected = sample_local_items(items, limit=limit, seed=seed, balance_key=balance_key)
    preset = Preset(
        name=name,
        kind='local',
        description=description,
        source_relpath=source_relpath,
        local_mode='expanded_ordinal',
        balance_key=balance_key,
    )
    count = materialize_local_items(
        preset=preset,
        dataset_root=dataset_root,
        items=selected,
        max_edge=max_edge,
        max_source_pixels=max_source_pixels,
        keep_limit=limit if int(limit) > 0 else 0,
    )
    add_minmax_low_level_columns(dataset_root, max_edge=max_edge, overwrite=overwrite_low_level)
    return int(count)


def log10_or_empty(value: object) -> str:
    val = finite_float(value)
    if val is None or val <= 0:
        return ''
    return fmt_float(math.log10(val))
