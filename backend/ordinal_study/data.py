from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

try:
    from ..gallery_backend import ImageGalleryEngine
except ImportError:
    from gallery_backend import ImageGalleryEngine

from .config import AxisSpec, DATASET_SPECS, DatasetSpec


@dataclass(frozen=True)
class OrdinalTask:
    dataset: str
    domain: str
    axis_name: str
    axis_field: str
    query: str
    low_text: str
    high_text: str
    ladder: tuple[str, ...]
    label_source: str
    dataset_root: Path
    ids: List[str]
    paths: List[str]
    X: np.ndarray
    y_raw: np.ndarray
    y01: np.ndarray
    axis_category: str = 'high_level'
    raw_min_value: float = 0.0
    raw_max_value: float = 1.0

    @property
    def task_id(self) -> str:
        return f'{self.dataset}__{self.axis_field}'

    @property
    def id_to_index(self) -> dict[str, int]:
        return {image_id: idx for idx, image_id in enumerate(self.ids)}


def normalize_with_bounds(values: np.ndarray, axis: AxisSpec) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    lo = float(axis.min_value)
    hi = float(axis.max_value)
    span = hi - lo
    if span <= 0.0:
        raise ValueError(f'Invalid normalization bounds for axis {axis.name}: {lo}, {hi}')
    return np.clip((arr - lo) / span, 0.0, 1.0).astype(np.float32)


def _load_metadata(dataset_root: Path, ids: Sequence[str]) -> pd.DataFrame:
    metadata_path = dataset_root / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f'Missing metadata.csv for ordinal dataset: {metadata_path}')
    meta = pd.read_csv(metadata_path)
    meta.columns = [str(col).strip() for col in meta.columns]
    if 'image' not in meta.columns:
        raise RuntimeError(f'metadata.csv must contain image column: {metadata_path}')
    meta['image'] = meta['image'].astype(str).str.strip()
    meta = meta.drop_duplicates(subset=['image'], keep='first').set_index('image', drop=False)
    aligned = meta.reindex([str(image_id) for image_id in ids])
    aligned['image'] = [str(image_id) for image_id in ids]
    return aligned.reset_index(drop=True)


def _load_clip_embeddings(dataset_root: Path, engine: ImageGalleryEngine, entries: Sequence[object]) -> np.ndarray:
    embeddings = engine._load_embeddings_only(list(entries), method='clip')
    if embeddings is None:
        raise RuntimeError(f'CLIP embedding cache missing for ordinal dataset: {dataset_root}')
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-8)).astype(np.float32)


def load_tasks(dataset_names: Sequence[str]) -> List[OrdinalTask]:
    tasks: List[OrdinalTask] = []
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_SPECS:
            raise RuntimeError(f'Unknown ordinal dataset: {dataset_name}')
        spec = DATASET_SPECS[dataset_name]
        tasks.extend(load_dataset_tasks(spec))
    return tasks


def load_dataset_tasks(spec: DatasetSpec) -> List[OrdinalTask]:
    dataset_root = Path(spec.root)
    if not dataset_root.exists():
        raise FileNotFoundError(f'Prepared ordinal dataset not found: {dataset_root}')
    engine = ImageGalleryEngine(str(dataset_root))
    entries = engine.list_images()
    if not entries:
        raise RuntimeError(f'No images found in prepared ordinal dataset: {dataset_root}')
    ids = [str(entry.id) for entry in entries]
    paths = [str(entry.path) for entry in entries]
    metadata = _load_metadata(dataset_root, ids)
    X = _load_clip_embeddings(dataset_root, engine, entries)
    out: List[OrdinalTask] = []
    for axis in spec.axes:
        if axis.field not in metadata.columns:
            raise RuntimeError(f'Axis field {axis.field!r} missing in {dataset_root / "metadata.csv"}')
        values = pd.to_numeric(metadata[axis.field], errors='coerce')
        keep = values.notna().to_numpy(dtype=bool)
        if int(np.sum(keep)) < 3:
            raise RuntimeError(f'Axis {axis.name} has fewer than 3 numeric labels in {dataset_root}')
        y_raw = values.to_numpy(dtype=np.float32)[keep]
        out.append(OrdinalTask(
            dataset=spec.name,
            domain=spec.domain,
            axis_name=axis.name,
            axis_field=axis.field,
            query=axis.query,
            low_text=axis.low_text,
            high_text=axis.high_text,
            ladder=tuple(axis.ladder),
            label_source=axis.label_source,
            dataset_root=dataset_root,
            ids=[ids[idx] for idx, flag in enumerate(keep) if bool(flag)],
            paths=[paths[idx] for idx, flag in enumerate(keep) if bool(flag)],
            X=np.asarray(X[keep], dtype=np.float32),
            y_raw=np.asarray(y_raw, dtype=np.float32),
            y01=normalize_with_bounds(y_raw, axis),
            axis_category=axis.category,
            raw_min_value=float(axis.min_value),
            raw_max_value=float(axis.max_value),
        ))
    return out
