#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

try:
    from .import_curated_dataset import (
        DATASETS_ROOT,
        DEFAULT_LOCAL_ROOT,
        LocalImportItem,
        Preset,
        build_path_item,
        ensure_empty_dataset_dir,
        log,
        materialize_local_items,
        parse_methods,
        precompute_dataset,
        sample_local_items,
    )
except ImportError:
    from import_curated_dataset import (
        DATASETS_ROOT,
        DEFAULT_LOCAL_ROOT,
        LocalImportItem,
        Preset,
        build_path_item,
        ensure_empty_dataset_dir,
        log,
        materialize_local_items,
        parse_methods,
        precompute_dataset,
        sample_local_items,
    )


EMOTION_PROXY_SCORES: Dict[str, tuple[float, float]] = {
    'anger': (-0.75, 0.80),
    'contempt': (-0.55, 0.45),
    'disgust': (-0.80, 0.65),
    'fear': (-0.85, 0.90),
    'happy': (0.90, 0.70),
    'neutral': (0.00, 0.10),
    'sad': (-0.70, 0.25),
    'surprise': (0.35, 0.95),
}

GENDER_LABELS = {'0': 'male', '1': 'female'}
RACE_LABELS = {
    '0': 'white',
    '1': 'black',
    '2': 'asian',
    '3': 'indian',
    '4': 'other',
}


@dataclass(frozen=True)
class OrdinalImportSpec:
    name: str
    output_name: str
    source_relpath: str
    collector: Callable[[Path], List[LocalImportItem]]
    balance_key: str
    description: str


def _bin01(value: float, edges: Sequence[float]) -> str:
    idx = int(np.searchsorted(np.asarray(edges, dtype=np.float32), float(value), side='right'))
    return f'bin_{idx:02d}'


def collect_utkface_records(source_root: Path) -> List[LocalImportItem]:
    image_root = source_root / 'UTKFace'
    if not image_root.exists():
        raise RuntimeError(f'UTKFace image folder not found: {image_root}')
    items: List[LocalImportItem] = []
    for image_path in sorted(image_root.iterdir()):
        if not image_path.is_file():
            continue
        parts = image_path.name.split('_')
        if len(parts) < 4:
            continue
        try:
            age = int(parts[0])
        except ValueError:
            continue
        gender_code = str(parts[1]).strip()
        race_code = str(parts[2]).strip()
        age_bin = f'{min(age // 10, 11) * 10:03d}s'
        metadata = {
            'age': str(age),
            'age_bin': age_bin,
            'gender_code': gender_code,
            'gender': GENDER_LABELS.get(gender_code, ''),
            'race_code': race_code,
            'race': RACE_LABELS.get(race_code, ''),
            'ordinal_axis_age': str(age),
        }
        items.append(build_path_item(
            source_root=source_root,
            image_path=image_path,
            metadata=metadata,
            filename_hint=f'age_{age:03d}_{image_path.stem}',
            sample_label=age_bin,
        ))
    if not items:
        raise RuntimeError(f'No UTKFace images parsed under {image_root}')
    return items


def _resolve_affectnet_path(root: Path, rel_path: str) -> tuple[Path, str]:
    rel = Path(str(rel_path).strip())
    candidates = [root / 'Train' / rel, root / 'Test' / rel]
    if len(rel.parts) >= 2:
        candidates.append(root / 'Test' / Path(rel.parts[0].capitalize(), *rel.parts[1:]))
    for path in candidates:
        if path.exists():
            split = path.relative_to(root).parts[0]
            return path, split
    raise FileNotFoundError(str(rel))


def collect_affectnet_records(source_root: Path) -> List[LocalImportItem]:
    labels_path = source_root / 'labels.csv'
    if not labels_path.exists():
        raise RuntimeError(f'AffectNet labels.csv not found: {labels_path}')
    items: List[LocalImportItem] = []
    with labels_path.open('r', newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            label = str(row.get('label', '')).strip().lower()
            if label not in EMOTION_PROXY_SCORES:
                continue
            image_path, split = _resolve_affectnet_path(source_root, str(row.get('pth', '')))
            valence, arousal = EMOTION_PROXY_SCORES[label]
            metadata = {
                'emotion': label,
                'source_split': split,
                'relFCs': str(row.get('relFCs', '')).strip(),
                'valence_proxy': f'{valence:.4f}',
                'arousal_proxy': f'{arousal:.4f}',
                'ordinal_proxy_source': 'emotion_category_circumplex_proxy',
            }
            items.append(build_path_item(
                source_root=source_root,
                image_path=image_path,
                metadata=metadata,
                filename_hint=f'{label}_{image_path.stem}',
                sample_label=label,
            ))
    if not items:
        raise RuntimeError(f'No AffectNet rows resolved from {labels_path}')
    return items


def collect_koniq_records(source_root: Path) -> List[LocalImportItem]:
    csv_path = source_root / 'koniq10k_distributions_sets.csv'
    image_root = source_root / '512x384'
    if not csv_path.exists():
        raise RuntimeError(f'KonIQ metadata CSV not found: {csv_path}')
    if not image_root.exists():
        raise RuntimeError(f'KonIQ image folder not found: {image_root}')
    rows: List[dict[str, str]] = []
    with csv_path.open('r', newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    mos_values = [float(row['MOS']) for row in rows if str(row.get('MOS', '')).strip()]
    if not mos_values:
        raise RuntimeError(f'KonIQ CSV has no MOS values: {csv_path}')
    edges = np.quantile(np.asarray(mos_values, dtype=np.float32), [0.2, 0.4, 0.6, 0.8]).tolist()
    items: List[LocalImportItem] = []
    for row in rows:
        image_name = str(row.get('image_name', '')).strip()
        if not image_name:
            continue
        image_path = image_root / image_name
        if not image_path.exists():
            continue
        mos = float(row['MOS'])
        quality_bin = _bin01(mos, edges)
        metadata = {
            'mos': f'{mos:.6f}',
            'sd': str(row.get('SD', '')).strip(),
            'rating_count': str(row.get('c_total', '')).strip(),
            'source_split': str(row.get('set', '')).strip(),
            'quality_bin': quality_bin,
            'ordinal_axis_quality_mos': f'{mos:.6f}',
        }
        for key in ('c1', 'c2', 'c3', 'c4', 'c5'):
            metadata[f'rating_distribution_{key}'] = str(row.get(key, '')).strip()
        items.append(build_path_item(
            source_root=source_root,
            image_path=image_path,
            metadata=metadata,
            filename_hint=f'quality_{quality_bin}_{image_path.stem}',
            sample_label=quality_bin,
        ))
    if not items:
        raise RuntimeError(f'No KonIQ images matched metadata under {image_root}')
    return items


ORDINAL_IMPORT_SPECS: Dict[str, OrdinalImportSpec] = {
    'utkface': OrdinalImportSpec(
        name='utkface',
        output_name='UTKFace',
        source_relpath='utkface-new',
        collector=collect_utkface_records,
        balance_key='age_bin',
        description='UTKFace age-labeled face images.',
    ),
    'affectnet': OrdinalImportSpec(
        name='affectnet',
        output_name='AffectNet',
        source_relpath='affectnet/archive (3)',
        collector=collect_affectnet_records,
        balance_key='emotion',
        description='Local AffectNet class export with explicit emotion-derived valence/arousal proxies.',
    ),
    'koniq10k': OrdinalImportSpec(
        name='koniq10k',
        output_name='KonIQ10k',
        source_relpath='koniq-10k-dataset',
        collector=collect_koniq_records,
        balance_key='quality_bin',
        description='KonIQ-10k image-quality images with MOS annotations.',
    ),
}


def import_ordinal_dataset(
    spec: OrdinalImportSpec,
    *,
    local_root: Path = Path(DEFAULT_LOCAL_ROOT),
    limit: int = 2000,
    seed: int = 7,
    max_edge: int = 512,
    max_source_pixels: int = 80_000_000,
    overwrite: bool = False,
) -> tuple[Path, int, bool]:
    source_root = (Path(local_root) / spec.source_relpath).resolve()
    if not source_root.exists():
        raise RuntimeError(f'Ordinal source not found: {source_root}')
    dataset_root = DATASETS_ROOT / spec.output_name
    reused_existing, existing_count = ensure_empty_dataset_dir(dataset_root, overwrite=overwrite)
    if reused_existing:
        return dataset_root, int(existing_count), True
    items = spec.collector(source_root)
    selected = sample_local_items(items, limit=limit, seed=seed, balance_key=spec.balance_key)
    preset = Preset(
        name=spec.name,
        kind='local',
        description=spec.description,
        source_relpath=spec.source_relpath,
        local_mode='ordinal',
        balance_key=spec.balance_key,
    )
    count = materialize_local_items(
        preset=preset,
        dataset_root=dataset_root,
        items=selected,
        max_edge=max_edge,
        max_source_pixels=max_source_pixels,
        keep_limit=limit if limit > 0 else 0,
    )
    return dataset_root, int(count), False


def precompute_ordinal_dataset(dataset_root: Path, methods_csv: str, reduction: str) -> None:
    methods = parse_methods(methods_csv)
    precompute_dataset(dataset_root=dataset_root, methods=methods, reduction=reduction)


def selected_specs(only: Sequence[str]) -> List[OrdinalImportSpec]:
    wanted = {str(value).strip().lower() for value in only if str(value).strip()}
    if not wanted:
        return list(ORDINAL_IMPORT_SPECS.values())
    out: List[OrdinalImportSpec] = []
    matched: set[str] = set()
    for key, spec in ORDINAL_IMPORT_SPECS.items():
        aliases = {key.lower(), spec.name.lower(), spec.output_name.lower()}
        if aliases.intersection(wanted):
            out.append(spec)
            matched.update(aliases.intersection(wanted))
    missing = sorted(wanted - matched)
    if missing:
        raise RuntimeError(f'Unknown ordinal dataset(s): {", ".join(missing)}')
    return out
