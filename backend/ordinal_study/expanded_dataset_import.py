from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    from ..import_curated_dataset import (
        DATASETS_ROOT,
        DEFAULT_LOCAL_ROOT,
        LocalImportItem,
        build_path_item,
        ensure_empty_dataset_dir,
        log,
        parse_methods,
        precompute_dataset,
    )
except ImportError:
    from import_curated_dataset import (
        DATASETS_ROOT,
        DEFAULT_LOCAL_ROOT,
        LocalImportItem,
        build_path_item,
        ensure_empty_dataset_dir,
        log,
        parse_methods,
        precompute_dataset,
    )

from .expanded_import_utils import (
    clean_text,
    fmt_float,
    log10_or_empty,
    materialize_expanded_items,
    quantile_bin,
    quantile_edges,
)


@dataclass(frozen=True)
class ExpandedImportSpec:
    name: str
    output_name: str
    source_relpath: str
    collector: Callable[[Path], List[LocalImportItem]]
    balance_key: str
    description: str


def collect_scut_records(root: Path) -> List[LocalImportItem]:
    rows = []
    for line in (root / 'labels.txt').read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], float(parts[1])))
    edges = quantile_edges(score for _, score in rows)
    image_root = root / 'Images' / 'Images'
    items = []
    for filename, score in rows:
        path = image_root / filename
        if not path.exists():
            continue
        prefix = filename[:2].upper()
        ethnicity = {'A': 'Asian', 'C': 'Caucasian'}.get(prefix[:1], '')
        gender = {'F': 'female', 'M': 'male'}.get(prefix[1:2], '')
        score_bin = quantile_bin(score, edges)
        items.append(build_path_item(root, path, {
            'source_image_id': filename,
            'attractiveness_score': fmt_float(score),
            'attractiveness_bin': score_bin,
            'face_ethnicity_group': ethnicity,
            'face_gender_group': gender,
        }, f'attractiveness_{score_bin}_{Path(filename).stem}', score_bin))
    return items


def collect_isic2024_records(root: Path) -> List[LocalImportItem]:
    metadata = pd.read_csv(root / 'metadata.csv', low_memory=False)
    supplement = pd.read_csv(root / 'ISIC_2024_Training_Supplement.csv', low_memory=False)
    metadata = metadata.merge(supplement[['isic_id', 'iddx_1', 'iddx_full', 'tbp_lv_dnn_lesion_confidence']], on='isic_id', how='left')
    keep_cols = [
        'isic_id', 'patient_id', 'age_approx', 'sex', 'anatom_site_general',
        'clin_size_long_diam_mm', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio',
        'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL',
        'tbp_lv_deltaLB', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
        'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM',
        'tbp_lv_radial_color_std_max', 'tbp_lv_symm_2axis',
        'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
        'iddx_1', 'iddx_full', 'tbp_lv_dnn_lesion_confidence',
    ]
    metadata = metadata[[col for col in keep_cols if col in metadata.columns]].copy()
    edges = quantile_edges(metadata['clin_size_long_diam_mm'])
    items = []
    for row in metadata.itertuples(index=False):
        rowd = row._asdict()
        image_id = clean_text(rowd.get('isic_id'))
        path = root / f'{image_id}.jpg'
        if not image_id or not path.exists():
            continue
        size_bin = quantile_bin(rowd.get('clin_size_long_diam_mm'), edges)
        meta = {key: clean_text(value) for key, value in rowd.items()}
        meta['lesion_diameter_bin'] = size_bin
        items.append(build_path_item(root, path, meta, f'lesion_{size_bin}_{image_id}', size_bin))
    return items


def collect_messidor2_records(root: Path) -> List[LocalImportItem]:
    frame = pd.read_csv(root / 'messidor_data.csv')
    image_root = root / 'messidor-2' / 'messidor-2' / 'preprocess'
    items = []
    for row in frame.itertuples(index=False):
        image_name = clean_text(row.id_code)
        path = image_root / image_name
        if not path.exists():
            continue
        diagnosis = int(row.diagnosis)
        label = f'dr_{diagnosis}'
        items.append(build_path_item(root, path, {
            'source_image_id': image_name,
            'diagnosis': str(diagnosis),
            'dr_grade': str(diagnosis),
            'adjudicated_dme': clean_text(row.adjudicated_dme),
            'adjudicated_gradable': clean_text(row.adjudicated_gradable),
            'disease_name': 'diabetic retinopathy',
        }, f'diabetic_retinopathy_{label}_{Path(image_name).stem}', label))
    return items


def _birads_grade(value: object) -> str:
    match = re.search(r'([0-9]+)', clean_text(value))
    return match.group(1) if match else ''


def collect_vindr_mammo_records(root: Path) -> List[LocalImportItem]:
    frame = pd.read_csv(root / 'metadata.csv', low_memory=False)
    path_by_name = {path.name: path for path in (root / 'images_png').glob('*/*.png')}
    items = []
    for image_name, group in frame.groupby('image_id', dropna=False):
        image_name = clean_text(image_name)
        path = path_by_name.get(image_name)
        if path is None:
            continue
        first = group.iloc[0]
        birads = _birads_grade(first.get('breast_birads', ''))
        density = clean_text(first.get('density', ''))
        width = float(first.get('width') or 1.0)
        height = float(first.get('height') or 1.0)
        box_fracs = []
        finding_count = 0
        for _, row in group.iterrows():
            no_finding = int(row.get('No_Finding') or 0) == 1
            if no_finding:
                continue
            finding_count += 1
            vals = [row.get(col) for col in ('xmin', 'ymin', 'xmax', 'ymax')]
            if any(pd.isna(v) for v in vals):
                continue
            box_w = max(0.0, float(vals[2]) - float(vals[0]))
            box_h = max(0.0, float(vals[3]) - float(vals[1]))
            box_fracs.append((box_w * box_h) / max(1.0, width * height))
        finding_area = max(box_fracs) if box_fracs else 0.0
        label = f'birads_{birads or "missing"}'
        items.append(build_path_item(root, path, {
            'source_image_id': image_name,
            'patient_id': clean_text(first.get('patient_id')),
            'laterality': clean_text(first.get('laterality')),
            'view': clean_text(first.get('view')),
            'breast_birads': clean_text(first.get('breast_birads')),
            'breast_birads_grade': birads,
            'breast_density': clean_text(first.get('breast_density')),
            'breast_density_grade': density,
            'finding_categories': clean_text(first.get('finding_categories')),
            'finding_count': str(finding_count),
            'finding_box_area_fraction': fmt_float(finding_area, digits=8),
        }, f'mammo_{label}_{Path(image_name).stem}', label))
    return items


def collect_aadb_records(root: Path) -> List[LocalImportItem]:
    train = pd.read_csv(root / 'Dataset.csv')
    test = pd.read_csv(root / 'Dataset_test.csv')
    train['source_split'] = 'train'
    test['source_split'] = 'test'
    frame = pd.concat([train, test], ignore_index=True)
    image_root = root / 'datasetImages_warp256' / 'datasetImages_warp256'
    edges = quantile_edges(frame['score'])
    items = []
    for row in frame.itertuples(index=False):
        image_name = clean_text(row.ImageFile)
        path = image_root / image_name
        if not path.exists():
            continue
        score_bin = quantile_bin(row.score, edges)
        meta = {
            'source_image_id': image_name,
            'aesthetic_score': fmt_float(row.score),
            'score': fmt_float(row.score),
            'source_split': clean_text(row.source_split),
        }
        for key in ['BalacingElements', 'ColorHarmony', 'Content', 'DoF', 'Light', 'MotionBlur', 'Object', 'Repetition', 'RuleOfThirds', 'Symmetry', 'VividColor']:
            meta[key] = fmt_float(getattr(row, key))
        items.append(build_path_item(root, path, meta, f'aesthetic_{score_bin}_{Path(image_name).stem}', score_bin))
    return items


def collect_lamem_records(root: Path) -> List[LocalImportItem]:
    image_root = root / 'lamem' / 'images'
    rows = []
    for split in ('train', 'val', 'test'):
        for line in (root / 'lamem' / 'splits' / f'{split}_1.txt').read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) >= 2:
                rows.append((split, parts[0], float(parts[1])))
    edges = quantile_edges(score for _, _, score in rows)
    items = []
    for split, filename, score in rows:
        path = image_root / filename
        if not path.exists():
            continue
        score_bin = quantile_bin(score, edges)
        items.append(build_path_item(root, path, {
            'source_image_id': filename,
            'source_split': split,
            'memorability_score': fmt_float(score),
            'memorability_bin': score_bin,
        }, f'memorability_{score_bin}_{Path(filename).stem}', score_bin))
    return items


def collect_oasis_records(root: Path) -> List[LocalImportItem]:
    frame = pd.read_csv(root / 'OASIS.csv')
    beauty = pd.read_csv(root / 'Beauty ratings from Brielmann and Pelli (2019)' / 'OASIS_beauty_ratings.csv')
    beauty = beauty.rename(columns={'item': 'item_id'})
    frame['item_id'] = frame['Unnamed: 0'].astype(str).str.extract(r'([0-9]+)').astype(int)
    frame = frame.merge(beauty[['item_id', 'beauty_mean', 'beauty_SD', 'N']], on='item_id', how='left')
    image_root = root / 'Images'
    edges = quantile_edges(frame['Valence_mean'])
    items = []
    for row in frame.itertuples(index=False):
        theme = clean_text(row.Theme)
        path = image_root / f'{theme}.jpg'
        if not path.exists():
            continue
        val_bin = quantile_bin(row.Valence_mean, edges)
        items.append(build_path_item(root, path, {
            'source_image_id': clean_text(getattr(row, '_0')),
            'theme': theme,
            'category': clean_text(row.Category),
            'source': clean_text(row.Source),
            'valence_mean': fmt_float(row.Valence_mean),
            'valence_sd': fmt_float(row.Valence_SD),
            'arousal_mean': fmt_float(row.Arousal_mean),
            'arousal_sd': fmt_float(row.Arousal_SD),
            'beauty_mean': fmt_float(row.beauty_mean),
            'beauty_sd': fmt_float(row.beauty_SD),
        }, f'oasis_valence_{val_bin}_{theme}', val_bin))
    return items


def collect_house_records(root: Path) -> List[LocalImportItem]:
    info = pd.read_csv(root / 'Houses Dataset' / 'HousesInfo.txt', sep=r'\s+', header=None, names=['bedrooms', 'bathrooms', 'area_sqft', 'zipcode', 'price_usd'])
    image_root = root / 'Houses Dataset'
    edges = quantile_edges(info['price_usd'])
    items = []
    for idx, row in enumerate(info.itertuples(index=False), start=1):
        price_bin = quantile_bin(row.price_usd, edges)
        for view in ('frontal', 'bedroom', 'bathroom', 'kitchen'):
            path = image_root / f'{idx}_{view}.jpg'
            if not path.exists():
                continue
            items.append(build_path_item(root, path, {
                'house_id': str(idx),
                'view': view,
                'bedrooms': clean_text(row.bedrooms),
                'bathrooms': clean_text(row.bathrooms),
                'area_sqft': clean_text(row.area_sqft),
                'zipcode': clean_text(row.zipcode),
                'price_usd': clean_text(row.price_usd),
                'price_log10': log10_or_empty(row.price_usd),
                'price_bin': price_bin,
            }, f'house_{price_bin}_{idx}_{view}', price_bin))
    return items


EXPANDED_IMPORT_SPECS: Dict[str, ExpandedImportSpec] = {
    'scut': ExpandedImportSpec('scut', 'SCUTFBP5500', 'scut-fbp5500-v2-facial-beauty-scores', collect_scut_records, 'attractiveness_bin', 'SCUT-FBP5500 face images with attractiveness scores.'),
    'isic2024': ExpandedImportSpec('isic2024', 'ISIC2024', 'ISIC_2024_Training_Input', collect_isic2024_records, 'lesion_diameter_bin', 'ISIC 2024 dermoscopy images with TBP lesion geometry and color features.'),
    'messidor2': ExpandedImportSpec('messidor2', 'Messidor2', 'messidor2preprocess', collect_messidor2_records, 'diagnosis', 'Messidor-2 retinal fundus images with diabetic retinopathy grades.'),
    'vindr': ExpandedImportSpec('vindr', 'VinDrMammo', 'vindr-mammogram-dataset-dicom-to-png', collect_vindr_mammo_records, 'breast_birads_grade', 'VinDr-Mammo mammograms with BI-RADS, density, and finding metadata.'),
    'aadb': ExpandedImportSpec('aadb', 'AADB', 'aadb-imagedatabase', collect_aadb_records, 'score', 'AADB general photographs with aesthetic attribute ratings.'),
    'lamem': ExpandedImportSpec('lamem', 'LaMem', 'lamemdatset', collect_lamem_records, 'memorability_bin', 'LaMem images with memorability scores.'),
    'oasis': ExpandedImportSpec('oasis', 'OASIS', 'oasis_emotion', collect_oasis_records, 'valence_mean', 'OASIS emotion images with valence, arousal, and beauty ratings.'),
    'house': ExpandedImportSpec('house', 'HousePrices', 'house-dataset', collect_house_records, 'price_bin', 'House view images with property prices and size metadata.'),
}


def selected_specs(only: Sequence[str]) -> List[ExpandedImportSpec]:
    wanted = {str(value).strip().lower() for value in only if str(value).strip()}
    if not wanted:
        return list(EXPANDED_IMPORT_SPECS.values())
    out = []
    matched = set()
    for key, spec in EXPANDED_IMPORT_SPECS.items():
        aliases = {key, spec.name.lower(), spec.output_name.lower()}
        if aliases.intersection(wanted):
            out.append(spec)
            matched.update(aliases.intersection(wanted))
    missing = sorted(wanted - matched)
    if missing:
        raise RuntimeError(f'Unknown expanded ordinal dataset(s): {", ".join(missing)}')
    return out


def import_expanded_dataset(
    spec: ExpandedImportSpec,
    *,
    local_root: Path = Path(DEFAULT_LOCAL_ROOT),
    limit: int = 1500,
    seed: int = 7,
    max_edge: int = 512,
    max_source_pixels: int = 80_000_000,
    overwrite: bool = False,
    overwrite_low_level: bool = False,
) -> tuple[Path, int, bool]:
    source_root = (Path(local_root) / spec.source_relpath).resolve()
    if not source_root.exists():
        raise RuntimeError(f'Expanded source not found: {source_root}')
    dataset_root = DATASETS_ROOT / spec.output_name
    reused_existing, existing_count = ensure_empty_dataset_dir(dataset_root, overwrite=overwrite)
    if reused_existing:
        return dataset_root, int(existing_count), True
    items = spec.collector(source_root)
    if not items:
        raise RuntimeError(f'No importable images found for {spec.name}: {source_root}')
    count = materialize_expanded_items(
        name=spec.name,
        output_name=spec.output_name,
        description=spec.description,
        source_relpath=spec.source_relpath,
        dataset_root=dataset_root,
        items=items,
        limit=limit,
        seed=seed,
        balance_key=spec.balance_key,
        max_edge=max_edge,
        max_source_pixels=max_source_pixels,
        overwrite_low_level=overwrite_low_level,
    )
    return dataset_root, count, False


def precompute_expanded_dataset(dataset_root: Path, methods_csv: str, reduction: str) -> None:
    precompute_dataset(dataset_root=dataset_root, methods=parse_methods(methods_csv), reduction=reduction)
