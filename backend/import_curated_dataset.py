#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import warnings
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from .gallery_backend import ImageGalleryEngine
    from .embeddings import normalize_multimodal_method
except ImportError:
    from gallery_backend import ImageGalleryEngine
    from embeddings import normalize_multimodal_method


DATASETS_ROOT = (Path(__file__).resolve().parent.parent / 'data' / 'datasets').resolve()
DEFAULT_LOCAL_ROOT = Path('/mnt/raid/mario/datasets')
DEFAULT_METHODS = ('color_rgb', 'siglip2', 'clip', 'dino')
DEFAULT_REDUCTION = 'pca'
DEFAULT_MAX_EDGE = 512
DEFAULT_LIMIT = 2000
DEFAULT_MAX_SOURCE_PIXELS = 80_000_000
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.heic', '.heif'}
IMAGENETTE_LABELS = {
    'n01440764': 'tench',
    'n02102040': 'english springer',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'french horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute',
}


@dataclass(frozen=True)
class Preset:
    name: str
    kind: str
    description: str
    hf_dataset: str = ''
    hf_config: Optional[str] = None
    hf_split: str = 'train'
    hf_image_column: Optional[str] = None
    filter_regex: Optional[str] = None
    archive_url: str = ''
    archive_subdir: str = ''
    split_dirs: Tuple[str, ...] = ()
    label_map: Optional[Dict[str, str]] = None
    source_relpath: str = ''
    local_mode: str = ''
    balance_key: str = ''


@dataclass(frozen=True)
class LocalImportItem:
    source: Any
    relative_source: str
    filename_hint: str
    metadata: Dict[str, str]
    sample_label: str = ''


PRESETS: Dict[str, Preset] = {
    'stars_hubble': Preset(
        name='stars_hubble',
        kind='hf',
        description='ESA/Hubble astronomy images filtered for star and stellar content.',
        hf_dataset='Supermaxman/esa-hubble',
        hf_split='train',
        hf_image_column='image',
        filter_regex=r'\b(star|stars|stellar|starburst|star[- ]forming|star cluster|open cluster|globular cluster)\b',
    ),
    'paintings_wikiart': Preset(
        name='paintings_wikiart',
        kind='hf',
        description='WikiArt paintings with artist, genre, and style metadata.',
        hf_dataset='huggan/wikiart',
        hf_split='train',
        hf_image_column='image',
    ),
    'imagenette_160': Preset(
        name='imagenette_160',
        kind='archive',
        description='Fast.ai Imagenette 160px, balanced across 10 ImageNet classes.',
        archive_url='https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',
        archive_subdir='imagenette2-160',
        split_dirs=('train', 'val'),
        label_map=IMAGENETTE_LABELS,
    ),
    'tiny_imagenet_200': Preset(
        name='tiny_imagenet_200',
        kind='archive',
        description='Stanford Tiny-ImageNet-200, balanced across 200 ImageNet synsets.',
        archive_url='https://cs231n.stanford.edu/tiny-imagenet-200.zip',
        archive_subdir='tiny-imagenet-200',
        split_dirs=('train',),
    ),
    'ancient-tamil-stone-inscriptions': Preset(
        name='ancient-tamil-stone-inscriptions',
        kind='local',
        description='Local inscription image collection with multiple inscription groups.',
        source_relpath='ancient-tamil-stone-inscriptions',
        local_mode='ancient_tamil',
        balance_key='inscription_id',
    ),
    'archaeomind-images': Preset(
        name='archaeomind-images',
        kind='local',
        description='Local ArchaeoMind artifact vs non-artifact image dataset.',
        source_relpath='archaeomind-images/ArchaeoMind_dataset',
        local_mode='split_label_tree',
        balance_key='label',
    ),
    'broden1_224': Preset(
        name='broden1_224',
        kind='local',
        description='Local Broden dataset with source split and annotation availability metadata.',
        source_relpath='broden1_224',
        local_mode='broden',
        balance_key='broden_source',
    ),
    'brain-mri-images-for-brain-tumor-detection': Preset(
        name='brain-mri-images-for-brain-tumor-detection',
        kind='local',
        description='Local brain MRI yes/no tumor detection dataset.',
        source_relpath='brain-mri-images-for-brain-tumor-detection',
        local_mode='brain_mri',
        balance_key='label',
    ),
    'celeba-dataset': Preset(
        name='celeba-dataset',
        kind='local',
        description='Local CelebA face dataset with attributes, boxes, landmarks, and partitions.',
        source_relpath='celeba-dataset',
        local_mode='celeba',
    ),
    'chest-xray-pneumonia-balanced-dataset': Preset(
        name='chest-xray-pneumonia-balanced-dataset',
        kind='local',
        description='Local chest X-ray dataset with train/val/test splits and pneumonia labels.',
        source_relpath='chest-xray-pneumonia-balanced-dataset',
        local_mode='split_label_tree',
        balance_key='label',
    ),
    'cub_local': Preset(
        name='cub',
        kind='local',
        description='Local CUB bird dataset with train/test species folders.',
        source_relpath='CUB',
        local_mode='split_label_tree',
        balance_key='label',
    ),
    'egyptian-archaeological-stone-statue-dataset': Preset(
        name='egyptian-archaeological-stone-statue-dataset',
        kind='local',
        description='Local Egyptian statue image collection with statue metadata from CSV.',
        source_relpath='egyptian-archaeological-stone-statue-dataset',
        local_mode='egyptian_statue',
        balance_key='statue',
    ),
    'ham10000_local': Preset(
        name='ham10000',
        kind='local',
        description='Local HAM10000 dermoscopy dataset with lesion metadata CSV.',
        source_relpath='HAM10000',
        local_mode='ham10000',
        balance_key='dx',
    ),
    'imagenet_n02958343_local': Preset(
        name='imagenet_n02958343',
        kind='local',
        description='Local single-class ImageNet subset containing only synset n02958343 images.',
        source_relpath='ImageNet_subsets/images/n02958343',
        local_mode='label_tree',
        balance_key='label',
    ),
    'imagenet_r_local': Preset(
        name='imagenet_r',
        kind='local',
        description='Local ImageNet-R rendition dataset with split/style/class directory layout.',
        source_relpath='Imagenet-R',
        local_mode='imagenet_r',
        balance_key='label',
    ),
    'imagenette_local': Preset(
        name='imagenette',
        kind='local',
        description='Local Imagenette dataset with train/val splits.',
        source_relpath='imagenette',
        local_mode='imagenette',
        balance_key='label',
    ),
    'inat2021birds_local': Preset(
        name='inat2021birds',
        kind='local',
        description='Local iNat2021 bird dataset with joined taxonomy and observation metadata.',
        source_relpath='inat2021birds',
        local_mode='inat2021birds',
        balance_key='label',
    ),
    'mapillary_vistas_local': Preset(
        name='mapillary_vistas',
        kind='local',
        description='Local Mapillary Vistas dataset with v2.0 semantic label summaries.',
        source_relpath='Mapillary Vistas',
        local_mode='mapillary_vistas',
    ),
    'metal-albums-artwork': Preset(
        name='metal-albums-artwork',
        kind='local',
        description='Local album artwork dataset stored as base64 images inside CSV.',
        source_relpath='metal-albums-artwork',
        local_mode='metal_album',
        balance_key='artist_main_genre',
    ),
    'qajar-dynasty-iranian-carpet-dataset': Preset(
        name='qajar-dynasty-iranian-carpet-dataset',
        kind='local',
        description='Local Qajar carpet dataset with metadata in XLSX.',
        source_relpath='qajar-dynasty-iranian-carpet-dataset',
        local_mode='qajar_carpet',
        balance_key='design_type',
    ),
    'sinhala-early-brahmi-inscription-dataset': Preset(
        name='sinhala-early-brahmi-inscription-dataset',
        kind='local',
        description='Local Sinhala Brahmi inscription dataset organized by inscription id.',
        source_relpath='sinhala-early-brahmi-inscription-dataset',
        local_mode='sinhala_brahmi',
        balance_key='group_id',
    ),
}


def slugify(text: str) -> str:
    s = re.sub(r'[^a-z0-9]+', '-', str(text or '').strip().lower())
    s = re.sub(r'-+', '-', s).strip('-')
    return s or 'item'


def log(message: str) -> None:
    print(f'[import] {message}', flush=True)


def ensure_empty_dataset_dir(dataset_root: Path, overwrite: bool) -> tuple[bool, int]:
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    if dataset_root.exists():
        if overwrite:
            shutil.rmtree(dataset_root)
        else:
            engine = ImageGalleryEngine(str(dataset_root))
            image_count = len(engine.list_images())
            if image_count > 0:
                log(f'reusing existing dataset with {image_count} images: {dataset_root}')
                return True, image_count
            existing_files = list(dataset_root.iterdir())
            if existing_files:
                raise RuntimeError(
                    f'Dataset directory exists but has no images: {dataset_root}. '
                    'Pass --overwrite to replace it.'
                )
    dataset_root.mkdir(parents=True, exist_ok=True)
    return False, 0


def load_hf_split(dataset_id: str, split: str, config: Optional[str], seed: int, limit: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('The datasets package is required. Install with `pip install datasets`.') from exc

    ds = load_dataset(dataset_id, name=config, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=max(1000, limit * 10))
    return ds


def infer_hf_image_column(dataset: Any, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    features = getattr(dataset, 'features', None) or {}
    try:
        from datasets.features import Image as HFImage
    except ImportError:
        HFImage = None
    for key, feature in features.items():
        if HFImage is not None and isinstance(feature, HFImage):
            return key
        if feature.__class__.__name__ == 'Image':
            return key
    sample = next(iter(dataset))
    for key, value in sample.items():
        if isinstance(value, Image.Image):
            return key
        if isinstance(value, dict) and ('bytes' in value or 'path' in value):
            return key
    raise RuntimeError('Could not infer image column from Hugging Face dataset')


def value_to_string(value: Any, feature: Any = None) -> str:
    if value is None:
        return ''
    try:
        from datasets.features import ClassLabel
    except ImportError:
        ClassLabel = None
    if ClassLabel is not None and feature is not None and isinstance(feature, ClassLabel):
        try:
            if isinstance(value, int):
                return feature.int2str(value)
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        if len(value) > 12:
            return ''
        return json.dumps(list(value), ensure_ascii=True)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(item, (str, int, float, bool)) or item is None:
                out[str(key)] = item
        if not out:
            return ''
        return json.dumps(out, ensure_ascii=True)
    return ''


def extract_pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    if isinstance(value, dict):
        img_bytes = value.get('bytes')
        if img_bytes:
            return Image.open(io.BytesIO(img_bytes))
        img_path = value.get('path')
        if img_path:
            return Image.open(img_path)
    raise RuntimeError('Unsupported image payload in source dataset')


def open_checked_image(source: Any, max_source_pixels: int) -> Optional[Image.Image]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            if isinstance(source, (str, Path)):
                image = Image.open(source)
            else:
                image = extract_pil_image(source)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError):
        return None
    width, height = image.size
    if (int(width) * int(height)) > int(max_source_pixels):
        scale = (float(max_source_pixels) / float(int(width) * int(height))) ** 0.5
        resized = image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
        image.close()
        return resized
    return image


def prepare_image(image: Image.Image, max_edge: int) -> Image.Image:
    img = ImageOps.exif_transpose(image).convert('RGB')
    width, height = img.size
    max_dim = max(width, height)
    if max_dim > max_edge:
        scale = float(max_edge) / float(max_dim)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def save_prepared_image(image: Image.Image, out_path: Path, max_edge: int) -> None:
    prepared = prepare_image(image, max_edge=max_edge)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.save(out_path, format='JPEG', quality=92, optimize=True)


def row_matches_filter(row: Dict[str, Any], features: Dict[str, Any], image_column: str, pattern: Optional[re.Pattern[str]]) -> bool:
    if pattern is None:
        return True
    chunks: List[str] = []
    for key, value in row.items():
        if key == image_column:
            continue
        text = value_to_string(value, features.get(key))
        if text:
            chunks.append(text.lower())
    return bool(pattern.search(' '.join(chunks)))


def write_metadata_csv(dataset_root: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with (dataset_root / 'metadata.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fields})


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f'downloading {url} -> {out_path}')
    with urllib.request.urlopen(url) as response, out_path.open('wb') as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    log(f'extracting {archive_path.name}')
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(dest_dir)
        return
    with tarfile.open(archive_path, 'r:*') as tf:
        tf.extractall(dest_dir)


def list_images_recursive(root: Path) -> List[Path]:
    out: List[Path] = []
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            out.append(path)
    return sorted(out)


def balanced_sample_by_label(items: Sequence[Tuple[str, Path]], limit: int, seed: int) -> List[Tuple[str, Path]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Path]] = {}
    for label, path in items:
        buckets.setdefault(label, []).append(path)
    for paths in buckets.values():
        rng.shuffle(paths)
    labels = sorted(buckets.keys())
    if not labels:
        return []
    selected: List[Tuple[str, Path]] = []
    cursor = {label: 0 for label in labels}
    while len(selected) < limit:
        made_progress = False
        for label in labels:
            idx = cursor[label]
            bucket = buckets[label]
            if idx >= len(bucket):
                continue
            selected.append((label, bucket[idx]))
            cursor[label] = idx + 1
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    return selected


def balanced_sample_indices_by_label(labels: Sequence[str], limit: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    buckets: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        key = str(label or '').strip() or '__unlabeled__'
        buckets.setdefault(key, []).append(idx)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets.keys())
    selected: List[int] = []
    cursor = {key: 0 for key in keys}
    while len(selected) < limit:
        made_progress = False
        for key in keys:
            idx = cursor[key]
            bucket = buckets[key]
            if idx >= len(bucket):
                continue
            selected.append(bucket[idx])
            cursor[key] = idx + 1
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    return selected


def sample_local_items(items: Sequence[LocalImportItem], limit: int, seed: int, balance_key: str = '') -> List[LocalImportItem]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    if balance_key:
        labels = [item.sample_label or item.metadata.get(balance_key, '') for item in items]
        if any(str(label or '').strip() for label in labels):
            indices = balanced_sample_indices_by_label(labels=labels, limit=limit, seed=seed)
            return [items[idx] for idx in indices]
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    chosen = sorted(indices[:limit])
    return [items[idx] for idx in chosen]


def build_local_image_name(index: int, hint: str) -> str:
    stem = slugify(hint)[:72]
    if stem == 'item':
        return f'{index:05d}.jpg'
    return f'{index:05d}_{stem}.jpg'


def materialize_local_items(
    preset: Preset,
    dataset_root: Path,
    items: Sequence[LocalImportItem],
    max_edge: int,
    max_source_pixels: int,
    keep_limit: int = 0,
) -> int:
    metadata_rows: List[Dict[str, str]] = []
    skipped_invalid = 0
    kept = 0
    for idx, item in enumerate(items):
        if keep_limit > 0 and kept >= keep_limit:
            break
        image = open_checked_image(item.source, max_source_pixels=max_source_pixels)
        if image is None:
            skipped_invalid += 1
            if skipped_invalid <= 10 or (skipped_invalid % 50) == 0:
                log(f'skipping unreadable image source={item.relative_source or item.filename_hint}')
            continue
        image_name = build_local_image_name(index=kept, hint=item.filename_hint)
        try:
            save_prepared_image(image=image, out_path=dataset_root / image_name, max_edge=max_edge)
        finally:
            image.close()
        metadata = {
            'image': image_name,
            'source_dataset': preset.name,
        }
        if item.relative_source:
            metadata['source_path'] = item.relative_source
        for key, value in item.metadata.items():
            text = value_to_string(value)
            if text:
                metadata[key] = text
        metadata_rows.append(metadata)
        kept += 1
        if kept % 100 == 0:
            log(f'saved {kept} images')
    if skipped_invalid:
        log(f'skipped invalid images={skipped_invalid}')
    write_metadata_csv(dataset_root, metadata_rows)
    return kept


def load_csv_metadata_by_key(
    csv_path: Path,
    key_field: str,
    wanted_keys: Optional[set[str]] = None,
    prefix: str = '',
) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with csv_path.open('r', newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = str(row.get(key_field, '')).strip()
            if not key:
                continue
            if wanted_keys is not None and key not in wanted_keys:
                continue
            metadata: Dict[str, str] = {}
            for column, value in row.items():
                if column == key_field:
                    continue
                text = value_to_string(value)
                if text:
                    metadata[f'{prefix}{column}'] = text
            out[key] = metadata
    return out


def excel_column_to_index(cell_ref: str) -> int:
    letters = ''.join(ch for ch in cell_ref if ch.isalpha()).upper()
    value = 0
    for ch in letters:
        value = (value * 26) + (ord(ch) - ord('A') + 1)
    return max(0, value - 1)


def load_simple_xlsx_rows(xlsx_path: Path) -> List[Dict[str, str]]:
    ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings: List[str] = []
        if 'xl/sharedStrings.xml' in zf.namelist():
            root = ET.fromstring(zf.read('xl/sharedStrings.xml'))
            for node in root.findall('a:si', ns):
                text = ''.join(piece.text or '' for piece in node.findall('.//a:t', ns))
                shared_strings.append(text)
        workbook = ET.fromstring(zf.read('xl/workbook.xml'))
        rels = ET.fromstring(zf.read('xl/_rels/workbook.xml.rels'))
        rel_map = {rel.attrib['Id']: rel.attrib['Target'] for rel in rels}
        first_sheet = workbook.find('a:sheets/a:sheet', ns)
        if first_sheet is None:
            return []
        rel_id = first_sheet.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id', '')
        target = rel_map.get(rel_id, '')
        if not target:
            return []
        sheet_root = ET.fromstring(zf.read(f'xl/{target.lstrip("/")}'))
        rows_raw: List[Dict[int, str]] = []
        for row in sheet_root.findall('.//a:sheetData/a:row', ns):
            cells: Dict[int, str] = {}
            for cell in row.findall('a:c', ns):
                cell_ref = cell.attrib.get('r', '')
                idx = excel_column_to_index(cell_ref)
                cell_type = cell.attrib.get('t', '')
                text = ''
                if cell_type == 'inlineStr':
                    is_node = cell.find('a:is', ns)
                    if is_node is not None:
                        text = ''.join(piece.text or '' for piece in is_node.findall('.//a:t', ns))
                else:
                    value_node = cell.find('a:v', ns)
                    if value_node is not None and value_node.text is not None:
                        text = value_node.text
                if cell_type == 's' and text:
                    try:
                        text = shared_strings[int(text)]
                    except (IndexError, ValueError):
                        pass
                cells[idx] = text
            if cells:
                rows_raw.append(cells)
    if not rows_raw:
        return []
    max_cols = max(max(row.keys(), default=0) for row in rows_raw) + 1
    header = [rows_raw[0].get(idx, '').strip() for idx in range(max_cols)]
    out: List[Dict[str, str]] = []
    for row in rows_raw[1:]:
        record: Dict[str, str] = {}
        for idx, column in enumerate(header):
            if not column:
                continue
            text = row.get(idx, '').strip()
            if text:
                record[column] = text
        if record:
            out.append(record)
    return out


def load_columnar_json_rows(json_path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(json_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(f'Expected a column-oriented JSON object: {json_path}')
    rows_by_key: Dict[str, Dict[str, Any]] = {}
    for column, values in raw.items():
        if not isinstance(values, dict):
            continue
        for row_key, value in values.items():
            rows_by_key.setdefault(str(row_key), {})[str(column)] = value
    def sort_key(row_key: str) -> tuple[int, Any]:
        text = str(row_key)
        if text.isdigit():
            return (0, int(text))
        return (1, text)
    return [rows_by_key[row_key] for row_key in sorted(rows_by_key.keys(), key=sort_key)]


def maybe_load_tiny_imagenet_words(extract_root: Path) -> Dict[str, str]:
    words_path = extract_root / 'tiny-imagenet-200' / 'words.txt'
    mapping: Dict[str, str] = {}
    if not words_path.exists():
        return mapping
    with words_path.open('r', encoding='utf-8') as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or '\t' not in raw:
                continue
            wnid, label = raw.split('\t', 1)
            mapping[wnid.strip()] = label.strip().split(',')[0]
    return mapping


def build_path_item(source_root: Path, image_path: Path, metadata: Dict[str, str], filename_hint: str, sample_label: str = '') -> LocalImportItem:
    rel = image_path.relative_to(source_root)
    return LocalImportItem(
        source=image_path,
        relative_source=str(rel),
        filename_hint=filename_hint or image_path.stem,
        metadata=metadata,
        sample_label=sample_label,
    )


def collect_split_label_tree_records(source_root: Path, allowed_splits: Optional[set[str]] = None) -> List[LocalImportItem]:
    items: List[LocalImportItem] = []
    for image_path in list_images_recursive(source_root):
        rel = image_path.relative_to(source_root)
        parts = rel.parts
        metadata: Dict[str, str] = {}
        split = parts[0] if len(parts) >= 3 else ''
        if allowed_splits is not None and split and split not in allowed_splits:
            continue
        label = parts[1] if len(parts) >= 3 else (parts[0] if len(parts) >= 2 else '')
        if split:
            metadata['source_split'] = split
        if label:
            metadata['label'] = label
        metadata['source_parent'] = image_path.parent.name
        items.append(build_path_item(
            source_root=source_root,
            image_path=image_path,
            metadata=metadata,
            filename_hint=f'{label}_{image_path.stem}',
            sample_label=label,
        ))
    return items


def collect_label_tree_records(source_root: Path, label_field: str = 'label') -> List[LocalImportItem]:
    items: List[LocalImportItem] = []
    for image_path in list_images_recursive(source_root):
        rel = image_path.relative_to(source_root)
        parts = rel.parts
        label = parts[0] if len(parts) >= 2 else image_path.parent.name
        metadata: Dict[str, str] = {
            label_field: label,
            'source_parent': image_path.parent.name,
        }
        if label_field != 'label':
            metadata['label'] = label
        items.append(build_path_item(
            source_root=source_root,
            image_path=image_path,
            metadata=metadata,
            filename_hint=f'{label}_{image_path.stem}',
            sample_label=label,
        ))
    return items


def collect_brain_mri_records(source_root: Path) -> List[LocalImportItem]:
    image_root = source_root / 'brain_tumor_dataset'
    if not image_root.exists():
        image_root = source_root
    return collect_label_tree_records(image_root, label_field='label')


def collect_ham10000_records(source_root: Path) -> List[LocalImportItem]:
    metadata_path = source_root / 'HAM10000_metadata.csv'
    if not metadata_path.exists():
        raise RuntimeError(f'HAM10000 metadata CSV not found: {metadata_path}')
    metadata_by_id = load_csv_metadata_by_key(
        csv_path=metadata_path,
        key_field='image_id',
        prefix='',
    )
    image_paths: List[Path] = []
    for folder_name in ('ham10000_images_part_1', 'ham10000_images_part_2'):
        folder = source_root / folder_name
        if folder.exists():
            image_paths.extend(list_images_recursive(folder))
    if not image_paths:
        image_paths = list_images_recursive(source_root)
    items: List[LocalImportItem] = []
    for image_path in image_paths:
        image_id = image_path.stem
        metadata = dict(metadata_by_id.get(image_id, {}))
        metadata['image_id'] = image_id
        metadata['source_parent'] = image_path.parent.name
        items.append(build_path_item(
            source_root=source_root,
            image_path=image_path,
            metadata=metadata,
            filename_hint=f"{metadata.get('dx', '')}_{image_id}",
            sample_label=metadata.get('dx', ''),
        ))
    return items


def collect_broden_records(source_root: Path) -> List[LocalImportItem]:
    image_root = source_root / 'images'
    index_path = source_root / 'index.csv'
    if not image_root.exists():
        raise RuntimeError(f'Broden image directory not found: {image_root}')
    if not index_path.exists():
        raise RuntimeError(f'Broden index CSV not found: {index_path}')
    rows_by_image = load_csv_metadata_by_key(
        csv_path=index_path,
        key_field='image',
        prefix='',
    )
    items: List[LocalImportItem] = []
    for rel_path, row in rows_by_image.items():
        image_path = image_root / rel_path
        if not image_path.exists():
            continue
        rel_parts = Path(rel_path).parts
        source_name = rel_parts[0] if len(rel_parts) >= 2 else image_path.parent.name
        metadata: Dict[str, str] = {
            'broden_source': source_name,
            'broden_image_relpath': rel_path,
        }
        for key in ('split', 'ih', 'iw', 'sh', 'sw'):
            text = row.get(key, '')
            if text:
                metadata[key] = text
        for category in ('color', 'object', 'part', 'material', 'scene', 'texture'):
            raw = str(row.get(category, '')).strip()
            metadata[f'has_{category}'] = '1' if raw else '0'
            if raw:
                metadata[f'{category}_annotation'] = raw
        items.append(build_path_item(
            source_root=image_root,
            image_path=image_path,
            metadata=metadata,
            filename_hint=f'{source_name}_{image_path.stem}',
            sample_label=source_name,
        ))
    return items


def collect_imagenet_r_records(source_root: Path) -> List[LocalImportItem]:
    items: List[LocalImportItem] = []
    for image_path in list_images_recursive(source_root):
        rel = image_path.relative_to(source_root)
        parts = rel.parts
        split = parts[0] if len(parts) >= 4 else ''
        style = parts[1] if len(parts) >= 4 else ''
        label = parts[2] if len(parts) >= 4 else (parts[1] if len(parts) >= 3 else image_path.parent.name)
        metadata: Dict[str, str] = {
            'source_parent': image_path.parent.name,
        }
        if split:
            metadata['source_split'] = split
        if style:
            metadata['rendition_style'] = style
        if label:
            metadata['label'] = label
        wnid = image_path.stem.split('_', 1)[0]
        if wnid.startswith('n'):
            metadata['wnid'] = wnid
        items.append(build_path_item(
            source_root=source_root,
            image_path=image_path,
            metadata=metadata,
            filename_hint=f'{style}_{label}_{image_path.stem}',
            sample_label=label,
        ))
    return items


def collect_imagenette_records(source_root: Path) -> List[LocalImportItem]:
    return collect_split_label_tree_records(source_root, allowed_splits={'train', 'val'})


def collect_inat2021birds_records(source_root: Path) -> List[LocalImportItem]:
    image_root = source_root / 'bird_train'
    if not image_root.exists():
        raise RuntimeError(f'iNat2021 birds image directory not found: {image_root}')
    classes_path = source_root / 'bird_classes.json'
    images_path = source_root / 'bird_images.json'
    annotations_path = source_root / 'bird_annotations.json'
    for required_path in (classes_path, images_path, annotations_path):
        if not required_path.exists():
            raise RuntimeError(f'iNat2021 birds metadata file not found: {required_path}')
    class_rows = load_columnar_json_rows(classes_path)
    image_rows = load_columnar_json_rows(images_path)
    annotation_rows = load_columnar_json_rows(annotations_path)
    class_by_id = {
        value_to_string(row.get('id', '')): row
        for row in class_rows
        if value_to_string(row.get('id', ''))
    }
    image_row_by_stem: Dict[str, Dict[str, Any]] = {}
    for row in image_rows:
        file_name = value_to_string(row.get('file_name', ''))
        if file_name:
            image_row_by_stem.setdefault(Path(file_name).stem, row)
    annotation_by_image_id = {
        value_to_string(row.get('image_id', '')): row
        for row in annotation_rows
        if value_to_string(row.get('image_id', ''))
    }
    items: List[LocalImportItem] = []
    taxonomy_fields = (
        ('supercategory', 'supercategory'),
        ('kingdom', 'taxon_kingdom'),
        ('phylum', 'taxon_phylum'),
        ('class', 'taxon_class'),
        ('order', 'taxon_order'),
        ('family', 'taxon_family'),
        ('genus', 'taxon_genus'),
        ('specific_epithet', 'specific_epithet'),
    )
    image_fields = (
        ('id', 'inat_image_id'),
        ('width', 'source_width'),
        ('height', 'source_height'),
        ('license', 'license_id'),
        ('rights_holder', 'rights_holder'),
        ('date', 'observed_at_ms'),
        ('latitude', 'latitude'),
        ('longitude', 'longitude'),
        ('location_uncertainty', 'location_uncertainty'),
    )
    for image_path in list_images_recursive(image_root):
        class_dir = image_path.parent.name
        image_row = image_row_by_stem.get(image_path.stem, {})
        image_id = value_to_string(image_row.get('id', ''))
        annotation_row = annotation_by_image_id.get(image_id, {}) if image_id else {}
        category_id = value_to_string(annotation_row.get('category_id', ''))
        class_row = class_by_id.get(category_id, {})
        scientific_name = value_to_string(class_row.get('name', ''))
        if not scientific_name:
            species_tokens = class_dir.split('_')[-2:]
            scientific_name = ' '.join(token for token in species_tokens if token)
        metadata: Dict[str, str] = {
            'source_split': 'bird_train',
            'source_parent': class_dir,
            'image_dir_name': value_to_string(class_row.get('image_dir_name', '')) or class_dir,
        }
        if scientific_name:
            metadata['label'] = scientific_name
            metadata['scientific_name'] = scientific_name
        common_name = value_to_string(class_row.get('common_name', ''))
        if common_name:
            metadata['common_name'] = common_name
        if category_id:
            metadata['inat_category_id'] = category_id
        for source_field, output_field in taxonomy_fields:
            text = value_to_string(class_row.get(source_field, ''))
            if text:
                metadata[output_field] = text
        file_name = value_to_string(image_row.get('file_name', ''))
        if file_name:
            metadata['source_json_file_name'] = file_name
        for source_field, output_field in image_fields:
            text = value_to_string(image_row.get(source_field, ''))
            if text:
                metadata[output_field] = text
        items.append(build_path_item(
            source_root=image_root,
            image_path=image_path,
            metadata=metadata,
            filename_hint=f'{scientific_name or class_dir}_{image_path.stem}',
            sample_label=scientific_name or class_dir,
        ))
    return items


def collect_mapillary_vistas_records(source_root: Path) -> List[LocalImportItem]:
    items: List[LocalImportItem] = []
    for split in ('training', 'validation'):
        image_root = source_root / split / 'images'
        label_root = source_root / split / 'v2.0' / 'labels'
        if not image_root.exists():
            raise RuntimeError(f'Mapillary image directory not found: {image_root}')
        if not label_root.exists():
            raise RuntimeError(f'Mapillary label directory not found: {label_root}')
        for image_path in list_images_recursive(image_root):
            label_path = label_root / f'{image_path.stem}.png'
            if not label_path.exists():
                continue
            items.append(build_path_item(
                source_root=source_root,
                image_path=image_path,
                metadata={
                    'source_split': split,
                    'image_id': image_path.stem,
                    'mapillary_version': 'v2.0',
                    'label_mask_path': str(label_path.relative_to(source_root)),
                },
                filename_hint=f'{split}_{image_path.stem}',
            ))
    return items


def enrich_mapillary_vistas_items(source_root: Path, items: Sequence[LocalImportItem]) -> List[LocalImportItem]:
    config_path = source_root / 'config_v2.0.json'
    if not config_path.exists():
        raise RuntimeError(f'Mapillary config not found: {config_path}')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    labels = config.get('labels', []) if isinstance(config, dict) else []
    label_names = {
        idx: str(label.get('readable') or label.get('name') or idx)
        for idx, label in enumerate(labels)
        if isinstance(label, dict)
    }
    evaluated_ids = {
        idx for idx, label in enumerate(labels)
        if isinstance(label, dict) and bool(label.get('evaluate'))
    }
    instance_ids = {
        idx for idx, label in enumerate(labels)
        if isinstance(label, dict) and bool(label.get('instances'))
    }
    enriched: List[LocalImportItem] = []
    for item in items:
        label_relpath = item.metadata.get('label_mask_path', '')
        label_path = (source_root / label_relpath).resolve()
        if not label_path.exists():
            continue
        with Image.open(label_path) as label_image:
            mask = np.array(label_image, dtype=np.uint8)
        label_ids, label_counts = np.unique(mask, return_counts=True)
        total_pixels = int(label_counts.sum())
        ranked = sorted(
            (
                (int(label_id), int(count))
                for label_id, count in zip(label_ids.tolist(), label_counts.tolist())
                if int(count) > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        metadata = dict(item.metadata)
        top_names: List[str] = []
        for rank, (label_id, count) in enumerate(ranked[:3], start=1):
            label_name = label_names.get(label_id, str(label_id))
            share = (float(count) / float(total_pixels)) if total_pixels else 0.0
            metadata[f'mapillary_top_{rank}_label'] = label_name
            metadata[f'mapillary_top_{rank}_share'] = f'{share:.4f}'
            top_names.append(label_name)
            if rank == 1:
                metadata['label'] = label_name
                metadata['mapillary_primary_label'] = label_name
                metadata['mapillary_primary_share'] = f'{share:.4f}'
        if top_names:
            metadata['mapillary_top_labels'] = '|'.join(top_names)
        metadata['mapillary_unique_label_count'] = str(len(ranked))
        metadata['mapillary_evaluated_label_count'] = str(sum(1 for label_id, _ in ranked if label_id in evaluated_ids))
        metadata['mapillary_instance_label_count'] = str(sum(1 for label_id, _ in ranked if label_id in instance_ids))
        enriched.append(LocalImportItem(
            source=item.source,
            relative_source=item.relative_source,
            filename_hint=f"{metadata.get('label', 'scene')}_{item.filename_hint}",
            metadata=metadata,
            sample_label=metadata.get('label', ''),
        ))
    return enriched


def collect_celeba_records(source_root: Path, limit: int, seed: int) -> List[LocalImportItem]:
    image_root = source_root / 'img_align_celeba'
    if not image_root.exists():
        raise RuntimeError(f'CelebA image directory not found: {image_root}')
    image_paths = list_images_recursive(image_root)
    if not image_paths:
        return []
    if limit > 0 and len(image_paths) > limit:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(image_paths)), limit))
        image_paths = [image_paths[idx] for idx in indices]
    selected_names = {path.name for path in image_paths}
    partition_map = load_csv_metadata_by_key(
        csv_path=source_root / 'list_eval_partition.csv',
        key_field='image_id',
        wanted_keys=selected_names,
        prefix='',
    )
    attr_map = load_csv_metadata_by_key(
        csv_path=source_root / 'list_attr_celeba.csv',
        key_field='image_id',
        wanted_keys=selected_names,
        prefix='attr_',
    )
    bbox_map = load_csv_metadata_by_key(
        csv_path=source_root / 'list_bbox_celeba.csv',
        key_field='image_id',
        wanted_keys=selected_names,
        prefix='bbox_',
    )
    landmarks_map = load_csv_metadata_by_key(
        csv_path=source_root / 'list_landmarks_align_celeba.csv',
        key_field='image_id',
        wanted_keys=selected_names,
        prefix='landmark_',
    )
    partition_names = {'0': 'train', '1': 'val', '2': 'test'}
    items: List[LocalImportItem] = []
    for image_path in image_paths:
        name = image_path.name
        metadata: Dict[str, str] = {}
        metadata.update(partition_map.get(name, {}))
        metadata.update(attr_map.get(name, {}))
        metadata.update(bbox_map.get(name, {}))
        metadata.update(landmarks_map.get(name, {}))
        partition = metadata.get('partition', '')
        if partition:
            metadata['partition_name'] = partition_names.get(partition, partition)
        items.append(build_path_item(
            source_root=image_root,
            image_path=image_path,
            metadata=metadata,
            filename_hint=image_path.stem,
        ))
    return items


def collect_egyptian_statue_records(source_root: Path) -> List[LocalImportItem]:
    image_root = source_root / 'images_resized' / 'upload_resized'
    if not image_root.exists():
        raise RuntimeError(f'Egyptian statue image directory not found: {image_root}')
    metadata_rows = load_csv_metadata_by_key(
        csv_path=source_root / 'meta_data.csv',
        key_field='statue',
        prefix='',
    )
    items: List[LocalImportItem] = []
    for image_path in list_images_recursive(image_root):
        rel = image_path.relative_to(image_root)
        statue = rel.parts[0] if len(rel.parts) >= 2 else image_path.parent.name
        metadata = dict(metadata_rows.get(statue, {}))
        metadata['statue'] = statue
        metadata['source_parent'] = image_path.parent.name
        items.append(build_path_item(
            source_root=image_root,
            image_path=image_path,
            metadata=metadata,
            filename_hint=f'{statue}_{image_path.stem}',
            sample_label=statue,
        ))
    return items


def collect_qajar_carpet_records(source_root: Path) -> List[LocalImportItem]:
    image_root = source_root / 'Qajar Dynasty Iranian Carpet Dataset'
    if not image_root.exists():
        raise RuntimeError(f'Qajar image directory not found: {image_root}')
    metadata_rows = load_simple_xlsx_rows(source_root / 'metadata.xlsx')
    metadata_by_name = {row.get('file-name', ''): row for row in metadata_rows if row.get('file-name')}
    items: List[LocalImportItem] = []
    for image_path in list_images_recursive(image_root):
        row = dict(metadata_by_name.get(image_path.name, {}))
        if row:
            row.pop('file-name', None)
        row['file_name'] = image_path.name
        items.append(build_path_item(
            source_root=image_root,
            image_path=image_path,
            metadata=row,
            filename_hint=image_path.stem,
            sample_label=row.get('design_type', ''),
        ))
    return items


def collect_sinhala_brahmi_records(source_root: Path) -> List[LocalImportItem]:
    image_root = source_root / 'Dataset Modified'
    if not image_root.exists():
        raise RuntimeError(f'Sinhala image directory not found: {image_root}')
    return collect_label_tree_records(image_root, label_field='group_id')


def collect_ancient_tamil_records(source_root: Path) -> List[LocalImportItem]:
    excluded_roots = {
        'agisoftmodels',
        'lidarcameramodels',
        'lidar pics',
        'bounding box results',
    }
    items: List[LocalImportItem] = []
    for image_path in list_images_recursive(source_root):
        rel = image_path.relative_to(source_root)
        top = rel.parts[0] if rel.parts else ''
        if top.lower() in excluded_roots:
            continue
        top_slug = slugify(top)
        inscription_id = ''
        match = re.search(r'(?<![a-z])ins\s*([0-9]+)', ' '.join(rel.parts[:-1]) + ' ' + image_path.stem, flags=re.IGNORECASE)
        if match:
            inscription_id = f'ins{match.group(1)}'
        metadata = {
            'source_group': top,
            'source_parent': image_path.parent.name,
        }
        if inscription_id:
            metadata['inscription_id'] = inscription_id
        items.append(build_path_item(
            source_root=source_root,
            image_path=image_path,
            metadata=metadata,
            filename_hint=f'{top_slug}_{image_path.stem}',
            sample_label=inscription_id or top,
        ))
    return items


def import_metal_album_preset(
    preset: Preset,
    dataset_root: Path,
    source_root: Path,
    limit: int,
    seed: int,
    max_edge: int,
    max_source_pixels: int,
) -> int:
    csv.field_size_limit(sys.maxsize)
    csv_path = source_root / 'metal_albums_artwork_images.csv'
    if not csv_path.exists():
        raise RuntimeError(f'Metal artwork CSV not found: {csv_path}')
    skeleton_items: List[LocalImportItem] = []
    with csv_path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            artist = str(row.get('artist_name', '')).strip()
            album = str(row.get('album_name', '')).strip()
            genre = str(row.get('artist_main_genre', '')).strip()
            metadata = {
                'artist_name': artist,
                'artist_country': str(row.get('artist_country', '')).strip(),
                'artist_status': str(row.get('artist_status', '')).strip(),
                'artist_main_genre': genre,
                'artist_alt_genre': str(row.get('artist_alt_genre', '')).strip(),
                'album_name': album,
                'album_cover_url': str(row.get('album_cover_url', '')).strip(),
                'source_row': str(row_idx),
            }
            skeleton_items.append(LocalImportItem(
                source=row_idx,
                relative_source=f'metal_albums_artwork_images.csv:{row_idx}',
                filename_hint=f'{artist}_{album}' if artist or album else f'row_{row_idx}',
                metadata=metadata,
                sample_label=genre,
            ))
    selected = sample_local_items(items=skeleton_items, limit=limit, seed=seed, balance_key=preset.balance_key)
    selected_by_row = {int(item.source): item for item in selected}
    metadata_rows: List[Dict[str, str]] = []
    kept = 0
    skipped_invalid = 0
    with csv_path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            selected_item = selected_by_row.get(row_idx)
            if selected_item is None:
                continue
            raw = row.get('album_cover_image', '')
            if not raw:
                skipped_invalid += 1
                continue
            try:
                image_bytes = base64.b64decode(raw)
            except Exception:
                skipped_invalid += 1
                continue
            image = open_checked_image(image_bytes, max_source_pixels=max_source_pixels)
            if image is None:
                skipped_invalid += 1
                continue
            image_name = build_local_image_name(index=kept, hint=selected_item.filename_hint)
            try:
                save_prepared_image(image=image, out_path=dataset_root / image_name, max_edge=max_edge)
            finally:
                image.close()
            metadata = {
                'image': image_name,
                'source_dataset': preset.name,
                'source_path': selected_item.relative_source,
            }
            for key, value in selected_item.metadata.items():
                text = value_to_string(value)
                if text:
                    metadata[key] = text
            metadata_rows.append(metadata)
            kept += 1
            if kept % 100 == 0:
                log(f'saved {kept} images')
    if skipped_invalid:
        log(f'skipped invalid images={skipped_invalid}')
    write_metadata_csv(dataset_root, metadata_rows)
    return kept


def import_local_preset(
    preset: Preset,
    dataset_root: Path,
    local_root: Path,
    limit: int,
    seed: int,
    max_edge: int,
    max_source_pixels: int,
) -> int:
    source_root = (local_root / preset.source_relpath).resolve()
    if not source_root.exists():
        raise RuntimeError(f'Local dataset source not found: {source_root}')
    if preset.local_mode == 'metal_album':
        return import_metal_album_preset(
            preset=preset,
            dataset_root=dataset_root,
            source_root=source_root,
            limit=limit,
            seed=seed,
            max_edge=max_edge,
            max_source_pixels=max_source_pixels,
        )
    if preset.local_mode == 'split_label_tree':
        items = collect_split_label_tree_records(source_root)
    elif preset.local_mode == 'label_tree':
        items = collect_label_tree_records(source_root)
    elif preset.local_mode == 'brain_mri':
        items = collect_brain_mri_records(source_root)
    elif preset.local_mode == 'broden':
        items = collect_broden_records(source_root)
    elif preset.local_mode == 'celeba':
        items = collect_celeba_records(source_root=source_root, limit=limit, seed=seed)
    elif preset.local_mode == 'ham10000':
        items = collect_ham10000_records(source_root)
    elif preset.local_mode == 'inat2021birds':
        items = collect_inat2021birds_records(source_root)
    elif preset.local_mode == 'imagenet_r':
        items = collect_imagenet_r_records(source_root)
    elif preset.local_mode == 'imagenette':
        items = collect_imagenette_records(source_root)
    elif preset.local_mode == 'mapillary_vistas':
        items = collect_mapillary_vistas_records(source_root)
    elif preset.local_mode == 'egyptian_statue':
        items = collect_egyptian_statue_records(source_root)
    elif preset.local_mode == 'qajar_carpet':
        items = collect_qajar_carpet_records(source_root)
    elif preset.local_mode == 'sinhala_brahmi':
        items = collect_sinhala_brahmi_records(source_root)
    elif preset.local_mode == 'ancient_tamil':
        items = collect_ancient_tamil_records(source_root)
    else:
        raise RuntimeError(f'Unsupported local preset mode: {preset.local_mode}')
    keep_limit = limit if limit > 0 else 0
    if preset.local_mode != 'celeba':
        sample_limit = keep_limit
        if keep_limit > 0 and len(items) > keep_limit:
            sample_buffer = max(50, min(256, keep_limit // 5))
            sample_limit = min(len(items), keep_limit + sample_buffer)
        items = sample_local_items(items=items, limit=sample_limit, seed=seed, balance_key=preset.balance_key)
    if preset.local_mode == 'mapillary_vistas':
        items = enrich_mapillary_vistas_items(source_root=source_root, items=items)
    return materialize_local_items(
        preset=preset,
        dataset_root=dataset_root,
        items=items,
        max_edge=max_edge,
        max_source_pixels=max_source_pixels,
        keep_limit=keep_limit,
    )


def import_hf_preset(
    preset: Preset,
    dataset_root: Path,
    limit: int,
    seed: int,
    max_edge: int,
    max_source_pixels: int,
) -> int:
    from datasets.features import Image as HFImage

    dataset = load_hf_split(
        dataset_id=preset.hf_dataset,
        split=preset.hf_split,
        config=preset.hf_config,
        seed=seed,
        limit=limit,
    )
    features = getattr(dataset, 'features', None) or {}
    image_column = infer_hf_image_column(dataset, preset.hf_image_column)
    dataset = dataset.cast_column(image_column, HFImage(decode=False))
    pattern = re.compile(preset.filter_regex, flags=re.IGNORECASE) if preset.filter_regex else None
    metadata_rows: List[Dict[str, str]] = []
    kept = 0
    skipped_oversized = 0
    for row_idx, row in enumerate(dataset):
        if kept >= limit:
            break
        if not row_matches_filter(row=row, features=features, image_column=image_column, pattern=pattern):
            continue
        image = open_checked_image(row[image_column], max_source_pixels=max_source_pixels)
        if image is None:
            skipped_oversized += 1
            if skipped_oversized <= 10 or (skipped_oversized % 50) == 0:
                log(f'skipping unreadable image row={row_idx} max_source_pixels={max_source_pixels}')
            continue
        source_id = value_to_string(row.get('id') if isinstance(row, dict) else None) or str(row_idx)
        title = value_to_string(row.get('title') if isinstance(row, dict) else None)
        stem_parts = [f'{kept:05d}']
        if title:
            stem_parts.append(slugify(title)[:48])
        else:
            stem_parts.append(slugify(source_id)[:48])
        image_name = '_'.join([part for part in stem_parts if part]) + '.jpg'
        try:
            save_prepared_image(image=image, out_path=dataset_root / image_name, max_edge=max_edge)
        finally:
            image.close()
        metadata: Dict[str, str] = {
            'image': image_name,
            'source_dataset': preset.hf_dataset,
            'source_split': preset.hf_split,
            'source_row': str(row_idx),
        }
        for key, value in row.items():
            if key == image_column:
                continue
            text = value_to_string(value, features.get(key))
            if text:
                metadata[key] = text
        metadata_rows.append(metadata)
        kept += 1
        if kept % 100 == 0:
            log(f'saved {kept} images')
    if skipped_oversized:
        log(f'skipped oversized images={skipped_oversized}')
    write_metadata_csv(dataset_root, metadata_rows)
    return kept


def import_archive_preset(
    preset: Preset,
    dataset_root: Path,
    limit: int,
    seed: int,
    max_edge: int,
    max_source_pixels: int,
) -> int:
    with tempfile.TemporaryDirectory(prefix='promptherder_import_') as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        archive_name = preset.archive_url.rsplit('/', 1)[-1]
        archive_path = tmp_dir / archive_name
        download_file(preset.archive_url, archive_path)
        extract_archive(archive_path, tmp_dir)
        label_map = dict(preset.label_map or {})
        if preset.name == 'tiny_imagenet_200':
            label_map.update(maybe_load_tiny_imagenet_words(tmp_dir))
        base_dir = tmp_dir / preset.archive_subdir
        if not base_dir.exists():
            raise RuntimeError(f'Expected extracted directory not found: {base_dir}')
        candidates: List[Tuple[str, Path, str]] = []
        for split in preset.split_dirs:
            split_root = base_dir / split
            if not split_root.exists():
                continue
            for image_path in list_images_recursive(split_root):
                if split == 'val' and preset.name == 'tiny_imagenet_200':
                    continue
                label_code = image_path.parent.name
                if preset.name == 'tiny_imagenet_200' and image_path.parent.name == 'images':
                    label_code = image_path.parent.parent.name
                label = label_map.get(label_code, label_code)
                candidates.append((label, image_path, split))
        selected = balanced_sample_by_label(
            items=[(label, path) for label, path, _split in candidates],
            limit=limit,
            seed=seed,
        )
        split_by_path = {str(path): split for _label, path, split in candidates}
        metadata_rows: List[Dict[str, str]] = []
        skipped_oversized = 0
        kept = 0
        for idx, (label, image_path) in enumerate(selected):
            split = split_by_path.get(str(image_path), '')
            image_name = f'{idx:05d}_{slugify(label)[:40]}_{slugify(image_path.stem)[:28]}.jpg'
            img = open_checked_image(image_path, max_source_pixels=max_source_pixels)
            if img is None:
                skipped_oversized += 1
                if skipped_oversized <= 10 or (skipped_oversized % 50) == 0:
                    log(f'skipping unreadable image path={image_path} max_source_pixels={max_source_pixels}')
                continue
            try:
                save_prepared_image(image=img, out_path=dataset_root / image_name, max_edge=max_edge)
            finally:
                img.close()
            metadata_rows.append({
                'image': image_name,
                'label': label,
                'label_code': image_path.parent.name,
                'source_dataset': preset.name,
                'source_split': split,
                'source_path': str(image_path.relative_to(base_dir)),
            })
            kept += 1
            if kept % 100 == 0 and kept > 0:
                log(f'saved {kept} images')
        if skipped_oversized:
            log(f'skipped oversized images={skipped_oversized}')
        write_metadata_csv(dataset_root, metadata_rows)
        return kept


def save_embedding_cache(dataset_root: Path, method: str, embeddings: np.ndarray, entries: Sequence[Any]) -> None:
    cache_dir = dataset_root / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_method = normalize_multimodal_method(method)
    paths = np.array([entry.path for entry in entries])
    mtimes = np.array([int(Path(path).stat().st_mtime) if Path(path).exists() else 0 for path in paths], dtype=np.int64)
    np.savez_compressed(
        cache_dir / f'embeddings_{cache_method}.npz',
        paths=paths,
        mtimes=mtimes,
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def save_pca_cache(dataset_root: Path, method: str, coords2d: np.ndarray, entries: Sequence[Any]) -> None:
    cache_dir = dataset_root / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_method = normalize_multimodal_method(method)
    paths = np.array([entry.path for entry in entries])
    mtimes = np.array([int(Path(path).stat().st_mtime) if Path(path).exists() else 0 for path in paths], dtype=np.int64)
    np.savez_compressed(
        cache_dir / f'coords_pca2d_{cache_method}.npz',
        paths=paths,
        mtimes=mtimes,
        coords=np.asarray(coords2d, dtype=np.float32),
    )


def precompute_dataset(dataset_root: Path, methods: Sequence[str], reduction: str) -> None:
    engine = ImageGalleryEngine(str(dataset_root))
    entries = engine.list_images()
    if not entries:
        raise RuntimeError(f'No images found under {dataset_root}')
    for method in methods:
        cached_embeddings = engine._load_embeddings_only(entries, method=method)
        if cached_embeddings is not None:
            embeddings = np.asarray(cached_embeddings, dtype=np.float32)
            log(f'skipping embedding compute for {method} (cache hit)')
        else:
            log(f'computing embeddings for {method}')
            embeddings = engine.estimate_embeddings(entries, method=method)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(entries):
                raise RuntimeError(f'Unsupported embedding shape for method={method}: {tuple(embeddings.shape)}')
            save_embedding_cache(dataset_root=dataset_root, method=method, embeddings=embeddings, entries=entries)

        cached_coords = engine._load_cached_coords(entries, method=method)
        if cached_coords is not None:
            log(f'skipping {reduction.upper()} coords for {method} (cache hit)')
            continue
        log(f'computing {reduction.upper()} coords for {method}')
        coords2d = engine.reduce_to_2d(embeddings, method=reduction)
        save_pca_cache(dataset_root=dataset_root, method=method, coords2d=coords2d, entries=entries)


def parse_methods(raw: str) -> List[str]:
    methods = [part.strip() for part in str(raw or '').split(',') if part.strip()]
    return methods or list(DEFAULT_METHODS)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Download or normalize curated datasets for the Promptherder UI.',
    )
    parser.add_argument('--preset', default='', choices=sorted(PRESETS.keys()))
    parser.add_argument('--name', default='', help='Output dataset name under data/datasets/. Defaults to the preset name.')
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help='Maximum number of images to keep.')
    parser.add_argument('--max-edge', type=int, default=DEFAULT_MAX_EDGE, help='Resize the largest image edge to this value.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS), help='Comma-separated embedding methods to precompute.')
    parser.add_argument('--reduction', default=DEFAULT_REDUCTION, choices=('pca', 'umap'))
    parser.add_argument('--max-source-pixels', type=int, default=DEFAULT_MAX_SOURCE_PIXELS, help='Downscale source images above this pixel count before the normal import resize step.')
    parser.add_argument('--local-root', default=str(DEFAULT_LOCAL_ROOT), help='Base folder for local curated dataset presets.')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing dataset directory.')
    parser.add_argument('--skip-precompute', action='store_true')
    parser.add_argument('--list-presets', action='store_true')
    return parser


def list_presets() -> None:
    print('Available presets:')
    for preset in PRESETS.values():
        source = preset.kind
        if preset.kind == 'local':
            source = f'local:{preset.source_relpath}'
        print(f'  - {preset.name} [{source}]: {preset.description}')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_presets:
        list_presets()
        return 0
    if not args.preset:
        parser.error('--preset is required unless --list-presets is used')
    preset = PRESETS[args.preset]
    dataset_name = args.name.strip() or preset.name
    dataset_root = DATASETS_ROOT / dataset_name
    local_root = Path(args.local_root).expanduser().resolve()
    methods = parse_methods(args.methods)
    reused_existing, existing_count = ensure_empty_dataset_dir(dataset_root, overwrite=bool(args.overwrite))
    if reused_existing:
        count = existing_count
        log(f'skipping download/import and using existing images count={count}')
    else:
        log(f'preparing preset={preset.name} dataset={dataset_root}')
        if preset.kind == 'hf':
            count = import_hf_preset(
                preset=preset,
                dataset_root=dataset_root,
                limit=max(1, int(args.limit)),
                seed=int(args.seed),
                max_edge=max(64, int(args.max_edge)),
                max_source_pixels=max(1, int(args.max_source_pixels)),
            )
        elif preset.kind == 'archive':
            count = import_archive_preset(
                preset=preset,
                dataset_root=dataset_root,
                limit=max(1, int(args.limit)),
                seed=int(args.seed),
                max_edge=max(64, int(args.max_edge)),
                max_source_pixels=max(1, int(args.max_source_pixels)),
            )
        elif preset.kind == 'local':
            count = import_local_preset(
                preset=preset,
                dataset_root=dataset_root,
                local_root=local_root,
                limit=max(1, int(args.limit)),
                seed=int(args.seed),
                max_edge=max(64, int(args.max_edge)),
                max_source_pixels=max(1, int(args.max_source_pixels)),
            )
        else:
            raise RuntimeError(f'Unsupported preset kind: {preset.kind}')
        log(f'imported {count} images')
    if not args.skip_precompute:
        precompute_dataset(dataset_root=dataset_root, methods=methods, reduction=args.reduction)
        log(f'precomputed methods={methods} reduction={args.reduction}')
    log(f'dataset ready at {dataset_root}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        raise SystemExit(1)
