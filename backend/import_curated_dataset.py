#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

try:
    from .gallery_backend import ImageGalleryEngine
except ImportError:
    from gallery_backend import ImageGalleryEngine


DATASETS_ROOT = (Path(__file__).resolve().parent.parent / 'data' / 'datasets').resolve()
DEFAULT_METHODS = ('color_rgb', 'clip', 'dino')
DEFAULT_REDUCTION = 'pca'
DEFAULT_MAX_EDGE = 512
DEFAULT_LIMIT = 1500
DEFAULT_MAX_SOURCE_PIXELS = 80_000_000
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
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
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        return None
    width, height = image.size
    if (int(width) * int(height)) > int(max_source_pixels):
        image.close()
        return None
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
                log(f'skipping oversized image row={row_idx} max_source_pixels={max_source_pixels}')
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
                    log(f'skipping oversized image path={image_path} max_source_pixels={max_source_pixels}')
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
    paths = np.array([entry.path for entry in entries])
    mtimes = np.array([int(Path(path).stat().st_mtime) if Path(path).exists() else 0 for path in paths], dtype=np.int64)
    np.savez_compressed(
        cache_dir / f'embeddings_{method.lower()}.npz',
        paths=paths,
        mtimes=mtimes,
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def save_pca_cache(dataset_root: Path, method: str, coords2d: np.ndarray, entries: Sequence[Any]) -> None:
    cache_dir = dataset_root / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = np.array([entry.path for entry in entries])
    mtimes = np.array([int(Path(path).stat().st_mtime) if Path(path).exists() else 0 for path in paths], dtype=np.int64)
    np.savez_compressed(
        cache_dir / f'coords_pca2d_{method.lower()}.npz',
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
        description='Download and prepare curated datasets for the Promptherder UI.',
    )
    parser.add_argument('--preset', default='', choices=sorted(PRESETS.keys()))
    parser.add_argument('--name', default='', help='Output dataset name under data/datasets/. Defaults to the preset name.')
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help='Maximum number of images to keep.')
    parser.add_argument('--max-edge', type=int, default=DEFAULT_MAX_EDGE, help='Resize the largest image edge to this value.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS), help='Comma-separated embedding methods to precompute.')
    parser.add_argument('--reduction', default=DEFAULT_REDUCTION, choices=('pca', 'umap'))
    parser.add_argument('--max-source-pixels', type=int, default=DEFAULT_MAX_SOURCE_PIXELS, help='Skip source images above this pixel count.')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing dataset directory.')
    parser.add_argument('--skip-precompute', action='store_true')
    parser.add_argument('--list-presets', action='store_true')
    return parser


def list_presets() -> None:
    print('Available presets:')
    for preset in PRESETS.values():
        print(f'  - {preset.name}: {preset.description}')


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
