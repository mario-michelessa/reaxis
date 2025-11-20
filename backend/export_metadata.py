#!/usr/bin/env python3
"""
Export metadata axes for standalone usage.

Reads a metadata.csv from a dataset folder and writes a gallery_metadata.json
containing per-field 1D axes compatible with the frontend's `metadata` view.

The CSV must include an `image` column identifying each row (by filename or id),
and any number of additional columns treated as categorical/numeric fields.

Usage:
  python backend/export_metadata.py <DATASET_DIR> <OUT_DIR>

Writes: <OUT_DIR>/gallery_metadata.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.append(str(Path(__file__).parent))
from gallery_backend import ImageGalleryEngine


def _load_metadata_axes(dataset_root: Path, entries: List[Any]) -> List[Dict[str, Any]]:
    meta_path = dataset_root / 'metadata.csv'
    if not meta_path.exists():
        print(f"[metadata] metadata.csv not found at {meta_path}")
        return []

    # Build id match helpers
    entry_ids = [e.id for e in entries]
    id_set = set(entry_ids)
    basename_to_id: Dict[str, str] = {}
    stem_to_id: Dict[str, str] = {}
    for e in entries:
        p = Path(e.path)
        base = p.name.lower()
        stem = p.stem.lower()
        # First occurrence wins; avoids overwriting duplicates
        if base not in basename_to_id:
            basename_to_id[base] = e.id
        if stem not in stem_to_id:
            stem_to_id[stem] = e.id

    metadata_axes: List[Dict[str, Any]] = []
    try:
        with meta_path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = [h.strip() for h in (reader.fieldnames or [])]
            if 'image' not in [h.lower() for h in headers]:
                print('[metadata] metadata.csv missing image column; skipping')
                return []
            image_col = next(h for h in headers if h.lower() == 'image')
            field_cols = [h for h in headers if h != image_col]
            print('[metadata] headers:', headers, 'fields:', field_cols)
            rows = []
            for r in reader:
                rows.append({(k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
            print('[metadata] rows:', len(rows))
            for field in field_cols:
                raw_map: Dict[str, Any] = {}
                for r in rows:
                    key = str(r.get(image_col) or '').strip()
                    if not key:
                        continue
                    match_id = None
                    if key in id_set:
                        match_id = key
                    else:
                        kp = Path(key)
                        base = kp.name.lower()
                        stem = kp.stem.lower()
                        if base in basename_to_id:
                            match_id = basename_to_id[base]
                        elif stem in stem_to_id:
                            match_id = stem_to_id[stem]
                    if not match_id:
                        continue
                    raw_map[match_id] = r.get(field)
                if not raw_map:
                    print('[metadata] field has no matches:', field)
                    continue
                # Unique values preserving order
                uniq_vals_raw: List[str] = []
                for v in raw_map.values():
                    sv = '' if v is None else str(v)
                    if sv not in uniq_vals_raw:
                        uniq_vals_raw.append(sv)
                # Try numeric sort; fallback to alpha (case-insensitive)
                as_num = []
                all_numeric = True
                for sv in uniq_vals_raw:
                    try:
                        as_num.append((float(sv), sv))
                    except Exception:
                        all_numeric = False
                        break
                if all_numeric:
                    uniq_vals_sorted = [sv for _, sv in sorted(as_num, key=lambda x: x[0])]
                else:
                    uniq_vals_sorted = sorted(uniq_vals_raw, key=lambda s: s.lower())
                n = max(1, len(uniq_vals_sorted))
                val_to_idx = {v: i for i, v in enumerate(uniq_vals_sorted)}
                coords: Dict[str, float] = {}
                for img_id, v in raw_map.items():
                    sv = '' if v is None else str(v)
                    idx = val_to_idx.get(sv, 0)
                    base = (idx + 1) / float(n)
                    # Deterministic jitter in [-1/(2n), 0]
                    jitter = 0.0
                    if n > 0:
                        jitter_range = 1.0 / (2.0 * float(n))
                        h = hashlib.md5(f'{field}|{img_id}'.encode('utf-8')).digest()
                        u = int.from_bytes(h[:8], 'big') / float(2**64 - 1)
                        jitter = (u - 1.0) * jitter_range
                    val = base + jitter
                    if val < 0.0:
                        val = 0.0
                    if val > 1.0:
                        val = 1.0
                    coords[img_id] = float(val)
                label_positions: List[float] = []
                for i in range(len(uniq_vals_sorted)):
                    label_positions.append((i + 0.75) / float(n))
                axis_id = f'axis:meta:{field}'
                axis_name = f'{field}'
                metadata_axes.append({
                    'id': axis_id,
                    'name': axis_name,
                    'coords': coords,
                    'labels': uniq_vals_sorted,
                    'label_positions': label_positions,
                })
    except Exception as e:
        print('[metadata] parse error:', e)
        return []

    print(f"[metadata] built axes: {len(metadata_axes)}")
    return metadata_axes


def export_metadata(dataset: str, out_dir: str) -> str:
    dataset_root = Path(dataset)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    eng = ImageGalleryEngine(str(dataset_root))
    entries = eng.list_images()
    axes = _load_metadata_axes(dataset_root, entries)
    out_path = out_root / 'gallery_metadata.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump({'metadata_axes': axes}, f, indent=2)
    print(f"✅ Wrote {out_path} with {len(axes)} axes")
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(description='Export gallery_metadata.json for standalone')
    ap.add_argument('dataset', help='Path to dataset root containing images and metadata.csv')
    ap.add_argument('out_dir', help='Directory to write gallery_metadata.json (e.g., frontend/public/datasets/<NAME>)')
    args = ap.parse_args()
    export_metadata(args.dataset, args.out_dir)


if __name__ == '__main__':
    main()
