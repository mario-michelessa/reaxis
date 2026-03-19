#!/usr/bin/env python3
"""
Image Gallery Backend utilities

Provides an engine to:
- List images in a dataset
- Estimate image embeddings (placeholder implementation, pluggable)
- Reduce embeddings to 2D (UMAP placeholder with PCA fallback)
- Pack 2D points into a non-overlapping image grid using grid.py

Note: Server endpoints are not implemented here; this module focuses on
data preparation. You can wire it into a FastAPI/Flask app later.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from json import dumps
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from .embeddings import (
        DEFAULT_SEMANTIC_EMBED_METHOD,
        SUPPORTED_FORMATS,
        ImageEntry,
        EmbeddingEngine,
        normalize_multimodal_method,
    )
    from . import layout as layout_utils
except ImportError:
    from embeddings import (
        DEFAULT_SEMANTIC_EMBED_METHOD,
        SUPPORTED_FORMATS,
        ImageEntry,
        EmbeddingEngine,
        normalize_multimodal_method,
    )
    import layout as layout_utils

class ImageGalleryEngine:
    """Orchestrates image discovery, embedding, 2D layout, and grid packing."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        # delegate embeddings to dedicated engine
        self.emb = EmbeddingEngine(dataset_path)

    def list_images(self) -> List[ImageEntry]:
        return self.emb.list_images()

    def estimate_embeddings(self, images: Sequence[ImageEntry], method: str = DEFAULT_SEMANTIC_EMBED_METHOD, resize: Tuple[int, int] = (32, 32)) -> np.ndarray:
        return self.emb.estimate_embeddings(images, method=method, resize=resize)

    # Embedding extraction implementations moved to embeddings. No local copies here.

    def reduce_to_2d(self, embeddings: np.ndarray, method: str = "pca", random_state: int = 42) -> np.ndarray:
        return layout_utils.reduce_to_2d(embeddings, method=method, random_state=random_state)

    # PCA implementation moved to layout utils

    def pack_to_grid(self, coords01: np.ndarray, n_layer: int = 64, n_tile: int = 8,
                     filter_fn: Optional[Callable[[int, Dict], bool]] = None) -> Tuple[np.ndarray, int]:
        return layout_utils.pack_to_grid(coords01, n_layer=n_layer, n_tile=n_tile, filter_fn=filter_fn)

    def build_gallery(self, n_layer: int = 64, n_tile: int = 8,
                      method: str = "pca", embed_method: str = DEFAULT_SEMANTIC_EMBED_METHOD) -> Tuple[List[ImageEntry], np.ndarray, np.ndarray, int]:
        """End-to-end pipeline returning entries, reduced coords, and packed coords.

        Embeddings and PCA coordinates are cached per dataset and model.
        """
        entries = self.list_images()
        embs = self._load_or_compute_embeddings(entries, embed_method)
        coords2d = self._load_or_compute_coords(entries, embs, embed_method, method)
        packed, eff_layer = self.pack_to_grid(coords2d, n_layer=n_layer, n_tile=n_tile)
        return entries, coords2d, packed, eff_layer

    def _cache_dir(self) -> Path:
        return self.emb._cache_dir()

    def _load_or_compute_embeddings(self, entries: List[ImageEntry], method: str) -> np.ndarray:
        cache_method = normalize_multimodal_method(method)
        # Try fast-path: plain embeddings npz without path metadata (e.g., dift_sd_partXY)
        cache = self._cache_dir() / f'embeddings_{cache_method}.npz'
        if cache.exists():
            try:
                data = np.load(cache, allow_pickle=False)
                # Accept minimal format: just an 'embeddings' array or first array
                arr = None
                if 'embeddings' in data.files:
                    arr = data['embeddings']
                elif len(data.files) > 0:
                    arr = data[data.files[0]]
                if arr is not None and arr.ndim == 2 and arr.shape[0] == len(entries):
                    print(f"[emb] loaded minimal cache {cache.name} shape={tuple(arr.shape)}")
                    return arr
            except Exception:
                # Fall through to normal path
                print(f"[emb] failed to load minimal cache: {cache.name}")
        # Normal cached format with path+mtime checks
        print(f"[emb] probing cache (full) embeddings_{cache_method}.npz with paths/mtimes")
        embs = self.emb.load_embeddings_only(entries, method=method)
        if embs is not None:
            print(f"[emb] loaded full cache shape={tuple(embs.shape)}")
            return embs
        print(f"[emb] cache miss, computing embeddings method={method}")
        return self.emb.compute_and_cache_embeddings(entries, method=method)

    def _load_embeddings_only(self, entries: List[ImageEntry], method: str) -> Optional[np.ndarray]:
        """Load cached embeddings only; supports both minimal and full cache formats.

        Minimal: npz with only an embeddings array (any key, prefers 'embeddings').
        Full: npz with paths/mtimes + embeddings; validates against current dataset.
        """
        cache_method = normalize_multimodal_method(method)
        cache = self._cache_dir() / f'embeddings_{cache_method}.npz'
        if not cache.exists():
            print(f"[emb] cache file not found: {cache}")
            return None
        try:
            print(f"[emb] attempting to load cache: {cache}")
            data = np.load(cache, allow_pickle=False)
            print(f"[emb] cache loaded, available keys: {data.files}")
            
            # Full format with validation
            if {'paths', 'embeddings'}.issubset(set(data.files)):
                print(f"[emb] found full format with paths/embeddings")
                embs = data['embeddings']
                if embs.ndim != 2 or embs.shape[0] != len(entries):
                    print(f"[emb] shape mismatch in full format: expected ({len(entries)}, N), got {embs.shape}")
                    print(f"[emb] cache format invalid or does not match dataset: {cache}")
                    return None

                # First try strict absolute-path comparison.
                current_paths = np.array([str(Path(e.path).resolve()) for e in entries])
                cached_paths = np.array([str(Path(p).resolve()) for p in data['paths']])
                paths_match = len(cached_paths) == len(current_paths) and np.all(cached_paths == current_paths)
                print(f"[emb] validation results - paths match: {paths_match}")
                if paths_match:
                    print(f"[emb] paths match, embeddings shape={embs.shape}")
                    print(f"[emb] returning full format embeddings")
                    return embs

                # If path roots differ (common when cache was built from another cwd),
                # validate and align by image id (filename).
                current_ids = [str(e.id) for e in entries]
                cached_ids = [Path(str(p)).name for p in data['paths']]

                if len(cached_ids) == len(current_ids):
                    if np.all(np.asarray(cached_ids, dtype=object) == np.asarray(current_ids, dtype=object)):
                        print("[emb] paths mismatch but ids match in order; accepting cache by id")
                        return embs

                    id_to_idx: Dict[str, int] = {}
                    duplicate_id = False
                    for idx, image_id in enumerate(cached_ids):
                        if image_id in id_to_idx:
                            duplicate_id = True
                            break
                        id_to_idx[image_id] = idx

                    if (not duplicate_id) and all(image_id in id_to_idx for image_id in current_ids):
                        remap = np.asarray([id_to_idx[image_id] for image_id in current_ids], dtype=np.int64)
                        print("[emb] paths mismatch; remapping embeddings by image id order")
                        return embs[remap]

                print(f"[emb] cache validation failed for full format (paths/id mismatch)")
        except Exception:
            print(f"[emb] failed to load/validate cache: {cache}")
            return None
        print(f"[emb] cache format invalid or does not match dataset: {cache}")
        return None

    def _load_or_compute_coords(self, entries: List[ImageEntry], embs: np.ndarray, method: str, red_method: str) -> np.ndarray:
        cached_coords = self._load_cached_coords(entries, method=method)
        if red_method.lower() == 'pca' and cached_coords is not None:
            return cached_coords
        cache_method = normalize_multimodal_method(method)
        cache = self._cache_dir() / f'coords_pca2d_{cache_method}.npz'
        print(f"[coords] computing coords method={red_method} for embs shape={tuple(embs.shape)}")
        coords2d = self.reduce_to_2d(embs, method=red_method)
        if red_method.lower() == 'pca':
            try:
                # Persist identifiers for validation on reload
                paths = np.array([e.path for e in entries])
                mtimes = np.array([int(Path(p).stat().st_mtime) if Path(p).exists() else 0 for p in paths], dtype=np.int64)
                np.savez_compressed(cache, paths=paths, mtimes=mtimes, coords=coords2d)
                print(f"[coords] cached PCA coords at {cache}")
            except Exception:
                pass
        return coords2d

    def _load_cached_coords(self, entries: List[ImageEntry], method: str) -> Optional[np.ndarray]:
        # Always cache PCA coordinates as primary.
        cache_method = normalize_multimodal_method(method)
        cache = self._cache_dir() / f'coords_pca2d_{cache_method}.npz'
        current_paths = np.array([str(Path(e.path).resolve()) for e in entries])
        current_names = np.array([Path(e.path).name for e in entries])
        if not cache.exists():
            return None
        try:
            data = np.load(cache, allow_pickle=False)
            if 'paths' in data.files:
                cached_paths = np.array([str(Path(p).resolve()) for p in data['paths']])
                cached_names = np.array([Path(p).name for p in data['paths']])
            else:
                cached_paths = None
                cached_names = None
            paths_match = (
                cached_paths is not None
                and len(cached_paths) == len(current_paths)
                and np.all(cached_paths == current_paths)
            )
            names_match = (
                cached_names is not None
                and len(cached_names) == len(current_names)
                and np.all(cached_names == current_names)
            )
            if paths_match or names_match:
                print(f"[coords] loaded cached PCA coords shape={tuple(data['coords'].shape)}")
                return data['coords']
        except Exception:
            return None
        return None

    def export_gallery_json(self, out_path: str, base_url: Optional[str] = None,
                             n_layer: int = 64, n_tile: int = 8,
                             method: str = "umap", embed_method: str = DEFAULT_SEMANTIC_EMBED_METHOD) -> str:
        """Generate a JSON file with image metadata and coordinates.

        - base_url: optional URL prefix to serve images (e.g., '/images')
        Returns the path to the written file.
        """
        entries, coords2d, packed, eff_layer = self.build_gallery(n_layer=n_layer, n_tile=n_tile, method=method, embed_method=embed_method)

        items = []
        for i, e in enumerate(entries):
            url = e.path
            if base_url:
                # Attempt to map to base_url by taking filename
                url = f"{base_url}/{Path(e.path).name}"
            items.append({
                "id": e.id,
                "path": e.path,
                "url": url,
                "className": e.class_name,
                "x": float(coords2d[i, 0]),
                "y": float(coords2d[i, 1]),
                "gx": float(packed[i, 0]),
                "gy": float(packed[i, 1]),
            })

        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(dumps({"items": items, "n_layer": eff_layer, "n_tile": n_tile, "method": method, "embed": embed_method}, indent=2))
        return out_path

    def build_gallery_from_precomputed(self, n_layer: int = 64, n_tile: int = 8,
                                       method: str = "pca", embed_method: str = DEFAULT_SEMANTIC_EMBED_METHOD):
        """Build gallery using ONLY precomputed embeddings.

        Loads cached embeddings; if unavailable returns (None, None, None, 0).
        """
        entries = self.list_images()
        if not entries:
            return [], np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), 0
        embs = self._load_embeddings_only(entries, method=embed_method)
        if embs is None:
            print(f"[build] No cached embeddings found for method={embed_method}")
            return None, None, None, 0
        print(f"[build] embs shape={tuple(embs.shape)} method={method}")
        # If cached embeddings are transposed (D x N), fix by transposing
        if embs.ndim == 2 and embs.shape[0] == 2 and embs.shape[1] != 2 and embs.shape[1] == len(entries):
            print(f"Transposing embeddings from {embs.shape} to ({len(entries)}, 2) assumption")
            embs = embs.T
        coords2d = self._load_cached_coords(entries, method=embed_method)
        if coords2d is None:
            print(f"[build] No cached 2D coords found for method={embed_method}")
            return None, None, None, 0
        # Sanity: coords must be (N,2). If (2,N), transpose.
        if coords2d.ndim == 2 and coords2d.shape[0] == 2 and coords2d.shape[1] == len(entries):
            print(f"Transposing coords2d from {coords2d.shape} to ({len(entries)}, 2)")
            coords2d = coords2d.T
        packed, eff_layer = self.pack_to_grid(coords2d, n_layer=n_layer, n_tile=n_tile)
        print(f"[build] coords2d={tuple(coords2d.shape)} packed={tuple(packed.shape)} eff_layer={eff_layer}")
        return entries, coords2d, packed, eff_layer

    # Note: for part-specific embeddings (e.g., dift_sd_partXY), use embed_method naming
    # and rely on the existing cache loader which maps to embeddings_{embed_method}.npz

from PIL import Image

def export_all(dataset: str, out_dir: str, methods: Optional[List[str]] = None,
               reduction: str = "pca", n_layer: int = 64, n_tile: int = 8,
               default_method: Optional[str] = None, copy_images: bool = True) -> Dict[str, Any]:
    """Export a standalone dataset folder with images and multiple gallery_* JSONs.

    - Copies all images into <out_dir>/images
    - For each method in `methods`, computes 2D coords (PCA) and writes <out_dir>/gallery_<method>.json
    - Writes <out_dir>/gallery.json for the default_method (or first in list)
    - Writes <out_dir>/methods.json listing available methods
    Returns mapping with written files and counts.
    """
    dataset_path = Path(dataset)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    eng = ImageGalleryEngine(str(dataset_path))
    entries = eng.list_images()
    if not entries:
        raise RuntimeError(f"No images found in dataset: {dataset_path}")

    # Copy images
    copied = 0
    if copy_images:
        img_dir = out_root / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)
        for e in entries:
            src = Path(e.path)
            dst = img_dir / src.name
            if not dst.exists() or (src.stat().st_mtime > dst.stat().st_mtime):
                img = Image.open(src)
                w,h = img.size
                new_w = 256  # Desired new width
                new_h = int(h * (new_w / w))
                img = img.resize((new_w, new_h), Image.LANCZOS)
                print(f"Copying and resizing image {src} -> {dst} ({new_w}x{new_h})")
                img.save(dst)
            copied += 1

    # Methods processing
    written: Dict[str, str] = {}
    for m in methods:
        out_path = out_root / f"gallery_{m}.json"
        eng.export_gallery_json(str(out_path), base_url='images',
                                n_layer=n_layer, n_tile=n_tile,
                                method=reduction, embed_method=m)
        written[m] = str(out_path)

    # Write default link
    default_m = default_method or (methods[0] if methods else None)
    if default_m:
        default_json = out_root / 'gallery.json'
        src_json = out_root / f"gallery_{default_m}.json"
        if src_json.exists():
            # Copy to gallery.json (do not symlink for portability)
            shutil.copy2(src_json, default_json)

    # Methods manifest
    with open(out_root / 'methods.json', 'w') as f:
        f.write(dumps({ 'methods': methods, 'default': default_m }, indent=2))

    return { 'out_dir': str(out_root), 'copied_images': copied, 'written': written, 'default_method': default_m }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build gallery embeddings and coords")
    parser.add_argument("dataset", help="Path to image dataset root")
    parser.add_argument("output", nargs='?', help="Path to output JSON, e.g., frontend/public/gallery.json")
    parser.add_argument("--n_layer", type=int, default=64)
    parser.add_argument("--n_tile", type=int, default=8)
    parser.add_argument("--method", type=str, default="pca", help="'umap' or 'pca'")
    parser.add_argument("--base_url", type=str, default=None, help="Optional URL prefix for images")
    parser.add_argument("--embed", type=str, default=DEFAULT_SEMANTIC_EMBED_METHOD, help="Embedding method: 'siglip2', 'color_rgb', 'clip', 'dino', 'sd'")
    # Standalone export-all
    parser.add_argument("--export_all_dir", type=str, default=None, help="Output folder for standalone dataset (copies images and writes gallery_*.json)")
    parser.add_argument("--methods", type=str, default=f"color_rgb,{DEFAULT_SEMANTIC_EMBED_METHOD},clip,dino,dift_sd", help="Comma-separated embedding methods to export")
    parser.add_argument("--default_method", type=str, default=DEFAULT_SEMANTIC_EMBED_METHOD, help="Default method for gallery.json link")
    args = parser.parse_args()

    if args.export_all_dir:
        methods = [m.strip() for m in (args.methods or '').split(',') if m.strip()]
        info = export_all(args.dataset, args.export_all_dir, methods=methods,
                          reduction=args.method or 'pca', n_layer=args.n_layer, n_tile=args.n_tile,
                          default_method=args.default_method or None, copy_images=True)
        print("✅ Exported standalone dataset:", info)
    else:
        if not args.output:
            raise SystemExit("Missing output path for single gallery export")
        engine = ImageGalleryEngine(args.dataset)
        out = engine.export_gallery_json(args.output, base_url=args.base_url,
                                         n_layer=args.n_layer, n_tile=args.n_tile,
                                         method=args.method, embed_method=args.embed)
        print(f"✅ Gallery JSON written to {out}")
