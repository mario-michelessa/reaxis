#!/usr/bin/env python3
"""
Simple Flask server that serves a gallery with embeddings, 2D layout, and
non-overlapping packed coordinates for a minimap and image grid.

Endpoints
- GET /health
- GET /gallery.json?method=umap|pca&n_layer=64&n_tile=8
  Returns items with fields: id, url, className, x, y, gx, gy
- GET /images/<path:relpath>
  Serves images relative to the configured dataset root

Note: This is a minimal prototype. For multi-dataset use, consider including
the dataset in the image URL or isolating per-dataset blueprints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS

from gallery_backend import ImageGalleryEngine


app = Flask(__name__)
CORS(app)

# Configure your dataset root here. Set this to your images root folder.
# Example: DATASET_PATH = Path('/data/my_images')
DATASET_PATH = Path('../data/datasets/CUB')
# Keep the currently active dataset root for serving images
app.config['DATASET_ROOT'] = str(DATASET_PATH.resolve()) if DATASET_PATH.exists() else None
print(f'Using DATASET_PATH: {DATASET_PATH}, exists: {DATASET_PATH.exists()}, length: {len(list(DATASET_PATH.rglob("*")))}')


@app.get('/health')
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get('/gallery.json')
def gallery() -> Any:
    # Use configured dataset; do not accept dataset via query params
    dataset = str(DATASET_PATH)
    method = request.args.get('method', 'pca').lower()
    embed_method = request.args.get('embed', 'avg').lower()

    print("[gallery] start",
          f"dataset={dataset}",
          f"method={method}",
          f"embed={embed_method}")
    # Parse grid params; if n_layer missing, allow auto mode by passing 0
    n_layer_raw = request.args.get('n_layer')
    n_tile_raw = request.args.get('n_tile', '8')
    try:
        n_layer = int(n_layer_raw) if n_layer_raw is not None else 0
        n_tile = int(n_tile_raw)
    except ValueError:
        return abort(400, description='Invalid n_layer or n_tile')

    if not Path(dataset).exists():
        print("[gallery] invalid dataset path:", dataset)
        return jsonify({'items': [], 'dataset': dataset, 'n_layer': 0, 'n_tile': n_tile, 'method': method, 'embed': embed_method, 'warning': f'DATASET_PATH does not exist: {dataset}'}), 200

    # Be explicit and let errors surface (no swallowing). Easier debugging.
    print(f"[gallery] resolved dataset={Path(dataset).resolve()} n_layer={n_layer} n_tile={n_tile}")
    engine = ImageGalleryEngine(dataset)
    entries_probe = engine.list_images()
    print(f"[gallery] images found: {len(entries_probe)}")
    # Load only precomputed embeddings; do not compute on the fly.
    # Note: embed_method may include part suffix (e.g., dift_sd_part11), which maps to embeddings_{embed_method}.npz
    entries, coords2d, packed, eff_layer = engine.build_gallery_from_precomputed(
        n_layer=n_layer, n_tile=n_tile, method=method, embed_method=embed_method
    )
    if entries is None or coords2d is None or packed is None:
        # Return explicit error to surface missing precompute
        abort(500, description=f'Precomputed embeddings not found for method {embed_method}')
    if len(entries) == 0:
        abort(500, description='No images found in dataset')
    print(f"[gallery] built ok: N={len(entries)} coords={tuple(coords2d.shape)} packed={tuple(packed.shape)} eff_layer={eff_layer}")

    # Store dataset root for serving images
    app.config['DATASET_ROOT'] = str(Path(dataset).resolve())

    items: List[Dict[str, Any]] = []
    dataset_root = Path(dataset).resolve()
    for i, e in enumerate(entries):
        # URL for serving via /images endpoint (path relative to dataset root)
        rel = os.path.relpath(Path(e.path).resolve(), dataset_root)
        items.append({
            'id': e.id,
            'url': f'/images/{rel}',
            'className': e.class_name,
            'x': float(coords2d[i, 0]),
            'y': float(coords2d[i, 1]),
            'gx': float(packed[i, 0]),
            'gy': float(packed[i, 1]),
        })
    print(f"[gallery] returning items={len(items)}")
    return jsonify({'items': items, 'dataset': dataset, 'n_layer': eff_layer, 'n_tile': n_tile, 'method': method, 'embed': embed_method, 'warning': None})


@app.get('/images/<path:relpath>')
def serve_image(relpath: str):
    dataset_root = app.config.get('DATASET_ROOT')
    if not dataset_root:
        return abort(400, description='Dataset root not set. Call /gallery.json first with a valid dataset.')
    # Security: resolve and ensure within dataset_root
    root = Path(dataset_root).resolve()
    target = (root / relpath).resolve()
    if root not in target.parents and root != target:
        return abort(403)
    directory = str(target.parent)
    filename = target.name
    if not Path(target).exists():
        return abort(404)
    return send_from_directory(directory, filename)


@app.post('/upload')
def upload_image():
    """Accept an image upload and save it under the dataset root.

    Form fields:
    - file: image file (required)
    - class: optional subfolder to place the image in
    - dataset: optional dataset root; defaults to current DATASET_ROOT or backend/uploads
    """
    f = request.files.get('file')
    if not f:
        return abort(400, description='Missing file')
    class_name = request.form.get('class', '').strip()
    dataset = request.form.get('dataset')

    root = Path(dataset) if dataset else (Path(app.config.get('DATASET_ROOT') or Path(__file__).parent / 'uploads'))
    root.mkdir(parents=True, exist_ok=True)
    target_dir = root / class_name if class_name else root
    target_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    fname = Path(f.filename).name
    if not fname:
        return abort(400, description='Invalid filename')
    out_path = target_dir / fname
    f.save(str(out_path))

    # Return relative path for use in /images
    rel = os.path.relpath(out_path.resolve(), root.resolve())
    return jsonify({'ok': True, 'path': f'/images/{rel}', 'class': class_name or None})


def main():
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '5001'))
    app.run(host=host, port=port, debug=True)


if __name__ == '__main__':
    main()
