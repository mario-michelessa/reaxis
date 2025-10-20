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
from PIL import Image
import io
from flask_cors import CORS

from gallery_backend import ImageGalleryEngine
from embeddings import EmbeddingEngine


app = Flask(__name__)
CORS(app)

# Configure your dataset root here. Set this to your images root folder.
# Example: DATASET_PATH = Path('/data/my_images')
DATASET_PATH = Path('../data/datasets/ISIC2017/')
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
    # 'text' is a frontend-only view; map to a real embedding for gallery fallbacks
    if embed_method == 'text':
        embed_method = 'clip'

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


@app.get('/thumb/<int:size>/<path:relpath>')
def serve_thumbnail(size: int, relpath: str):
    """Serve or generate a cached thumbnail sized to fit within size x size.

    Thumbnails are cached under <DATASET_ROOT>/.cache/thumbs/<size>/<relpath>.jpg
    """
    dataset_root = app.config.get('DATASET_ROOT')
    if not dataset_root:
        return abort(400, description='Dataset root not set. Call /gallery.json first with a valid dataset.')
    if size <= 0 or size > 2048:
        return abort(400, description='Invalid size')
    root = Path(dataset_root).resolve()
    src = (root / relpath).resolve()
    if root not in src.parents and root != src:
        return abort(403)
    if not src.exists():
        return abort(404)

    cache_dir = root / '.cache' / 'thumbs' / str(size) / Path(relpath).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (Path(relpath).stem + '.jpg')

    try:
        if not cache_file.exists() or cache_file.stat().st_mtime < src.stat().st_mtime:
            # Generate thumb
            with Image.open(str(src)) as im:
                # Convert to RGB (flatten alpha on white background)
                if im.mode in ('RGBA', 'LA'):
                    bg = Image.new('RGB', im.size, (255, 255, 255))
                    alpha = im.split()[-1]
                    bg.paste(im, mask=alpha)
                    im = bg
                elif im.mode != 'RGB':
                    im = im.convert('RGB')
                im.thumbnail((size, size), Image.Resampling.LANCZOS)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                im.save(str(cache_file), format='JPEG', quality=85, optimize=True, progressive=True)
    except Exception as e:
        # As a fallback, try to stream a resized image without caching
        try:
            with Image.open(str(src)) as im:
                if im.mode in ('RGBA', 'LA'):
                    bg = Image.new('RGB', im.size, (255, 255, 255))
                    alpha = im.split()[-1]
                    bg.paste(im, mask=alpha)
                    im = bg
                elif im.mode != 'RGB':
                    im = im.convert('RGB')
                im.thumbnail((size, size), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format='JPEG', quality=85, optimize=True, progressive=True)
                buf.seek(0)
                from flask import send_file
                return send_file(buf, mimetype='image/jpeg')
        except Exception:
            return abort(500, description=f'Failed to generate thumbnail: {e}')

    return send_from_directory(str(cache_file.parent), cache_file.name)


@app.post('/text_force')
def text_force():
    """Apply an attractive force from a text region to image coordinates.

    Body JSON:
    - text: string (required)
    - rect: { x, y, w, h } in normalized [0,1] (required)
    - embed: embedding method for text/image space, e.g., 'clip' (default 'clip')
    - alpha: float force strength (default 0.25)
    - method: 'pca' or 'umap' for 2D reduction (default 'pca')
    """
    dataset = app.config.get('DATASET_ROOT') or str(DATASET_PATH)
    if not dataset or not Path(dataset).exists():
        abort(400, description='Invalid dataset path')
    payload = request.get_json(silent=True) or {}
    text = payload.get('text', '').strip()
    rect = payload.get('rect') or {}
    embed_method = (payload.get('embed') or 'clip').lower()
    red_method = (payload.get('method') or 'pca').lower()
    alpha = float(payload.get('alpha') or 0.25)
    if not text or not isinstance(rect, dict) or not all(k in rect for k in ('x','y','w','h')):
        abort(400, description='Missing text or rect')

    engine = ImageGalleryEngine(dataset)
    entries = engine.list_images()
    if not entries:
        abort(400, description='No images')

    import numpy as np
    # Reduce to 2D base coords from the selected embedding method
    embs_for_layout = engine._load_embeddings_only(entries, method=embed_method)
    if embs_for_layout is None:
        abort(400, description=f'Embeddings not available for method {embed_method}')
    coords2d = engine.reduce_to_2d(embs_for_layout, method=red_method)

    # Always compute similarities in CLIP space to match text embedding dimension
    embs_for_sim = engine._load_embeddings_only(entries, method='clip')
    if embs_for_sim is None:
        abort(400, description='CLIP embeddings not available. Precompute with --methods clip')
    emb_engine = EmbeddingEngine(dataset)
    tvec = emb_engine.text_embedding('clip', text)
    if tvec is None:
        # No-op similarities
        sims = np.zeros((coords2d.shape[0],), dtype=np.float32)
    else:
        X = embs_for_sim.astype(np.float32)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        tv = tvec.astype(np.float32)
        tv = tv / (np.linalg.norm(tv) + 1e-8)
        sims = (Xn @ tv)
        sims = np.maximum(0.0, sims)  # only pull, no push

    # Attractive force towards rect center
    cx = float(rect['x']) + float(rect['w']) * 0.5
    cy = float(rect['y']) + float(rect['h']) * 0.5
    c = np.array([cx, cy], dtype=np.float32)
    delta = (c[None, :] - coords2d)
    new_coords = coords2d + alpha * sims[:, None] * delta
    new_coords = np.clip(new_coords, 0.0, 1.0)

    # Optionally pack to grid for minimap display
    packed, eff_layer = engine.pack_to_grid(new_coords, n_layer=0, n_tile=8)

    return jsonify({
        'similarities': sims.tolist(),
        'coords': new_coords.tolist(),
        'packed': packed.tolist(),
        'n_layer': eff_layer,
    })


@app.post('/text_forces')
def text_forces():
    """Apply attraction from multiple text labels to image coordinates.

    Body JSON:
    - texts: [{ text: str, rect: {x,y,w,h} }]
    - ids: [str] optional — order of images in base_coords and desired output order
    - base_coords: [[x,y], ...] optional — initial positions; if missing, uses 2D coords from embed
    - embed: embedding method for similarity space (default 'clip')
    - alpha: float force scale (default 0.25)
    - method: dimensionality reduction method for fallback base coords (default 'pca')
    """
    dataset = app.config.get('DATASET_ROOT') or str(DATASET_PATH)
    if not dataset or not Path(dataset).exists():
        abort(400, description='Invalid dataset path')
    payload = request.get_json(silent=True) or {}
    texts = payload.get('texts') or []
    ids = payload.get('ids') or []
    base_coords = payload.get('base_coords')
    embed_method = (payload.get('embed') or 'clip').lower()
    red_method = (payload.get('method') or 'pca').lower()
    alpha = float(payload.get('alpha') or 0.25)
    if (not isinstance(texts, list)) or len(texts) == 0:
        abort(400, description='Missing texts array')

    engine = ImageGalleryEngine(dataset)
    entries = engine.list_images()
    if not entries:
        abort(400, description='No images')

    # Build id -> index map for ordering
    id_to_idx = {e.id: i for i, e in enumerate(entries)}
    if ids and not isinstance(ids, list):
        abort(400, description='ids must be a list')
    order = [id_to_idx.get(i) for i in ids] if ids else list(range(len(entries)))
    if ids and any(o is None for o in order):
        abort(400, description='Some ids not found in dataset')

    import numpy as np
    # Base coordinates
    if isinstance(base_coords, list) and len(base_coords) == len(order):
        base = np.array(base_coords, dtype=np.float32)
        if base.ndim != 2 or base.shape[1] != 2:
            abort(400, description='base_coords must be Nx2')
    else:
        # Fallback: compute base from selected embedding
        embs_for_layout = engine._load_embeddings_only(entries, method=embed_method)
        if embs_for_layout is None:
            abort(400, description=f'Embeddings not available for method {embed_method}')
        coords2d = engine.reduce_to_2d(embs_for_layout, method=red_method)
        base = coords2d[order, :].astype(np.float32)

    # Similarities in CLIP space (or chosen embed method if desired)
    embs_for_sim = engine._load_embeddings_only(entries, method='clip')
    if embs_for_sim is None:
        abort(400, description='CLIP embeddings not available. Precompute with --methods clip')
    emb_engine = EmbeddingEngine(dataset)
    X = embs_for_sim.astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # Compute per-text normalized similarities and aggregate force
    accum = np.zeros_like(base, dtype=np.float32)  # Nx2
    out_sims = []
    for t in texts:
        txt = (t.get('text') or '').strip()
        rect = t.get('rect') or {}
        if not txt or not all(k in rect for k in ('x','y','w','h')):
            # Skip invalid entries
            out_sims.append([0.0] * len(order))
            continue
        tvec = emb_engine.text_embedding('clip', txt)
        if tvec is None:
            sims = np.zeros((len(entries),), dtype=np.float32)
        else:
            tv = tvec.astype(np.float32)
            tv = tv / (np.linalg.norm(tv) + 1e-8)
            sims = (Xn @ tv)
        # Reorder to requested order
        sims_ord = sims[order]
        # Normalize to [0,1] across dataset for this text
        smin = float(np.min(sims_ord))
        smax = float(np.max(sims_ord))
        denom = (smax - smin) if (smax - smin) > 1e-8 else 1.0
        s_norm = (sims_ord - smin) / denom
        out_sims.append(s_norm.tolist())
        # Attraction to rect center
        cx = float(rect['x']) + float(rect['w']) * 0.5
        cy = float(rect['y']) + float(rect['h']) * 0.5
        c = np.array([cx, cy], dtype=np.float32)[None, :]
        delta = (c - base)
        accum += (s_norm[:, None].astype(np.float32)) * delta

    new_coords = base + alpha * accum
    new_coords = np.clip(new_coords, 0.0, 1.0)

    # Pack to grid for minimap display (optional use on frontend)
    packed, eff_layer = engine.pack_to_grid(new_coords, n_layer=0, n_tile=8)
    return jsonify({
        'ids': ids if ids else [entries[i].id for i in order],
        'similarities': out_sims,  # list per text
        'coords': new_coords.tolist(),
        'packed': packed.tolist(),
        'n_layer': eff_layer,
    })

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
