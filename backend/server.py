#!/usr/bin/env python3
"""
Simple Flask server that serves a gallery with embeddings, 2D layout, and
non-overlapping packed coordinates for a minimap and image grid.

Endpoints
- GET /health
- GET /gallery.json?method=pca|umap|tsne&n_layer=64&n_tile=8
  Returns items with fields: id, url, className, x, y, gx, gy
- GET /images/<path:relpath>
  Serves images relative to the configured dataset root

Note: This is a minimal prototype. For multi-dataset use, consider including
the dataset in the image URL or isolating per-dataset blueprints.
"""

from __future__ import annotations

import os
import re
import time
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory, abort
from PIL import Image
import io
from flask_cors import CORS
import numpy as np

import csv
import hashlib
try:
    from .gallery_backend import ImageGalleryEngine
    from .embeddings import EmbeddingEngine, DEFAULT_SEMANTIC_EMBED_METHOD, embedding_cache_filename, normalize_multimodal_method
    from .reduction_cache import normalize_reduction_method
    from .session_store import (
        DEFAULT_SESSION_NAME,
        LEGACY_SESSION_NAME,
        append_session_log,
        iter_session_names,
        normalize_session_name,
        session_activity_log_path,
        session_axes_path,
        session_dir,
        session_subsets_path,
        session_summary,
        session_visualizations_path,
    )
    from .constants import (
        BACKEND_HOST,
        BACKEND_PORT,
        INITIAL_GALLERY_PROJECTION_METHOD,
        DEFAULT_MAX_ATTRIBUTES,
        DEFAULT_VALUE_COUNT,
        AXIS_MODEL_TYPE,
        AXIS_BAYES_ALPHA,
        AXIS_BAYES_DINO_ALPHA,
        AXIS_BAYES_BIAS_ALPHA,
        AXIS_BAYES_MODE,
        AXIS_BAYES_FEATURE_SPACE,
        AXIS_BAYES_SEMANTIC_METHOD,
        AXIS_BAYES_NORM,
        AXIS_BAYES_CLIP_WEIGHT,
        AXIS_BAYES_DINO_WEIGHT,
        AXIS_PIECEWISE_NUM_EXPERTS,
        AXIS_PIECEWISE_USE_GATING,
        AXIS_PIECEWISE_AGGREGATOR,
        AXIS_PIECEWISE_CLIP_SCALE,
        AXIS_PIECEWISE_DINO_SCALE,
        AXIS_PIECEWISE_PAIRWISE_FROM_SCALAR_MARGIN,
        AXIS_PIECEWISE_PRIOR_STRENGTH,
        AXIS_PIECEWISE_EXPERT_DIVERSITY_STRENGTH,
        AXIS_PIECEWISE_L2_REG,
        AXIS_PIECEWISE_LEARNING_RATE,
        AXIS_PIECEWISE_MAX_REFINE_STEPS,
        AXIS_BAYES_SIGMA2,
        AXIS_BAYES_GRAPH_KNN_K,
        AXIS_BAYES_GRAPH_LAMBDA_SMOOTH,
        AXIS_BAYES_GRAPH_LAMBDA_PRIOR,
        AXIS_BAYES_GRAPH_JITTER,
        AXIS_BAYES_MOVE_TRUST,
        AXIS_BAYES_MOVE_MAG_GAIN,
        AXIS_BAYES_RANK_ETA,
        AXIS_BAYES_RANK_ANCHOR_K,
        AXIS_BAYES_RANK_ANCHOR_DELTA,
        AXIS_BAYES_RANK_MAX_PAIRS,
        AXIS_BAYES_MAX_MOVES,
        AXIS_BAYES_HOTSPOT_BOUNDARY,
        AXIS_BAYES_HOTSPOT_TAU,
        AXIS_BAYES_HOTSPOT_K,
        AXIS_BAYES_EXEMPLAR_K,
        AXIS_RESIDUAL_ALPHA,
        AXIS_RESIDUAL_BETA,
        AXIS_RESIDUAL_LAMBDA,
        AXIS_RESIDUAL_SIGMA_Y,
        AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER,
        AXIS_RESIDUAL_JITTER,
        ZERO_SHOT_HISTOGRAM_BINS,
    )
    from .axis_bayes import AxisBayesEngine
    from .llm_inference import LightweightLLMEngine
    from .zero_shot_regressor import ZeroShotAttributeRegressor, slugify
except ImportError:
    from gallery_backend import ImageGalleryEngine
    from embeddings import EmbeddingEngine, DEFAULT_SEMANTIC_EMBED_METHOD, embedding_cache_filename, normalize_multimodal_method
    from reduction_cache import normalize_reduction_method
    from session_store import (
        DEFAULT_SESSION_NAME,
        LEGACY_SESSION_NAME,
        append_session_log,
        iter_session_names,
        normalize_session_name,
        session_activity_log_path,
        session_axes_path,
        session_dir,
        session_subsets_path,
        session_summary,
        session_visualizations_path,
    )
    from constants import (
        BACKEND_HOST,
        BACKEND_PORT,
        INITIAL_GALLERY_PROJECTION_METHOD,
        DEFAULT_MAX_ATTRIBUTES,
        DEFAULT_VALUE_COUNT,
        AXIS_MODEL_TYPE,
        AXIS_BAYES_ALPHA,
        AXIS_BAYES_DINO_ALPHA,
        AXIS_BAYES_BIAS_ALPHA,
        AXIS_BAYES_MODE,
        AXIS_BAYES_FEATURE_SPACE,
        AXIS_BAYES_SEMANTIC_METHOD,
        AXIS_BAYES_NORM,
        AXIS_BAYES_CLIP_WEIGHT,
        AXIS_BAYES_DINO_WEIGHT,
        AXIS_PIECEWISE_NUM_EXPERTS,
        AXIS_PIECEWISE_USE_GATING,
        AXIS_PIECEWISE_AGGREGATOR,
        AXIS_PIECEWISE_CLIP_SCALE,
        AXIS_PIECEWISE_DINO_SCALE,
        AXIS_PIECEWISE_PAIRWISE_FROM_SCALAR_MARGIN,
        AXIS_PIECEWISE_PRIOR_STRENGTH,
        AXIS_PIECEWISE_EXPERT_DIVERSITY_STRENGTH,
        AXIS_PIECEWISE_L2_REG,
        AXIS_PIECEWISE_LEARNING_RATE,
        AXIS_PIECEWISE_MAX_REFINE_STEPS,
        AXIS_BAYES_SIGMA2,
        AXIS_BAYES_GRAPH_KNN_K,
        AXIS_BAYES_GRAPH_LAMBDA_SMOOTH,
        AXIS_BAYES_GRAPH_LAMBDA_PRIOR,
        AXIS_BAYES_GRAPH_JITTER,
        AXIS_BAYES_MOVE_TRUST,
        AXIS_BAYES_MOVE_MAG_GAIN,
        AXIS_BAYES_RANK_ETA,
        AXIS_BAYES_RANK_ANCHOR_K,
        AXIS_BAYES_RANK_ANCHOR_DELTA,
        AXIS_BAYES_RANK_MAX_PAIRS,
        AXIS_BAYES_MAX_MOVES,
        AXIS_BAYES_HOTSPOT_BOUNDARY,
        AXIS_BAYES_HOTSPOT_TAU,
        AXIS_BAYES_HOTSPOT_K,
        AXIS_BAYES_EXEMPLAR_K,
        AXIS_RESIDUAL_ALPHA,
        AXIS_RESIDUAL_BETA,
        AXIS_RESIDUAL_LAMBDA,
        AXIS_RESIDUAL_SIGMA_Y,
        AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER,
        AXIS_RESIDUAL_JITTER,
        ZERO_SHOT_HISTOGRAM_BINS,
    )
    from axis_bayes import AxisBayesEngine
    from llm_inference import LightweightLLMEngine
    from zero_shot_regressor import ZeroShotAttributeRegressor, slugify


app = Flask(__name__)
CORS(app)

"""Dataset configuration

- DATASETS_ROOT: base folder containing available datasets
- DATASET_PATH: default dataset folder (used if no dataset query is provided)
"""
DATASETS_ROOT = (Path(__file__).parent.parent / 'data' / 'datasets').resolve()
DATASET_PATH = (DATASETS_ROOT / 'ISIC2017').resolve()
SESSIONS_ROOT = (Path(__file__).parent.parent / 'data' / 'sessions').resolve()
LEGACY_AXIS_LIBRARY_PATH = (Path(__file__).parent.parent / 'data' / 'axis_library.json').resolve()

# Keep the currently active dataset root for serving images
app.config['DATASET_ROOT'] = str(DATASET_PATH) if DATASET_PATH.exists() else None
print(f'[server] DATASETS_ROOT={DATASETS_ROOT} exists={DATASETS_ROOT.exists()}')
print(f'[server] default DATASET_PATH={DATASET_PATH} exists={DATASET_PATH.exists()}')

# Lightweight LLM + zero-shot inference modules.
LLM_ENGINE = LightweightLLMEngine()
ZERO_SHOT_REGRESSOR = ZeroShotAttributeRegressor()


def _make_axis_engine(
    *,
    model_type: Optional[str] = None,
    mode: Optional[str] = None,
    feature_space: Optional[str] = None,
    semantic_method: Optional[str] = None,
    norm: Optional[bool] = None,
    clip_weight: Optional[float] = None,
    dino_weight: Optional[float] = None,
    piecewise_num_experts: Optional[int] = None,
    piecewise_use_gating: Optional[bool] = None,
    piecewise_aggregator: Optional[str] = None,
    piecewise_clip_scale: Optional[float] = None,
    piecewise_dino_scale: Optional[float] = None,
    pairwise_from_scalar_margin: Optional[float] = None,
    piecewise_prior_strength: Optional[float] = None,
    piecewise_expert_diversity_strength: Optional[float] = None,
    piecewise_l2_reg: Optional[float] = None,
    piecewise_learning_rate: Optional[float] = None,
    piecewise_max_refine_steps: Optional[int] = None,
    alpha: Optional[float] = None,
    dino_alpha: Optional[float] = None,
    bias_alpha: Optional[float] = None,
    sigma2: Optional[float] = None,
    graph_knn_k: Optional[int] = None,
    graph_lambda_smooth: Optional[float] = None,
    graph_lambda_prior: Optional[float] = None,
    graph_jitter: Optional[float] = None,
    rank_eta: Optional[float] = None,
    rank_anchor_k: Optional[int] = None,
    rank_anchor_delta: Optional[float] = None,
    rank_max_pairs: Optional[int] = None,
    residual_alpha: Optional[float] = None,
    residual_beta: Optional[float] = None,
    residual_lambda: Optional[float] = None,
    residual_sigma_y: Optional[float] = None,
    residual_lengthscale_multiplier: Optional[float] = None,
    residual_jitter: Optional[float] = None,
) -> AxisBayesEngine:
    return AxisBayesEngine(
        llm_engine=LLM_ENGINE,
        model_type=str(model_type or AXIS_MODEL_TYPE),
        mode=str(mode or AXIS_BAYES_MODE),
        feature_space=str(feature_space or AXIS_BAYES_FEATURE_SPACE),
        semantic_method=str(semantic_method or AXIS_BAYES_SEMANTIC_METHOD),
        norm=AXIS_BAYES_NORM if norm is None else bool(norm),
        clip_weight=float(clip_weight if clip_weight is not None else AXIS_BAYES_CLIP_WEIGHT),
        dino_weight=float(dino_weight if dino_weight is not None else AXIS_BAYES_DINO_WEIGHT),
        piecewise_num_experts=int(piecewise_num_experts or AXIS_PIECEWISE_NUM_EXPERTS),
        piecewise_use_gating=AXIS_PIECEWISE_USE_GATING if piecewise_use_gating is None else bool(piecewise_use_gating),
        piecewise_aggregator=str(piecewise_aggregator or AXIS_PIECEWISE_AGGREGATOR),
        piecewise_clip_scale=float(piecewise_clip_scale if piecewise_clip_scale is not None else AXIS_PIECEWISE_CLIP_SCALE),
        piecewise_dino_scale=float(piecewise_dino_scale if piecewise_dino_scale is not None else AXIS_PIECEWISE_DINO_SCALE),
        pairwise_from_scalar_margin=float(
            pairwise_from_scalar_margin if pairwise_from_scalar_margin is not None else AXIS_PIECEWISE_PAIRWISE_FROM_SCALAR_MARGIN
        ),
        piecewise_prior_strength=float(piecewise_prior_strength if piecewise_prior_strength is not None else AXIS_PIECEWISE_PRIOR_STRENGTH),
        piecewise_expert_diversity_strength=float(
            piecewise_expert_diversity_strength if piecewise_expert_diversity_strength is not None else AXIS_PIECEWISE_EXPERT_DIVERSITY_STRENGTH
        ),
        piecewise_l2_reg=float(piecewise_l2_reg if piecewise_l2_reg is not None else AXIS_PIECEWISE_L2_REG),
        piecewise_learning_rate=float(piecewise_learning_rate if piecewise_learning_rate is not None else AXIS_PIECEWISE_LEARNING_RATE),
        piecewise_max_refine_steps=int(piecewise_max_refine_steps or AXIS_PIECEWISE_MAX_REFINE_STEPS),
        alpha=float(alpha if alpha is not None else AXIS_BAYES_ALPHA),
        dino_alpha=float(dino_alpha if dino_alpha is not None else AXIS_BAYES_DINO_ALPHA),
        bias_alpha=float(bias_alpha if bias_alpha is not None else AXIS_BAYES_BIAS_ALPHA),
        sigma2=float(sigma2 if sigma2 is not None else AXIS_BAYES_SIGMA2),
        graph_knn_k=int(graph_knn_k or AXIS_BAYES_GRAPH_KNN_K),
        graph_lambda_smooth=float(graph_lambda_smooth if graph_lambda_smooth is not None else AXIS_BAYES_GRAPH_LAMBDA_SMOOTH),
        graph_lambda_prior=float(graph_lambda_prior if graph_lambda_prior is not None else AXIS_BAYES_GRAPH_LAMBDA_PRIOR),
        graph_jitter=float(graph_jitter if graph_jitter is not None else AXIS_BAYES_GRAPH_JITTER),
        move_trust=AXIS_BAYES_MOVE_TRUST,
        move_mag_gain=AXIS_BAYES_MOVE_MAG_GAIN,
        rank_eta=float(rank_eta if rank_eta is not None else AXIS_BAYES_RANK_ETA),
        rank_anchor_k=int(rank_anchor_k or AXIS_BAYES_RANK_ANCHOR_K),
        rank_anchor_delta=float(rank_anchor_delta if rank_anchor_delta is not None else AXIS_BAYES_RANK_ANCHOR_DELTA),
        rank_max_pairs=int(rank_max_pairs or AXIS_BAYES_RANK_MAX_PAIRS),
        residual_alpha=float(residual_alpha if residual_alpha is not None else AXIS_RESIDUAL_ALPHA),
        residual_beta=float(residual_beta if residual_beta is not None else AXIS_RESIDUAL_BETA),
        residual_lambda=float(residual_lambda if residual_lambda is not None else AXIS_RESIDUAL_LAMBDA),
        residual_sigma_y=float(residual_sigma_y if residual_sigma_y is not None else AXIS_RESIDUAL_SIGMA_Y),
        residual_lengthscale_multiplier=float(
            residual_lengthscale_multiplier if residual_lengthscale_multiplier is not None else AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER
        ),
        residual_jitter=float(residual_jitter if residual_jitter is not None else AXIS_RESIDUAL_JITTER),
        max_moves=AXIS_BAYES_MAX_MOVES,
        hotspot_boundary=AXIS_BAYES_HOTSPOT_BOUNDARY,
        hotspot_tau=AXIS_BAYES_HOTSPOT_TAU,
        hotspot_k=AXIS_BAYES_HOTSPOT_K,
        exemplar_k=AXIS_BAYES_EXEMPLAR_K,
    )


AXIS_BAYES_ENGINE = _make_axis_engine()


def _llm_api_log(req_id: str, message: str):
    print(f'[llm-api:{req_id}] {message}')

def _dataset_has_required_axis_embeddings(dataset_dir: Path) -> bool:
    cache_dir = dataset_dir / '.cache'
    semantic_candidates = []
    preferred = normalize_multimodal_method(AXIS_BAYES_SEMANTIC_METHOD)
    if preferred in {'clip', 'siglip2'}:
        semantic_candidates.append(preferred)
    if 'clip' not in semantic_candidates:
        semantic_candidates.append('clip')
    if not any((cache_dir / f'embeddings_{method}.npz').exists() for method in semantic_candidates):
        return False
    if str(AXIS_BAYES_FEATURE_SPACE).strip().lower() == 'clip_dino':
        dino_cache = cache_dir / 'embeddings_dino.npz'
        return dino_cache.exists()
    return True

def list_available_datasets():
    out = []
    if not DATASETS_ROOT.exists():
        return out
    for p in sorted(DATASETS_ROOT.iterdir()):
        try:
            if p.is_dir() and not p.name.startswith('.'):
                # Heuristic: only include if contains at least one file in subtree
                any_file = next(p.rglob('*.*'), None)
                if any_file is not None and _dataset_has_required_axis_embeddings(p):
                    out.append({'label': p.name, 'value': p.name})
        except Exception:
            continue
    return out

def resolve_dataset_root(name_or_none: str | None) -> Path:
    if not name_or_none:
        return DATASET_PATH
    # Only allow names that resolve under DATASETS_ROOT to avoid arbitrary paths
    candidate = (DATASETS_ROOT / name_or_none).resolve()
    try:
        candidate.relative_to(DATASETS_ROOT)
    except Exception:
        # Outside datasets root; reject by falling back to default
        return DATASET_PATH
    return candidate if candidate.exists() else DATASET_PATH


def _normalize_session_name(value: Any) -> str:
    return normalize_session_name(value, default=DEFAULT_SESSION_NAME)


def _resolve_session_from_payload(payload: Dict[str, Any] | None) -> str:
    session_name = None
    if isinstance(payload, dict):
        session_name = payload.get('session')
    if session_name is None:
        session_name = request.args.get('session')
    return _normalize_session_name(session_name)


def _session_dir(session_name: str) -> Path:
    return session_dir(SESSIONS_ROOT, _normalize_session_name(session_name), create=True)


def _session_axes_path(session_name: str) -> Path:
    return session_axes_path(SESSIONS_ROOT, _normalize_session_name(session_name))


def _session_visualizations_path(session_name: str) -> Path:
    return session_visualizations_path(SESSIONS_ROOT, _normalize_session_name(session_name))


def _session_subsets_path(session_name: str) -> Path:
    return session_subsets_path(SESSIONS_ROOT, _normalize_session_name(session_name))


def _session_activity_path(session_name: str) -> Path:
    return session_activity_log_path(SESSIONS_ROOT, _normalize_session_name(session_name))


def _append_session_log(session_name: str, action: Any, detail: Any) -> Path:
    return append_session_log(SESSIONS_ROOT, _normalize_session_name(session_name), action, detail)


def _axis_library_key(raw: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(raw.get('name') or '').strip().lower(),
        str(raw.get('q') or '').strip().lower(),
        str(raw.get('origin_dataset') or '').strip().lower(),
    )


def _axis_library_item(raw: Any, include_artifact: bool = False) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    item_id = str(raw.get('id') or '').strip()
    name = str(raw.get('name') or '').strip()
    q = str(raw.get('q') or raw.get('query') or '').strip()
    origin_dataset = str(raw.get('origin_dataset') or '').strip()
    created_at = str(raw.get('created_at') or '').strip()
    updated_at = str(raw.get('updated_at') or '').strip()
    mode = str(raw.get('mode') or '').strip()
    model_type = str(raw.get('model_type') or '').strip()
    if not item_id or not name or not q:
        return None
    item = {
        'id': item_id,
        'name': name,
        'q': q,
        'origin_dataset': origin_dataset,
        'mode': mode,
        'model_type': model_type,
        'created_at': created_at,
        'updated_at': updated_at,
        'has_artifact': bool(isinstance(raw.get('serialized_axis'), dict)),
    }
    if include_artifact and isinstance(raw.get('serialized_axis'), dict):
        item['serialized_axis'] = raw.get('serialized_axis')
    return item


def _read_library_file(
    path: Path,
    item_parser,
    include_artifact: bool = False,
) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception:
        return []
    rows = payload if isinstance(payload, list) else payload.get('items', [])
    out: List[Dict[str, Any]] = []
    if isinstance(rows, list):
        for row in rows:
            item = item_parser(row, include_artifact=include_artifact)
            if item:
                out.append(item)
    out.sort(key=lambda it: str(it.get('updated_at') or it.get('created_at') or ''), reverse=True)
    return out


def _write_library_file(path: Path, items: List[Dict[str, Any]], item_parser) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = []
    for row in items:
        item = item_parser(row, include_artifact=True)
        if item:
            clean.append(item)
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2)
    tmp.replace(path)


def _migrate_legacy_axes_to_default_session() -> None:
    legacy_path = LEGACY_AXIS_LIBRARY_PATH
    if not legacy_path.exists():
        return
    target_path = _session_axes_path(LEGACY_SESSION_NAME)
    existing = _read_library_file(target_path, _axis_library_item, include_artifact=True)
    existing_keys = {_axis_library_key(item) for item in existing}
    legacy_items = _read_library_file(legacy_path, _axis_library_item, include_artifact=True)
    merged = list(existing)
    changed = False
    for item in legacy_items:
        key = _axis_library_key(item)
        if key in existing_keys:
            continue
        merged.append(item)
        existing_keys.add(key)
        changed = True
    if changed or (legacy_items and not target_path.exists()):
        _write_library_file(target_path, merged, _axis_library_item)


def _axis_library_read(session_name: str = DEFAULT_SESSION_NAME, include_artifact: bool = False) -> List[Dict[str, Any]]:
    _migrate_legacy_axes_to_default_session()
    return _read_library_file(_session_axes_path(session_name), _axis_library_item, include_artifact=include_artifact)


def _axis_library_write(session_name: str, items: List[Dict[str, Any]]) -> None:
    _write_library_file(_session_axes_path(session_name), items, _axis_library_item)


def _visualization_library_item(raw: Any, include_artifact: bool = False) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    item_id = str(raw.get('id') or '').strip()
    name = str(raw.get('name') or '').strip()
    dataset = str(raw.get('dataset') or '').strip()
    created_at = str(raw.get('created_at') or '').strip()
    updated_at = str(raw.get('updated_at') or '').strip()
    selected_x = str(raw.get('selected_x') or '').strip()
    selected_y = str(raw.get('selected_y') or '').strip()
    selected_x_name = str(raw.get('selected_x_name') or '').strip()
    selected_y_name = str(raw.get('selected_y_name') or '').strip()
    if not item_id or not name or not dataset:
        return None
    custom_axes = raw.get('custom_axes') if isinstance(raw.get('custom_axes'), list) else []
    subset_chips = raw.get('subset_chips') if isinstance(raw.get('subset_chips'), list) else []
    subset_filters = raw.get('subset_filters') if isinstance(raw.get('subset_filters'), list) else []
    subset_filter = raw.get('subset_filter') if isinstance(raw.get('subset_filter'), dict) else None
    histogram_slices = raw.get('histogram_slices') if isinstance(raw.get('histogram_slices'), list) else []
    item = {
        'id': item_id,
        'name': name,
        'dataset': dataset,
        'selected_x': selected_x,
        'selected_y': selected_y,
        'selected_x_name': selected_x_name,
        'selected_y_name': selected_y_name,
        'created_at': created_at,
        'updated_at': updated_at,
        'custom_axis_count': len(custom_axes),
        'subset_active': bool(
            len(subset_chips) > 0
            or len(subset_filters) > 0
            or len(histogram_slices) > 0
            or (isinstance(subset_filter, dict) and subset_filter.get('ids'))
        ),
    }
    if include_artifact:
        item['histogram_slices'] = histogram_slices
        item['subset_chips'] = subset_chips
        item['subset_filters'] = subset_filters
        item['subset_filter'] = subset_filter or None
        item['view_state'] = raw.get('view_state') if isinstance(raw.get('view_state'), dict) else {}
        item['minimap_size_offset'] = float(raw.get('minimap_size_offset') or 0.0)
        item['custom_axes'] = custom_axes
        item['axes_manifest'] = raw.get('axes_manifest') if isinstance(raw.get('axes_manifest'), list) else []
    return item


def _visualization_library_read(
    session_name: str = DEFAULT_SESSION_NAME,
    include_artifact: bool = False,
) -> List[Dict[str, Any]]:
    return _read_library_file(
        _session_visualizations_path(session_name),
        _visualization_library_item,
        include_artifact=include_artifact,
    )


def _visualization_library_write(session_name: str, items: List[Dict[str, Any]]) -> None:
    _write_library_file(_session_visualizations_path(session_name), items, _visualization_library_item)


def _subset_library_item(raw: Any, include_artifact: bool = False) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    item_id = str(raw.get('id') or '').strip()
    name = str(raw.get('name') or '').strip()
    dataset = str(raw.get('dataset') or '').strip()
    created_at = str(raw.get('created_at') or '').strip()
    updated_at = str(raw.get('updated_at') or '').strip()
    subset_chips = raw.get('subset_chips') if isinstance(raw.get('subset_chips'), list) else []
    image_count = int(raw.get('image_count') or 0)
    if not item_id or not name or not dataset:
        return None
    item = {
        'id': item_id,
        'name': name,
        'dataset': dataset,
        'created_at': created_at,
        'updated_at': updated_at,
        'chip_count': int(len(subset_chips)),
        'image_count': image_count,
        'subset_active': bool(len(subset_chips) > 0),
    }
    if include_artifact:
        item['subset_chips'] = subset_chips
    return item


def _subset_library_read(
    session_name: str = DEFAULT_SESSION_NAME,
    include_artifact: bool = False,
) -> List[Dict[str, Any]]:
    return _read_library_file(
        _session_subsets_path(session_name),
        _subset_library_item,
        include_artifact=include_artifact,
    )


def _subset_library_write(session_name: str, items: List[Dict[str, Any]]) -> None:
    _write_library_file(_session_subsets_path(session_name), items, _subset_library_item)


def _project_saved_axis_payload(
    *,
    serialized_axis: Optional[Dict[str, Any]],
    dataset_root: Path,
    collection_id: str,
    q: str,
    axis_name: str,
    mode: Optional[str],
    model_type: Optional[str],
) -> Dict[str, Any]:
    if serialized_axis is None:
        return AXIS_BAYES_ENGINE.create_axis(
            collection_id=collection_id,
            dataset_root=str(dataset_root),
            q=q,
            mode=mode,
            model_type=model_type,
        )

    clip_scale = float(serialized_axis.get('clip_scale') or 1.0)
    dino_scale = float(serialized_axis.get('dino_scale') or 0.0)
    artifact_engine = AxisBayesEngine(
        model_type=str(serialized_axis.get('model_type') or model_type or AXIS_MODEL_TYPE),
        mode=str(serialized_axis.get('mode') or mode or AXIS_BAYES_MODE),
        feature_space=str(serialized_axis.get('feature_space') or AXIS_BAYES_FEATURE_SPACE),
        semantic_method=str(serialized_axis.get('semantic_method') or AXIS_BAYES_SEMANTIC_METHOD),
        norm=bool(serialized_axis.get('norm') if 'norm' in serialized_axis else AXIS_BAYES_NORM),
        clip_weight=max(1e-6, clip_scale * clip_scale),
        dino_weight=max(1e-6, dino_scale * dino_scale) if dino_scale > 0 else 1e-6,
        piecewise_num_experts=int(serialized_axis.get('piecewise_num_experts') or AXIS_PIECEWISE_NUM_EXPERTS),
        piecewise_use_gating=bool(
            serialized_axis.get('piecewise_use_gating')
            if 'piecewise_use_gating' in serialized_axis else AXIS_PIECEWISE_USE_GATING
        ),
        piecewise_aggregator=str(serialized_axis.get('piecewise_aggregator') or AXIS_PIECEWISE_AGGREGATOR),
        piecewise_clip_scale=float(serialized_axis.get('piecewise_clip_scale') or AXIS_PIECEWISE_CLIP_SCALE),
        piecewise_dino_scale=float(serialized_axis.get('piecewise_dino_scale') or AXIS_PIECEWISE_DINO_SCALE),
        pairwise_from_scalar_margin=float(
            serialized_axis.get('pairwise_from_scalar_margin') or AXIS_PIECEWISE_PAIRWISE_FROM_SCALAR_MARGIN
        ),
        piecewise_prior_strength=float(serialized_axis.get('piecewise_prior_strength') or AXIS_PIECEWISE_PRIOR_STRENGTH),
        piecewise_expert_diversity_strength=float(
            serialized_axis.get('piecewise_diversity_strength') or AXIS_PIECEWISE_EXPERT_DIVERSITY_STRENGTH
        ),
        piecewise_l2_reg=float(serialized_axis.get('piecewise_l2_reg') or AXIS_PIECEWISE_L2_REG),
        piecewise_learning_rate=float(serialized_axis.get('piecewise_learning_rate') or AXIS_PIECEWISE_LEARNING_RATE),
        piecewise_max_refine_steps=int(
            serialized_axis.get('piecewise_max_refine_steps') or AXIS_PIECEWISE_MAX_REFINE_STEPS
        ),
        alpha=float(serialized_axis.get('alpha') or AXIS_BAYES_ALPHA),
        dino_alpha=float(serialized_axis.get('dino_alpha') or AXIS_BAYES_DINO_ALPHA),
        bias_alpha=float(serialized_axis.get('bias_alpha') or AXIS_BAYES_BIAS_ALPHA),
        sigma2=float(serialized_axis.get('sigma2') or AXIS_BAYES_SIGMA2),
        graph_knn_k=int(serialized_axis.get('graph_knn_k') or AXIS_BAYES_GRAPH_KNN_K),
        graph_lambda_smooth=float(
            serialized_axis.get('graph_lambda_smooth') or AXIS_BAYES_GRAPH_LAMBDA_SMOOTH
        ),
        graph_lambda_prior=float(serialized_axis.get('graph_lambda_prior') or AXIS_BAYES_GRAPH_LAMBDA_PRIOR),
        graph_jitter=float(serialized_axis.get('graph_jitter') or AXIS_BAYES_GRAPH_JITTER),
        move_trust=AXIS_BAYES_MOVE_TRUST,
        move_mag_gain=AXIS_BAYES_MOVE_MAG_GAIN,
        rank_eta=float(serialized_axis.get('rank_eta') or AXIS_BAYES_RANK_ETA),
        rank_anchor_k=int(serialized_axis.get('rank_anchor_k') or AXIS_BAYES_RANK_ANCHOR_K),
        rank_anchor_delta=float(serialized_axis.get('rank_anchor_delta') or AXIS_BAYES_RANK_ANCHOR_DELTA),
        rank_max_pairs=int(serialized_axis.get('rank_max_pairs') or AXIS_BAYES_RANK_MAX_PAIRS),
        residual_alpha=float(serialized_axis.get('residual_alpha') or AXIS_RESIDUAL_ALPHA),
        residual_beta=float(serialized_axis.get('residual_beta') or AXIS_RESIDUAL_BETA),
        residual_lambda=float(serialized_axis.get('residual_lambda') or AXIS_RESIDUAL_LAMBDA),
        residual_sigma_y=float(serialized_axis.get('residual_sigma_y') or AXIS_RESIDUAL_SIGMA_Y),
        residual_lengthscale_multiplier=float(
            serialized_axis.get('residual_lengthscale_multiplier') or AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER
        ),
        residual_jitter=float(serialized_axis.get('residual_jitter') or AXIS_RESIDUAL_JITTER),
        hotspot_boundary=AXIS_BAYES_HOTSPOT_BOUNDARY,
        hotspot_tau=AXIS_BAYES_HOTSPOT_TAU,
        hotspot_k=AXIS_BAYES_HOTSPOT_K,
        exemplar_k=AXIS_BAYES_EXEMPLAR_K,
        max_moves=AXIS_BAYES_MAX_MOVES,
        llm_engine=LLM_ENGINE,
    )
    projected_blob = artifact_engine.project_serialized_axis(
        payload=serialized_axis,
        collection_id=collection_id,
        dataset_root=str(dataset_root),
        axis_name=axis_name,
    )
    state = AXIS_BAYES_ENGINE.deserialize_axis(projected_blob)
    return AXIS_BAYES_ENGINE._state_payload(state)


def _resolve_dataset_from_payload(payload: Dict[str, Any] | None) -> Path:
    dataset_name = None
    if isinstance(payload, dict):
        dataset_name = payload.get('dataset')
        if dataset_name is None:
            dataset_name = payload.get('collection_id')
    if dataset_name is None:
        dataset_name = request.args.get('dataset')
    if dataset_name is None:
        dataset_name = request.args.get('collection_id')
    dataset_path = resolve_dataset_root(dataset_name)
    app.config['DATASET_ROOT'] = str(dataset_path.resolve())
    return dataset_path


def _metadata_field_names(dataset_root: Path) -> List[str]:
    out: List[str] = []
    meta_path = dataset_root / 'metadata.csv'
    if not meta_path.exists():
        return out
    try:
        with meta_path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = [h.strip() for h in (reader.fieldnames or [])]
        if not headers:
            return out
        image_col = None
        for h in headers:
            if h.lower() == 'image':
                image_col = h
                break
        out = [h for h in headers if h != image_col]
    except Exception:
        return []
    return out


def _semantic_method_or_default(method: str) -> str:
    normalized = normalize_multimodal_method(method)
    if normalized in {'clip', 'siglip2'}:
        return normalized
    return DEFAULT_SEMANTIC_EMBED_METHOD


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if raw in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return None


def _requested_axis_semantic_method(payload: Dict[str, Any]) -> Optional[str]:
    normalized = normalize_multimodal_method(
        payload.get('semantic_method') or payload.get('semanticMethod') or payload.get('axis_embed') or payload.get('axisEmbed') or ''
    )
    if normalized in {'clip', 'siglip2'}:
        return normalized
    return None


def _requested_axis_norm(payload: Dict[str, Any]) -> Optional[bool]:
    return _parse_optional_bool(payload.get('norm') if 'norm' in payload else payload.get('axis_norm', payload.get('axisNorm')))


def _load_semantic_embeddings(dataset_root: Path, preferred_method: str = DEFAULT_SEMANTIC_EMBED_METHOD):
    engine = ImageGalleryEngine(str(dataset_root))
    entries = engine.list_images()
    if not entries:
        abort(400, description='No images found in dataset')
    methods = []
    preferred = _semantic_method_or_default(preferred_method)
    for candidate in (preferred, 'clip'):
        if candidate not in methods:
            methods.append(candidate)
    for method in methods:
        embs = engine._load_embeddings_only(entries, method=method)
        if embs is None:
            continue
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim != 2 or embs.shape[0] != len(entries):
            abort(500, description=f'Invalid {method} embedding shape')
        return entries, embs, method
    abort(
        400,
        description=(
            f'No semantic embeddings available. Tried {methods}. '
            f'Precompute with --methods {preferred}'
        ),
    )


def _string_list(v: Any) -> List[str]:
    if not isinstance(v, list):
        return []
    out: List[str] = []
    for x in v:
        s = str(x or '').strip()
        if s:
            out.append(s)
    return out


def _normalize_attribute_type(v: Any) -> str:
    s = str(v or '').strip().lower()
    if s in {'categorical', 'ordinal', 'continuous'}:
        return s
    if s in {'nominal', 'class', 'classes', 'discrete', 'category', 'categories'}:
        return 'categorical'
    if s in {'ordered', 'ranked', 'rank', 'ranking'}:
        return 'ordinal'
    if s in {'numeric', 'number', 'scalar', 'real'}:
        return 'continuous'
    if 'categor' in s:
        return 'categorical'
    if 'ordin' in s or 'rank' in s:
        return 'ordinal'
    if 'contin' in s or 'numeric' in s or 'scalar' in s:
        return 'continuous'
    return ''


def _resolve_dataset_from_form_or_query(default_dataset: Any = None) -> Path:
    dataset_name = None
    try:
        dataset_name = request.form.get('dataset')
    except Exception:
        dataset_name = None
    if (dataset_name is None or str(dataset_name).strip() == '') and default_dataset is not None:
        dataset_name = default_dataset
    if dataset_name is None:
        dataset_name = request.args.get('dataset')
    dataset_path = resolve_dataset_root(str(dataset_name or '').strip() or None)
    app.config['DATASET_ROOT'] = str(dataset_path.resolve())
    return dataset_path


def _safe_float(v: Any) -> Optional[float]:
    try:
        n = float(v)
    except Exception:
        return None
    if not np.isfinite(n):
        return None
    return float(n)


def _format_numeric_label(v: float) -> str:
    n = float(v)
    if abs(n - round(n)) < 1e-6:
        return str(int(round(n)))
    return f'{n:.3f}'.rstrip('0').rstrip('.')


def _read_metadata_table(meta_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {'headers': [], 'image_col': None, 'rows': []}
    if not meta_path.exists():
        return out
    with meta_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = [str(h).strip() for h in (reader.fieldnames or []) if str(h or '').strip()]
        if not headers:
            return out
        image_col = next((h for h in headers if h.lower() == 'image'), None)
        if not image_col:
            return out
        rows: List[Dict[str, str]] = []
        for raw_row in reader:
            row: Dict[str, str] = {}
            for k, v in (raw_row or {}).items():
                key = str(k).strip() if k is not None else ''
                if not key:
                    continue
                row[key] = '' if v is None else str(v).strip()
            rows.append(row)
    out['headers'] = headers
    out['image_col'] = image_col
    out['rows'] = rows
    return out


def _normalize_metadata_rows(rows: List[Dict[str, Any]], image_col: str, headers: Optional[List[str]] = None) -> Tuple[List[str], List[Dict[str, str]]]:
    fields: List[str] = []
    out_rows: List[Dict[str, str]] = []

    def track_field(name: str):
        if name and name not in fields:
            fields.append(name)

    candidate_order = [str(h).strip() for h in (headers or []) if str(h or '').strip()]
    image_col_l = str(image_col or 'image').strip().lower()

    for raw in rows:
        if not isinstance(raw, dict):
            continue
        image_key = str(raw.get(image_col) or raw.get('image') or raw.get('Image') or '').strip()
        if not image_key:
            continue
        norm: Dict[str, str] = {'image': image_key}

        ordered_keys = []
        seen = set()
        for key in candidate_order:
            if key in seen:
                continue
            seen.add(key)
            ordered_keys.append(key)
        for key in raw.keys():
            k = str(key).strip()
            if not k or k in seen:
                continue
            seen.add(k)
            ordered_keys.append(k)

        for key in ordered_keys:
            k = str(key).strip()
            if not k or k.lower() == image_col_l or k.lower() == 'image':
                continue
            val = str(raw.get(key) or '').strip()
            if val == '':
                continue
            norm[k] = val
            track_field(k)
        out_rows.append(norm)

    return fields, out_rows


def _write_metadata_csv(meta_path: Path, fields: List[str], rows: List[Dict[str, str]]):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_fields = [f for f in fields if f and f != 'image']
    headers = ['image'] + ordered_fields
    with meta_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            out = {h: str(row.get(h) or '') for h in headers}
            writer.writerow(out)


def _build_entry_lookup(entries: List[Any], dataset_root: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}

    def add(key: str, image_id: str):
        s = str(key or '').strip()
        if not s:
            return
        variants = [s, s.lower(), s.replace('\\', '/'), s.replace('\\', '/').lower()]
        for v in variants:
            if v and v not in lookup:
                lookup[v] = image_id

    for e in entries:
        image_id = str(getattr(e, 'id', '') or '').strip()
        if not image_id:
            continue
        add(image_id, image_id)
        p = Path(str(getattr(e, 'path', '') or ''))
        add(p.name, image_id)
        add(p.stem, image_id)
        try:
            rel = os.path.relpath(Path(e.path).resolve(), dataset_root)
            add(rel, image_id)
            add(Path(rel).name, image_id)
            add(Path(rel).stem, image_id)
        except Exception:
            pass
    return lookup


def _resolve_entry_id_from_key(raw_key: Any, lookup: Dict[str, str]) -> Optional[str]:
    key = str(raw_key or '').strip()
    if not key:
        return None
    candidates = [key, key.replace('\\', '/'), Path(key).name, Path(key).stem]
    for c in candidates:
        if not c:
            continue
        if c in lookup:
            return lookup[c]
        cl = c.lower()
        if cl in lookup:
            return lookup[cl]
    return None


def _infer_metadata_field_type(values: List[str]) -> str:
    numeric_vals: List[float] = []
    non_numeric = 0
    for raw in values:
        s = str(raw or '').strip()
        if not s:
            continue
        n = _safe_float(s)
        if n is None:
            non_numeric += 1
        else:
            numeric_vals.append(float(n))
    total = len(numeric_vals) + non_numeric
    if total == 0:
        return 'categorical'
    mostly_numeric = len(numeric_vals) >= max(1, int(0.85 * total))
    if mostly_numeric:
        uniq = len({round(v, 6) for v in numeric_vals})
        if uniq >= 12:
            return 'continuous'
        if uniq >= 3:
            return 'ordinal'
        return 'categorical'
    return 'categorical'


def _load_metadata_field_maps(dataset_root: Path, entries: List[Any]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], int, List[str]]:
    meta_path = dataset_root / 'metadata.csv'
    table = _read_metadata_table(meta_path)
    image_col = table.get('image_col')
    headers = table.get('headers') or []
    rows = table.get('rows') or []
    if not image_col or not headers:
        return {}, {}, 0, []

    field_order = [h for h in headers if h != image_col]
    field_maps: Dict[str, Dict[str, str]] = {field: {} for field in field_order}
    lookup = _build_entry_lookup(entries, dataset_root)

    matched_rows = 0
    for row in rows:
        image_key = str(row.get(image_col) or '').strip()
        image_id = _resolve_entry_id_from_key(image_key, lookup)
        if not image_id:
            continue
        matched_rows += 1
        for field in field_order:
            val = str(row.get(field) or '').strip()
            if not val:
                continue
            field_maps[field][image_id] = val

    # Remove empty fields and infer types.
    filtered_maps: Dict[str, Dict[str, str]] = {}
    field_types: Dict[str, str] = {}
    filtered_order: List[str] = []
    for field in field_order:
        fmap = field_maps.get(field) or {}
        if len(fmap) == 0:
            continue
        filtered_maps[field] = fmap
        filtered_order.append(field)
        field_types[field] = _infer_metadata_field_type(list(fmap.values()))
    return filtered_maps, field_types, matched_rows, filtered_order


def _build_metadata_axes(entries: List[Any], dataset_root: Path) -> List[Dict[str, Any]]:
    field_maps, field_types, _, field_order = _load_metadata_field_maps(dataset_root, entries)
    axes: List[Dict[str, Any]] = []

    for field in field_order:
        raw_map = field_maps.get(field) or {}
        if len(raw_map) == 0:
            continue
        field_type = field_types.get(field) or 'categorical'

        # Numeric continuous field -> direct min/max normalization in [0,1].
        if field_type == 'continuous':
            numeric_map: Dict[str, float] = {}
            all_numeric = True
            for image_id, raw_val in raw_map.items():
                n = _safe_float(raw_val)
                if n is None:
                    all_numeric = False
                    break
                numeric_map[image_id] = float(n)
            if all_numeric and len(numeric_map) > 0:
                vals = np.asarray(list(numeric_map.values()), dtype=np.float32)
                lo = float(np.min(vals))
                hi = float(np.max(vals))
                span = hi - lo
                coords = {}
                for image_id, num_val in numeric_map.items():
                    if span <= 1e-8:
                        score = 0.5
                    else:
                        score = (float(num_val) - lo) / span
                    coords[image_id] = float(max(0.0, min(1.0, score)))
                tick_pos = [0.0, 0.25, 0.5, 0.75, 1.0]
                labels = [_format_numeric_label(lo + (hi - lo) * t) for t in tick_pos]
                axes.append({
                    'id': f'axis:meta:{field}',
                    'name': field,
                    'coords': coords,
                    'labels': labels,
                    'label_positions': tick_pos,
                    'group': 'meta',
                    'attribute_type': 'continuous',
                })
                continue

        # Ordinal/categorical fallback -> ordered bins + deterministic jitter.
        uniq_vals_raw = []
        for v in raw_map.values():
            s = str(v or '').strip()
            if s and s not in uniq_vals_raw:
                uniq_vals_raw.append(s)
        if len(uniq_vals_raw) == 0:
            continue

        as_num: List[Tuple[float, str]] = []
        all_numeric = True
        for sv in uniq_vals_raw:
            n = _safe_float(sv)
            if n is None:
                all_numeric = False
                break
            as_num.append((float(n), sv))
        if all_numeric:
            uniq_vals_sorted = [sv for _, sv in sorted(as_num, key=lambda x: x[0])]
        else:
            uniq_vals_sorted = sorted(uniq_vals_raw, key=lambda s: s.lower())

        n_vals = len(uniq_vals_sorted)
        val_to_idx = {v: i for i, v in enumerate(uniq_vals_sorted)}
        coords: Dict[str, float] = {}
        for image_id, raw_val in raw_map.items():
            sv = str(raw_val or '').strip()
            idx = int(val_to_idx.get(sv, 0))
            base = (idx + 1) / float(max(1, n_vals))
            jitter_range = 1.0 / (2.0 * float(max(1, n_vals)))
            h = hashlib.md5(f'{field}|{image_id}'.encode('utf-8')).digest()
            u = int.from_bytes(h[:8], 'big') / float(2**64 - 1)
            jitter = (u - 1.0) * jitter_range
            val = max(0.0, min(1.0, base + jitter))
            coords[image_id] = float(val)
        label_positions = [((i + 0.75) / float(max(1, n_vals))) for i in range(n_vals)]
        axes.append({
            'id': f'axis:meta:{field}',
            'name': field,
            'coords': coords,
            'labels': uniq_vals_sorted,
            'label_positions': label_positions,
            'group': 'meta',
            'attribute_type': field_type if field_type in {'categorical', 'ordinal'} else 'categorical',
        })
    return axes


def _summarize_metadata_fields(entries: List[Any], dataset_root: Path) -> Dict[str, Any]:
    field_maps, field_types, matched_rows, field_order = _load_metadata_field_maps(dataset_root, entries)
    n_images = len(entries)
    fields = []
    for field in field_order:
        fmap = field_maps.get(field) or {}
        vals = [str(v).strip() for v in fmap.values() if str(v or '').strip()]
        unique_vals = len({v.lower() for v in vals})
        fields.append({
            'name': field,
            'type': field_types.get(field) or 'categorical',
            'coverage': int(len(fmap)),
            'coverage_ratio': float(len(fmap) / float(max(1, n_images))),
            'unique_values': int(unique_vals),
        })
    return {
        'field_maps': field_maps,
        'field_types': field_types,
        'field_order': field_order,
        'matched_rows': int(matched_rows),
        'fields': fields,
        'image_count': int(n_images),
    }


def _normalize_axis_payload(raw_axis: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_axis, dict):
        return None
    axis_id = str(raw_axis.get('id') or '').strip()
    if not axis_id:
        return None
    coords_raw = raw_axis.get('coords')
    if not isinstance(coords_raw, dict):
        return None
    coords: Dict[str, float] = {}
    for k, v in coords_raw.items():
        key = str(k or '').strip()
        n = _safe_float(v)
        if not key or n is None:
            continue
        coords[key] = float(max(0.0, min(1.0, n)))
    if len(coords) < 2:
        return None
    return {
        'id': axis_id,
        'name': str(raw_axis.get('name') or axis_id).strip(),
        'coords': coords,
        'attribute_type': _normalize_attribute_type(raw_axis.get('attribute_type') or raw_axis.get('type') or ''),
        'group': str(raw_axis.get('group') or '').strip(),
        'scoring_method': str(raw_axis.get('scoring_method') or '').strip(),
    }


def _normalize_axes_payload(raw_axes: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_axes, list):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for raw in raw_axes:
        axis = _normalize_axis_payload(raw)
        if not axis:
            continue
        axis_id = axis['id']
        if axis_id in seen:
            continue
        seen.add(axis_id)
        out.append(axis)
    return out


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float32).reshape(-1)
    yv = np.asarray(y, dtype=np.float32).reshape(-1)
    if xv.size < 2 or yv.size < 2:
        return 0.0
    sx = float(np.std(xv))
    sy = float(np.std(yv))
    if sx <= 1e-8 or sy <= 1e-8:
        return 0.0
    c = float(np.corrcoef(xv, yv)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def _build_labels_for_field_separability(field_map: Dict[str, str], ids: List[str], field_type: str) -> Tuple[np.ndarray, str, List[str]]:
    labels = np.array([''] * len(ids), dtype=object)
    ftype = field_type if field_type in {'categorical', 'ordinal', 'continuous'} else 'categorical'

    if ftype == 'continuous':
        valid_idx: List[int] = []
        vals: List[float] = []
        for i, image_id in enumerate(ids):
            n = _safe_float(field_map.get(image_id))
            if n is None:
                continue
            valid_idx.append(i)
            vals.append(float(n))
        if len(valid_idx) < 12:
            return labels, ftype, []
        arr = np.asarray(vals, dtype=np.float32)
        uniq = np.unique(np.round(arr, 6))
        if uniq.size < 3:
            return labels, ftype, []
        n_bins = int(min(5, max(3, uniq.size)))
        edges = np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1))
        if np.any(np.diff(edges) <= 1e-8):
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            if hi - lo <= 1e-8:
                return labels, ftype, []
            edges = np.linspace(lo, hi, num=n_bins + 1, dtype=np.float32)
        bin_ids = np.digitize(arr, edges[1:-1], right=False)
        for j, i in enumerate(valid_idx):
            labels[i] = f'q{int(bin_ids[j]) + 1}'
        value_names = [f'q{i + 1}' for i in range(int(n_bins))]
        return labels, ftype, value_names

    # Categorical/ordinal: case-insensitive symbolic labels.
    for i, image_id in enumerate(ids):
        v = str(field_map.get(image_id) or '').strip().lower()
        if v:
            labels[i] = v
    uniq_vals = sorted({str(v) for v in labels.tolist() if str(v)})
    return labels, ftype, uniq_vals


def _compute_knn_indices(points: np.ndarray, k_neighbors: int) -> Optional[np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    n = int(pts.shape[0]) if pts.ndim == 2 else 0
    if n < 2:
        return None
    k = int(max(1, min(max(1, k_neighbors), n - 1)))
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    return np.argpartition(d2, kth=k - 1, axis=1)[:, :k]


def _label_separability_from_knn(labels: np.ndarray, knn_idx: np.ndarray) -> Optional[Dict[str, Any]]:
    if labels.size == 0 or knn_idx is None or knn_idx.size == 0:
        return None
    valid = np.array([bool(str(v)) for v in labels.tolist()], dtype=bool)
    n_valid = int(np.sum(valid))
    if n_valid < max(12, knn_idx.shape[1] + 1):
        return None
    vals, counts = np.unique(labels[valid], return_counts=True)
    if len(vals) < 2:
        return None
    priors = counts.astype(np.float64) / float(max(1, n_valid))
    baseline = float(np.sum(priors * priors))

    purities: List[float] = []
    valid_idx = np.where(valid)[0]
    for i in valid_idx:
        neigh = knn_idx[i]
        neigh_valid = valid[neigh]
        n = int(np.sum(neigh_valid))
        if n <= 0:
            continue
        same = int(np.sum(labels[neigh[neigh_valid]] == labels[i]))
        purities.append(float(same) / float(n))
    if len(purities) < max(8, len(vals)):
        return None
    purity = float(np.mean(purities))
    norm = (purity - baseline) / float(max(1e-8, 1.0 - baseline))
    norm = float(max(0.0, min(1.0, norm)))
    return {
        'score': norm,
        'purity': purity,
        'baseline': baseline,
        'coverage': float(n_valid / float(max(1, labels.size))),
        'n_classes': int(len(vals)),
        'n_labeled': int(n_valid),
    }


def _compute_pair_separability(
    ids: List[str],
    x: np.ndarray,
    y: np.ndarray,
    field_maps: Dict[str, Dict[str, str]],
    field_types: Dict[str, str],
    field_filter: Optional[List[str]] = None,
    k_neighbors: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    if len(ids) < 2:
        return 0.0, []
    filter_set = {str(f).strip().lower() for f in (field_filter or []) if str(f).strip()}
    points = np.stack([x, y], axis=1).astype(np.float32)
    knn_idx = _compute_knn_indices(points, k_neighbors=k_neighbors)
    if knn_idx is None:
        return 0.0, []

    breakdown: List[Dict[str, Any]] = []
    for field, fmap in field_maps.items():
        if filter_set and field.lower() not in filter_set:
            continue
        labels, ftype, values = _build_labels_for_field_separability(
            field_map=fmap,
            ids=ids,
            field_type=field_types.get(field) or 'categorical',
        )
        sep = _label_separability_from_knn(labels, knn_idx)
        if not sep:
            continue
        breakdown.append({
            'field': field,
            'field_type': ftype,
            'score': float(sep['score']),
            'purity': float(sep['purity']),
            'baseline': float(sep['baseline']),
            'coverage': float(sep['coverage']),
            'n_classes': int(sep['n_classes']),
            'n_labeled': int(sep['n_labeled']),
            'values': values[:8],
        })
    breakdown.sort(key=lambda r: float(r.get('score') or 0.0), reverse=True)
    if len(breakdown) == 0:
        return 0.0, []
    top = breakdown[:3]
    score = float(np.mean([float(row.get('score') or 0.0) for row in top]))
    return score, breakdown


def _sanitize_axes_for_export(raw_axes: Any, valid_ids: set[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for axis in _normalize_axes_payload(raw_axes):
        coords = {
            image_id: float(v)
            for image_id, v in (axis.get('coords') or {}).items()
            if image_id in valid_ids
        }
        out.append({
            'id': axis.get('id'),
            'name': axis.get('name'),
            'coords': coords,
            'attribute_type': axis.get('attribute_type') or '',
            'group': axis.get('group') or '',
            'scoring_method': axis.get('scoring_method') or '',
        })
    return out


@app.get('/health')
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get('/gallery.json')
def gallery() -> Any:
    # Accept optional dataset name via query params (must exist under DATASETS_ROOT)
    dataset_name = request.args.get('dataset')
    dataset_path = resolve_dataset_root(dataset_name)
    dataset = str(dataset_path)
    try:
        method = normalize_reduction_method(request.args.get('method', INITIAL_GALLERY_PROJECTION_METHOD))
    except ValueError as exc:
        abort(400, description=str(exc))
    embed_method = normalize_multimodal_method(request.args.get('embed', DEFAULT_SEMANTIC_EMBED_METHOD))
    # 'text' is a frontend-only view; map to a real embedding for gallery fallbacks
    if embed_method == 'text':
        embed_method = DEFAULT_SEMANTIC_EMBED_METHOD

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
    warning = None
    entries, coords2d, packed, eff_layer = engine.build_gallery_from_precomputed(
        n_layer=n_layer, n_tile=n_tile, method=method, embed_method=embed_method
    )
    if (entries is None or coords2d is None or packed is None) and embed_method == DEFAULT_SEMANTIC_EMBED_METHOD:
        fallback_method = 'clip'
        entries, coords2d, packed, eff_layer = engine.build_gallery_from_precomputed(
            n_layer=n_layer, n_tile=n_tile, method=method, embed_method=fallback_method
        )
        if entries is not None and coords2d is not None and packed is not None:
            warning = f'{DEFAULT_SEMANTIC_EMBED_METHOD} cache missing; fell back to {fallback_method}'
            embed_method = fallback_method
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
    # Attempt to load per-image metadata axes from metadata.csv in dataset root
    metadata_axes = []
    try:
        metadata_axes = _build_metadata_axes(entries=entries, dataset_root=dataset_root)
    except Exception as e:
        print('[gallery] metadata parse error:', e)

    print(f"[gallery] returning items={len(items)} meta_axes={len(metadata_axes)}")
    return jsonify({'items': items,
                    'dataset': dataset,
                    'n_layer': eff_layer,
                    'n_tile': n_tile,
                    'method': method,
                    'embed': embed_method,
                    'embedding_file': embedding_cache_filename(embed_method, normalize=True),
                    'metadata_axes': metadata_axes,
                    'warning': warning})


@app.get('/datasets')
def datasets_list():
    """List available datasets under DATASETS_ROOT.

    Returns an array of { label, value } objects, where value can be passed
    as the 'dataset' query parameter to /gallery.json.
    """
    datasets = list_available_datasets()
    # Always include a default entry at top
    default_label = DATASET_PATH.name if DATASET_PATH.exists() else 'Default'
    base = (
        [{'label': default_label, 'value': DATASET_PATH.name}]
        if DATASET_PATH.exists() and _dataset_has_required_axis_embeddings(DATASET_PATH)
        else []
    )
    # De-duplicate by value while preserving order
    seen = set()
    out = []
    for d in base + datasets:
        v = d.get('value')
        if v in seen:
            continue
        seen.add(v)
        out.append({'label': d.get('label') or v, 'value': v})
    return jsonify(out)


@app.get('/metadata/summary')
def metadata_summary():
    dataset_name = request.args.get('dataset')
    dataset_root = resolve_dataset_root(dataset_name)
    app.config['DATASET_ROOT'] = str(dataset_root.resolve())

    engine = ImageGalleryEngine(str(dataset_root))
    entries = engine.list_images()
    summary = _summarize_metadata_fields(entries=entries, dataset_root=dataset_root)
    try:
        axes = _build_metadata_axes(entries=entries, dataset_root=dataset_root)
    except Exception:
        axes = []

    return jsonify({
        'dataset': str(dataset_root),
        'dataset_name': dataset_root.name,
        'image_count': int(summary.get('image_count') or 0),
        'matched_rows': int(summary.get('matched_rows') or 0),
        'field_count': int(len(summary.get('fields') or [])),
        'fields': summary.get('fields') or [],
        'metadata_axes': axes,
        'metadata_path': str(dataset_root / 'metadata.csv'),
    })


@app.post('/metadata/upload')
def metadata_upload():
    incoming_fields: List[str] = []
    incoming_rows: List[Dict[str, str]] = []
    source = 'json'

    if request.files and request.files.get('file'):
        source = 'file'
        dataset_root = _resolve_dataset_from_form_or_query()
        mode = str(request.form.get('mode') or 'replace').strip().lower()

        f = request.files.get('file')
        if not f:
            abort(400, description='Missing file')
        try:
            text = f.read().decode('utf-8-sig', errors='replace')
            reader = csv.DictReader(io.StringIO(text))
            headers = [str(h).strip() for h in (reader.fieldnames or []) if str(h or '').strip()]
            if len(headers) == 0:
                abort(400, description='metadata CSV has no headers')
            image_col = next((h for h in headers if h.lower() == 'image'), None)
            if not image_col:
                abort(400, description='metadata CSV must contain an image column')
            rows = []
            for raw in reader:
                row = {str(k).strip(): ('' if v is None else str(v).strip()) for k, v in (raw or {}).items()}
                rows.append(row)
            incoming_fields, incoming_rows = _normalize_metadata_rows(rows=rows, image_col=image_col, headers=headers)
        except Exception as e:
            abort(400, description=f'Could not parse metadata CSV: {e}')
    else:
        payload = request.get_json(silent=True) or {}
        dataset_root = _resolve_dataset_from_payload(payload)
        mode = str(payload.get('mode') or 'replace').strip().lower()
        rows_raw = payload.get('rows')
        if not isinstance(rows_raw, list):
            abort(400, description='Provide rows as a JSON array of objects')
        headers: List[str] = []
        for r in rows_raw:
            if not isinstance(r, dict):
                continue
            for k in r.keys():
                key = str(k).strip()
                if key and key not in headers:
                    headers.append(key)
        image_col = next((h for h in headers if h.lower() == 'image'), None)
        if not image_col:
            abort(400, description='JSON metadata rows must contain an image key')
        incoming_fields, incoming_rows = _normalize_metadata_rows(rows=rows_raw, image_col=image_col, headers=headers)

    if not dataset_root.exists():
        abort(400, description=f'Dataset does not exist: {dataset_root}')
    if mode not in {'replace', 'merge'}:
        mode = 'replace'
    if len(incoming_rows) == 0:
        abort(400, description='No valid metadata rows were parsed')

    meta_path = dataset_root / 'metadata.csv'

    if mode == 'merge' and meta_path.exists():
        existing = _read_metadata_table(meta_path)
        existing_fields, existing_rows = _normalize_metadata_rows(
            rows=existing.get('rows') or [],
            image_col=str(existing.get('image_col') or 'image'),
            headers=existing.get('headers') or [],
        )
        merged: Dict[str, Dict[str, str]] = {}
        order: List[str] = []
        for row in existing_rows:
            image_id = str(row.get('image') or '').strip()
            if not image_id or image_id in merged:
                continue
            merged[image_id] = dict(row)
            order.append(image_id)
        for row in incoming_rows:
            image_id = str(row.get('image') or '').strip()
            if not image_id:
                continue
            if image_id not in merged:
                merged[image_id] = {'image': image_id}
                order.append(image_id)
            for field, value in row.items():
                if field == 'image':
                    continue
                sval = str(value or '').strip()
                if sval == '':
                    continue
                merged[image_id][field] = sval
        final_fields = existing_fields + [f for f in incoming_fields if f not in existing_fields]
        final_rows = [merged[image_id] for image_id in order]
    else:
        final_fields = incoming_fields
        final_rows = incoming_rows

    _write_metadata_csv(meta_path=meta_path, fields=final_fields, rows=final_rows)

    engine = ImageGalleryEngine(str(dataset_root))
    entries = engine.list_images()
    summary = _summarize_metadata_fields(entries=entries, dataset_root=dataset_root)
    axes = _build_metadata_axes(entries=entries, dataset_root=dataset_root)

    return jsonify({
        'ok': True,
        'dataset': str(dataset_root),
        'dataset_name': dataset_root.name,
        'mode': mode,
        'source': source,
        'metadata_path': str(meta_path),
        'row_count': int(len(final_rows)),
        'field_count': int(len(summary.get('fields') or [])),
        'fields': summary.get('fields') or [],
        'matched_rows': int(summary.get('matched_rows') or 0),
        'metadata_axes': axes,
    })


@app.post('/analysis/recommend_scatterplots')
def recommend_scatterplots():
    payload = request.get_json(silent=True) or {}
    dataset_root = _resolve_dataset_from_payload(payload)

    axes = _normalize_axes_payload(payload.get('axes'))
    if len(axes) < 2:
        abort(400, description='Provide at least two axes with coords to recommend scatterplots')

    top_k = int(payload.get('top_k') or 8)
    top_k = max(1, min(40, top_k))
    min_overlap = int(payload.get('min_overlap') or 40)
    min_overlap = max(8, min(10000, min_overlap))
    k_neighbors = int(payload.get('k_neighbors') or 10)
    k_neighbors = max(3, min(64, k_neighbors))

    corr_w = float(payload.get('correlation_weight') or 0.45)
    sep_w = float(payload.get('separability_weight') or 0.55)
    corr_w = max(0.0, corr_w)
    sep_w = max(0.0, sep_w)
    if (corr_w + sep_w) <= 1e-8:
        corr_w, sep_w = 0.45, 0.55
    w_sum = corr_w + sep_w
    corr_w /= w_sum
    sep_w /= w_sum

    subset_ids_raw = _string_list(payload.get('subset_ids'))
    subset_ids = set(subset_ids_raw) if subset_ids_raw else None
    field_filter = _string_list(payload.get('metadata_fields'))

    engine = ImageGalleryEngine(str(dataset_root))
    entries = engine.list_images()
    valid_ids = {str(e.id) for e in entries}
    meta_summary = _summarize_metadata_fields(entries=entries, dataset_root=dataset_root)
    field_maps = meta_summary.get('field_maps') or {}
    field_types = meta_summary.get('field_types') or {}

    recommendations: List[Dict[str, Any]] = []
    candidate_pairs = 0

    for i in range(len(axes)):
        ax = axes[i]
        x_coords = ax.get('coords') or {}
        if len(x_coords) < 2:
            continue
        for j in range(i + 1, len(axes)):
            ay = axes[j]
            y_coords = ay.get('coords') or {}
            if len(y_coords) < 2:
                continue
            candidate_pairs += 1

            common_ids = [image_id for image_id in x_coords.keys() if image_id in y_coords]
            common_ids = [image_id for image_id in common_ids if image_id in valid_ids]
            if subset_ids is not None:
                common_ids = [image_id for image_id in common_ids if image_id in subset_ids]
            if len(common_ids) < min_overlap:
                continue

            x = np.asarray([float(x_coords[image_id]) for image_id in common_ids], dtype=np.float32)
            y = np.asarray([float(y_coords[image_id]) for image_id in common_ids], dtype=np.float32)
            corr = _pearson_corr(x, y)
            corr_score = float(max(0.0, 1.0 - min(1.0, abs(corr))))

            sep_score, sep_breakdown = _compute_pair_separability(
                ids=common_ids,
                x=x,
                y=y,
                field_maps=field_maps,
                field_types=field_types,
                field_filter=field_filter,
                k_neighbors=k_neighbors,
            )
            final_score = float((sep_w * sep_score) + (corr_w * corr_score))
            best_meta = sep_breakdown[0] if len(sep_breakdown) > 0 else None
            recommendations.append({
                'x_axis_id': ax.get('id'),
                'x_axis_name': ax.get('name') or ax.get('id'),
                'y_axis_id': ay.get('id'),
                'y_axis_name': ay.get('name') or ay.get('id'),
                'score': final_score,
                'correlation': float(corr),
                'correlation_score': corr_score,
                'separability_score': float(sep_score),
                'n_points': int(len(common_ids)),
                'best_metadata_field': best_meta.get('field') if best_meta else None,
                'best_metadata_field_type': best_meta.get('field_type') if best_meta else None,
                'best_metadata_field_score': float(best_meta.get('score')) if best_meta else None,
                'separability_breakdown': sep_breakdown[:6],
            })

    recommendations.sort(key=lambda r: float(r.get('score') or 0.0), reverse=True)
    recommendations = recommendations[:top_k]

    return jsonify({
        'dataset': str(dataset_root),
        'dataset_name': dataset_root.name,
        'weights': {
            'correlation': float(corr_w),
            'separability': float(sep_w),
        },
        'metadata_field_count': int(len(meta_summary.get('fields') or [])),
        'metadata_fields': meta_summary.get('fields') or [],
        'candidate_pairs': int(candidate_pairs),
        'recommended_pairs': int(len(recommendations)),
        'recommendations': recommendations,
    })


@app.post('/artifacts/export')
def export_artifacts():
    payload = request.get_json(silent=True) or {}
    dataset_root = _resolve_dataset_from_payload(payload)

    engine = ImageGalleryEngine(str(dataset_root))
    entries = engine.list_images()
    valid_ids = {str(e.id) for e in entries}
    all_ids = [str(e.id) for e in entries]

    subset_ids_raw = _string_list(payload.get('subset_ids'))
    if len(subset_ids_raw) > 0:
        subset_ids = [image_id for image_id in subset_ids_raw if image_id in valid_ids]
    else:
        subset_ids = list(all_ids)
    if len(subset_ids) == 0:
        subset_ids = list(all_ids)

    axes = _sanitize_axes_for_export(payload.get('axes'), valid_ids=valid_ids)
    selected_x = str(payload.get('selected_x') or payload.get('selectedX') or '').strip() or None
    selected_y = str(payload.get('selected_y') or payload.get('selectedY') or '').strip() or None
    prompt = str(payload.get('prompt') or '').strip()
    notes = str(payload.get('notes') or '').strip()
    artifact_name = str(payload.get('name') or payload.get('artifact_name') or '').strip()

    now = datetime.now(timezone.utc)
    timestamp = now.strftime('%Y%m%dT%H%M%SZ')
    base_name = slugify(artifact_name or f'{dataset_root.name}-scatterplot-artifact')
    filename = f'{base_name}-{timestamp}.json'
    out_dir = dataset_root / 'artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    meta_summary = _summarize_metadata_fields(entries=entries, dataset_root=dataset_root)
    metadata_fields = meta_summary.get('fields') or []

    artifact: Dict[str, Any] = {
        'schema_version': '1.0',
        'created_at': now.isoformat(),
        'dataset': {
            'name': dataset_root.name,
            'path': str(dataset_root),
            'image_count': int(len(entries)),
        },
        'prompt': prompt or None,
        'notes': notes or None,
        'selection': {
            'subset_size': int(len(subset_ids)),
            'subset_ids': subset_ids,
        },
        'view': {
            'selected_x': selected_x,
            'selected_y': selected_y,
        },
        'axes': axes,
        'metadata': {
            'path': str(dataset_root / 'metadata.csv'),
            'fields': metadata_fields,
        },
    }

    recommendations = payload.get('recommendations')
    if isinstance(recommendations, list) and len(recommendations) > 0:
        artifact['recommendations'] = recommendations[:20]
    if isinstance(payload.get('slice'), dict):
        artifact['slice'] = payload.get('slice')

    with out_path.open('w', encoding='utf-8') as f:
        json.dump(artifact, f, indent=2)

    return jsonify({
        'ok': True,
        'dataset': str(dataset_root),
        'dataset_name': dataset_root.name,
        'filename': filename,
        'artifact_path': str(out_path),
        'subset_size': int(len(subset_ids)),
        'axis_count': int(len(axes)),
        'artifact': artifact,
    })


@app.post('/axis/create')
def axis_create():
    payload = request.get_json(silent=True) or {}
    dataset_root = _resolve_dataset_from_payload(payload)
    q = str(payload.get('q') or payload.get('attribute') or '').strip()
    mode = str(payload.get('mode') or '').strip() or None
    model_type = str(payload.get('model_type') or payload.get('modelType') or '').strip() or None
    requested_semantic_method = _requested_axis_semantic_method(payload)
    requested_norm = _requested_axis_norm(payload)
    collection_id = str(payload.get('collection_id') or payload.get('dataset') or dataset_root.name).strip()
    if not q:
        abort(400, description='Missing q')

    try:
        engine = AXIS_BAYES_ENGINE
        if requested_semantic_method is not None or requested_norm is not None or mode is not None or model_type is not None:
            engine = _make_axis_engine(
                model_type=model_type or AXIS_MODEL_TYPE,
                mode=mode or AXIS_BAYES_MODE,
                semantic_method=requested_semantic_method or AXIS_BAYES_SEMANTIC_METHOD,
                norm=AXIS_BAYES_NORM if requested_norm is None else requested_norm,
            )
        result = engine.create_axis(
            collection_id=collection_id,
            dataset_root=str(dataset_root),
            q=q,
            mode=mode,
            model_type=model_type,
        )
        axis_id = str(result.get('axis_id') or '').strip()
        state = getattr(engine, '_axes', {}).get(axis_id)
        if state is not None:
            AXIS_BAYES_ENGINE._axes[axis_id] = state
    except Exception as e:
        abort(500, description=f'Axis creation failed: {e}')
    return jsonify(result)


@app.post('/axis/move')
def axis_move():
    payload = request.get_json(silent=True) or {}
    axis_id = str(payload.get('axis_id') or payload.get('axisId') or '').strip()
    image_id = str(payload.get('image_id') or payload.get('imageId') or '').strip()
    move_type = str(payload.get('move_type') or payload.get('moveType') or 'score').strip().lower() or 'score'
    raw_score = payload.get('new_score_0_100', payload.get('newScore0To100'))
    if not axis_id:
        abort(400, description='Missing axis_id')
    if not image_id:
        abort(400, description='Missing image_id')
    if move_type in {'undefined', 'exclude', 'delete', 'remove'}:
        new_score_0_100 = 0.0
    else:
        try:
            new_score_0_100 = float(raw_score)
        except Exception:
            abort(400, description='Invalid new_score_0_100')

    try:
        result = AXIS_BAYES_ENGINE.move_axis(
            axis_id=axis_id,
            image_id=image_id,
            new_score_0_100=new_score_0_100,
            move_type=move_type,
        )
    except KeyError as e:
        abort(404, description=str(e))
    except ValueError as e:
        abort(400, description=str(e))
    except Exception as e:
        abort(500, description=f'Axis update failed: {e}')
    return jsonify(result)


@app.post('/axis/update_prompts')
def axis_update_prompts():
    payload = request.get_json(silent=True) or {}
    axis_id = str(payload.get('axis_id') or payload.get('axisId') or '').strip()
    pos_prompts = payload.get('pos_prompts', payload.get('posPrompts'))
    neg_prompts = payload.get('neg_prompts', payload.get('negPrompts'))
    if not axis_id:
        abort(400, description='Missing axis_id')
    if not isinstance(pos_prompts, list) or not isinstance(neg_prompts, list):
        abort(400, description='pos_prompts and neg_prompts must be arrays')

    try:
        result = AXIS_BAYES_ENGINE.update_axis_prompts(
            axis_id=axis_id,
            pos_prompts=pos_prompts,
            neg_prompts=neg_prompts,
        )
    except KeyError as e:
        abort(404, description=str(e))
    except ValueError as e:
        abort(400, description=str(e))
    except Exception as e:
        abort(500, description=f'Axis prompt update failed: {e}')
    return jsonify(result)


@app.get('/sessions')
def sessions_list():
    _migrate_legacy_axes_to_default_session()
    items = []
    for session_name in iter_session_names(SESSIONS_ROOT, include_default=True):
        axes_count = len(_axis_library_read(session_name=session_name, include_artifact=False))
        visualizations_count = len(_visualization_library_read(session_name=session_name, include_artifact=False))
        items.append(session_summary(
            SESSIONS_ROOT,
            session_name,
            axes_count=axes_count,
            visualizations_count=visualizations_count,
        ))
    return jsonify({'items': items, 'default': DEFAULT_SESSION_NAME})


@app.post('/sessions/open')
def sessions_open():
    payload = request.get_json(silent=True) or {}
    requested = payload.get('session') or payload.get('name') or payload.get('value')
    session_name = _normalize_session_name(requested)
    axes_count = len(_axis_library_read(session_name=session_name, include_artifact=False))
    visualizations_count = len(_visualization_library_read(session_name=session_name, include_artifact=False))
    summary = session_summary(
        SESSIONS_ROOT,
        session_name,
        axes_count=axes_count,
        visualizations_count=visualizations_count,
    )
    _append_session_log(session_name, 'enter session', session_name)
    return jsonify({'ok': True, 'session': session_name, 'item': summary})


@app.post('/session/log')
def session_log():
    payload = request.get_json(silent=True) or {}
    session_name = _resolve_session_from_payload(payload)
    action = payload.get('action')
    detail = payload.get('detail')
    if not str(action or '').strip():
        abort(400, description='Missing action')
    path = _append_session_log(session_name, action, detail)
    return jsonify({
        'ok': True,
        'session': session_name,
        'path': str(path),
    })


@app.get('/axis/library')
def axis_library_list():
    session_name = _resolve_session_from_payload(None)
    items = _axis_library_read(session_name=session_name, include_artifact=False)
    return jsonify({'session': session_name, 'items': items})


@app.post('/axis/library/save')
def axis_library_save():
    payload = request.get_json(silent=True) or {}
    session_name = _resolve_session_from_payload(payload)
    dataset_root = _resolve_dataset_from_payload(payload)
    axis_id = str(payload.get('axis_id') or payload.get('axisId') or '').strip()
    name = str(payload.get('name') or payload.get('axis_name') or '').strip()
    q = str(payload.get('q') or payload.get('query') or '').strip()
    mode = str(payload.get('mode') or '').strip()
    model_type = str(payload.get('model_type') or payload.get('modelType') or '').strip()
    origin_dataset = str(
        payload.get('origin_dataset')
        or payload.get('dataset')
        or payload.get('collection_id')
        or dataset_root.name
    ).strip() or dataset_root.name
    if not q:
        abort(400, description='Missing q')
    if not name:
        name = q

    serialized_axis = None
    if axis_id:
        try:
            serialized_axis = AXIS_BAYES_ENGINE.serialize_portable_axis(axis_id)
        except KeyError as e:
            abort(404, description=str(e))
        except Exception as e:
            abort(500, description=f'Axis serialization failed: {e}')
        if not mode:
            mode = str(serialized_axis.get('mode') or '').strip()
        if not model_type:
            model_type = str(serialized_axis.get('model_type') or '').strip()
        if not q:
            q = str(serialized_axis.get('q') or '').strip() or q

    items = _axis_library_read(session_name=session_name, include_artifact=True)
    key_name = name.lower()
    key_q = q.lower()
    key_origin = origin_dataset.lower()
    existing = next(
        (
            it for it in items
            if str(it.get('name') or '').strip().lower() == key_name
            and str(it.get('q') or '').strip().lower() == key_q
            and str(it.get('origin_dataset') or '').strip().lower() == key_origin
        ),
        None,
    )
    if existing:
        if serialized_axis is not None:
            existing['serialized_axis'] = serialized_axis
            existing['mode'] = mode
            existing['model_type'] = model_type
            existing['updated_at'] = datetime.now(timezone.utc).isoformat()
            _axis_library_write(session_name, items)
        return jsonify({
            'ok': True,
            'session': session_name,
            'item': _axis_library_item(existing),
            'items': _axis_library_read(session_name=session_name),
        })

    item = {
        'id': f'axlib:{uuid.uuid4().hex[:12]}',
        'name': name,
        'q': q,
        'origin_dataset': origin_dataset,
        'mode': mode,
        'model_type': model_type,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'serialized_axis': serialized_axis,
    }
    items.append(item)
    _axis_library_write(session_name, items)
    return jsonify({
        'ok': True,
        'session': session_name,
        'item': _axis_library_item(item),
        'items': _axis_library_read(session_name=session_name),
    })


@app.delete('/axis/library/<path:item_id>')
def axis_library_delete(item_id: str):
    session_name = _resolve_session_from_payload(None)
    axis_lib_id = str(item_id or '').strip()
    if not axis_lib_id:
        abort(400, description='Missing item id')
    items = _axis_library_read(session_name=session_name, include_artifact=True)
    next_items = [it for it in items if str(it.get('id') or '').strip() != axis_lib_id]
    if len(next_items) == len(items):
        abort(404, description='Axis library item not found')
    _axis_library_write(session_name, next_items)
    return jsonify({'ok': True, 'session': session_name, 'items': _axis_library_read(session_name=session_name)})


@app.post('/axis/library/project')
def axis_library_project():
    payload = request.get_json(silent=True) or {}
    session_name = _resolve_session_from_payload(payload)
    axis_lib_id = str(payload.get('library_axis_id') or payload.get('id') or '').strip()
    if not axis_lib_id:
        abort(400, description='Missing library_axis_id')

    items = _axis_library_read(session_name=session_name, include_artifact=True)
    item = next((it for it in items if str(it.get('id') or '').strip() == axis_lib_id), None)
    if not item:
        abort(404, description='Axis library item not found')

    dataset_root = _resolve_dataset_from_payload(payload)
    collection_id = str(payload.get('collection_id') or payload.get('dataset') or dataset_root.name).strip()
    q = str(item.get('q') or '').strip()
    if not q:
        abort(400, description='Saved axis has no query')

    mode = str(item.get('mode') or '').strip() or None
    model_type = str(item.get('model_type') or '').strip() or None
    try:
        serialized_axis = item.get('serialized_axis') if isinstance(item.get('serialized_axis'), dict) else None
        result = _project_saved_axis_payload(
            serialized_axis=serialized_axis,
            dataset_root=dataset_root,
            collection_id=collection_id,
            q=q,
            axis_name=str(item.get('name') or q),
            mode=mode,
            model_type=model_type,
        )
    except Exception as e:
        abort(500, description=f'Saved axis projection failed: {e}')

    axis_name = str(item.get('name') or q).strip() or q
    axis_id = str(result.get('axis_id') or result.get('axis', {}).get('id') or '').strip()
    axis_obj = result.get('axis')
    if isinstance(axis_obj, dict):
        axis_obj['name'] = axis_name

    try:
        state = getattr(AXIS_BAYES_ENGINE, '_axes', {}).get(axis_id)
        if state is not None:
            state.axis_name = axis_name
    except Exception:
        pass

    result['library_axis'] = {
        'id': str(item.get('id') or ''),
        'name': axis_name,
        'origin_dataset': str(item.get('origin_dataset') or ''),
        'session': session_name,
    }
    return jsonify(result)


@app.get('/visualization/library')
def visualization_library_list():
    session_name = _resolve_session_from_payload(None)
    items = _visualization_library_read(session_name=session_name, include_artifact=False)
    return jsonify({'session': session_name, 'items': items})


@app.post('/visualization/library/save')
def visualization_library_save():
    payload = request.get_json(silent=True) or {}
    session_name = _resolve_session_from_payload(payload)
    dataset_root = _resolve_dataset_from_payload(payload)
    dataset_name = str(payload.get('dataset') or dataset_root.name).strip() or dataset_root.name
    name = str(payload.get('name') or '').strip()
    if not name:
        abort(400, description='Missing visualization name')

    axis_ids = [str(v).strip() for v in (payload.get('axis_ids') or []) if str(v).strip()]
    axes_manifest = payload.get('axes_manifest') if isinstance(payload.get('axes_manifest'), list) else []
    manifest_by_id = {
        str(entry.get('id') or '').strip(): entry
        for entry in axes_manifest
        if isinstance(entry, dict) and str(entry.get('id') or '').strip()
    }
    custom_axes = []
    for axis_id in axis_ids:
        try:
            serialized_axis = AXIS_BAYES_ENGINE.serialize_portable_axis(axis_id)
        except Exception:
            continue
        manifest = manifest_by_id.get(axis_id) or {}
        custom_axes.append({
            'source_axis_id': axis_id,
            'name': str(manifest.get('name') or serialized_axis.get('axis_name') or serialized_axis.get('q') or axis_id),
            'group': str(manifest.get('group') or '').strip(),
            'serialized_axis': serialized_axis,
        })

    histogram_slices = payload.get('histogram_slices') if isinstance(payload.get('histogram_slices'), list) else []
    subset_chips = payload.get('subset_chips') if isinstance(payload.get('subset_chips'), list) else []
    subset_filters = payload.get('subset_filters') if isinstance(payload.get('subset_filters'), list) else []
    subset_filter = payload.get('subset_filter') if isinstance(payload.get('subset_filter'), dict) else None
    view_state = payload.get('view_state') if isinstance(payload.get('view_state'), dict) else {}
    minimap_size_offset = float(payload.get('minimap_size_offset') or 0.0)
    selected_x = str(payload.get('selected_x') or '').strip()
    selected_y = str(payload.get('selected_y') or '').strip()
    selected_x_name = str(payload.get('selected_x_name') or '').strip()
    selected_y_name = str(payload.get('selected_y_name') or '').strip()

    items = _visualization_library_read(session_name=session_name, include_artifact=True)
    existing = next(
        (
            it for it in items
            if str(it.get('name') or '').strip().lower() == name.lower()
            and str(it.get('dataset') or '').strip().lower() == dataset_name.lower()
        ),
        None,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    if existing:
        existing.update({
            'dataset': dataset_name,
            'selected_x': selected_x,
            'selected_y': selected_y,
            'selected_x_name': selected_x_name,
            'selected_y_name': selected_y_name,
            'histogram_slices': histogram_slices,
            'subset_chips': subset_chips,
            'subset_filters': subset_filters,
            'subset_filter': subset_filter,
            'view_state': view_state,
            'minimap_size_offset': minimap_size_offset,
            'custom_axes': custom_axes,
            'axes_manifest': axes_manifest,
            'updated_at': now_iso,
        })
        _visualization_library_write(session_name, items)
        return jsonify({
            'ok': True,
            'session': session_name,
            'item': _visualization_library_item(existing),
            'items': _visualization_library_read(session_name=session_name),
        })

    item = {
        'id': f'vizlib:{uuid.uuid4().hex[:12]}',
        'name': name,
        'dataset': dataset_name,
        'selected_x': selected_x,
        'selected_y': selected_y,
        'selected_x_name': selected_x_name,
        'selected_y_name': selected_y_name,
        'histogram_slices': histogram_slices,
        'subset_chips': subset_chips,
        'subset_filters': subset_filters,
        'subset_filter': subset_filter,
        'view_state': view_state,
        'minimap_size_offset': minimap_size_offset,
        'custom_axes': custom_axes,
        'axes_manifest': axes_manifest,
        'created_at': now_iso,
        'updated_at': now_iso,
    }
    items.append(item)
    _visualization_library_write(session_name, items)
    return jsonify({
        'ok': True,
        'session': session_name,
        'item': _visualization_library_item(item),
        'items': _visualization_library_read(session_name=session_name),
    })


@app.delete('/visualization/library/<path:item_id>')
def visualization_library_delete(item_id: str):
    session_name = _resolve_session_from_payload(None)
    viz_id = str(item_id or '').strip()
    if not viz_id:
        abort(400, description='Missing item id')
    items = _visualization_library_read(session_name=session_name, include_artifact=True)
    next_items = [it for it in items if str(it.get('id') or '').strip() != viz_id]
    if len(next_items) == len(items):
        abort(404, description='Visualization item not found')
    _visualization_library_write(session_name, next_items)
    return jsonify({'ok': True, 'session': session_name, 'items': _visualization_library_read(session_name=session_name)})


@app.post('/visualization/library/project')
def visualization_library_project():
    payload = request.get_json(silent=True) or {}
    session_name = _resolve_session_from_payload(payload)
    visualization_id = str(payload.get('visualization_id') or payload.get('id') or '').strip()
    if not visualization_id:
        abort(400, description='Missing visualization_id')

    items = _visualization_library_read(session_name=session_name, include_artifact=True)
    item = next((it for it in items if str(it.get('id') or '').strip() == visualization_id), None)
    if not item:
        abort(404, description='Visualization item not found')

    dataset_name = str(payload.get('dataset') or item.get('dataset') or '').strip()
    dataset_root = resolve_dataset_root(dataset_name)
    collection_id = str(payload.get('collection_id') or dataset_root.name).strip() or dataset_root.name

    projected_axes: List[Dict[str, Any]] = []
    axis_id_map: Dict[str, str] = {}
    for raw_axis in item.get('custom_axes') if isinstance(item.get('custom_axes'), list) else []:
        if not isinstance(raw_axis, dict):
            continue
        serialized_axis = raw_axis.get('serialized_axis') if isinstance(raw_axis.get('serialized_axis'), dict) else None
        source_axis_id = str(raw_axis.get('source_axis_id') or '').strip()
        axis_name = str(
            raw_axis.get('name')
            or (serialized_axis or {}).get('axis_name')
            or (serialized_axis or {}).get('q')
            or source_axis_id
            or 'Axis'
        ).strip()
        q = str((serialized_axis or {}).get('q') or axis_name).strip() or axis_name
        mode = str((serialized_axis or {}).get('mode') or '').strip() or None
        model_type = str((serialized_axis or {}).get('model_type') or '').strip() or None
        try:
            projected = _project_saved_axis_payload(
                serialized_axis=serialized_axis,
                dataset_root=dataset_root,
                collection_id=collection_id,
                q=q,
                axis_name=axis_name,
                mode=mode,
                model_type=model_type,
            )
        except Exception as e:
            abort(500, description=f'Visualization projection failed for axis {axis_name}: {e}')
        axis_obj = projected.get('axis')
        axis_id = str(projected.get('axis_id') or (axis_obj or {}).get('id') or '').strip()
        if source_axis_id and axis_id:
            axis_id_map[source_axis_id] = axis_id
        projected_axes.append(projected)

    visualization = {
        'id': str(item.get('id') or ''),
        'name': str(item.get('name') or ''),
        'dataset': dataset_root.name,
        'selected_x': str(item.get('selected_x') or ''),
        'selected_y': str(item.get('selected_y') or ''),
        'selected_x_name': str(item.get('selected_x_name') or ''),
        'selected_y_name': str(item.get('selected_y_name') or ''),
        'histogram_slices': item.get('histogram_slices') if isinstance(item.get('histogram_slices'), list) else [],
        'subset_chips': item.get('subset_chips') if isinstance(item.get('subset_chips'), list) else [],
        'subset_filters': item.get('subset_filters') if isinstance(item.get('subset_filters'), list) else [],
        'subset_filter': item.get('subset_filter') if isinstance(item.get('subset_filter'), dict) else None,
        'view_state': item.get('view_state') if isinstance(item.get('view_state'), dict) else {},
        'minimap_size_offset': float(item.get('minimap_size_offset') or 0.0),
        'axes_manifest': item.get('axes_manifest') if isinstance(item.get('axes_manifest'), list) else [],
        'session': session_name,
    }
    return jsonify({
        'ok': True,
        'session': session_name,
        'visualization': visualization,
        'projected_axes': projected_axes,
        'axis_id_map': axis_id_map,
    })


@app.get('/subset/library')
def subset_library_list():
    session_name = _resolve_session_from_payload(None)
    items = _subset_library_read(session_name=session_name, include_artifact=False)
    return jsonify({'session': session_name, 'items': items})


@app.post('/subset/library/save')
def subset_library_save():
    payload = request.get_json(silent=True) or {}
    session_name = _resolve_session_from_payload(payload)
    dataset_root = _resolve_dataset_from_payload(payload)
    dataset_name = str(payload.get('dataset') or dataset_root.name).strip() or dataset_root.name
    name = str(payload.get('name') or '').strip()
    if not name:
        abort(400, description='Missing subset name')

    subset_chips = payload.get('subset_chips') if isinstance(payload.get('subset_chips'), list) else []
    image_count = int(payload.get('image_count') or 0)
    if len(subset_chips) == 0:
        abort(400, description='Missing subset_chips')

    items = _subset_library_read(session_name=session_name, include_artifact=True)
    existing = next(
        (
            it for it in items
            if str(it.get('name') or '').strip().lower() == name.lower()
            and str(it.get('dataset') or '').strip().lower() == dataset_name.lower()
        ),
        None,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    if existing:
        existing.update({
            'dataset': dataset_name,
            'subset_chips': subset_chips,
            'image_count': image_count,
            'updated_at': now_iso,
        })
        _subset_library_write(session_name, items)
        return jsonify({
            'ok': True,
            'session': session_name,
            'item': _subset_library_item(existing),
            'items': _subset_library_read(session_name=session_name),
        })

    item = {
        'id': f'subsetlib:{uuid.uuid4().hex[:12]}',
        'name': name,
        'dataset': dataset_name,
        'subset_chips': subset_chips,
        'image_count': image_count,
        'created_at': now_iso,
        'updated_at': now_iso,
    }
    items.append(item)
    _subset_library_write(session_name, items)
    return jsonify({
        'ok': True,
        'session': session_name,
        'item': _subset_library_item(item),
        'items': _subset_library_read(session_name=session_name),
    })


@app.delete('/subset/library/<path:item_id>')
def subset_library_delete(item_id: str):
    session_name = _resolve_session_from_payload(None)
    subset_id = str(item_id or '').strip()
    if not subset_id:
        abort(400, description='Missing item id')
    items = _subset_library_read(session_name=session_name, include_artifact=True)
    next_items = [it for it in items if str(it.get('id') or '').strip() != subset_id]
    if len(next_items) == len(items):
        abort(404, description='Subset library item not found')
    _subset_library_write(session_name, next_items)
    return jsonify({'ok': True, 'session': session_name, 'items': _subset_library_read(session_name=session_name)})


@app.post('/subset/library/project')
def subset_library_project():
    payload = request.get_json(silent=True) or {}
    session_name = _resolve_session_from_payload(payload)
    subset_id = str(payload.get('subset_id') or payload.get('id') or '').strip()
    if not subset_id:
        abort(400, description='Missing subset_id')

    items = _subset_library_read(session_name=session_name, include_artifact=True)
    item = next((it for it in items if str(it.get('id') or '').strip() == subset_id), None)
    if not item:
        abort(404, description='Subset library item not found')

    subset = {
        'id': str(item.get('id') or ''),
        'name': str(item.get('name') or ''),
        'dataset': str(item.get('dataset') or ''),
        'subset_chips': item.get('subset_chips') if isinstance(item.get('subset_chips'), list) else [],
        'image_count': int(item.get('image_count') or 0),
        'session': session_name,
    }
    return jsonify({
        'ok': True,
        'session': session_name,
        'subset': subset,
    })


@app.post('/llm/extract_attributes')
def llm_extract_attributes():
    """Extract measurable attributes from a free-form visualization prompt."""
    req_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get('prompt') or '').strip()
    _llm_api_log(req_id, f'extract_attributes start prompt_len={len(prompt)}')
    if not prompt:
        _llm_api_log(req_id, 'extract_attributes error missing prompt')
        abort(400, description='Missing prompt')

    max_attributes = payload.get('max_attributes', DEFAULT_MAX_ATTRIBUTES)
    try:
        max_attributes = int(max_attributes)
    except Exception:
        max_attributes = DEFAULT_MAX_ATTRIBUTES
    if max_attributes < 1:
        max_attributes = DEFAULT_MAX_ATTRIBUTES

    dataset_root = _resolve_dataset_from_payload(payload)
    _llm_api_log(
        req_id,
        f'extract_attributes request dataset={dataset_root.name} max_attributes={max_attributes}',
    )

    result = LLM_ENGINE.extract_attributes(
        prompt=prompt,
        max_attributes=max_attributes,
        dataset_name=dataset_root.name,
    )
    attrs = result.get('attributes') or []
    if len(attrs) == 0:
        _llm_api_log(
            req_id,
            f"extract_attributes failed provider={result.get('provider') or 'unknown'} error={result.get('error') or 'no attributes returned'}",
        )
        abort(500, description=f"LLM attribute extraction failed: {result.get('error') or 'no attributes returned'}")
    _llm_api_log(
        req_id,
        f"extract_attributes done provider={result.get('provider') or 'unknown'} attributes={len(attrs)} elapsed={time.time()-t0:.2f}s",
    )
    attribute_names = result.get('attribute_names')
    if not isinstance(attribute_names, list) or len(attribute_names) == 0:
        attribute_names = []
        for a in attrs:
            if isinstance(a, dict):
                name = str(a.get('name') or '').strip()
                if name:
                    attribute_names.append(name)
            else:
                s = str(a or '').strip()
                if s:
                    attribute_names.append(s)
    return jsonify({
        'prompt': prompt,
        'attributes': attrs,
        'attribute_names': attribute_names,
        'provider': result.get('provider') or 'unknown',
        'support_provider': result.get('support_provider') or 'unknown',
    })


@app.post('/llm/suggest_values')
def llm_suggest_values():
    """Suggest ordered values (low -> high) for one attribute."""
    req_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    payload = request.get_json(silent=True) or {}
    attribute = str(payload.get('attribute') or '').strip()
    _llm_api_log(req_id, f'suggest_values start attribute="{attribute}"')
    if not attribute:
        _llm_api_log(req_id, 'suggest_values error missing attribute')
        abort(400, description='Missing attribute')

    n_values = payload.get('n_values', DEFAULT_VALUE_COUNT)
    try:
        n_values = int(n_values)
    except Exception:
        n_values = DEFAULT_VALUE_COUNT
    if n_values < 2:
        n_values = DEFAULT_VALUE_COUNT

    provided_values = _string_list(payload.get('values'))
    _llm_api_log(
        req_id,
        f'suggest_values request n_values={n_values} provided_values={len(provided_values)}',
    )
    result = LLM_ENGINE.suggest_values(
        attribute=attribute,
        n_values=n_values,
        provided_values=provided_values if provided_values else None,
    )
    values = result.get('values') or []
    if len(values) < 2:
        _llm_api_log(
            req_id,
            f"suggest_values failed provider={result.get('provider') or 'unknown'} error={result.get('error') or 'no ordered values returned'}",
        )
        abort(500, description=f"LLM value suggestion failed: {result.get('error') or 'no ordered values returned'}")
    _llm_api_log(
        req_id,
        f"suggest_values done provider={result.get('provider') or 'unknown'} values={len(values)} elapsed={time.time()-t0:.2f}s",
    )
    return jsonify({
        'attribute': attribute,
        'values': values,
        'provider': result.get('provider') or 'unknown',
    })


@app.post('/llm/attribute_distribution')
def llm_attribute_distribution():
    """Score all images in [0,1] for an attribute, with type-dependent LLM+VLM logic."""
    req_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    payload = request.get_json(silent=True) or {}
    attribute = str(payload.get('attribute') or '').strip()
    _llm_api_log(req_id, f'attribute_distribution start attribute="{attribute}"')
    if not attribute:
        _llm_api_log(req_id, 'attribute_distribution error missing attribute')
        abort(400, description='Missing attribute')

    n_values = payload.get('n_values', DEFAULT_VALUE_COUNT)
    try:
        n_values = int(n_values)
    except Exception:
        n_values = DEFAULT_VALUE_COUNT
    if n_values < 2:
        n_values = DEFAULT_VALUE_COUNT

    histogram_bins = payload.get('histogram_bins', ZERO_SHOT_HISTOGRAM_BINS)
    try:
        histogram_bins = int(histogram_bins)
    except Exception:
        histogram_bins = ZERO_SHOT_HISTOGRAM_BINS
    if histogram_bins < 2:
        histogram_bins = ZERO_SHOT_HISTOGRAM_BINS

    include_probabilities = bool(payload.get('include_probabilities', False))
    provided_values = _string_list(payload.get('values'))
    if len(provided_values) == 0:
        provided_values = _string_list(payload.get('categories'))

    attribute_type = _normalize_attribute_type(payload.get('attribute_type'))
    if not attribute_type:
        _llm_api_log(req_id, 'attribute_distribution error missing attribute_type')
        abort(400, description='Missing attribute_type')
    attribute_type_provider = 'request'

    anchors_payload = payload.get('anchors') if isinstance(payload.get('anchors'), dict) else {}
    provided_low = str(
        payload.get('low_anchor')
        or payload.get('anchor_low')
        or anchors_payload.get('low')
        or anchors_payload.get('min')
        or ''
    ).strip()
    provided_high = str(
        payload.get('high_anchor')
        or payload.get('anchor_high')
        or anchors_payload.get('high')
        or anchors_payload.get('max')
        or ''
    ).strip()
    _llm_api_log(
        req_id,
        (
            f'attribute_distribution request n_values={n_values} histogram_bins={histogram_bins} '
            f'include_probabilities={include_probabilities} provided_values={len(provided_values)} '
            f'attribute_type={attribute_type}'
        ),
    )

    dataset_root = _resolve_dataset_from_payload(payload)
    entries, image_embs, semantic_method = _load_semantic_embeddings(
        dataset_root,
        preferred_method=DEFAULT_SEMANTIC_EMBED_METHOD,
    )
    regressor = ZeroShotAttributeRegressor(text_method=semantic_method)
    _llm_api_log(
        req_id,
        (
            f'attribute_distribution embeddings loaded dataset={dataset_root.name} '
            f'images={len(entries)} dim={int(image_embs.shape[1])} method={semantic_method}'
        ),
    )
    ids = [e.id for e in entries]

    suggestion_provider = 'unknown'
    reg: Dict[str, Any]
    values: List[str] = []
    anchors: Dict[str, str] | None = None
    if attribute_type in {'categorical', 'ordinal'}:
        suggested = LLM_ENGINE.suggest_categories(
            attribute=attribute,
            attribute_type=attribute_type,
            n_values=n_values,
            provided_categories=provided_values if provided_values else None,
        )
        suggestion_provider = suggested.get('provider') or 'unknown'
        values = suggested.get('categories') or []
        if len(values) < 2:
            _llm_api_log(
                req_id,
                (
                    f'attribute_distribution failed category suggestion provider={suggestion_provider} '
                    f'error={suggested.get("error") or "empty categories"}'
                ),
            )
            abort(500, description=f'Could not produce categories for this attribute: {suggested.get("error") or "empty categories"}')
        _llm_api_log(
            req_id,
            f'attribute_distribution categories ready provider={suggestion_provider} categories={values}',
        )
        try:
            _llm_api_log(req_id, 'attribute_distribution zero_shot_classification start')
            reg = regressor.score_discrete(
                image_embeddings=image_embs,
                attribute=attribute,
                categories=values,
                attribute_type=attribute_type,
                histogram_bins=histogram_bins,
                include_probabilities=include_probabilities,
            )
        except Exception as e:
            _llm_api_log(req_id, f'attribute_distribution zero_shot_classification failed error={e}')
            abort(500, description=f'Zero-shot classification failed: {e}')
    elif attribute_type == 'continuous':
        suggested = LLM_ENGINE.suggest_continuous_anchors(
            attribute=attribute,
            provided_low=provided_low or None,
            provided_high=provided_high or None,
        )
        suggestion_provider = suggested.get('provider') or 'unknown'
        low_anchor = str(suggested.get('low') or '').strip()
        high_anchor = str(suggested.get('high') or '').strip()
        if not low_anchor or not high_anchor:
            _llm_api_log(
                req_id,
                (
                    f'attribute_distribution failed anchor suggestion provider={suggestion_provider} '
                    f'error={suggested.get("error") or "missing anchors"}'
                ),
            )
            abort(500, description=f'Could not produce continuous anchors: {suggested.get("error") or "missing anchors"}')
        anchors = {'low': low_anchor, 'high': high_anchor}
        values = [low_anchor, high_anchor]
        _llm_api_log(
            req_id,
            f'attribute_distribution anchors ready provider={suggestion_provider} low="{low_anchor}" high="{high_anchor}"',
        )
        try:
            _llm_api_log(req_id, 'attribute_distribution clip_direction_projection start')
            reg = regressor.score_continuous_anchors(
                image_embeddings=image_embs,
                attribute=attribute,
                low_anchor=low_anchor,
                high_anchor=high_anchor,
                histogram_bins=histogram_bins,
            )
        except Exception as e:
            _llm_api_log(req_id, f'attribute_distribution clip_direction_projection failed error={e}')
            abort(500, description=f'Continuous CLIP projection failed: {e}')
    else:
        abort(500, description=f'Unsupported attribute type: {attribute_type}')

    scores = reg.get('scores') or []
    raw_scores = reg.get('scores_raw') or []
    coords = {ids[i]: float(scores[i]) for i in range(min(len(ids), len(scores)))}
    axis_id = f'axis:llm:{slugify(attribute)}'
    axis_name = re.sub(r'\s+', ' ', attribute).strip().title() or 'Attribute'

    resp: Dict[str, Any] = {
        'dataset': str(dataset_root),
        'attribute': attribute,
        'attribute_type': attribute_type,
        'attribute_type_provider': attribute_type_provider,
        'values': reg.get('values') or values,
        'value_positions': reg.get('value_positions') or [],
        'ids': ids,
        'scores': scores,
        'scores_raw': raw_scores,
        'predicted_values': reg.get('predicted_values') or [],
        'value_histogram': reg.get('value_histogram') or [],
        'score_histogram': reg.get('score_histogram') or {},
        'score_transform': reg.get('score_transform') or {},
        'scoring_method': reg.get('scoring_method') or 'unknown',
        'provider': suggestion_provider,
        'axis': {
            'id': axis_id,
            'name': axis_name,
            'coords': coords,
            'labels': reg.get('values') or values,
            'label_positions': reg.get('value_positions') or [],
            'group': 'llm',
        },
    }
    if anchors:
        resp['anchors'] = anchors
    if include_probabilities and 'value_probabilities' in reg:
        resp['value_probabilities'] = reg['value_probabilities']
    _llm_api_log(
        req_id,
        (
            f'attribute_distribution done type={attribute_type} scores={len(scores)} raw_scores={len(raw_scores)} '
            f'bins={len((reg.get("score_histogram") or {}).get("counts") or [])} '
            f'score_transform={((reg.get("score_transform") or {}).get("method") or "none")} '
            f'elapsed={time.time()-t0:.2f}s'
        ),
    )
    return jsonify(resp)


@app.post('/llm/score_attribute')
def llm_score_attribute_alias():
    # Backward-compatible alias for attribute_distribution.
    print('[llm-api] /llm/score_attribute alias -> /llm/attribute_distribution')
    return llm_attribute_distribution()


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
    resp = send_from_directory(directory, filename)
    try:
        resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    except Exception:
        pass
    return resp


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
                resp = send_file(buf, mimetype='image/jpeg')
                try:
                    resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
                except Exception:
                    pass
                return resp
        except Exception:
            return abort(500, description=f'Failed to generate thumbnail: {e}')

    resp = send_from_directory(str(cache_file.parent), cache_file.name)
    try:
        resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    except Exception:
        pass
    return resp


@app.post('/text_force')
def text_force():
    """Apply an attractive force from a text region to image coordinates.

    Body JSON:
    - text: string (required)
    - rect: { x, y, w, h } in normalized [0,1] (required)
    - embed: embedding method for text/image space, e.g., 'siglip2' (default 'siglip2')
    - alpha: float force strength (default 0.25)
    - method: 'pca', 'umap', or 'tsne' for 2D reduction (default is the backend gallery default)
    """
    dataset = app.config.get('DATASET_ROOT') or str(DATASET_PATH)
    if not dataset or not Path(dataset).exists():
        abort(400, description='Invalid dataset path')
    payload = request.get_json(silent=True) or {}
    text = payload.get('text', '').strip()
    rect = payload.get('rect') or {}
    embed_method = normalize_multimodal_method(payload.get('embed') or DEFAULT_SEMANTIC_EMBED_METHOD)
    try:
        red_method = normalize_reduction_method(payload.get('method', INITIAL_GALLERY_PROJECTION_METHOD))
    except ValueError as exc:
        abort(400, description=str(exc))
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

    # Use the selected semantic space when available; otherwise fall back to the default VLM.
    _, embs_for_sim, sim_method = _load_semantic_embeddings(Path(dataset), preferred_method=embed_method)
    emb_engine = EmbeddingEngine(dataset)
    tvec = emb_engine.text_embedding(sim_method, text)
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
    - embed: embedding method for similarity space (default 'siglip2')
    - alpha: float force scale (default 0.25)
    - method: dimensionality reduction method for fallback base coords (pca, umap, tsne)
    """
    dataset = app.config.get('DATASET_ROOT') or str(DATASET_PATH)
    if not dataset or not Path(dataset).exists():
        abort(400, description='Invalid dataset path')
    payload = request.get_json(silent=True) or {}
    texts = payload.get('texts') or []
    ids = payload.get('ids') or []
    base_coords = payload.get('base_coords')
    embed_method = normalize_multimodal_method(payload.get('embed') or DEFAULT_SEMANTIC_EMBED_METHOD)
    try:
        red_method = normalize_reduction_method(payload.get('method', INITIAL_GALLERY_PROJECTION_METHOD))
    except ValueError as exc:
        abort(400, description=str(exc))
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

    # Similarities in the selected semantic space, falling back to the default VLM.
    _, embs_for_sim, sim_method = _load_semantic_embeddings(Path(dataset), preferred_method=embed_method)
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
        tvec = emb_engine.text_embedding(sim_method, txt)
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
    app.run(host=BACKEND_HOST, port=int(BACKEND_PORT), debug=True)


if __name__ == '__main__':
    main()
