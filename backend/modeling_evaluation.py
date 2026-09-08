#!/usr/bin/env python3
from __future__ import annotations

"""Config-first modeling evaluation harness for semantic-axis methods.

This script is intended for the paper-side modeling study. It keeps the setup
editable in one place:
- datasets and metadata fields are declared in DATASET_REGISTRY
- methods are declared in METHOD_REGISTRY
- sections (prior, refinement, uncertainty, undefined, ablations) can be
  turned on/off with booleans below

Outputs are append-only CSV files written under REAXIS_OUTPUT_ROOT/modeling_eval.
The script logs rows incrementally so long runs still leave usable partial data.
"""

import csv
import zlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    from .axis_bayes import AxisBayesEngine, CollectionCache
    from .constants import (
        AXIS_BAYES_ALPHA,
        AXIS_BAYES_BIAS_ALPHA,
        AXIS_BAYES_CLIP_WEIGHT,
        AXIS_BAYES_DINO_ALPHA,
        AXIS_BAYES_DINO_WEIGHT,
        AXIS_BAYES_FEATURE_SPACE,
        AXIS_BAYES_GRAPH_JITTER,
        AXIS_BAYES_GRAPH_KNN_K,
        AXIS_BAYES_GRAPH_LAMBDA_PRIOR,
        AXIS_BAYES_GRAPH_LAMBDA_SMOOTH,
        AXIS_BAYES_MODE,
        AXIS_BAYES_RANK_ANCHOR_DELTA,
        AXIS_BAYES_RANK_ANCHOR_K,
        AXIS_BAYES_RANK_ETA,
        AXIS_BAYES_RANK_MAX_PAIRS,
        AXIS_BAYES_SIGMA2,
        AXIS_MODEL_TYPE,
        AXIS_RESIDUAL_ALPHA,
        AXIS_RESIDUAL_BETA,
        AXIS_RESIDUAL_JITTER,
        AXIS_RESIDUAL_LAMBDA,
        AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER,
        AXIS_RESIDUAL_SIGMA_Y,
        AXIS_PIECEWISE_AGGREGATOR,
        AXIS_PIECEWISE_CLIP_SCALE,
        AXIS_PIECEWISE_DINO_SCALE,
        AXIS_PIECEWISE_EXPERT_DIVERSITY_STRENGTH,
        AXIS_PIECEWISE_L2_REG,
        AXIS_PIECEWISE_LEARNING_RATE,
        AXIS_PIECEWISE_MAX_REFINE_STEPS,
        AXIS_PIECEWISE_NUM_EXPERTS,
        AXIS_PIECEWISE_PAIRWISE_FROM_SCALAR_MARGIN,
        AXIS_PIECEWISE_PRIOR_STRENGTH,
        AXIS_PIECEWISE_USE_GATING,
    )
    from .gallery_backend import ImageGalleryEngine
    from .modeling_eval_prompt_cache import hydrate_prompt_cache
    from .runtime_config import DATASETS_ROOT, OUTPUT_ROOT
except ImportError:
    from axis_bayes import AxisBayesEngine, CollectionCache
    from constants import (
        AXIS_BAYES_ALPHA,
        AXIS_BAYES_BIAS_ALPHA,
        AXIS_BAYES_CLIP_WEIGHT,
        AXIS_BAYES_DINO_ALPHA,
        AXIS_BAYES_DINO_WEIGHT,
        AXIS_BAYES_FEATURE_SPACE,
        AXIS_BAYES_GRAPH_JITTER,
        AXIS_BAYES_GRAPH_KNN_K,
        AXIS_BAYES_GRAPH_LAMBDA_PRIOR,
        AXIS_BAYES_GRAPH_LAMBDA_SMOOTH,
        AXIS_BAYES_MODE,
        AXIS_BAYES_RANK_ANCHOR_DELTA,
        AXIS_BAYES_RANK_ANCHOR_K,
        AXIS_BAYES_RANK_ETA,
        AXIS_BAYES_RANK_MAX_PAIRS,
        AXIS_BAYES_SIGMA2,
        AXIS_MODEL_TYPE,
        AXIS_RESIDUAL_ALPHA,
        AXIS_RESIDUAL_BETA,
        AXIS_RESIDUAL_JITTER,
        AXIS_RESIDUAL_LAMBDA,
        AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER,
        AXIS_RESIDUAL_SIGMA_Y,
        AXIS_PIECEWISE_AGGREGATOR,
        AXIS_PIECEWISE_CLIP_SCALE,
        AXIS_PIECEWISE_DINO_SCALE,
        AXIS_PIECEWISE_EXPERT_DIVERSITY_STRENGTH,
        AXIS_PIECEWISE_L2_REG,
        AXIS_PIECEWISE_LEARNING_RATE,
        AXIS_PIECEWISE_MAX_REFINE_STEPS,
        AXIS_PIECEWISE_NUM_EXPERTS,
        AXIS_PIECEWISE_PAIRWISE_FROM_SCALAR_MARGIN,
        AXIS_PIECEWISE_PRIOR_STRENGTH,
        AXIS_PIECEWISE_USE_GATING,
    )
    from gallery_backend import ImageGalleryEngine
    from modeling_eval_prompt_cache import hydrate_prompt_cache
    from runtime_config import DATASETS_ROOT, OUTPUT_ROOT


# =============================
# Top-level run configuration
# =============================
OUTPUT_DIR = OUTPUT_ROOT / 'modeling_eval'
CACHE_DIR = OUTPUT_DIR / 'cache'

RUN_NAME = 'modeling_eval_v1'
RUN_NOTE = 'Edit DATASET_REGISTRY, ACTIVE_DATASETS, ACTIVE_METHODS, and budgets below to change the study.'

ACTIVE_DATASETS = [
    'EmoSet',
    'paintings_wikiart',
    'celeba_dataset',
    'chest_xray_pneumonia',
    'brain_mri_images',
    'HAM10000',
]

# Optional substring filter over task ids. Leave empty to run all tasks coming
# from the active datasets.
ACTIVE_TASK_SUBSTRINGS: List[str] = []

ACTIVE_METHODS = [
    'clip_text_similarity',
    'prompt_ensemble',
    'label_only_linear',
    'knn_label_propagation',
    'request_bayes_linear_gaussian',
    'request_bayes_linear_rank',
    'residual',
    'request_piecewise',
    'request_graph',
]

# Ablations are declared separately so they can be run without cluttering the
# main method list. Edit this list freely.
ACTIVE_ABLATIONS = [
    'ablate_no_text_prior',
]

RUN_E1_PRIOR = True
RUN_E2_REFINEMENT = True
RUN_E3_UNCERTAINTY = True
RUN_E4_UNDEFINED = True
RUN_ABLATIONS = True

REFINEMENT_BUDGETS = [0, 1, 3, 5, 10, 20]
MAIN_QUERY_POLICY = 'hybrid'
QUERY_POLICIES = ['random', 'diversity', 'uncertainty', 'hybrid']
REPRESENTATIVE_TASKS_PER_DATASET = 4
REPRESENTATIVE_IMAGES_PER_TASK = 20
REPRESENTATIVE_LOG_BUDGETS = [0, max(REFINEMENT_BUDGETS)]

# Global evaluation knobs.
GLOBAL_SEED = 20260313
MAX_PAIRWISE_METRIC_PAIRS = 500
TOPK_FRAC = 0.10
FUTURE_CORRECTION_ERROR_THRESHOLD = 0.20
COVERAGE_POINTS = [0.25, 0.50, 0.75, 1.0]

# If a dataset is larger than this and the spec does not already represent a
# curated subset, set a positive cap. Leave at 0 to keep all images.
DEFAULT_MAX_IMAGES_PER_DATASET = 0

# When undefined supervision is enabled on a task, images inside the undefined
# mask are sent as move_type='undefined' instead of a scalar target.
DEFAULT_UNDEFINED_BAND = 0.10
CONTRASTIVE_PROMPT_SOURCE = 'llm_cache'  # 'llm_cache' | 'template'
CONTRASTIVE_PROMPT_CACHE_PATH = CACHE_DIR / 'contrastive_prompts.txt'
CONTRASTIVE_PROMPT_COUNT = 3
CONTRASTIVE_PROMPT_REGENERATE = False
CONTRASTIVE_PROMPT_MAX_RETRIES = 4
CONTRASTIVE_PROMPT_RETRY_WAIT_SEC = 65.0


# =============================
# Config schema
# =============================
@dataclass(frozen=True)
class ConceptSpec:
    kind: str
    name: str
    field: str = ''
    field_prefix: str = ''
    query_prefix: str = ''
    min_count: int = 1
    max_values: int = 0
    exclude_values: Tuple[str, ...] = ()
    positive_values: Tuple[str, ...] = ()
    value_aliases: Tuple[Tuple[str, str], ...] = ()
    undefined_band: float = 0.0
    enabled: bool = True


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_root: Path
    concepts: Tuple[ConceptSpec, ...]
    subset_size: int = DEFAULT_MAX_IMAGES_PER_DATASET
    subset_seed: int = GLOBAL_SEED
    subset_stratify_field: str = ''
    enabled: bool = True


@dataclass
class LoadedDataset:
    name: str
    dataset_root: Path
    ids: List[str]
    paths: List[str]
    id_to_index: Dict[str, int]
    metadata: pd.DataFrame
    image_col: str
    X_clip: np.ndarray
    X_dino: np.ndarray
    fused_default: np.ndarray


@dataclass
class EvaluationTask:
    task_id: str
    dataset_name: str
    dataset_root: Path
    collection_id: str
    concept_kind: str
    concept_name: str
    metadata_field: str
    metadata_value: str
    query: str
    pos_prompts: List[str]
    neg_prompts: List[str]
    ids: List[str]
    paths: List[str]
    id_to_index: Dict[str, int]
    X_clip: np.ndarray
    X_dino: np.ndarray
    fused_default: np.ndarray
    reference_scores01: np.ndarray
    undefined_mask: np.ndarray
    reference_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodSpec:
    name: str
    kind: str
    supports_prior: bool
    supports_feedback: bool
    supports_uncertainty: bool
    supports_undefined: bool
    model_type: str = ''
    mode: str = ''
    feature_space: str = AXIS_BAYES_FEATURE_SPACE
    clip_weight: float = AXIS_BAYES_CLIP_WEIGHT
    dino_weight: float = AXIS_BAYES_DINO_WEIGHT
    params: Dict[str, Any] = field(default_factory=dict)


# =============================
# Dataset registry
# =============================
DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    'EmoSet': DatasetSpec(
        name='EmoSet',
        dataset_root=DATASETS_ROOT / 'EmoSet',
        concepts=(
            ConceptSpec(kind='metadata_one_vs_rest', name='emotion', field='emotion', query_prefix='emotion', min_count=1, max_values=8),
            ConceptSpec(kind='derived_metric', name='brightness', undefined_band=DEFAULT_UNDEFINED_BAND),
            ConceptSpec(kind='derived_metric', name='saturation', undefined_band=DEFAULT_UNDEFINED_BAND),
            ConceptSpec(kind='derived_metric', name='blur', undefined_band=DEFAULT_UNDEFINED_BAND),
        ),
    ),
    'paintings_wikiart': DatasetSpec(
        name='paintings_wikiart',
        dataset_root=DATASETS_ROOT / 'paintings_wikiart',
        concepts=(
            ConceptSpec(kind='metadata_one_vs_rest', name='genre', field='genre', query_prefix='genre', min_count=30, exclude_values=('Unknown Genre',)),
            ConceptSpec(kind='metadata_one_vs_rest', name='style', field='style', query_prefix='style', min_count=20),
            ConceptSpec(kind='derived_metric', name='brightness', undefined_band=DEFAULT_UNDEFINED_BAND),
            ConceptSpec(kind='derived_metric', name='saturation', undefined_band=DEFAULT_UNDEFINED_BAND),
            ConceptSpec(kind='derived_metric', name='blur', undefined_band=DEFAULT_UNDEFINED_BAND),
        ),
    ),
    'celeba_dataset': DatasetSpec(
        name='celeba_dataset',
        dataset_root=DATASETS_ROOT / 'celeba_dataset',
        concepts=(
            ConceptSpec(kind='metadata_binary_prefix', name='face_attribute', field_prefix='attr_', min_count=1),
        ),
    ),
    'chest_xray_pneumonia': DatasetSpec(
        name='chest_xray_pneumonia',
        dataset_root=DATASETS_ROOT / 'chest_xray_pneumonia',
        concepts=(
            ConceptSpec(kind='metadata_binary_value', name='pneumonia', field='label', positive_values=('PNEUMONIA',), min_count=1),
        ),
    ),
    'brain_mri_images': DatasetSpec(
        name='brain_mri_images',
        dataset_root=DATASETS_ROOT / 'brain_mri_images',
        concepts=(
            ConceptSpec(kind='metadata_binary_value', name='tumor', field='label', positive_values=('yes',), min_count=1),
        ),
    ),
    'HAM10000': DatasetSpec(
        name='HAM10000',
        dataset_root=DATASETS_ROOT / 'HAM10000',
        concepts=(
            ConceptSpec(
                kind='metadata_one_vs_rest',
                name='lesion_type',
                field='dx',
                query_prefix='lesion type',
                min_count=20,
                max_values=7,
                value_aliases=(
                    ('akiec', 'actinic keratosis / Bowen disease'),
                    ('bcc', 'basal cell carcinoma'),
                    ('bkl', 'benign keratosis-like lesion'),
                    ('df', 'dermatofibroma'),
                    ('mel', 'melanoma'),
                    ('nv', 'melanocytic nevus'),
                    ('vasc', 'vascular lesion'),
                ),
            ),
        ),
    ),
}


# =============================
# Method registry
# =============================
#
# Keep the modeling-study configs explicit here instead of inheriting the
# runtime defaults from backend/constants.py. The server defaults mix together
# values chosen for interactive use, while the paper study should use the
# best sweep result for each evaluated method.
GAUSSIAN_SWEEP_BEST: Dict[str, Any] = {
    'feature_space': 'clip',
    'clip_weight': 1.0,
    'dino_weight': 0.0,
    'alpha': 0.17392955514866384,
    'dino_alpha': 0.0,
    # 'feature_space': 'clip_dino',
    # 'clip_weight': 0.5913735456374828,
    # 'dino_weight': 0.4086264543625172,
    # 'alpha': 0.17392955514866384,
    # 'dino_alpha': 0.7508090208834363,
    'bias_alpha': 0.02885457255148477,
    'sigma2': 0.0023543092401027846,
}

# Trial 69 from backend/experiments/emoset_optuna_rank.csv.
# The rank sweep did not tune bias_alpha or sigma2, so those stay explicit here
# as shared linear defaults for the study.
RANK_SWEEP_BEST: Dict[str, Any] = {
    'feature_space': 'clip',
    # 'feature_space': 'clip_dino',
    'clip_weight': 1.0,
    'dino_weight': 0.0,
    # 'clip_weight': 0.8041519207887364,
    # 'dino_weight': 0.19584807921126357,
    'alpha': 0.34361647953904884,
    'dino_alpha': 0.0,
    'bias_alpha': AXIS_BAYES_BIAS_ALPHA,
    'sigma2': AXIS_BAYES_SIGMA2,
    'rank_eta': 1.3185334419106902,
    'rank_anchor_k': 3,
    'rank_anchor_delta': 0.3181991587400454,
    'rank_max_pairs': 224,
}

GRAPH_SWEEP_BEST: Dict[str, Any] = {
    'feature_space': 'clip',
    'clip_weight': 1.0,
    'dino_weight': 0.0,
    'sigma2': 0.0006123800590318016,
    'graph_knn_k': 4,
    'graph_lambda_smooth': 3.1392207160296035,
    'graph_lambda_prior': 0.05044615326613909,
    'graph_jitter': 1e-6,
}

PIECEWISE_SWEEP_BEST: Dict[str, Any] = {
    'feature_space': 'clip',
    'clip_weight': 1.0,
    'dino_weight': 0.0,
    'piecewise_num_experts': 1,
    'piecewise_use_gating': True,
    'piecewise_aggregator': 'max',
    'piecewise_clip_scale': 1.3508877482680062,
    'piecewise_dino_scale': 0.0,
    'pairwise_from_scalar_margin': 0.34936776596459196,
    'piecewise_prior_strength': 0.11917005611546902,
    'piecewise_expert_diversity_strength': 0.5197325862847438,
    'piecewise_l2_reg': 0.04912946914323385,
    'piecewise_learning_rate': 0.07989580454753624,
    'piecewise_max_refine_steps': 48,
}

RESIDUAL_DEFAULTS: Dict[str, Any] = {
    'feature_space': 'clip',
    'clip_weight': 1.0,
    'dino_weight': 0.0,
    'residual_alpha': AXIS_RESIDUAL_ALPHA,
    'residual_beta': AXIS_RESIDUAL_BETA,
    'residual_lambda': AXIS_RESIDUAL_LAMBDA,
    'residual_sigma_y': AXIS_RESIDUAL_SIGMA_Y,
    'residual_lengthscale_multiplier': AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER,
    'residual_jitter': AXIS_RESIDUAL_JITTER,
}

METHOD_REGISTRY: Dict[str, MethodSpec] = {
    'clip_text_similarity': MethodSpec(
        name='clip_text_similarity',
        kind='clip_text',
        supports_prior=True,
        supports_feedback=False,
        supports_uncertainty=False,
        supports_undefined=False,
        feature_space='clip',
        clip_weight=1.0,
        dino_weight=0.0,
        params={'use_ensemble': False},
    ),
    'prompt_ensemble': MethodSpec(
        name='prompt_ensemble',
        kind='clip_text',
        supports_prior=True,
        supports_feedback=False,
        supports_uncertainty=False,
        supports_undefined=False,
        feature_space='clip',
        clip_weight=1.0,
        dino_weight=0.0,
        params={'use_ensemble': True},
    ),
    'label_only_linear': MethodSpec(
        name='label_only_linear',
        kind='label_only_linear',
        supports_prior=False,
        supports_feedback=True,
        supports_uncertainty=False,
        supports_undefined=False,
        feature_space='clip',
        clip_weight=1.0,
        dino_weight=0.0,
        params={'ridge_lambda': 1.0},
    ),
    'knn_label_propagation': MethodSpec(
        name='knn_label_propagation',
        kind='knn_propagation',
        supports_prior=False,
        supports_feedback=True,
        supports_uncertainty=False,
        supports_undefined=False,
        feature_space='clip',
        clip_weight=1.0,
        dino_weight=0.0,
        params={'k': 15, 'temperature': 12.0},
    ),
    'request_bayes_linear_gaussian': MethodSpec(
        name='request_bayes_linear_gaussian',
        kind='engine',
        supports_prior=True,
        supports_feedback=True,
        supports_uncertainty=True,
        supports_undefined=True,
        model_type='bayes_linear',
        mode='gaussian',
        feature_space=str(GAUSSIAN_SWEEP_BEST['feature_space']),
        clip_weight=float(GAUSSIAN_SWEEP_BEST['clip_weight']),
        dino_weight=float(GAUSSIAN_SWEEP_BEST['dino_weight']),
        params={
            'alpha': float(GAUSSIAN_SWEEP_BEST['alpha']),
            'dino_alpha': float(GAUSSIAN_SWEEP_BEST['dino_alpha']),
            'bias_alpha': float(GAUSSIAN_SWEEP_BEST['bias_alpha']),
            'sigma2': float(GAUSSIAN_SWEEP_BEST['sigma2']),
        },
    ),
    'request_bayes_linear_rank': MethodSpec(
        name='request_bayes_linear_rank',
        kind='engine',
        supports_prior=True,
        supports_feedback=True,
        supports_uncertainty=True,
        supports_undefined=True,
        model_type='bayes_linear',
        mode='rank',
        feature_space=str(RANK_SWEEP_BEST['feature_space']),
        clip_weight=float(RANK_SWEEP_BEST['clip_weight']),
        dino_weight=float(RANK_SWEEP_BEST['dino_weight']),
        params={
            'alpha': float(RANK_SWEEP_BEST['alpha']),
            'dino_alpha': float(RANK_SWEEP_BEST['dino_alpha']),
            'bias_alpha': float(RANK_SWEEP_BEST['bias_alpha']),
            'sigma2': float(RANK_SWEEP_BEST['sigma2']),
            'rank_eta': float(RANK_SWEEP_BEST['rank_eta']),
            'rank_anchor_k': int(RANK_SWEEP_BEST['rank_anchor_k']),
            'rank_anchor_delta': float(RANK_SWEEP_BEST['rank_anchor_delta']),
            'rank_max_pairs': int(RANK_SWEEP_BEST['rank_max_pairs']),
        },
    ),
    'residual': MethodSpec(
        name='residual',
        kind='engine',
        supports_prior=True,
        supports_feedback=True,
        supports_uncertainty=True,
        supports_undefined=True,
        model_type='residual',
        mode='gaussian',
        feature_space=str(RESIDUAL_DEFAULTS['feature_space']),
        clip_weight=float(RESIDUAL_DEFAULTS['clip_weight']),
        dino_weight=float(RESIDUAL_DEFAULTS['dino_weight']),
        params={
            'residual_alpha': float(RESIDUAL_DEFAULTS['residual_alpha']),
            'residual_beta': float(RESIDUAL_DEFAULTS['residual_beta']),
            'residual_lambda': float(RESIDUAL_DEFAULTS['residual_lambda']),
            'residual_sigma_y': float(RESIDUAL_DEFAULTS['residual_sigma_y']),
            'residual_lengthscale_multiplier': float(RESIDUAL_DEFAULTS['residual_lengthscale_multiplier']),
            'residual_jitter': float(RESIDUAL_DEFAULTS['residual_jitter']),
        },
    ),
    'request_piecewise': MethodSpec(
        name='request_piecewise',
        kind='engine',
        supports_prior=True,
        supports_feedback=True,
        supports_uncertainty=True,
        supports_undefined=True,
        model_type='piecewise_linear',
        mode='rank',
        feature_space=str(PIECEWISE_SWEEP_BEST['feature_space']),
        clip_weight=float(PIECEWISE_SWEEP_BEST['clip_weight']),
        dino_weight=float(PIECEWISE_SWEEP_BEST['dino_weight']),
        params={
            'piecewise_num_experts': int(PIECEWISE_SWEEP_BEST['piecewise_num_experts']),
            'piecewise_use_gating': bool(PIECEWISE_SWEEP_BEST['piecewise_use_gating']),
            'piecewise_aggregator': str(PIECEWISE_SWEEP_BEST['piecewise_aggregator']),
            'piecewise_clip_scale': float(PIECEWISE_SWEEP_BEST['piecewise_clip_scale']),
            'piecewise_dino_scale': float(PIECEWISE_SWEEP_BEST['piecewise_dino_scale']),
            'pairwise_from_scalar_margin': float(PIECEWISE_SWEEP_BEST['pairwise_from_scalar_margin']),
            'piecewise_prior_strength': float(PIECEWISE_SWEEP_BEST['piecewise_prior_strength']),
            'piecewise_expert_diversity_strength': float(PIECEWISE_SWEEP_BEST['piecewise_expert_diversity_strength']),
            'piecewise_l2_reg': float(PIECEWISE_SWEEP_BEST['piecewise_l2_reg']),
            'piecewise_learning_rate': float(PIECEWISE_SWEEP_BEST['piecewise_learning_rate']),
            'piecewise_max_refine_steps': int(PIECEWISE_SWEEP_BEST['piecewise_max_refine_steps']),
        },
    ),
    'request_graph': MethodSpec(
        name='request_graph',
        kind='engine',
        supports_prior=True,
        supports_feedback=True,
        supports_uncertainty=True,
        supports_undefined=True,
        model_type='bayes_linear',
        mode='graph',
        feature_space=str(GRAPH_SWEEP_BEST['feature_space']),
        clip_weight=float(GRAPH_SWEEP_BEST['clip_weight']),
        dino_weight=float(GRAPH_SWEEP_BEST['dino_weight']),
        params={
            'sigma2': float(GRAPH_SWEEP_BEST['sigma2']),
            'graph_knn_k': int(GRAPH_SWEEP_BEST['graph_knn_k']),
            'graph_lambda_smooth': float(GRAPH_SWEEP_BEST['graph_lambda_smooth']),
            'graph_lambda_prior': float(GRAPH_SWEEP_BEST['graph_lambda_prior']),
            'graph_jitter': float(GRAPH_SWEEP_BEST['graph_jitter']),
        },
    ),
}

ABLATION_REGISTRY: Dict[str, MethodSpec] = {
    'ablate_clip_only': replace(
        METHOD_REGISTRY['request_piecewise'],
        name='ablate_clip_only',
        feature_space='clip',
        clip_weight=1.0,
        dino_weight=0.0,
    ),
    'ablate_clip_dino': replace(
        METHOD_REGISTRY['request_piecewise'],
        name='ablate_clip_dino',
        feature_space='clip_dino',
        clip_weight=float(PIECEWISE_SWEEP_BEST['clip_weight']),
        dino_weight=float(PIECEWISE_SWEEP_BEST['dino_weight']),
    ),
    'ablate_no_text_prior': replace(
        METHOD_REGISTRY['request_piecewise'],
        name='ablate_no_text_prior',
        params={**METHOD_REGISTRY['request_piecewise'].params, 'piecewise_prior_strength': 0.0},
    ),
}


# =============================
# Utilities
# =============================
def utc_now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def slugify(value: str) -> str:
    text = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(value or '').strip())
    while '__' in text:
        text = text.replace('__', '_')
    return text.strip('_') or 'item'


def capitalize_first(text: str) -> str:
    s = str(text or '').strip()
    return s[:1].upper() + s[1:] if s else ''


def humanize_label(label: str) -> str:
    return str(label or '').strip().replace('_', ' ')


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return (arr / norms).astype(np.float32)


def rank_percentile_01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.array([0.5], dtype=np.float32)
    order = np.argsort(arr, kind='mergesort')
    sorted_vals = arr[order]
    out = np.zeros_like(arr, dtype=np.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(float(sorted_vals[j]) - float(sorted_vals[i])) <= 1e-12:
            j += 1
        rank = float(i + (j - i - 1) * 0.5) / float(max(1, n - 1))
        out[order[i:j]] = rank
        i = j
    return out


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if xa.size != ya.size or xa.size == 0:
        return float('nan')
    xa = xa - xa.mean()
    ya = ya - ya.mean()
    denom = float(np.linalg.norm(xa) * np.linalg.norm(ya))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xa, ya) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rank_percentile_01(x), rank_percentile_01(y))


def pairwise_order_metrics(pred: np.ndarray, ref: np.ndarray, *, max_pairs: int, rng: np.random.Generator) -> Dict[str, float]:
    p = np.asarray(pred, dtype=np.float32).reshape(-1)
    r = np.asarray(ref, dtype=np.float32).reshape(-1)
    n = int(p.size)
    if n < 2:
        return {
            'pairwise_acc': float('nan'),
            'kendall_tau': float('nan'),
        }
    total_pairs = (n * (n - 1)) // 2
    if total_pairs <= int(max_pairs):
        ii, jj = np.triu_indices(n, k=1)
    else:
        ii = rng.integers(0, n, size=int(max_pairs), endpoint=False)
        jj = rng.integers(0, n - 1, size=int(max_pairs), endpoint=False)
        jj = np.where(jj >= ii, jj + 1, jj)
    ref_diff = r[ii] - r[jj]
    pred_diff = p[ii] - p[jj]
    valid = np.abs(ref_diff) > 1e-6
    if not np.any(valid):
        return {
            'pairwise_acc': float('nan'),
            'kendall_tau': float('nan'),
        }
    signed = pred_diff[valid] * ref_diff[valid]
    concordant = signed > 0.0
    discordant = signed < 0.0
    ties = np.abs(pred_diff[valid]) <= 1e-6
    denom = float(max(1, int(np.sum(valid))))
    return {
        'pairwise_acc': float((np.sum(concordant) + (0.5 * np.sum(ties))) / denom),
        'kendall_tau': float((np.sum(concordant) - np.sum(discordant)) / denom),
    }


def pairwise_accuracy(pred: np.ndarray, ref: np.ndarray, *, max_pairs: int, rng: np.random.Generator) -> float:
    return float(pairwise_order_metrics(pred, ref, max_pairs=max_pairs, rng=rng)['pairwise_acc'])


def sampled_kendall_tau(pred: np.ndarray, ref: np.ndarray, *, max_pairs: int, rng: np.random.Generator) -> float:
    return float(pairwise_order_metrics(pred, ref, max_pairs=max_pairs, rng=rng)['kendall_tau'])


def topk_extreme_precision(pred: np.ndarray, ref: np.ndarray, frac: float) -> float:
    p = np.asarray(pred, dtype=np.float32).reshape(-1)
    r = np.asarray(ref, dtype=np.float32).reshape(-1)
    n = int(p.size)
    if n == 0:
        return float('nan')
    k = max(1, int(round(float(frac) * n)))
    pred_hi = set(np.argsort(p, kind='mergesort')[-k:].tolist())
    ref_hi = set(np.argsort(r, kind='mergesort')[-k:].tolist())
    pred_lo = set(np.argsort(p, kind='mergesort')[:k].tolist())
    ref_lo = set(np.argsort(r, kind='mergesort')[:k].tolist())
    hi = len(pred_hi.intersection(ref_hi)) / float(k)
    lo = len(pred_lo.intersection(ref_lo)) / float(k)
    return float(0.5 * (hi + lo))


def auroc_binary(scores: np.ndarray, labels01: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels01, dtype=np.int32).reshape(-1)
    if s.size == 0 or s.size != y.size:
        return float('nan')
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.zeros_like(s, dtype=np.float64)
    sorted_scores = s[order]
    i = 0
    n = int(s.size)
    while i < n:
        j = i + 1
        while j < n and abs(sorted_scores[j] - sorted_scores[i]) <= 1e-12:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    return float((sum_ranks_pos - (pos * (pos + 1) * 0.5)) / float(pos * neg))


def binary_auroc_if_binary(scores: np.ndarray, ref_scores01: np.ndarray) -> float:
    ref = np.asarray(ref_scores01, dtype=np.float32).reshape(-1)
    pred = np.asarray(scores, dtype=np.float32).reshape(-1)
    if ref.size == 0 or ref.size != pred.size:
        return float('nan')
    finite = np.isfinite(ref) & np.isfinite(pred)
    if int(np.sum(finite)) < 2:
        return float('nan')
    ref = ref[finite]
    pred = pred[finite]
    is_zero = np.abs(ref) <= 1e-6
    is_one = np.abs(ref - 1.0) <= 1e-6
    if not np.all(is_zero | is_one):
        return float('nan')
    labels01 = is_one.astype(np.int32)
    if int(np.min(labels01)) == int(np.max(labels01)):
        return float('nan')
    return auroc_binary(pred, labels01)


def area_under_learning_curve(xs: Sequence[int], ys: Sequence[float]) -> float:
    if len(xs) == 0 or len(xs) != len(ys):
        return float('nan')
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size == 1:
        return float(y[0])
    span = float(np.max(x) - np.min(x))
    area = float(np.trapz(y, x))
    return area / span if span > 0 else float(np.mean(y))


def resolve_image_column(meta: pd.DataFrame) -> str:
    if 'image' in meta.columns:
        return 'image'
    return str(meta.columns[0]) if len(meta.columns) > 0 else 'image'


def resolve_field_column(meta: pd.DataFrame, field: str) -> str:
    target = str(field or '').strip().lower()
    for col in meta.columns:
        if str(col).strip().lower() == target:
            return str(col)
    raise RuntimeError(f'Could not find field={field!r} in metadata columns={list(meta.columns)}')


def build_fused_features(x_clip: np.ndarray, x_dino: np.ndarray, clip_weight: float, dino_weight: float, feature_space: str) -> np.ndarray:
    x_clip = np.asarray(x_clip, dtype=np.float32)
    if str(feature_space) == 'clip' or x_dino.size == 0:
        return l2_normalize_rows(x_clip)
    w_clip = max(0.0, float(clip_weight))
    w_dino = max(0.0, float(dino_weight))
    total = w_clip + w_dino
    if total <= 1e-8:
        w_clip = 1.0
        w_dino = 0.0
        total = 1.0
    wc = math.sqrt(w_clip / total)
    wd = math.sqrt(w_dino / total)
    fused = np.concatenate([wc * x_clip, wd * np.asarray(x_dino, dtype=np.float32)], axis=1)
    return l2_normalize_rows(fused)


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float('nan')
    return float(np.mean(finite))


def safe_learning_curve_auc(xs: Sequence[int], ys: Sequence[float]) -> float:
    pairs = [(int(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(float(y))]
    if len(pairs) == 0:
        return float('nan')
    return area_under_learning_curve([x for x, _ in pairs], [y for _, y in pairs])


def compact_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(',', ':'), sort_keys=True)


def normalize_query_text(query: str) -> str:
    return ' '.join(str(query or '').strip().split())


def normalize_dataset_name(dataset_name: str) -> str:
    text = str(dataset_name or '').strip()
    return str(Path(text).name).strip() if text else ''


def prompt_override_key(dataset_name: str, query: str) -> str:
    return f'{normalize_dataset_name(dataset_name).lower()}::{normalize_query_text(query).lower()}'


def method_uses_dino(spec: MethodSpec) -> bool:
    feature_space = str(spec.feature_space or '').strip().lower()
    if feature_space == 'clip':
        return False
    if feature_space == 'clip_dino':
        return True
    if abs(float(spec.dino_weight)) > 1e-8:
        return True
    params = dict(spec.params or {})
    for key in ('dino_alpha', 'piecewise_dino_scale'):
        raw = params.get(key)
        try:
            if abs(float(raw)) > 1e-8:
                return True
        except (TypeError, ValueError):
            continue
    return False


def method_spec_payload(spec: MethodSpec) -> Dict[str, Any]:
    return {
        'name': str(spec.name),
        'kind': str(spec.kind),
        'model_type': str(spec.model_type or ''),
        'mode': str(spec.mode or ''),
        'feature_space': str(spec.feature_space),
        'clip_weight': float(spec.clip_weight),
        'dino_weight': float(spec.dino_weight),
        'params': dict(spec.params),
    }


def fixed_prompt_override(prefix: str, label: str) -> Tuple[List[str], List[str]]:
    prefix_key = str(prefix or '').strip().lower()
    label_text = humanize_label(label)
    if prefix_key == 'emotion':
        return (
            [
                f'an image that strongly conveys {label_text}',
                f'an image full of {label_text}',
                f'a scene expressing {label_text}',
            ],
            [
                f'an image without {label_text}',
                'an emotionally neutral image',
                f'a calm scene not expressing {label_text}',
            ],
        )
    if prefix_key == 'genre':
        return (
            [
                f'a painting in the {label_text} genre',
                f'an artwork of genre {label_text}',
                f'an example of {label_text} painting',
            ],
            [
                f'a painting not in the {label_text} genre',
                'an artwork of a different painting genre',
                f'a non-{label_text} painting',
            ],
        )
    if prefix_key == 'style':
        return (
            [
                f'a painting in the {label_text} style',
                f'an artwork in {label_text} style',
                f'an example of {label_text} painting style',
            ],
            [
                f'a painting not in the {label_text} style',
                'an artwork in a different painting style',
                f'a non-{label_text} painting style',
            ],
        )
    if prefix_key == 'brightness':
        return (
            ['a very bright image', 'an image with high brightness', 'an image full of light'],
            ['a very dark image', 'an image with low brightness', 'an image with very little light'],
        )
    if prefix_key == 'saturation':
        return (
            ['a highly saturated image', 'an image with vivid colors', 'an image with strong color saturation'],
            ['a desaturated image', 'an image with muted colors', 'an image with very low color saturation'],
        )
    if prefix_key == 'blur':
        return (
            ['a very blurry image', 'an image strongly out of focus', 'an image with heavy blur'],
            ['a very sharp image', 'an image in crisp focus', 'an image with high sharpness'],
        )
    if prefix_key == 'pneumonia':
        return (
            [
                'a chest x ray showing pneumonia',
                'a lung x ray with signs of pneumonia',
                'a chest radiograph with pneumonia',
            ],
            [
                'a normal chest x ray without pneumonia',
                'a healthy lung x ray',
                'a chest radiograph with no pneumonia',
            ],
        )
    if prefix_key == 'tumor':
        return (
            [
                'a brain mri showing a tumor',
                'a brain scan with a tumor',
                'a brain mri with abnormal tumor tissue',
            ],
            [
                'a normal brain mri without a tumor',
                'a healthy brain scan',
                'a brain mri with no tumor',
            ],
        )
    return (
        [
            f'an image with very strong presence of {label_text}',
            f'an image with high presence of {label_text}',
            f'an image with a lot of {label_text}',
        ],
        [
            f'an image with very weak presence of {label_text}',
            f'an image with low presence of {label_text}',
            f'an image with very little {label_text}',
        ],
    )


def stable_seed(*parts: object) -> int:
    joined = '||'.join(str(part) for part in parts)
    return int(zlib.crc32(joined.encode('utf-8')) & 0xFFFFFFFF)


def descending_rank_1based(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int32)
    order = np.argsort(-arr, kind='mergesort')
    ranks = np.zeros((arr.size,), dtype=np.int32)
    ranks[order] = np.arange(1, arr.size + 1, dtype=np.int32)
    return ranks


def select_representative_task_ids(tasks: Sequence[EvaluationTask], per_dataset: int) -> List[str]:
    chosen: List[str] = []
    tasks_sorted = sorted(tasks, key=lambda task: (task.dataset_name, task.concept_kind, task.task_id))
    by_dataset: Dict[str, List[EvaluationTask]] = {}
    for task in tasks_sorted:
        by_dataset.setdefault(task.dataset_name, []).append(task)
    for dataset_name, dataset_tasks in by_dataset.items():
        picked: List[EvaluationTask] = []
        seen_task_ids = set()
        by_kind: Dict[str, List[EvaluationTask]] = {}
        for task in dataset_tasks:
            by_kind.setdefault(task.concept_kind, []).append(task)
        for concept_kind in sorted(by_kind):
            if len(picked) >= int(per_dataset):
                break
            task = by_kind[concept_kind][0]
            if task.task_id not in seen_task_ids:
                picked.append(task)
                seen_task_ids.add(task.task_id)
        if len(picked) < int(per_dataset):
            for task in dataset_tasks:
                if len(picked) >= int(per_dataset):
                    break
                if task.task_id in seen_task_ids:
                    continue
                picked.append(task)
                seen_task_ids.add(task.task_id)
        chosen.extend(task.task_id for task in picked)
    return chosen


def select_representative_example_indices(task: EvaluationTask, count: int) -> np.ndarray:
    n = int(len(task.ids))
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(np.asarray(task.reference_scores01, dtype=np.float32), kind='mergesort')
    want = min(int(count), n)
    if want <= 0:
        return np.zeros((0,), dtype=np.int64)
    picks: List[int] = []
    seen = set()
    for pos in np.linspace(0, n - 1, num=want):
        idx = int(round(float(pos)))
        idx = max(0, min(n - 1, idx))
        chosen = int(order[idx])
        if chosen in seen:
            continue
        picks.append(chosen)
        seen.add(chosen)
    if len(picks) < want:
        for chosen in order.tolist():
            if int(chosen) in seen:
                continue
            picks.append(int(chosen))
            seen.add(int(chosen))
            if len(picks) >= want:
                break
    return np.asarray(picks[:want], dtype=np.int64)


class CSVAppender:
    def __init__(self, path: Path, fieldnames: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._initialized = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open('r', newline='', encoding='utf-8') as handle:
                header = next(csv.reader(handle), [])
            if list(header) == self.fieldnames:
                self._initialized = True
            else:
                with self.path.open('r', newline='', encoding='utf-8') as handle:
                    rows = list(csv.DictReader(handle))
                with self.path.open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({key: row.get(key, '') for key in self.fieldnames})
                self._initialized = True

    def append(self, row: Mapping[str, object]) -> None:
        payload = {key: row.get(key, '') for key in self.fieldnames}
        with self.path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(payload)
            handle.flush()


# =============================
# Dataset loading / task building
# =============================
def _sample_indices_stratified(values: Sequence[str], count: int, seed: int) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    rng = np.random.default_rng(seed)
    if len(arr) <= count:
        return np.arange(len(arr), dtype=np.int64)
    by_value: Dict[str, List[int]] = {}
    for idx, value in enumerate(arr):
        key = str(value)
        by_value.setdefault(key, []).append(idx)
    picks: List[int] = []
    groups = [list(v) for _, v in sorted(by_value.items()) if len(v) > 0]
    while len(picks) < count and any(len(group) > 0 for group in groups):
        for group in groups:
            if len(group) == 0 or len(picks) >= count:
                continue
            choice_idx = int(rng.integers(0, len(group)))
            picks.append(group.pop(choice_idx))
            if len(picks) >= count:
                break
    picks = sorted(set(picks))
    if len(picks) < count:
        remaining = [idx for idx in range(len(arr)) if idx not in set(picks)]
        extra = rng.choice(np.asarray(remaining, dtype=np.int64), size=int(count - len(picks)), replace=False).tolist()
        picks.extend(extra)
    return np.asarray(sorted(picks[:count]), dtype=np.int64)


def load_dataset(spec: DatasetSpec, *, require_dino: bool = False) -> LoadedDataset:
    dataset_root = Path(spec.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_root}')
    engine = ImageGalleryEngine(str(dataset_root))
    entries = engine.list_images()
    if len(entries) == 0:
        raise RuntimeError(f'No images found in dataset: {dataset_root}')
    ids_all = [str(entry.id) for entry in entries]
    paths_all = [str(entry.path) for entry in entries]

    metadata_path = dataset_root / 'metadata.csv'
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        meta.columns = [str(col).strip() for col in meta.columns]
        image_col = resolve_image_column(meta)
        meta[image_col] = meta[image_col].astype(str).str.strip()
        meta = meta.drop_duplicates(subset=[image_col], keep='first')
        meta = meta.set_index(image_col, drop=False)
        meta = meta.reindex(ids_all)
        meta[image_col] = ids_all
        meta = meta.reset_index(drop=True)
    else:
        image_col = 'image'
        meta = pd.DataFrame({image_col: ids_all})

    keep_idx = np.arange(len(ids_all), dtype=np.int64)
    if int(spec.subset_size) > 0 and len(keep_idx) > int(spec.subset_size):
        stratify_field = str(spec.subset_stratify_field or '').strip()
        if stratify_field:
            field_col = resolve_field_column(meta, stratify_field)
            sampled = _sample_indices_stratified(
                values=[str(v) for v in meta[field_col].fillna('').astype(str).tolist()],
                count=int(spec.subset_size),
                seed=int(spec.subset_seed),
            )
        else:
            rng = np.random.default_rng(int(spec.subset_seed))
            sampled = np.asarray(sorted(rng.choice(keep_idx, size=int(spec.subset_size), replace=False).tolist()), dtype=np.int64)
        keep_idx = sampled

    ids = [ids_all[int(i)] for i in keep_idx]
    paths = [paths_all[int(i)] for i in keep_idx]
    meta = meta.iloc[keep_idx].reset_index(drop=True)

    clip_all = engine._load_embeddings_only(entries, method='clip')
    if clip_all is None:
        raise RuntimeError(f'CLIP embeddings not available for {dataset_root}')
    x_clip = l2_normalize_rows(np.asarray(clip_all, dtype=np.float32)[keep_idx])
    if require_dino:
        dino_all = engine._load_embeddings_only(entries, method='dino')
        if dino_all is None:
            raise RuntimeError(f'DINO embeddings not available for {dataset_root}')
        x_dino = l2_normalize_rows(np.asarray(dino_all, dtype=np.float32)[keep_idx])
    else:
        x_dino = np.zeros((len(keep_idx), 0), dtype=np.float32)
    fused_default = build_fused_features(x_clip, x_dino, AXIS_BAYES_CLIP_WEIGHT, AXIS_BAYES_DINO_WEIGHT, AXIS_BAYES_FEATURE_SPACE)
    id_to_index = {image_id: idx for idx, image_id in enumerate(ids)}
    return LoadedDataset(
        name=spec.name,
        dataset_root=dataset_root,
        ids=ids,
        paths=paths,
        id_to_index=id_to_index,
        metadata=meta,
        image_col=image_col,
        X_clip=x_clip,
        X_dino=x_dino,
        fused_default=fused_default,
    )


def _derived_metrics_cache_path(dataset: LoadedDataset) -> Path:
    return CACHE_DIR / f'{slugify(dataset.name)}_derived_metrics.csv'


def _compute_image_metrics(path: str, max_edge: int = 160) -> Dict[str, float]:
    with Image.open(path) as image:
        image = image.convert('RGB')
        w, h = image.size
        scale = float(max(w, h))
        if scale > float(max_edge):
            ratio = float(max_edge) / scale
            image = image.resize((max(8, int(round(w * ratio))), max(8, int(round(h * ratio)))), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    brightness = float(np.mean(maxc))
    saturation = float(np.mean((maxc - minc) / (maxc + 1e-6)))
    gray = (0.2989 * r) + (0.5870 * g) + (0.1140 * b)
    contrast = float(np.std(gray))
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lap = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        sharpness = float(np.var(lap))
    else:
        sharpness = 0.0
    blur = float(1.0 / (sharpness + 1e-6))
    return {
        'brightness': brightness,
        'saturation': saturation,
        'contrast': contrast,
        'blur': blur,
    }


def load_or_compute_derived_metrics(dataset: LoadedDataset) -> pd.DataFrame:
    cache_path = _derived_metrics_cache_path(dataset)
    existing: pd.DataFrame
    if cache_path.exists() and cache_path.stat().st_size > 0:
        existing = pd.read_csv(cache_path)
        existing.columns = [str(col).strip() for col in existing.columns]
        if 'image' in existing.columns:
            existing['image'] = existing['image'].astype(str).str.strip()
        else:
            existing = pd.DataFrame(columns=['image', 'brightness', 'saturation', 'contrast', 'blur'])
    else:
        existing = pd.DataFrame(columns=['image', 'brightness', 'saturation', 'contrast', 'blur'])
    have = set(existing['image'].tolist()) if 'image' in existing.columns else set()
    missing_rows: List[Dict[str, Any]] = []
    for image_id, path in zip(dataset.ids, dataset.paths):
        if image_id in have:
            continue
        metrics = _compute_image_metrics(path)
        missing_rows.append({'image': image_id, **metrics})
    if missing_rows:
        logger = CSVAppender(cache_path, ['image', 'brightness', 'saturation', 'contrast', 'blur'])
        for row in missing_rows:
            logger.append(row)
        existing = pd.read_csv(cache_path)
        existing.columns = [str(col).strip() for col in existing.columns]
    existing['image'] = existing['image'].astype(str).str.strip()
    existing = existing.drop_duplicates(subset=['image'], keep='last').set_index('image', drop=False)
    return existing.reindex(dataset.ids)


def _task_query_and_prompts(concept_name: str, query_prefix: str, value: str = '') -> Tuple[str, List[str], List[str]]:
    if value:
        query = f'{query_prefix}: {humanize_label(value)}' if query_prefix else humanize_label(value)
        pos, neg = fixed_prompt_override(query_prefix or concept_name, value)
        return query, list(pos), list(neg)
    query = humanize_label(concept_name)
    pos, neg = fixed_prompt_override(concept_name, concept_name)
    return query, list(pos), list(neg)


def _normalize_reference(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    if span <= 1e-8:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return ((arr - lo) / span).astype(np.float32)


def concept_display_value(spec: ConceptSpec, value: str) -> str:
    raw_value = str(value or '').strip()
    if not raw_value:
        return ''
    alias_map = {
        str(raw_key).strip().lower(): str(display_value).strip()
        for raw_key, display_value in tuple(spec.value_aliases or ())
        if str(raw_key).strip() and str(display_value).strip()
    }
    return alias_map.get(raw_value.lower(), raw_value)


def build_metadata_one_vs_rest_tasks(dataset: LoadedDataset, spec: ConceptSpec) -> List[EvaluationTask]:
    meta = dataset.metadata.copy()
    field_col = resolve_field_column(meta, spec.field)
    values = meta[field_col].fillna('').astype(str).str.strip()
    exclude = {str(v).strip().lower() for v in spec.exclude_values}
    valid = (values.astype(str).str.len() > 0) & (~values.str.lower().isin(exclude))
    counts = values[valid].value_counts()
    kept_values = [str(v) for v, c in counts.items() if int(c) >= int(spec.min_count)]
    if int(spec.max_values) > 0:
        kept_values = kept_values[: int(spec.max_values)]
    tasks: List[EvaluationTask] = []
    keep_mask = valid.to_numpy(dtype=bool)
    for value in kept_values:
        mask = values.isin([value]).to_numpy(dtype=bool)
        display_value = concept_display_value(spec, value)
        ids = [dataset.ids[idx] for idx, flag in enumerate(keep_mask) if bool(flag)]
        paths = [dataset.paths[idx] for idx, flag in enumerate(keep_mask) if bool(flag)]
        x_clip = dataset.X_clip[keep_mask]
        x_dino = dataset.X_dino[keep_mask]
        fused = dataset.fused_default[keep_mask]
        ref = mask[keep_mask].astype(np.float32)
        query, pos, neg = _task_query_and_prompts(spec.name, spec.query_prefix or spec.field, display_value)
        task_id = f'{slugify(dataset.name)}__{slugify(spec.field)}__{slugify(value)}'
        tasks.append(
            EvaluationTask(
                task_id=task_id,
                dataset_name=dataset.name,
                dataset_root=dataset.dataset_root,
                collection_id=task_id,
                concept_kind=spec.kind,
                concept_name=capitalize_first(humanize_label(display_value)),
                metadata_field=str(spec.field),
                metadata_value=str(value),
                query=query,
                pos_prompts=pos,
                neg_prompts=neg,
                ids=ids,
                paths=paths,
                id_to_index={image_id: idx for idx, image_id in enumerate(ids)},
                X_clip=np.asarray(x_clip, dtype=np.float32),
                X_dino=np.asarray(x_dino, dtype=np.float32),
                fused_default=np.asarray(fused, dtype=np.float32),
                reference_scores01=np.asarray(ref, dtype=np.float32),
                undefined_mask=np.zeros((len(ids),), dtype=bool),
                reference_meta={'field': spec.field, 'value': value, 'display_value': display_value},
            ),
        )
    return tasks


def _binary_positive_mask(values: pd.Series, spec: ConceptSpec) -> np.ndarray:
    raw = values.fillna('')
    if spec.positive_values:
        wanted = {str(v).strip().lower() for v in spec.positive_values}
        return raw.astype(str).str.strip().str.lower().isin(wanted).to_numpy(dtype=bool)
    numeric = pd.to_numeric(raw, errors='coerce')
    if int(np.sum(numeric.notna().to_numpy(dtype=bool))) == len(raw):
        return (numeric.to_numpy(dtype=np.float32) > 0.0)
    text = raw.astype(str).str.strip().str.lower()
    return text.isin({'1', 'true', 'yes', 'y', 'present', 'positive'}).to_numpy(dtype=bool)


def build_metadata_binary_value_tasks(dataset: LoadedDataset, spec: ConceptSpec) -> List[EvaluationTask]:
    meta = dataset.metadata.copy()
    field_col = resolve_field_column(meta, spec.field)
    values = meta[field_col]
    positive = _binary_positive_mask(values, spec)
    keep = values.fillna('').astype(str).str.strip().str.len().to_numpy(dtype=np.int32) > 0
    pos_count = int(np.sum(positive & keep))
    neg_count = int(np.sum((~positive) & keep))
    if pos_count < int(spec.min_count) or neg_count < int(spec.min_count):
        return []
    ids = [dataset.ids[idx] for idx, flag in enumerate(keep) if bool(flag)]
    paths = [dataset.paths[idx] for idx, flag in enumerate(keep) if bool(flag)]
    x_clip = dataset.X_clip[keep]
    x_dino = dataset.X_dino[keep]
    fused = dataset.fused_default[keep]
    ref = positive[keep].astype(np.float32)
    query, pos, neg = _task_query_and_prompts(spec.name or spec.field, spec.query_prefix or spec.name or spec.field)
    task_id = f'{slugify(dataset.name)}__binary__{slugify(spec.field)}__{slugify(spec.name or "positive")}'
    return [
        EvaluationTask(
            task_id=task_id,
            dataset_name=dataset.name,
            dataset_root=dataset.dataset_root,
            collection_id=task_id,
            concept_kind=spec.kind,
            concept_name=capitalize_first(humanize_label(spec.name or spec.field)),
            metadata_field=str(spec.field),
            metadata_value='|'.join(str(v) for v in spec.positive_values),
            query=query,
            pos_prompts=pos,
            neg_prompts=neg,
            ids=ids,
            paths=paths,
            id_to_index={image_id: idx for idx, image_id in enumerate(ids)},
            X_clip=np.asarray(x_clip, dtype=np.float32),
            X_dino=np.asarray(x_dino, dtype=np.float32),
            fused_default=np.asarray(fused, dtype=np.float32),
            reference_scores01=np.asarray(ref, dtype=np.float32),
            undefined_mask=np.zeros((len(ids),), dtype=bool),
            reference_meta={'field': spec.field, 'positive_values': list(spec.positive_values)},
        )
    ]


def build_metadata_binary_prefix_tasks(dataset: LoadedDataset, spec: ConceptSpec) -> List[EvaluationTask]:
    meta = dataset.metadata.copy()
    prefix = str(spec.field_prefix or '').strip().lower()
    if not prefix:
        return []
    tasks: List[EvaluationTask] = []
    for field_col in [str(col) for col in meta.columns if str(col).strip().lower().startswith(prefix)]:
        values = meta[field_col]
        keep = values.fillna('').astype(str).str.strip().str.len().to_numpy(dtype=np.int32) > 0
        positive = _binary_positive_mask(values, spec)
        pos_count = int(np.sum(positive & keep))
        neg_count = int(np.sum((~positive) & keep))
        if pos_count < int(spec.min_count) or neg_count < int(spec.min_count):
            continue
        concept_label = field_col[len(spec.field_prefix):] if field_col.startswith(spec.field_prefix) else field_col
        ids = [dataset.ids[idx] for idx, flag in enumerate(keep) if bool(flag)]
        paths = [dataset.paths[idx] for idx, flag in enumerate(keep) if bool(flag)]
        x_clip = dataset.X_clip[keep]
        x_dino = dataset.X_dino[keep]
        fused = dataset.fused_default[keep]
        ref = positive[keep].astype(np.float32)
        query, pos, neg = _task_query_and_prompts(concept_label, spec.query_prefix or concept_label)
        task_id = f'{slugify(dataset.name)}__binary_prefix__{slugify(field_col)}'
        tasks.append(
            EvaluationTask(
                task_id=task_id,
                dataset_name=dataset.name,
                dataset_root=dataset.dataset_root,
                collection_id=task_id,
                concept_kind=spec.kind,
                concept_name=capitalize_first(humanize_label(concept_label)),
                metadata_field=str(field_col),
                metadata_value='positive',
                query=query,
                pos_prompts=pos,
                neg_prompts=neg,
                ids=ids,
                paths=paths,
                id_to_index={image_id: idx for idx, image_id in enumerate(ids)},
                X_clip=np.asarray(x_clip, dtype=np.float32),
                X_dino=np.asarray(x_dino, dtype=np.float32),
                fused_default=np.asarray(fused, dtype=np.float32),
                reference_scores01=np.asarray(ref, dtype=np.float32),
                undefined_mask=np.zeros((len(ids),), dtype=bool),
                reference_meta={'field': field_col, 'type': 'binary_prefix'},
            ),
        )
    return tasks


def build_derived_metric_task(dataset: LoadedDataset, spec: ConceptSpec) -> List[EvaluationTask]:
    derived = load_or_compute_derived_metrics(dataset)
    if spec.name not in derived.columns:
        raise RuntimeError(f'Derived metric {spec.name!r} not available for dataset {dataset.name}')
    values = pd.to_numeric(derived[spec.name], errors='coerce')
    keep = values.notna().to_numpy(dtype=bool)
    ids = [dataset.ids[idx] for idx, flag in enumerate(keep) if bool(flag)]
    paths = [dataset.paths[idx] for idx, flag in enumerate(keep) if bool(flag)]
    x_clip = dataset.X_clip[keep]
    x_dino = dataset.X_dino[keep]
    fused = dataset.fused_default[keep]
    ref_raw = values.to_numpy(dtype=np.float32)[keep]
    ref = _normalize_reference(ref_raw)
    undefined_mask = np.zeros((len(ref),), dtype=bool)
    band = float(spec.undefined_band)
    if band > 0.0:
        lo = max(0.0, 0.5 - (0.5 * band))
        hi = min(1.0, 0.5 + (0.5 * band))
        undefined_mask = (ref >= lo) & (ref <= hi)
    query, pos, neg = _task_query_and_prompts(spec.name, spec.name)
    task_id = f'{slugify(dataset.name)}__derived__{slugify(spec.name)}'
    return [
        EvaluationTask(
            task_id=task_id,
            dataset_name=dataset.name,
            dataset_root=dataset.dataset_root,
            collection_id=task_id,
            concept_kind=spec.kind,
            concept_name=capitalize_first(spec.name),
            metadata_field='',
            metadata_value='',
            query=query,
            pos_prompts=pos,
            neg_prompts=neg,
            ids=ids,
            paths=paths,
            id_to_index={image_id: idx for idx, image_id in enumerate(ids)},
            X_clip=np.asarray(x_clip, dtype=np.float32),
            X_dino=np.asarray(x_dino, dtype=np.float32),
            fused_default=np.asarray(fused, dtype=np.float32),
            reference_scores01=np.asarray(ref, dtype=np.float32),
            undefined_mask=np.asarray(undefined_mask, dtype=bool),
            reference_meta={'derived_metric': spec.name},
        )
    ]


def build_metadata_numeric_task(dataset: LoadedDataset, spec: ConceptSpec) -> List[EvaluationTask]:
    meta = dataset.metadata.copy()
    field_col = resolve_field_column(meta, spec.field)
    values = pd.to_numeric(meta[field_col], errors='coerce')
    keep = values.notna().to_numpy(dtype=bool)
    ids = [dataset.ids[idx] for idx, flag in enumerate(keep) if bool(flag)]
    paths = [dataset.paths[idx] for idx, flag in enumerate(keep) if bool(flag)]
    x_clip = dataset.X_clip[keep]
    x_dino = dataset.X_dino[keep]
    fused = dataset.fused_default[keep]
    ref = _normalize_reference(values.to_numpy(dtype=np.float32)[keep])
    undefined_mask = np.zeros((len(ref),), dtype=bool)
    band = float(spec.undefined_band)
    if band > 0.0:
        lo = max(0.0, 0.5 - (0.5 * band))
        hi = min(1.0, 0.5 + (0.5 * band))
        undefined_mask = (ref >= lo) & (ref <= hi)
    query, pos, neg = _task_query_and_prompts(spec.name or spec.field, spec.field or spec.name)
    task_id = f'{slugify(dataset.name)}__numeric__{slugify(spec.field)}'
    return [
        EvaluationTask(
            task_id=task_id,
            dataset_name=dataset.name,
            dataset_root=dataset.dataset_root,
            collection_id=task_id,
            concept_kind=spec.kind,
            concept_name=capitalize_first(spec.name or humanize_label(spec.field)),
            metadata_field=str(spec.field),
            metadata_value='',
            query=query,
            pos_prompts=pos,
            neg_prompts=neg,
            ids=ids,
            paths=paths,
            id_to_index={image_id: idx for idx, image_id in enumerate(ids)},
            X_clip=np.asarray(x_clip, dtype=np.float32),
            X_dino=np.asarray(x_dino, dtype=np.float32),
            fused_default=np.asarray(fused, dtype=np.float32),
            reference_scores01=np.asarray(ref, dtype=np.float32),
            undefined_mask=np.asarray(undefined_mask, dtype=bool),
            reference_meta={'field': spec.field, 'type': 'numeric'},
        )
    ]


def build_tasks_for_dataset(dataset: LoadedDataset, spec: DatasetSpec) -> List[EvaluationTask]:
    tasks: List[EvaluationTask] = []
    for concept in spec.concepts:
        if not concept.enabled:
            continue
        if concept.kind == 'metadata_one_vs_rest':
            tasks.extend(build_metadata_one_vs_rest_tasks(dataset, concept))
        elif concept.kind == 'metadata_binary_prefix':
            tasks.extend(build_metadata_binary_prefix_tasks(dataset, concept))
        elif concept.kind == 'metadata_binary_value':
            tasks.extend(build_metadata_binary_value_tasks(dataset, concept))
        elif concept.kind == 'metadata_numeric':
            tasks.extend(build_metadata_numeric_task(dataset, concept))
        elif concept.kind == 'derived_metric':
            tasks.extend(build_derived_metric_task(dataset, concept))
        else:
            raise ValueError(f'Unknown concept kind: {concept.kind}')
    return tasks


def task_is_active(task: EvaluationTask) -> bool:
    if not ACTIVE_TASK_SUBSTRINGS:
        return True
    task_key = f'{task.dataset_name}::{task.task_id}::{task.concept_name}'.lower()
    return any(str(token).lower() in task_key for token in ACTIVE_TASK_SUBSTRINGS)


# =============================
# Engine subclass for preloaded tasks
# =============================
class PreloadedAxisBayesEngine(AxisBayesEngine):
    _prompt_embed_cache: Dict[Tuple[str, str, str, str, bool], Tuple[np.ndarray, List[str], List[str], Dict[str, str]]] = {}

    def __init__(
        self,
        tasks_by_collection: Mapping[str, EvaluationTask],
        prompt_overrides: Mapping[str, Tuple[List[str], List[str]]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tasks_by_collection = dict(tasks_by_collection)
        self._prompt_overrides = {
            str(key).strip().lower(): (list(value[0]), list(value[1]))
            for key, value in prompt_overrides.items()
        }

    def build_prompt_ensemble(self, q: str, *, dataset_name: str = '') -> tuple[List[str], List[str], Dict[str, str]]:
        query = normalize_query_text(q)
        override = self._prompt_overrides.get(prompt_override_key(dataset_name, query))
        if override is None:
            override = self._prompt_overrides.get(query.lower())
        if override is not None:
            pos, neg = override
            return list(pos), list(neg), {'source': 'eval_override', 'provider': 'fixed_template'}
        return super().build_prompt_ensemble(query, dataset_name=dataset_name)

    def _embed_prompt_ensemble(
        self,
        q: str,
        semantic_method: str,
        norm: Optional[bool] = None,
        dataset_name: str = '',
    ) -> tuple[np.ndarray, List[str], List[str], Dict[str, str]]:
        query = ' '.join(str(q or '').strip().split())
        cache_key = (self.axis_bounds_text_source, dataset_name, query, semantic_method, bool(self.norm if norm is None else norm))
        cached = self._prompt_embed_cache.get(cache_key)
        if cached is not None:
            return (
                np.asarray(cached[0], dtype=np.float32).copy(),
                list(cached[1]),
                list(cached[2]),
                dict(cached[3]),
            )
        result = super()._embed_prompt_ensemble(
            query,
            semantic_method=semantic_method,
            norm=norm,
            dataset_name=dataset_name,
        )
        self._prompt_embed_cache[cache_key] = (
            np.asarray(result[0], dtype=np.float32).copy(),
            list(result[1]),
            list(result[2]),
            dict(result[3]),
        )
        return result

    def _get_collection(self, dataset_root: str, collection_id: str) -> CollectionCache:
        key = str(collection_id or '').strip()
        if key not in self._tasks_by_collection:
            raise KeyError(f'Unknown preloaded collection_id={key!r}')
        task = self._tasks_by_collection[key]
        cache_key = f'{key}|{self.feature_space}|cw={self.clip_scale:.6f}|dw={self.dino_scale:.6f}'
        cached = self._collections.get(cache_key)
        if cached is not None:
            return cached
        x_clip = np.asarray(task.X_clip, dtype=np.float32)
        x_dino = np.asarray(task.X_dino, dtype=np.float32)
        clip_dim = int(x_clip.shape[1])
        if self.feature_space == 'clip_dino':
            x = np.concatenate(
                [
                    float(self.clip_scale) * x_clip,
                    float(self.dino_scale) * x_dino,
                ],
                axis=1,
            ).astype(np.float32)
            dino_dim = int(x_dino.shape[1])
        else:
            x = l2_normalize_rows(x_clip)
            dino_dim = 0
        embedding_norms = np.linalg.norm(np.asarray(x, dtype=np.float32), axis=1).astype(np.float32)
        cached = CollectionCache(
            dataset_root=str(task.dataset_root.resolve()),
            collection_id=key,
            ids=list(task.ids),
            embeddings=np.asarray(x, dtype=np.float32),
            embedding_norms=embedding_norms,
            id_to_index=dict(task.id_to_index),
            feature_space=self.feature_space,
            semantic_method='clip',
            norm=True,
            clip_dim=clip_dim,
            dino_dim=dino_dim,
            clip_scale=float(self.clip_scale),
            dino_scale=float(self.dino_scale),
        )
        self._collections[cache_key] = cached
        return cached


# =============================
# Method sessions
# =============================
class BaseMethodSession:
    """Common API so the evaluation loop does not care about the backend.

    Scalar labels are converted into ranking behavior internally by some methods
    (notably the ReQuest piecewise model), but the evaluation loop only needs
    to pass image ids and scalar targets in [0, 1].
    """

    def __init__(self, spec: MethodSpec, task: EvaluationTask, rng: np.random.Generator):
        self.spec = spec
        self.task = task
        self.rng = rng
        self.queried_ids: List[str] = []
        self.undefined_ids: List[str] = []

    def current_scores(self) -> np.ndarray:
        raise NotImplementedError

    def current_uncertainty(self) -> np.ndarray:
        return np.full((len(self.task.ids),), np.nan, dtype=np.float32)

    def observe(self, image_id: str, target01: Optional[float], move_type: str = 'score') -> None:
        self.queried_ids.append(str(image_id))
        if move_type == 'undefined':
            self.undefined_ids.append(str(image_id))

    def queried_mask(self) -> np.ndarray:
        mask = np.zeros((len(self.task.ids),), dtype=bool)
        for image_id in self.queried_ids:
            idx = self.task.id_to_index.get(str(image_id))
            if idx is not None:
                mask[int(idx)] = True
        return mask


class ClipTextSession(BaseMethodSession):
    def __init__(self, spec: MethodSpec, task: EvaluationTask, rng: np.random.Generator, context: 'EvaluationContext'):
        super().__init__(spec, task, rng)
        engine = context.get_text_engine()
        if bool(spec.params.get('use_ensemble')):
            w0, _, _, _ = engine._embed_prompt_lists(
                task.pos_prompts,
                task.neg_prompts,
                semantic_method='clip',
                norm=True,
                source='eval_override',
                provider='precomputed_prompts',
            )
        else:
            w0, _, _, _ = engine._embed_prompt_lists(
                task.pos_prompts[:1],
                task.neg_prompts[:1],
                semantic_method='clip',
                norm=True,
                source='eval_override',
                provider='precomputed_prompts',
            )
        self._scores = (np.asarray(task.X_clip, dtype=np.float32) @ np.asarray(w0, dtype=np.float32)).astype(np.float32)

    def current_scores(self) -> np.ndarray:
        return np.asarray(self._scores, dtype=np.float32)


class LabelOnlyLinearSession(BaseMethodSession):
    def __init__(self, spec: MethodSpec, task: EvaluationTask, rng: np.random.Generator):
        super().__init__(spec, task, rng)
        self.features = build_fused_features(task.X_clip, task.X_dino, spec.clip_weight, spec.dino_weight, spec.feature_space)
        self.labeled_ids: List[str] = []
        self.labeled_targets: List[float] = []
        self._scores = np.full((len(task.ids),), 0.5, dtype=np.float32)

    def _fit(self) -> None:
        if len(self.labeled_ids) == 0:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        idx = np.asarray([self.task.id_to_index[image_id] for image_id in self.labeled_ids], dtype=np.int64)
        x = np.asarray(self.features[idx], dtype=np.float32)
        y = np.asarray(self.labeled_targets, dtype=np.float32)
        lam = float(self.spec.params.get('ridge_lambda', 1.0))
        gram = x @ x.T
        dual = np.linalg.solve(gram + (lam * np.eye(len(idx), dtype=np.float32)), y)
        weights = x.T @ dual
        bias = float(np.mean(y - (x @ weights)))
        self._scores = ((self.features @ weights) + bias).astype(np.float32)

    def current_scores(self) -> np.ndarray:
        return np.asarray(self._scores, dtype=np.float32)

    def observe(self, image_id: str, target01: Optional[float], move_type: str = 'score') -> None:
        super().observe(image_id, target01, move_type)
        if move_type == 'undefined' or target01 is None:
            return
        self.labeled_ids.append(str(image_id))
        self.labeled_targets.append(float(target01))
        self._fit()


class KNNPropagationSession(BaseMethodSession):
    def __init__(self, spec: MethodSpec, task: EvaluationTask, rng: np.random.Generator):
        super().__init__(spec, task, rng)
        self.features = build_fused_features(task.X_clip, task.X_dino, spec.clip_weight, spec.dino_weight, spec.feature_space)
        self.labeled_ids: List[str] = []
        self.labeled_targets: List[float] = []
        self._scores = np.full((len(task.ids),), 0.5, dtype=np.float32)

    def _fit(self) -> None:
        if len(self.labeled_ids) == 0:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        idx = np.asarray([self.task.id_to_index[image_id] for image_id in self.labeled_ids], dtype=np.int64)
        x_lab = self.features[idx]
        y_lab = np.asarray(self.labeled_targets, dtype=np.float32)
        sims = self.features @ x_lab.T
        k = max(1, min(int(self.spec.params.get('k', 15)), x_lab.shape[0]))
        top = np.argsort(sims, axis=1, kind='mergesort')[:, -k:]
        top_sims = np.take_along_axis(sims, top, axis=1)
        temp = float(self.spec.params.get('temperature', 12.0))
        weights = np.exp(temp * (top_sims - np.max(top_sims, axis=1, keepdims=True)))
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
        top_y = y_lab[top]
        self._scores = np.sum(weights * top_y, axis=1).astype(np.float32)

    def current_scores(self) -> np.ndarray:
        return np.asarray(self._scores, dtype=np.float32)

    def observe(self, image_id: str, target01: Optional[float], move_type: str = 'score') -> None:
        super().observe(image_id, target01, move_type)
        if move_type == 'undefined' or target01 is None:
            return
        self.labeled_ids.append(str(image_id))
        self.labeled_targets.append(float(target01))
        self._fit()


class EngineMethodSession(BaseMethodSession):
    def __init__(self, spec: MethodSpec, task: EvaluationTask, rng: np.random.Generator, context: 'EvaluationContext'):
        super().__init__(spec, task, rng)
        self.engine = context.get_engine(spec)
        payload = self.engine.create_axis(
            collection_id=task.collection_id,
            dataset_root=str(task.dataset_root),
            q=task.query,
            mode=spec.mode or AXIS_BAYES_MODE,
            model_type=spec.model_type or AXIS_MODEL_TYPE,
        )
        self.axis_id = str(payload['axis_id'])
        self.payload = payload

    def current_scores(self) -> np.ndarray:
        return np.asarray(self.payload.get('projection_values') or [], dtype=np.float32)

    def current_uncertainty(self) -> np.ndarray:
        return np.asarray(self.payload.get('std') or [], dtype=np.float32)

    def observe(self, image_id: str, target01: Optional[float], move_type: str = 'score') -> None:
        super().observe(image_id, target01, move_type)
        score100 = 50.0 if target01 is None else float(max(0.0, min(1.0, target01)) * 100.0)
        self.payload = self.engine.move_axis(
            axis_id=self.axis_id,
            image_id=str(image_id),
            new_score_0_100=score100,
            move_type=move_type,
        )


class EvaluationContext:
    def __init__(self, tasks: Sequence[EvaluationTask]):
        self.tasks = list(tasks)
        self.tasks_by_collection = {task.collection_id: task for task in self.tasks}
        self.prompt_overrides = {
            prompt_override_key(task.dataset_root.name, task.query): (list(task.pos_prompts), list(task.neg_prompts))
            for task in self.tasks
        }
        self._engine_cache: Dict[str, PreloadedAxisBayesEngine] = {}
        self._text_engine: Optional[PreloadedAxisBayesEngine] = None

    def get_text_engine(self) -> PreloadedAxisBayesEngine:
        if self._text_engine is None:
            self._text_engine = PreloadedAxisBayesEngine(
                tasks_by_collection=self.tasks_by_collection,
                prompt_overrides=self.prompt_overrides,
                model_type='bayes_linear',
                mode='gaussian',
                feature_space='clip',
                semantic_method='clip',
                norm=True,
                clip_weight=1.0,
                dino_weight=0.0,
                axis_bounds_text_source='template',
                use_llm_prompt_ensemble=False,
                max_moves=0,
            )
        return self._text_engine

    def get_engine(self, spec: MethodSpec) -> PreloadedAxisBayesEngine:
        key = compact_json(
            {
                'name': spec.name,
                'model_type': spec.model_type,
                'mode': spec.mode,
                'feature_space': spec.feature_space,
                'clip_weight': spec.clip_weight,
                'dino_weight': spec.dino_weight,
                'params': spec.params,
            },
        )
        cached = self._engine_cache.get(key)
        if cached is not None:
            return cached
        params = dict(spec.params)
        engine = PreloadedAxisBayesEngine(
            tasks_by_collection=self.tasks_by_collection,
            prompt_overrides=self.prompt_overrides,
            model_type=spec.model_type or AXIS_MODEL_TYPE,
            mode=spec.mode or AXIS_BAYES_MODE,
            feature_space=spec.feature_space,
            semantic_method='clip',
            norm=True,
            clip_weight=float(spec.clip_weight),
            dino_weight=float(spec.dino_weight),
            alpha=float(params.get('alpha', AXIS_BAYES_ALPHA)),
            dino_alpha=float(params.get('dino_alpha', AXIS_BAYES_DINO_ALPHA)),
            bias_alpha=float(params.get('bias_alpha', AXIS_BAYES_BIAS_ALPHA)),
            sigma2=float(params.get('sigma2', AXIS_BAYES_SIGMA2)),
            rank_eta=float(params.get('rank_eta', AXIS_BAYES_RANK_ETA)),
            rank_anchor_k=int(params.get('rank_anchor_k', AXIS_BAYES_RANK_ANCHOR_K)),
            rank_anchor_delta=float(params.get('rank_anchor_delta', AXIS_BAYES_RANK_ANCHOR_DELTA)),
            rank_max_pairs=int(params.get('rank_max_pairs', AXIS_BAYES_RANK_MAX_PAIRS)),
            residual_alpha=float(params.get('residual_alpha', AXIS_RESIDUAL_ALPHA)),
            residual_beta=float(params.get('residual_beta', AXIS_RESIDUAL_BETA)),
            residual_lambda=float(params.get('residual_lambda', AXIS_RESIDUAL_LAMBDA)),
            residual_sigma_y=float(params.get('residual_sigma_y', AXIS_RESIDUAL_SIGMA_Y)),
            residual_lengthscale_multiplier=float(
                params.get('residual_lengthscale_multiplier', AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER)
            ),
            residual_jitter=float(params.get('residual_jitter', AXIS_RESIDUAL_JITTER)),
            graph_knn_k=int(params.get('graph_knn_k', AXIS_BAYES_GRAPH_KNN_K)),
            graph_lambda_smooth=float(params.get('graph_lambda_smooth', AXIS_BAYES_GRAPH_LAMBDA_SMOOTH)),
            graph_lambda_prior=float(params.get('graph_lambda_prior', AXIS_BAYES_GRAPH_LAMBDA_PRIOR)),
            graph_jitter=float(params.get('graph_jitter', AXIS_BAYES_GRAPH_JITTER)),
            piecewise_num_experts=int(params.get('piecewise_num_experts', AXIS_PIECEWISE_NUM_EXPERTS)),
            piecewise_use_gating=bool(params.get('piecewise_use_gating', AXIS_PIECEWISE_USE_GATING)),
            piecewise_aggregator=str(params.get('piecewise_aggregator', AXIS_PIECEWISE_AGGREGATOR)),
            piecewise_clip_scale=float(params.get('piecewise_clip_scale', AXIS_PIECEWISE_CLIP_SCALE)),
            piecewise_dino_scale=float(params.get('piecewise_dino_scale', AXIS_PIECEWISE_DINO_SCALE)),
            pairwise_from_scalar_margin=float(params.get('pairwise_from_scalar_margin', AXIS_PIECEWISE_PAIRWISE_FROM_SCALAR_MARGIN)),
            piecewise_prior_strength=float(params.get('piecewise_prior_strength', AXIS_PIECEWISE_PRIOR_STRENGTH)),
            piecewise_expert_diversity_strength=float(params.get('piecewise_expert_diversity_strength', AXIS_PIECEWISE_EXPERT_DIVERSITY_STRENGTH)),
            piecewise_l2_reg=float(params.get('piecewise_l2_reg', AXIS_PIECEWISE_L2_REG)),
            piecewise_learning_rate=float(params.get('piecewise_learning_rate', AXIS_PIECEWISE_LEARNING_RATE)),
            piecewise_max_refine_steps=int(params.get('piecewise_max_refine_steps', AXIS_PIECEWISE_MAX_REFINE_STEPS)),
            axis_bounds_text_source='template',
            use_llm_prompt_ensemble=False,
            max_moves=0,
        )
        self._engine_cache[key] = engine
        return engine

    def create_session(self, spec: MethodSpec, task: EvaluationTask, seed: int) -> BaseMethodSession:
        rng = np.random.default_rng(seed)
        if spec.kind == 'clip_text':
            return ClipTextSession(spec, task, rng, self)
        if spec.kind == 'label_only_linear':
            return LabelOnlyLinearSession(spec, task, rng)
        if spec.kind == 'knn_propagation':
            return KNNPropagationSession(spec, task, rng)
        if spec.kind == 'engine':
            return EngineMethodSession(spec, task, rng, self)
        raise ValueError(f'Unknown method kind: {spec.kind}')


# =============================
# Evaluation helpers
# =============================
def compute_metrics(pred_scores: np.ndarray, ref_scores01: np.ndarray, *, rng: np.random.Generator) -> Dict[str, float]:
    pairwise = pairwise_order_metrics(pred_scores, ref_scores01, max_pairs=MAX_PAIRWISE_METRIC_PAIRS, rng=rng)
    return {
        'spearman': spearman_corr(pred_scores, ref_scores01),
        'kendall_tau': pairwise['kendall_tau'],
        'pairwise_acc': pairwise['pairwise_acc'],
        'topk_extreme_precision': topk_extreme_precision(pred_scores, ref_scores01, TOPK_FRAC),
        'auroc_binary': binary_auroc_if_binary(pred_scores, ref_scores01),
    }


def compute_metrics_with_mask(pred_scores: np.ndarray, ref_scores01: np.ndarray, mask: np.ndarray, *, rng: np.random.Generator) -> Dict[str, float]:
    valid = np.asarray(mask, dtype=bool)
    if int(np.sum(valid)) < 2:
        return {
            'spearman': float('nan'),
            'kendall_tau': float('nan'),
            'pairwise_acc': float('nan'),
            'topk_extreme_precision': float('nan'),
            'auroc_binary': float('nan'),
        }
    return compute_metrics(pred_scores[valid], ref_scores01[valid], rng=rng)


def select_query_index(policy: str, session: BaseMethodSession, task: EvaluationTask, rng: np.random.Generator) -> Optional[int]:
    queried = session.queried_mask()
    available = np.flatnonzero(~queried)
    if available.size == 0:
        return None
    if policy == 'random':
        return int(rng.choice(available))

    scores = np.asarray(session.current_scores(), dtype=np.float32)
    uncert = np.asarray(session.current_uncertainty(), dtype=np.float32)
    if uncert.size != len(task.ids) or not np.any(np.isfinite(uncert)):
        uncert = np.zeros((len(task.ids),), dtype=np.float32)
    uncert = np.where(np.isfinite(uncert), uncert, np.nanmin(np.where(np.isfinite(uncert), uncert, np.nan)) if np.any(np.isfinite(uncert)) else 0.0)
    uncert = uncert.astype(np.float32)

    feats = np.asarray(task.fused_default, dtype=np.float32)
    if np.any(queried):
        lab = feats[queried]
        sims = feats @ lab.T
        novelty = 1.0 - np.max(sims, axis=1)
    else:
        novelty = np.ones((len(task.ids),), dtype=np.float32)

    if policy == 'diversity':
        score = novelty
    elif policy == 'uncertainty':
        score = uncert
    elif policy == 'hybrid':
        score = (uncert - float(np.min(uncert))) / (float(np.max(uncert) - np.min(uncert)) + 1e-8)
        nov = (novelty - float(np.min(novelty))) / (float(np.max(novelty) - np.min(novelty)) + 1e-8)
        score = score * (0.5 + nov)
    else:
        raise ValueError(f'Unknown query policy: {policy}')
    score = np.asarray(score, dtype=np.float32)
    score[queried] = -np.inf
    best = int(np.argmax(score))
    return best if np.isfinite(score[best]) else None


def build_uncertainty_rows(task: EvaluationTask, method: MethodSpec, session: BaseMethodSession, interaction_count: int, variant: str, rng: np.random.Generator) -> List[Dict[str, Any]]:
    pred = np.asarray(session.current_scores(), dtype=np.float32)
    uncert = np.asarray(session.current_uncertainty(), dtype=np.float32)
    if uncert.size != len(task.ids) or not np.any(np.isfinite(uncert)):
        return []
    pred_rank = rank_percentile_01(pred)
    ref_rank = rank_percentile_01(task.reference_scores01)
    abs_error = np.abs(pred_rank - ref_rank)
    finite = np.isfinite(uncert)
    if int(np.sum(finite)) < 3:
        return []
    error_pos = (abs_error >= float(FUTURE_CORRECTION_ERROR_THRESHOLD)).astype(np.int32)
    rows: List[Dict[str, Any]] = [
        {
            'row_type': 'error_prediction',
            'dataset': task.dataset_name,
            'task_id': task.task_id,
            'concept_name': task.concept_name,
            'method': method.name,
            'variant': variant,
            'interaction_count': int(interaction_count),
            'coverage': '',
            'corr_uncert_abs_error': pearson_corr(uncert[finite], abs_error[finite]),
            'auroc_future_correction': auroc_binary(uncert[finite], error_pos[finite]),
            'spearman': '',
            'kendall_tau': '',
            'pairwise_acc': '',
            'topk_extreme_precision': '',
            'auroc_binary': '',
        },
    ]
    order = np.argsort(uncert[finite], kind='mergesort')
    finite_idx = np.flatnonzero(finite)[order]
    for coverage in COVERAGE_POINTS:
        keep_n = max(2, int(round(float(coverage) * len(finite_idx))))
        keep = finite_idx[:keep_n]
        metrics = compute_metrics(pred[keep], task.reference_scores01[keep], rng=rng)
        rows.append(
            {
                'row_type': 'coverage_risk',
                'dataset': task.dataset_name,
                'task_id': task.task_id,
                'concept_name': task.concept_name,
                'method': method.name,
                'variant': variant,
                'interaction_count': int(interaction_count),
                'coverage': float(coverage),
                'corr_uncert_abs_error': '',
                'auroc_future_correction': '',
                'spearman': metrics['spearman'],
                'kendall_tau': metrics['kendall_tau'],
                'pairwise_acc': metrics['pairwise_acc'],
                'topk_extreme_precision': metrics['topk_extreme_precision'],
                'auroc_binary': metrics['auroc_binary'],
            },
        )
    return rows


def append_representative_ranking_rows(
    logger: CSVAppender,
    *,
    run_id: str,
    task: EvaluationTask,
    method: MethodSpec,
    session: BaseMethodSession,
    policy: str,
    variant: str,
    interaction_count: int,
    example_indices: Sequence[int],
) -> None:
    sample_idx = np.asarray(list(example_indices), dtype=np.int64)
    if sample_idx.size == 0:
        return
    pred = np.asarray(session.current_scores(), dtype=np.float32)
    ref = np.asarray(task.reference_scores01, dtype=np.float32)
    pred_rank_global = descending_rank_1based(pred)
    ref_rank_global = descending_rank_1based(ref)
    pred_rank_sample = descending_rank_1based(pred[sample_idx])
    ref_rank_sample = descending_rank_1based(ref[sample_idx])
    sample_order = np.argsort(ref[sample_idx], kind='mergesort')
    ordered_sample_idx = sample_idx[sample_order]
    for slot, image_idx in enumerate(ordered_sample_idx.tolist(), start=1):
        sample_pos = int(np.where(sample_idx == int(image_idx))[0][0])
        logger.append(
            {
                'ts_utc': utc_now_iso(),
                'run_id': run_id,
                'variant': variant,
                'dataset': task.dataset_name,
                'task_id': task.task_id,
                'concept_name': task.concept_name,
                'method': method.name,
                'policy': policy,
                'interaction_count': int(interaction_count),
                'example_slot': int(slot),
                'image_id': str(task.ids[int(image_idx)]),
                'image_path': str(task.paths[int(image_idx)]),
                'reference_score': float(ref[int(image_idx)]),
                'reference_rank_global': int(ref_rank_global[int(image_idx)]),
                'reference_rank_sample': int(ref_rank_sample[sample_pos]),
                'pred_score': float(pred[int(image_idx)]),
                'pred_rank_global': int(pred_rank_global[int(image_idx)]),
                'pred_rank_sample': int(pred_rank_sample[sample_pos]),
                'is_undefined': bool(task.undefined_mask[int(image_idx)]),
            },
        )


def append_task_row(logger: CSVAppender, run_id: str, task: EvaluationTask) -> None:
    logger.append(
        {
            'ts_utc': utc_now_iso(),
            'run_id': run_id,
            'dataset': task.dataset_name,
            'task_id': task.task_id,
            'concept_name': task.concept_name,
            'concept_kind': task.concept_kind,
            'metadata_field': task.metadata_field,
            'metadata_value': task.metadata_value,
            'query': task.query,
            'image_count': int(len(task.ids)),
            'undefined_count': int(np.sum(task.undefined_mask)),
            'pos_prompts_json': compact_json(task.pos_prompts),
            'neg_prompts_json': compact_json(task.neg_prompts),
            'reference_meta_json': compact_json(task.reference_meta),
        },
    )


def evaluate_prior_quality(run_id: str, tasks: Sequence[EvaluationTask], methods: Sequence[MethodSpec], context: EvaluationContext, logger: CSVAppender) -> None:
    for task_idx, task in enumerate(tasks):
        for method in methods:
            if not method.supports_prior:
                logger.append(
                    {
                        'ts_utc': utc_now_iso(),
                        'run_id': run_id,
                        'dataset': task.dataset_name,
                        'task_id': task.task_id,
                        'concept_name': task.concept_name,
                        'method': method.name,
                        'spearman': '',
                        'kendall_tau': '',
                        'pairwise_acc': '',
                        'topk_extreme_precision': '',
                        'auroc_binary': '',
                        'image_count': int(len(task.ids)),
                        'undefined_count': int(np.sum(task.undefined_mask)),
                        'status': 'na_prior',
                    },
                )
                continue
            seed = GLOBAL_SEED + int(task_idx * 1000) + (stable_seed(method.name) % 997)
            session = context.create_session(method, task, seed)
            metrics = compute_metrics(np.asarray(session.current_scores(), dtype=np.float32), task.reference_scores01, rng=np.random.default_rng(seed))
            logger.append(
                {
                    'ts_utc': utc_now_iso(),
                    'run_id': run_id,
                    'dataset': task.dataset_name,
                    'task_id': task.task_id,
                    'concept_name': task.concept_name,
                    'method': method.name,
                    'spearman': metrics['spearman'],
                    'kendall_tau': metrics['kendall_tau'],
                    'pairwise_acc': metrics['pairwise_acc'],
                    'topk_extreme_precision': metrics['topk_extreme_precision'],
                    'auroc_binary': metrics['auroc_binary'],
                    'image_count': int(len(task.ids)),
                    'undefined_count': int(np.sum(task.undefined_mask)),
                    'status': 'ok',
                },
            )


def run_refinement_loop(
    *,
    run_id: str,
    task: EvaluationTask,
    method: MethodSpec,
    context: EvaluationContext,
    logger: CSVAppender,
    uncertainty_logger: Optional[CSVAppender],
    representative_logger: Optional[CSVAppender],
    budgets: Sequence[int],
    policy: str,
    allow_undefined: bool,
    row_prefix: str,
    representative_example_indices: Optional[Sequence[int]] = None,
) -> None:
    seed = GLOBAL_SEED + (stable_seed(task.task_id, method.name, policy, row_prefix) % 1_000_000)
    rng = np.random.default_rng(seed)
    session = context.create_session(method, task, seed)
    budget_set = set(int(v) for v in budgets)
    representative_budget_set = set(int(v) for v in REPRESENTATIVE_LOG_BUDGETS)
    logged_steps: List[int] = []
    logged_all: Dict[str, List[float]] = {
        'spearman': [],
        'kendall_tau': [],
        'pairwise_acc': [],
        'topk_extreme_precision': [],
        'auroc_binary': [],
    }
    logged_defined: Dict[str, List[float]] = {
        'spearman': [],
        'kendall_tau': [],
        'pairwise_acc': [],
        'topk_extreme_precision': [],
        'auroc_binary': [],
    }

    def emit_step(interaction_count: int, query_image_id: str = '', feedback_type: str = '') -> None:
        metrics_all = compute_metrics(np.asarray(session.current_scores(), dtype=np.float32), task.reference_scores01, rng=rng)
        metrics_defined = compute_metrics_with_mask(
            np.asarray(session.current_scores(), dtype=np.float32),
            task.reference_scores01,
            ~task.undefined_mask,
            rng=rng,
        )
        logger.append(
            {
                'ts_utc': utc_now_iso(),
                'run_id': run_id,
                'row_type': 'step',
                'variant': row_prefix,
                'dataset': task.dataset_name,
                'task_id': task.task_id,
                'concept_name': task.concept_name,
                'method': method.name,
                'policy': policy,
                'allow_undefined': bool(allow_undefined),
                'interaction_count': int(interaction_count),
                'query_image_id': query_image_id,
                'feedback_type': feedback_type,
                'numeric_label_count': int(len(session.queried_ids) - len(session.undefined_ids)),
                'undefined_feedback_count': int(len(session.undefined_ids)),
                'spearman_all': metrics_all['spearman'],
                'kendall_tau_all': metrics_all['kendall_tau'],
                'pairwise_acc_all': metrics_all['pairwise_acc'],
                'topk_extreme_precision_all': metrics_all['topk_extreme_precision'],
                'auroc_binary_all': metrics_all['auroc_binary'],
                'spearman_defined': metrics_defined['spearman'],
                'kendall_tau_defined': metrics_defined['kendall_tau'],
                'pairwise_acc_defined': metrics_defined['pairwise_acc'],
                'topk_extreme_precision_defined': metrics_defined['topk_extreme_precision'],
                'auroc_binary_defined': metrics_defined['auroc_binary'],
            },
        )
        logged_steps.append(int(interaction_count))
        for key in logged_all:
            logged_all[key].append(float(metrics_all[key]))
            logged_defined[key].append(float(metrics_defined[key]))
        if uncertainty_logger is not None and method.supports_uncertainty:
            for row in build_uncertainty_rows(task, method, session, interaction_count, row_prefix, rng):
                uncertainty_logger.append({'ts_utc': utc_now_iso(), 'run_id': run_id, **row})
        if (
            representative_logger is not None
            and representative_example_indices is not None
            and int(interaction_count) in representative_budget_set
        ):
            append_representative_ranking_rows(
                representative_logger,
                run_id=run_id,
                task=task,
                method=method,
                session=session,
                policy=policy,
                variant=row_prefix,
                interaction_count=interaction_count,
                example_indices=representative_example_indices,
            )

    if 0 in budget_set:
        emit_step(0)

    max_budget = max(int(v) for v in budgets) if budgets else 0
    if not method.supports_feedback or max_budget <= 0:
        logger.append(
            {
                'ts_utc': utc_now_iso(),
                'run_id': run_id,
                'row_type': 'summary',
                'variant': row_prefix,
                'dataset': task.dataset_name,
                'task_id': task.task_id,
                'concept_name': task.concept_name,
                'method': method.name,
                'policy': policy,
                'allow_undefined': bool(allow_undefined),
                'interaction_count': int(max_budget),
                'query_image_id': '',
                'feedback_type': '',
                'numeric_label_count': int(len(session.queried_ids) - len(session.undefined_ids)),
                'undefined_feedback_count': int(len(session.undefined_ids)),
                'spearman_all': safe_learning_curve_auc(logged_steps, logged_all['spearman']),
                'kendall_tau_all': safe_learning_curve_auc(logged_steps, logged_all['kendall_tau']),
                'pairwise_acc_all': safe_learning_curve_auc(logged_steps, logged_all['pairwise_acc']),
                'topk_extreme_precision_all': safe_learning_curve_auc(logged_steps, logged_all['topk_extreme_precision']),
                'auroc_binary_all': safe_learning_curve_auc(logged_steps, logged_all['auroc_binary']),
                'spearman_defined': safe_learning_curve_auc(logged_steps, logged_defined['spearman']),
                'kendall_tau_defined': safe_learning_curve_auc(logged_steps, logged_defined['kendall_tau']),
                'pairwise_acc_defined': safe_learning_curve_auc(logged_steps, logged_defined['pairwise_acc']),
                'topk_extreme_precision_defined': safe_learning_curve_auc(logged_steps, logged_defined['topk_extreme_precision']),
                'auroc_binary_defined': safe_learning_curve_auc(logged_steps, logged_defined['auroc_binary']),
            },
        )
        return

    for interaction_count in range(1, max_budget + 1):
        query_idx = select_query_index(policy, session, task, rng)
        if query_idx is None:
            break
        query_id = task.ids[int(query_idx)]
        if allow_undefined and method.supports_undefined and bool(task.undefined_mask[int(query_idx)]):
            feedback_type = 'undefined'
            session.observe(query_id, None, move_type='undefined')
        else:
            feedback_type = 'score'
            session.observe(query_id, float(task.reference_scores01[int(query_idx)]), move_type='score')
        if interaction_count in budget_set:
            emit_step(interaction_count, query_image_id=query_id, feedback_type=feedback_type)

    logger.append(
        {
            'ts_utc': utc_now_iso(),
            'run_id': run_id,
            'row_type': 'summary',
            'variant': row_prefix,
            'dataset': task.dataset_name,
            'task_id': task.task_id,
            'concept_name': task.concept_name,
            'method': method.name,
            'policy': policy,
            'allow_undefined': bool(allow_undefined),
            'interaction_count': int(max_budget),
            'query_image_id': '',
            'feedback_type': '',
            'numeric_label_count': int(len(session.queried_ids) - len(session.undefined_ids)),
            'undefined_feedback_count': int(len(session.undefined_ids)),
            'spearman_all': safe_learning_curve_auc(logged_steps, logged_all['spearman']),
            'kendall_tau_all': safe_learning_curve_auc(logged_steps, logged_all['kendall_tau']),
            'pairwise_acc_all': safe_learning_curve_auc(logged_steps, logged_all['pairwise_acc']),
            'topk_extreme_precision_all': safe_learning_curve_auc(logged_steps, logged_all['topk_extreme_precision']),
            'auroc_binary_all': safe_learning_curve_auc(logged_steps, logged_all['auroc_binary']),
            'spearman_defined': safe_learning_curve_auc(logged_steps, logged_defined['spearman']),
            'kendall_tau_defined': safe_learning_curve_auc(logged_steps, logged_defined['kendall_tau']),
            'pairwise_acc_defined': safe_learning_curve_auc(logged_steps, logged_defined['pairwise_acc']),
            'topk_extreme_precision_defined': safe_learning_curve_auc(logged_steps, logged_defined['topk_extreme_precision']),
            'auroc_binary_defined': safe_learning_curve_auc(logged_steps, logged_defined['auroc_binary']),
        },
    )


def evaluate_query_policies(run_id: str, tasks: Sequence[EvaluationTask], methods: Sequence[MethodSpec], context: EvaluationContext, logger: CSVAppender) -> None:
    for task in tasks:
        for method in methods:
            if not method.supports_feedback:
                continue
            for policy in QUERY_POLICIES:
                run_refinement_loop(
                    run_id=run_id,
                    task=task,
                    method=method,
                    context=context,
                    logger=logger,
                    uncertainty_logger=None,
                    representative_logger=None,
                    budgets=REFINEMENT_BUDGETS,
                    policy=policy,
                    allow_undefined=False,
                    row_prefix='query_policy',
                )


def evaluate_undefined(run_id: str, tasks: Sequence[EvaluationTask], methods: Sequence[MethodSpec], context: EvaluationContext, logger: CSVAppender) -> None:
    for task in tasks:
        if int(np.sum(task.undefined_mask)) == 0:
            continue
        for method in methods:
            if not method.supports_feedback:
                continue
            run_refinement_loop(
                run_id=run_id,
                task=task,
                method=method,
                context=context,
                logger=logger,
                uncertainty_logger=None,
                representative_logger=None,
                budgets=REFINEMENT_BUDGETS,
                policy=MAIN_QUERY_POLICY,
                allow_undefined=False,
                row_prefix='without_undefined',
            )
            if method.supports_undefined:
                run_refinement_loop(
                    run_id=run_id,
                    task=task,
                    method=method,
                    context=context,
                    logger=logger,
                    uncertainty_logger=None,
                    representative_logger=None,
                    budgets=REFINEMENT_BUDGETS,
                    policy=MAIN_QUERY_POLICY,
                    allow_undefined=True,
                    row_prefix='with_undefined',
                )


def evaluate_ablations(run_id: str, tasks: Sequence[EvaluationTask], methods: Sequence[MethodSpec], context: EvaluationContext, logger: CSVAppender) -> None:
    for task in tasks:
        for method in methods:
            run_refinement_loop(
                run_id=run_id,
                task=task,
                method=method,
                context=context,
                logger=logger,
                uncertainty_logger=None,
                representative_logger=None,
                budgets=REFINEMENT_BUDGETS,
                policy=MAIN_QUERY_POLICY,
                allow_undefined=False,
                row_prefix='ablation',
            )


# =============================
# Main
# =============================
def active_method_specs(names: Sequence[str], registry: Mapping[str, MethodSpec]) -> List[MethodSpec]:
    specs: List[MethodSpec] = []
    for name in names:
        if name not in registry:
            raise KeyError(f'Unknown method name: {name}')
        specs.append(registry[name])
    return specs


def apply_prompt_cache_to_tasks(tasks: Sequence[EvaluationTask], prompt_records: Mapping[str, Mapping[str, Any]]) -> int:
    applied = 0
    for task in tasks:
        record = prompt_records.get(task.task_id)
        if record is None:
            continue
        expected_dataset = normalize_dataset_name(task.dataset_root.name)
        cached_dataset = normalize_dataset_name(str(record.get('dataset_name') or ''))
        if cached_dataset and cached_dataset.lower() != expected_dataset.lower():
            raise RuntimeError(
                f'Prompt cache dataset mismatch for task_id={task.task_id}: '
                f'cached={cached_dataset!r} expected={expected_dataset!r}'
            )
        expected_query = normalize_query_text(task.query)
        cached_query = normalize_query_text(str(record.get('query') or ''))
        if cached_query and cached_query.lower() != expected_query.lower():
            raise RuntimeError(
                f'Prompt cache query mismatch for task_id={task.task_id}: '
                f'cached={cached_query!r} expected={expected_query!r}'
            )
        pos_prompts = [str(value).strip() for value in list(record.get('pos_prompts') or []) if str(value).strip()]
        neg_prompts = [str(value).strip() for value in list(record.get('neg_prompts') or []) if str(value).strip()]
        if len(pos_prompts) == 0 or len(neg_prompts) == 0:
            raise RuntimeError(f'Prompt cache record for task_id={task.task_id} is missing positive or negative prompts')
        task.pos_prompts = pos_prompts
        task.neg_prompts = neg_prompts
        applied += 1
    return applied


def load_active_tasks(*, require_dino: bool = False) -> List[EvaluationTask]:
    tasks: List[EvaluationTask] = []
    for dataset_name in ACTIVE_DATASETS:
        if dataset_name not in DATASET_REGISTRY:
            raise KeyError(f'Unknown dataset name: {dataset_name}')
        spec = DATASET_REGISTRY[dataset_name]
        if not spec.enabled:
            continue
        print(f'[modeling-eval] loading dataset={dataset_name} root={spec.dataset_root}')
        loaded = load_dataset(spec, require_dino=require_dino)
        dataset_tasks = build_tasks_for_dataset(loaded, spec)
        dataset_tasks = [task for task in dataset_tasks if task_is_active(task)]
        print(f'[modeling-eval] dataset={dataset_name} tasks={len(dataset_tasks)} images={len(loaded.ids)}')
        tasks.extend(dataset_tasks)
    return tasks


TABLE_FIELDNAMES: Dict[str, List[str]] = {
    'runs': [
        'ts_utc', 'run_id', 'run_name', 'note', 'active_datasets_json', 'active_methods_json',
        'method_specs_json', 'active_ablations_json', 'ablation_specs_json', 'budgets_json',
        'main_policy', 'query_policies_json', 'representative_task_ids_json',
        'representative_images_per_task',
    ],
    'tasks': [
        'ts_utc', 'run_id', 'dataset', 'task_id', 'concept_name', 'concept_kind', 'metadata_field',
        'metadata_value', 'query', 'image_count', 'undefined_count', 'pos_prompts_json',
        'neg_prompts_json', 'reference_meta_json',
    ],
    'prior': [
        'ts_utc', 'run_id', 'dataset', 'task_id', 'concept_name', 'method', 'spearman',
        'kendall_tau', 'pairwise_acc', 'topk_extreme_precision', 'auroc_binary',
        'image_count', 'undefined_count', 'status',
    ],
    'refinement': [
        'ts_utc', 'run_id', 'row_type', 'variant', 'dataset', 'task_id', 'concept_name', 'method',
        'policy', 'allow_undefined', 'interaction_count', 'query_image_id', 'feedback_type',
        'numeric_label_count', 'undefined_feedback_count', 'spearman_all', 'kendall_tau_all',
        'pairwise_acc_all', 'topk_extreme_precision_all', 'auroc_binary_all',
        'spearman_defined', 'kendall_tau_defined', 'pairwise_acc_defined',
        'topk_extreme_precision_defined', 'auroc_binary_defined',
    ],
    'uncertainty': [
        'ts_utc', 'run_id', 'row_type', 'dataset', 'task_id', 'concept_name', 'method', 'variant',
        'interaction_count', 'coverage', 'corr_uncert_abs_error', 'auroc_future_correction',
        'spearman', 'kendall_tau', 'pairwise_acc', 'topk_extreme_precision', 'auroc_binary',
    ],
    'query_policy': [
        'ts_utc', 'run_id', 'row_type', 'variant', 'dataset', 'task_id', 'concept_name', 'method',
        'policy', 'allow_undefined', 'interaction_count', 'query_image_id', 'feedback_type',
        'numeric_label_count', 'undefined_feedback_count', 'spearman_all', 'kendall_tau_all',
        'pairwise_acc_all', 'topk_extreme_precision_all', 'auroc_binary_all',
        'spearman_defined', 'kendall_tau_defined', 'pairwise_acc_defined',
        'topk_extreme_precision_defined', 'auroc_binary_defined',
    ],
    'undefined': [
        'ts_utc', 'run_id', 'row_type', 'variant', 'dataset', 'task_id', 'concept_name', 'method',
        'policy', 'allow_undefined', 'interaction_count', 'query_image_id', 'feedback_type',
        'numeric_label_count', 'undefined_feedback_count', 'spearman_all', 'kendall_tau_all',
        'pairwise_acc_all', 'topk_extreme_precision_all', 'auroc_binary_all',
        'spearman_defined', 'kendall_tau_defined', 'pairwise_acc_defined',
        'topk_extreme_precision_defined', 'auroc_binary_defined',
    ],
    'ablation': [
        'ts_utc', 'run_id', 'row_type', 'variant', 'dataset', 'task_id', 'concept_name', 'method',
        'policy', 'allow_undefined', 'interaction_count', 'query_image_id', 'feedback_type',
        'numeric_label_count', 'undefined_feedback_count', 'spearman_all', 'kendall_tau_all',
        'pairwise_acc_all', 'topk_extreme_precision_all', 'auroc_binary_all',
        'spearman_defined', 'kendall_tau_defined', 'pairwise_acc_defined',
        'topk_extreme_precision_defined', 'auroc_binary_defined',
    ],
    'representative_rankings': [
        'ts_utc', 'run_id', 'variant', 'dataset', 'task_id', 'concept_name', 'method', 'policy',
        'interaction_count', 'example_slot', 'image_id', 'image_path', 'reference_score',
        'reference_rank_global', 'reference_rank_sample', 'pred_score', 'pred_rank_global',
        'pred_rank_sample', 'is_undefined',
    ],
}


def build_loggers(output_dir: Path = OUTPUT_DIR) -> Dict[str, CSVAppender]:
    root = Path(output_dir)
    return {
        name: CSVAppender(root / f'{name}.csv', fieldnames)
        for name, fieldnames in TABLE_FIELDNAMES.items()
    }


def _load_run_frame(csv_name: str, run_id: str, *, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    path = Path(output_dir) / f'{csv_name}.csv'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'run_id' in df.columns:
        df = df[df['run_id'].astype(str) == str(run_id)].copy()
    return df


def latest_modeling_run_id(output_dir: Path = OUTPUT_DIR) -> str:
    runs_path = Path(output_dir) / 'runs.csv'
    if not runs_path.exists():
        raise FileNotFoundError(f'No runs.csv found at {runs_path}')
    runs = pd.read_csv(runs_path)
    if runs.empty:
        raise RuntimeError('runs.csv is empty')
    runs = runs.sort_values('ts_utc').reset_index(drop=True)
    return str(runs.iloc[-1]['run_id'])


def export_modeling_eval_figures(run_id: str, *, output_dir: Path = OUTPUT_DIR) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    figure_dir = root / 'figures' / str(run_id)
    figure_dir.mkdir(parents=True, exist_ok=True)

    prior = _load_run_frame('prior', run_id, output_dir=root)
    refinement = _load_run_frame('refinement', run_id, output_dir=root)
    query_policy = _load_run_frame('query_policy', run_id, output_dir=root)
    uncertainty = _load_run_frame('uncertainty', run_id, output_dir=root)
    undefined = _load_run_frame('undefined', run_id, output_dir=root)
    ablation = _load_run_frame('ablation', run_id, output_dir=root)
    representative = _load_run_frame('representative_rankings', run_id, output_dir=root)
    tasks = _load_run_frame('tasks', run_id, output_dir=root)

    def _save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(figure_dir / f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(figure_dir / f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)

    def _save_contact_sheet(df: pd.DataFrame, name: str) -> None:
        if df.empty:
            return
        cell_w = 136
        cell_h = 136
        label_h = 24
        cols = min(5, max(1, int(df.shape[0])))
        rows = int(math.ceil(df.shape[0] / float(cols)))
        canvas = Image.new('RGB', (cols * cell_w, rows * (cell_h + label_h)), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for row_idx, (_, item) in enumerate(df.iterrows()):
            x0 = int((row_idx % cols) * cell_w)
            y0 = int((row_idx // cols) * (cell_h + label_h))
            thumb = Image.new('RGB', (cell_w, cell_h), color=(245, 245, 245))
            try:
                with Image.open(str(item['image_path'])) as img:
                    img = img.convert('RGB')
                    img.thumbnail((cell_w - 8, cell_h - 8))
                    px = int((cell_w - img.width) * 0.5)
                    py = int((cell_h - img.height) * 0.5)
                    thumb.paste(img, (px, py))
            except (OSError, ValueError) as exc:
                print(f'[modeling-eval] thumbnail unavailable path={item["image_path"]}: {exc}')
            canvas.paste(thumb, (x0, y0))
            caption = f"{int(item['example_slot'])}: {float(item['reference_score']):.2f}"
            draw.text((x0 + 4, y0 + cell_h + 4), caption, fill=(20, 20, 20))
        canvas.save(figure_dir / f'{name}.png')
        canvas.save(figure_dir / f'{name}.pdf', 'PDF', resolution=180.0)

    def _plot_dataset_curves(frame: pd.DataFrame, metric_col: str, *, title: str, ylabel: str, file_name: str) -> None:
        metric_frame = frame.dropna(subset=[metric_col]).copy()
        if metric_frame.empty:
            return
        datasets = sorted(metric_frame['dataset'].dropna().astype(str).unique().tolist())
        if not datasets:
            return
        n_cols = min(2, len(datasets))
        n_rows = int(math.ceil(len(datasets) / max(1, n_cols)))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.2 * n_rows), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for idx, dataset_name in enumerate(datasets):
            ax = axes[idx // n_cols][idx % n_cols]
            ax.set_visible(True)
            sub = metric_frame[metric_frame['dataset'].astype(str) == dataset_name]
            curve = (
                sub.groupby(['method', 'interaction_count'], dropna=False)[metric_col]
                .mean()
                .reset_index()
            )
            for method, grp in curve.groupby('method', dropna=False):
                grp = grp.sort_values('interaction_count')
                ax.plot(grp['interaction_count'], grp[metric_col], marker='o', label=str(method))
            ax.set_title(str(dataset_name))
            ax.set_xlabel('Interaction count')
            ax.set_ylabel(ylabel)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=min(4, len(labels)), fontsize=8)
        _save(fig, file_name)

    def _plot_grouped_dataset_bars(frame: pd.DataFrame, metric_cols: Sequence[Tuple[str, str]], *, title_prefix: str, file_name: str) -> None:
        available = [(metric, label) for metric, label in metric_cols if metric in frame.columns and frame[metric].notna().any()]
        if frame.empty or not available:
            return
        n_cols = len(available)
        fig, axes = plt.subplots(1, n_cols, figsize=(5.6 * n_cols, 4.8), squeeze=False)
        axes_flat = axes[0]
        for ax, (metric, label) in zip(axes_flat, available):
            pivot = (
                frame.dropna(subset=[metric])
                .groupby(['dataset', 'method'], dropna=False)[metric]
                .mean()
                .unstack('method')
                .sort_index()
            )
            if pivot.empty:
                ax.set_visible(False)
                continue
            pivot.plot(kind='bar', ax=ax)
            ax.set_title(f'{title_prefix}: {label}')
            ax.set_xlabel('Dataset')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=0)
            ax.legend(title='Method', fontsize=8, title_fontsize=9)
        _save(fig, file_name)

    breakdown_frames: List[pd.DataFrame] = []

    if not tasks.empty:
        task_summary = (
            tasks.groupby(['dataset', 'concept_kind'], dropna=False)
            .agg(task_count=('task_id', 'nunique'))
            .reset_index()
        )
        pivot = task_summary.pivot(index='dataset', columns='concept_kind', values='task_count').fillna(0.0)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pivot.plot(kind='bar', ax=ax)
        ax.set_title('Task Count By Dataset And Concept Type')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Task count')
        ax.legend(title='Concept type', fontsize=8, title_fontsize=9)
        _save(fig, 'task_count_by_dataset_and_concept')

    if not prior.empty:
        for col in ['spearman', 'kendall_tau', 'pairwise_acc', 'topk_extreme_precision', 'auroc_binary']:
            prior[col] = pd.to_numeric(prior[col], errors='coerce')
        prior_ok = prior[prior['status'].astype(str) == 'ok'].copy()
        if not prior_ok.empty:
            prior_mean = prior_ok.groupby('method', dropna=False)['spearman'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            prior_mean.plot(kind='bar', ax=ax, color='#2563eb')
            ax.set_title('Prior Mean Spearman By Method')
            ax.set_xlabel('Method')
            ax.set_ylabel('Mean Spearman')
            ax.tick_params(axis='x', rotation=35)
            _save(fig, 'prior_mean_spearman_by_method')

            prior_dataset = (
                prior_ok.groupby(['dataset', 'method'], dropna=False)['spearman']
                .mean()
                .unstack('method')
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(10, 4.8))
            prior_dataset.plot(kind='bar', ax=ax)
            ax.set_title('Prior Spearman By Dataset')
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Mean Spearman')
            ax.tick_params(axis='x', rotation=0)
            ax.legend(title='Method', fontsize=8, title_fontsize=9)
            _save(fig, 'prior_spearman_by_dataset')

            prior_auc = prior_ok.dropna(subset=['auroc_binary']).copy()
            if not prior_auc.empty:
                auc_mean = prior_auc.groupby('method', dropna=False)['auroc_binary'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(9, 4.5))
                auc_mean.plot(kind='bar', ax=ax, color='#059669')
                ax.set_title('Prior Mean Binary AUROC By Method')
                ax.set_xlabel('Method')
                ax.set_ylabel('Mean AUROC')
                ax.tick_params(axis='x', rotation=35)
                _save(fig, 'prior_mean_binary_auroc_by_method')

                auc_dataset = (
                    prior_auc.groupby(['dataset', 'method'], dropna=False)['auroc_binary']
                    .mean()
                    .unstack('method')
                    .sort_index()
                )
                fig, ax = plt.subplots(figsize=(10, 4.8))
                auc_dataset.plot(kind='bar', ax=ax)
                ax.set_title('Prior Binary AUROC By Dataset')
                ax.set_xlabel('Dataset')
                ax.set_ylabel('Mean AUROC')
                ax.tick_params(axis='x', rotation=0)
                ax.legend(title='Method', fontsize=8, title_fontsize=9)
                _save(fig, 'prior_binary_auroc_by_dataset')

            for metric in ['spearman', 'kendall_tau', 'pairwise_acc', 'topk_extreme_precision', 'auroc_binary']:
                metric_frame = prior_ok.dropna(subset=[metric]).copy()
                if metric_frame.empty:
                    continue
                grouped = (
                    metric_frame.groupby(['dataset', 'method'], dropna=False)[metric]
                    .mean()
                    .reset_index()
                )
                grouped['stage'] = 'prior'
                grouped['metric'] = metric
                grouped['interaction_count'] = np.nan
                grouped['variant'] = 'prior'
                breakdown_frames.append(grouped[['stage', 'variant', 'dataset', 'method', 'interaction_count', 'metric', metric]].rename(columns={metric: 'value'}))

    if not refinement.empty:
        steps = refinement[
            (refinement['row_type'].astype(str) == 'step')
            & (refinement['variant'].astype(str) == 'main')
        ].copy()
        for col in [
            'interaction_count',
            'spearman_all', 'spearman_defined',
            'pairwise_acc_all', 'pairwise_acc_defined',
            'auroc_binary_all', 'auroc_binary_defined',
        ]:
            steps[col] = pd.to_numeric(steps[col], errors='coerce')
        if not steps.empty:
            steps['metric'] = np.where(steps['spearman_defined'].notna(), steps['spearman_defined'], steps['spearman_all'])
            steps['pairwise_metric'] = np.where(steps['pairwise_acc_defined'].notna(), steps['pairwise_acc_defined'], steps['pairwise_acc_all'])
            steps['auroc_metric'] = np.where(steps['auroc_binary_defined'].notna(), steps['auroc_binary_defined'], steps['auroc_binary_all'])
            overall = (
                steps.groupby(['method', 'interaction_count'], dropna=False)['metric']
                .mean()
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(9, 4.8))
            for method, grp in overall.groupby('method', dropna=False):
                grp = grp.sort_values('interaction_count')
                ax.plot(grp['interaction_count'], grp['metric'], marker='o', label=str(method))
            ax.set_title('Refinement Learning Curves')
            ax.set_xlabel('Interaction count')
            ax.set_ylabel('Mean Spearman')
            ax.legend(fontsize=8, ncol=2)
            _save(fig, 'refinement_learning_curves')

            pairwise_overall = (
                steps.groupby(['method', 'interaction_count'], dropna=False)['pairwise_metric']
                .mean()
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(9, 4.8))
            for method, grp in pairwise_overall.groupby('method', dropna=False):
                grp = grp.sort_values('interaction_count')
                ax.plot(grp['interaction_count'], grp['pairwise_metric'], marker='o', label=str(method))
            ax.set_title('Refinement Pairwise Accuracy Curves')
            ax.set_xlabel('Interaction count')
            ax.set_ylabel('Mean pairwise accuracy')
            ax.legend(fontsize=8, ncol=2)
            _save(fig, 'refinement_pairwise_accuracy_curves')

            auroc_steps = steps.dropna(subset=['auroc_metric']).copy()
            if not auroc_steps.empty:
                auroc_overall = (
                    auroc_steps.groupby(['method', 'interaction_count'], dropna=False)['auroc_metric']
                    .mean()
                    .reset_index()
                )
                fig, ax = plt.subplots(figsize=(9, 4.8))
                for method, grp in auroc_overall.groupby('method', dropna=False):
                    grp = grp.sort_values('interaction_count')
                    ax.plot(grp['interaction_count'], grp['auroc_metric'], marker='o', label=str(method))
                ax.set_title('Refinement Binary AUROC Curves')
                ax.set_xlabel('Interaction count')
                ax.set_ylabel('Mean AUROC')
                ax.legend(fontsize=8, ncol=2)
                _save(fig, 'refinement_binary_auroc_curves')
            _plot_dataset_curves(
                steps,
                'metric',
                title='Refinement Learning Curves By Dataset',
                ylabel='Mean Spearman',
                file_name='refinement_learning_curves_by_dataset',
            )
            _plot_dataset_curves(
                steps,
                'pairwise_metric',
                title='Refinement Pairwise Accuracy Curves By Dataset',
                ylabel='Mean pairwise accuracy',
                file_name='refinement_pairwise_accuracy_curves_by_dataset',
            )
            _plot_dataset_curves(
                steps,
                'auroc_metric',
                title='Refinement Binary AUROC Curves By Dataset',
                ylabel='Mean AUROC',
                file_name='refinement_binary_auroc_curves_by_dataset',
            )

            final_budget = int(pd.to_numeric(steps['interaction_count'], errors='coerce').max())
            final_steps = steps[steps['interaction_count'] == final_budget].copy()
            _plot_grouped_dataset_bars(
                final_steps,
                [
                    ('metric', 'Mean Spearman'),
                    ('pairwise_metric', 'Mean pairwise accuracy'),
                    ('auroc_metric', 'Mean AUROC'),
                ],
                title_prefix=f'Final Budget {final_budget} By Dataset',
                file_name='refinement_final_metrics_by_dataset',
            )

            for metric in ['metric', 'pairwise_metric', 'auroc_metric']:
                metric_frame = final_steps.dropna(subset=[metric]).copy()
                if metric_frame.empty:
                    continue
                grouped = (
                    metric_frame.groupby(['dataset', 'method'], dropna=False)[metric]
                    .mean()
                    .reset_index()
                )
                grouped['stage'] = 'refinement_final'
                grouped['variant'] = 'main'
                grouped['interaction_count'] = float(final_budget)
                grouped['metric_name'] = metric
                breakdown_frames.append(
                    grouped[['stage', 'variant', 'dataset', 'method', 'interaction_count', 'metric_name', metric]]
                    .rename(columns={metric: 'value'})
                )

            summaries = refinement[
                (refinement['row_type'].astype(str) == 'summary')
                & (refinement['variant'].astype(str) == 'main')
            ].copy()
            for col in ['spearman_all', 'spearman_defined', 'pairwise_acc_all', 'pairwise_acc_defined', 'auroc_binary_all', 'auroc_binary_defined']:
                summaries[col] = pd.to_numeric(summaries[col], errors='coerce')
            if not summaries.empty:
                summaries['metric'] = np.where(summaries['spearman_defined'].notna(), summaries['spearman_defined'], summaries['spearman_all'])
                summaries['pairwise_metric'] = np.where(summaries['pairwise_acc_defined'].notna(), summaries['pairwise_acc_defined'], summaries['pairwise_acc_all'])
                summaries['auroc_metric'] = np.where(summaries['auroc_binary_defined'].notna(), summaries['auroc_binary_defined'], summaries['auroc_binary_all'])
                for metric in ['metric', 'pairwise_metric', 'auroc_metric']:
                    metric_frame = summaries.dropna(subset=[metric]).copy()
                    if metric_frame.empty:
                        continue
                    grouped = (
                        metric_frame.groupby(['dataset', 'method'], dropna=False)[metric]
                        .mean()
                        .reset_index()
                    )
                    grouped['stage'] = 'refinement_aulc'
                    grouped['variant'] = 'main'
                    grouped['interaction_count'] = np.nan
                    grouped['metric_name'] = metric
                    breakdown_frames.append(
                        grouped[['stage', 'variant', 'dataset', 'method', 'interaction_count', 'metric_name', metric]]
                        .rename(columns={metric: 'value'})
                    )

    if not query_policy.empty:
        query_summary = query_policy[query_policy['row_type'].astype(str) == 'summary'].copy()
        query_steps = query_policy[query_policy['row_type'].astype(str) == 'step'].copy()
        for col in ['spearman_all', 'spearman_defined', 'interaction_count']:
            if col in query_summary.columns:
                query_summary[col] = pd.to_numeric(query_summary[col], errors='coerce')
            if col in query_steps.columns:
                query_steps[col] = pd.to_numeric(query_steps[col], errors='coerce')
        for col in ['auroc_binary_all', 'auroc_binary_defined']:
            if col in query_summary.columns:
                query_summary[col] = pd.to_numeric(query_summary[col], errors='coerce')
            if col in query_steps.columns:
                query_steps[col] = pd.to_numeric(query_steps[col], errors='coerce')
        if not query_summary.empty:
            query_summary['aulc_spearman'] = pd.to_numeric(query_summary['spearman_all'], errors='coerce')
            heat = (
                query_summary.groupby(['method', 'policy'], dropna=False)['aulc_spearman']
                .mean()
                .unstack('policy')
            )
            fig, ax = plt.subplots(figsize=(9, max(3.8, 0.45 * max(1, len(heat.index)))))
            im = ax.imshow(heat.fillna(np.nan).to_numpy(), aspect='auto', cmap='Blues')
            ax.set_title('Query Policy Comparison By AULC')
            ax.set_xticks(range(len(heat.columns)))
            ax.set_xticklabels(list(heat.columns), rotation=20, ha='right')
            ax.set_yticks(range(len(heat.index)))
            ax.set_yticklabels(list(heat.index))
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            _save(fig, 'query_policy_aulc_heatmap')

        if not query_steps.empty:
            max_budget = int(pd.to_numeric(query_steps['interaction_count'], errors='coerce').max())
            final = query_steps[query_steps['interaction_count'] == max_budget].copy()
            final['metric'] = np.where(final['spearman_defined'].notna(), final['spearman_defined'], final['spearman_all'])
            heat = (
                final.groupby(['method', 'policy'], dropna=False)['metric']
                .mean()
                .unstack('policy')
            )
            fig, ax = plt.subplots(figsize=(9, max(3.8, 0.45 * max(1, len(heat.index)))))
            im = ax.imshow(heat.fillna(np.nan).to_numpy(), aspect='auto', cmap='Blues')
            ax.set_title(f'Query Policy Final Spearman At Budget {max_budget}')
            ax.set_xticks(range(len(heat.columns)))
            ax.set_xticklabels(list(heat.columns), rotation=20, ha='right')
            ax.set_yticks(range(len(heat.index)))
            ax.set_yticklabels(list(heat.index))
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            _save(fig, 'query_policy_final_budget_heatmap')

            bayes_policy_steps = query_steps[
                (query_steps['method'].astype(str) == 'request_bayes_linear_gaussian')
                & (query_steps['policy'].astype(str).isin(['random', 'uncertainty', 'diversity']))
            ].copy()
            bayes_policy_steps = bayes_policy_steps.dropna(subset=['auroc_binary_all'])
            if not bayes_policy_steps.empty:
                auroc_curves = (
                    bayes_policy_steps.groupby(['policy', 'interaction_count'], dropna=False)['auroc_binary_all']
                    .agg(mean_auroc='mean', std_auroc='std', sample_count='count')
                    .reset_index()
                )
                auroc_curves['std_auroc'] = pd.to_numeric(auroc_curves['std_auroc'], errors='coerce').fillna(0.0)
                auroc_curves['sample_count'] = pd.to_numeric(auroc_curves['sample_count'], errors='coerce').fillna(0.0)
                auroc_curves['sem_auroc'] = auroc_curves['std_auroc'] / np.sqrt(np.clip(auroc_curves['sample_count'].to_numpy(dtype=float), 1.0, None))
                policy_colors = {
                    'random': '#64748b',
                    'uncertainty': '#dc2626',
                    'diversity': '#059669',
                }
                fig, ax = plt.subplots(figsize=(8.6, 4.8))
                for policy_name, grp in auroc_curves.groupby('policy', dropna=False):
                    grp = grp.sort_values('interaction_count')
                    color = policy_colors.get(str(policy_name), None)
                    xs = grp['interaction_count'].to_numpy(dtype=float)
                    mean_vals = grp['mean_auroc'].to_numpy(dtype=float)
                    sem_vals = grp['sem_auroc'].to_numpy(dtype=float)
                    ax.plot(xs, mean_vals, marker='o', label=str(policy_name), color=color)
                    ax.fill_between(xs, mean_vals - sem_vals, mean_vals + sem_vals, alpha=0.18, color=color)
                ax.set_title('Bayes Linear AUROC By Query Policy')
                ax.set_xlabel('Interaction count')
                ax.set_ylabel('Mean AUROC')
                ax.legend(title='Policy', fontsize=8, title_fontsize=9)
                _save(fig, 'query_policy_bayes_linear_auroc_curves')

    if not uncertainty.empty:
        err = uncertainty[uncertainty['row_type'].astype(str) == 'error_prediction'].copy()
        cov = uncertainty[uncertainty['row_type'].astype(str) == 'coverage_risk'].copy()
        for col in ['interaction_count', 'corr_uncert_abs_error', 'auroc_future_correction', 'coverage', 'spearman']:
            if col in err.columns:
                err[col] = pd.to_numeric(err[col], errors='coerce')
            if col in cov.columns:
                cov[col] = pd.to_numeric(cov[col], errors='coerce')
        if not err.empty:
            corr_curve = (
                err.groupby(['method', 'interaction_count'], dropna=False)['corr_uncert_abs_error']
                .mean()
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for method, grp in corr_curve.groupby('method', dropna=False):
                grp = grp.sort_values('interaction_count')
                ax.plot(grp['interaction_count'], grp['corr_uncert_abs_error'], marker='o', label=str(method))
            ax.set_title('Uncertainty Versus Absolute Error Correlation')
            ax.set_xlabel('Interaction count')
            ax.set_ylabel('Mean correlation')
            ax.legend(fontsize=8, ncol=2)
            _save(fig, 'uncertainty_error_correlation')

            auroc_curve = (
                err.groupby(['method', 'interaction_count'], dropna=False)['auroc_future_correction']
                .mean()
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for method, grp in auroc_curve.groupby('method', dropna=False):
                grp = grp.sort_values('interaction_count')
                ax.plot(grp['interaction_count'], grp['auroc_future_correction'], marker='o', label=str(method))
            ax.set_title('Uncertainty AUROC For Future Correction')
            ax.set_xlabel('Interaction count')
            ax.set_ylabel('Mean AUROC')
            ax.legend(fontsize=8, ncol=2)
            _save(fig, 'uncertainty_future_correction_auroc')

        if not cov.empty:
            latest_step = int(pd.to_numeric(cov['interaction_count'], errors='coerce').max())
            cov_latest = cov[cov['interaction_count'] == latest_step].copy()
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for method, grp in cov_latest.groupby('method', dropna=False):
                grp = grp.sort_values('coverage')
                ax.plot(grp['coverage'], grp['spearman'], marker='o', label=str(method))
            ax.set_title(f'Coverage-Risk Curves At Interaction {latest_step}')
            ax.set_xlabel('Coverage')
            ax.set_ylabel('Mean Spearman')
            ax.legend(fontsize=8, ncol=2)
            _save(fig, 'coverage_risk_curves')

    if not undefined.empty:
        undef_summary = undefined[undefined['row_type'].astype(str) == 'summary'].copy()
        undef_steps = undefined[undefined['row_type'].astype(str) == 'step'].copy()
        for col in ['spearman_all', 'spearman_defined', 'interaction_count']:
            if col in undef_summary.columns:
                undef_summary[col] = pd.to_numeric(undef_summary[col], errors='coerce')
            if col in undef_steps.columns:
                undef_steps[col] = pd.to_numeric(undef_steps[col], errors='coerce')
        if not undef_summary.empty:
            undef_summary['aulc_spearman'] = pd.to_numeric(undef_summary['spearman_all'], errors='coerce')
            bar = (
                undef_summary.groupby(['method', 'variant'], dropna=False)['aulc_spearman']
                .mean()
                .unstack('variant')
            )
            fig, ax = plt.subplots(figsize=(10, 4.8))
            bar.plot(kind='bar', ax=ax)
            ax.set_title('Undefined Supervision AULC Comparison')
            ax.set_xlabel('Method')
            ax.set_ylabel('Mean AULC Spearman')
            ax.tick_params(axis='x', rotation=35)
            ax.legend(title='Variant', fontsize=8, title_fontsize=9)
            _save(fig, 'undefined_aulc_comparison')

        if not undef_steps.empty:
            undef_steps['metric'] = np.where(undef_steps['spearman_defined'].notna(), undef_steps['spearman_defined'], undef_steps['spearman_all'])
            fig, ax = plt.subplots(figsize=(10, 5.0))
            grouped = (
                undef_steps.groupby(['variant', 'method', 'interaction_count'], dropna=False)['metric']
                .mean()
                .reset_index()
            )
            for (variant, method), grp in grouped.groupby(['variant', 'method'], dropna=False):
                grp = grp.sort_values('interaction_count')
                linestyle = '--' if str(variant) == 'with_undefined' else '-'
                ax.plot(grp['interaction_count'], grp['metric'], marker='o', linestyle=linestyle, label=f'{method} | {variant}')
            ax.set_title('Undefined Supervision Learning Curves')
            ax.set_xlabel('Interaction count')
            ax.set_ylabel('Mean Spearman')
            ax.legend(fontsize=7, ncol=2)
            _save(fig, 'undefined_learning_curves')

    if not ablation.empty:
        abl_summary = ablation[ablation['row_type'].astype(str) == 'summary'].copy()
        if not abl_summary.empty:
            abl_summary['aulc_spearman'] = pd.to_numeric(abl_summary['spearman_all'], errors='coerce')
            mean_by_method = (
                abl_summary.groupby('method', dropna=False)['aulc_spearman']
                .mean()
                .sort_values(ascending=False)
            )
            fig, ax = plt.subplots(figsize=(9, 4.5))
            mean_by_method.plot(kind='bar', ax=ax, color='#2563eb')
            ax.set_title('Ablation AULC Summary')
            ax.set_xlabel('Method')
            ax.set_ylabel('Mean AULC Spearman')
            ax.tick_params(axis='x', rotation=35)
            _save(fig, 'ablation_aulc_summary')

            by_dataset = (
                abl_summary.groupby(['dataset', 'method'], dropna=False)['aulc_spearman']
                .mean()
                .unstack('method')
            )
            fig, ax = plt.subplots(figsize=(10, 4.8))
            by_dataset.plot(kind='bar', ax=ax)
            ax.set_title('Ablation AULC By Dataset')
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Mean AULC Spearman')
            ax.tick_params(axis='x', rotation=0)
            ax.legend(title='Method', fontsize=8, title_fontsize=9)
            _save(fig, 'ablation_aulc_by_dataset')

    if not representative.empty:
        for col in ['interaction_count', 'example_slot', 'reference_score', 'pred_rank_sample']:
            representative[col] = pd.to_numeric(representative[col], errors='coerce')
        rep = representative[representative['variant'].astype(str) == 'main'].copy()
        if not rep.empty:
            methods_in_order = list(dict.fromkeys(rep['method'].astype(str).tolist()))
            final_step = int(pd.to_numeric(rep['interaction_count'], errors='coerce').max())
            final_rep = rep[rep['interaction_count'] == final_step].copy()
            for (dataset_name, task_id, concept_name), grp in final_rep.groupby(['dataset', 'task_id', 'concept_name'], dropna=False):
                grp = grp.copy()
                sheet_rows = (
                    grp.sort_values(['example_slot', 'method'], kind='mergesort')
                    .drop_duplicates(subset=['example_slot'], keep='first')
                    [['example_slot', 'image_path', 'reference_score']]
                )
                safe_slug = slugify(f'{dataset_name}_{task_id}')
                _save_contact_sheet(sheet_rows, f'representative_scale_{safe_slug}')

                fig, ax = plt.subplots(figsize=(10, 6.5))
                cmap = plt.cm.viridis
                for slot, slot_grp in grp.groupby('example_slot', dropna=False):
                    slot_grp = slot_grp.copy()
                    slot_grp['method'] = slot_grp['method'].astype(str)
                    slot_grp['method_idx'] = slot_grp['method'].apply(lambda value: methods_in_order.index(value) if value in methods_in_order else -1)
                    slot_grp = slot_grp[slot_grp['method_idx'] >= 0].sort_values('method_idx')
                    color_val = float(slot_grp['reference_score'].iloc[0]) if not slot_grp.empty else 0.5
                    ax.plot(
                        slot_grp['method_idx'],
                        slot_grp['pred_rank_sample'],
                        color=cmap(color_val),
                        linewidth=1.6,
                        alpha=0.85,
                    )
                    if not slot_grp.empty:
                        ax.text(
                            -0.15,
                            float(slot_grp['pred_rank_sample'].iloc[0]),
                            str(int(slot)),
                            fontsize=7,
                            ha='right',
                            va='center',
                            color=cmap(color_val),
                        )
                ax.set_title(f'Bump Chart: {concept_name} ({dataset_name}) at interaction {final_step}')
                ax.set_xlabel('Method')
                ax.set_ylabel('Rank within 20 representative images')
                ax.set_xticks(range(len(methods_in_order)))
                ax.set_xticklabels(methods_in_order, rotation=35, ha='right')
                ax.invert_yaxis()
                ax.grid(axis='y', alpha=0.2)
                _save(fig, f'representative_bump_{safe_slug}')

    manifest = {
        'run_id': str(run_id),
        'generated_at_utc': utc_now_iso(),
        'figure_dir': str(figure_dir),
        'files': sorted(
            path.name
            for path in figure_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {'.png', '.pdf'}
        ),
    }
    if breakdown_frames:
        breakdown = pd.concat(breakdown_frames, ignore_index=True)
        metric_col = 'metric_name' if 'metric_name' in breakdown.columns else 'metric'
        breakdown = breakdown.sort_values(['stage', metric_col, 'dataset', 'method'], kind='mergesort')
        breakdown.to_csv(figure_dir / 'dataset_metric_breakdown.csv', index=False)
    (figure_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return figure_dir


def main() -> None:
    run_id = uuid.uuid4().hex[:12]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    loggers = build_loggers()

    active_methods = active_method_specs(ACTIVE_METHODS, METHOD_REGISTRY)
    ablation_methods = active_method_specs(ACTIVE_ABLATIONS, ABLATION_REGISTRY)
    methods_in_run = list(active_methods)
    if RUN_ABLATIONS:
        methods_in_run.extend(ablation_methods)
    dino_methods = [spec.name for spec in methods_in_run if method_uses_dino(spec)]
    if dino_methods:
        raise RuntimeError(
            'Modeling evaluation is configured to run in CLIP space only, '
            f'but these active methods still require DINO: {sorted(dino_methods)}'
        )

    tasks = load_active_tasks(require_dino=False)
    if len(tasks) == 0:
        raise RuntimeError('No evaluation tasks were built from the active dataset configuration')
    prompt_records = hydrate_prompt_cache(
        tasks,
        cache_path=CONTRASTIVE_PROMPT_CACHE_PATH,
        source=CONTRASTIVE_PROMPT_SOURCE,
        n_prompts=CONTRASTIVE_PROMPT_COUNT,
        regenerate=CONTRASTIVE_PROMPT_REGENERATE,
        max_retries=CONTRASTIVE_PROMPT_MAX_RETRIES,
        retry_wait_sec=CONTRASTIVE_PROMPT_RETRY_WAIT_SEC,
        logger=print,
    )
    applied_prompt_count = apply_prompt_cache_to_tasks(tasks, prompt_records)
    print(
        '[modeling-eval] prompt cache ready '
        f'tasks={applied_prompt_count}/{len(tasks)} source={CONTRASTIVE_PROMPT_SOURCE} '
        f'path={CONTRASTIVE_PROMPT_CACHE_PATH}'
    )
    representative_task_ids = select_representative_task_ids(tasks, REPRESENTATIVE_TASKS_PER_DATASET)
    representative_task_id_set = set(representative_task_ids)

    loggers['runs'].append(
        {
            'ts_utc': utc_now_iso(),
            'run_id': run_id,
            'run_name': RUN_NAME,
            'note': RUN_NOTE,
            'active_datasets_json': compact_json(ACTIVE_DATASETS),
            'active_methods_json': compact_json(ACTIVE_METHODS),
            'method_specs_json': compact_json({spec.name: method_spec_payload(spec) for spec in active_methods}),
            'active_ablations_json': compact_json(ACTIVE_ABLATIONS),
            'ablation_specs_json': compact_json({spec.name: method_spec_payload(spec) for spec in ablation_methods}),
            'budgets_json': compact_json(REFINEMENT_BUDGETS),
            'main_policy': MAIN_QUERY_POLICY,
            'query_policies_json': compact_json(QUERY_POLICIES),
            'representative_task_ids_json': compact_json(representative_task_ids),
            'representative_images_per_task': int(REPRESENTATIVE_IMAGES_PER_TASK),
        },
    )

    print(f'[modeling-eval] run_id={run_id}')
    for task in tasks:
        append_task_row(loggers['tasks'], run_id, task)

    context = EvaluationContext(tasks)

    if RUN_E1_PRIOR:
        print(f'[modeling-eval] E1 prior tasks={len(tasks)} methods={len(active_methods)}')
        evaluate_prior_quality(run_id, tasks, active_methods, context, loggers['prior'])

    if RUN_E2_REFINEMENT:
        print(f'[modeling-eval] E2 refinement tasks={len(tasks)} methods={len(active_methods)} policy={MAIN_QUERY_POLICY}')
        for task in tasks:
            representative_indices: Optional[np.ndarray] = None
            if task.task_id in representative_task_id_set:
                representative_indices = select_representative_example_indices(task, REPRESENTATIVE_IMAGES_PER_TASK)
            for method in active_methods:
                run_refinement_loop(
                    run_id=run_id,
                    task=task,
                    method=method,
                    context=context,
                    logger=loggers['refinement'],
                    uncertainty_logger=loggers['uncertainty'] if RUN_E3_UNCERTAINTY else None,
                    representative_logger=loggers['representative_rankings'] if representative_indices is not None else None,
                    budgets=REFINEMENT_BUDGETS,
                    policy=MAIN_QUERY_POLICY,
                    allow_undefined=False,
                    row_prefix='main',
                    representative_example_indices=representative_indices,
                )

    if RUN_E3_UNCERTAINTY:
        print(f'[modeling-eval] E3 query policy tasks={len(tasks)} methods={len(active_methods)}')
        evaluate_query_policies(run_id, tasks, active_methods, context, loggers['query_policy'])

    if RUN_E4_UNDEFINED:
        print(f'[modeling-eval] E4 undefined tasks={len(tasks)} methods={len(active_methods)}')
        evaluate_undefined(run_id, tasks, active_methods, context, loggers['undefined'])

    if RUN_ABLATIONS and len(ablation_methods) > 0:
        print(f'[modeling-eval] ablations tasks={len(tasks)} methods={len(ablation_methods)}')
        evaluate_ablations(run_id, tasks, ablation_methods, context, loggers['ablation'])

    figure_dir = export_modeling_eval_figures(run_id)
    print(f'[modeling-eval] exported figures run_id={run_id} figure_dir={figure_dir}')
    print(f'[modeling-eval] finished run_id={run_id} output_dir={OUTPUT_DIR}')


if __name__ == '__main__':
    main()
