#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

try:
    from .constants import (
        AXIS_BAYES_SEMANTIC_METHOD,
        AXIS_BUILDER_AXIS_BOUNDS_TEXT_SOURCE,
        AXIS_BUILDER_LLM_PROMPT_COUNT,
        AXIS_BUILDER_USE_LLM_PROMPT_ENSEMBLE,
        AXIS_RESIDUAL_ALPHA,
        AXIS_RESIDUAL_BETA,
        AXIS_RESIDUAL_LAMBDA,
        AXIS_RESIDUAL_SIGMA_Y,
        AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER,
        AXIS_RESIDUAL_JITTER,
    )
    from .embeddings import build_multimodal_extractor, normalize_multimodal_method
    from .gallery_backend import ImageGalleryEngine
except ImportError:
    from constants import (
        AXIS_BAYES_SEMANTIC_METHOD,
        AXIS_BUILDER_AXIS_BOUNDS_TEXT_SOURCE,
        AXIS_BUILDER_LLM_PROMPT_COUNT,
        AXIS_BUILDER_USE_LLM_PROMPT_ENSEMBLE,
        AXIS_RESIDUAL_ALPHA,
        AXIS_RESIDUAL_BETA,
        AXIS_RESIDUAL_LAMBDA,
        AXIS_RESIDUAL_SIGMA_Y,
        AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER,
        AXIS_RESIDUAL_JITTER,
    )
    from embeddings import build_multimodal_extractor, normalize_multimodal_method
    from gallery_backend import ImageGalleryEngine


def _slugify(text: str) -> str:
    s = re.sub(r'[^a-z0-9]+', '-', str(text or '').strip().lower()).strip('-')
    return s or 'axis'


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError('Expected NxD array')
    denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / denom


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(arr)) + 1e-8
    return (arr / denom).astype(np.float32)


def _normalize_semantic_method(method: str) -> str:
    normalized = normalize_multimodal_method(method)
    if normalized in {'clip', 'siglip2'}:
        return normalized
    return 'clip'


def _rank_percentile_01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return arr
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


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0.0, 1.0 - float(np.dot(a, b))))


def _format_axis_value(v: float) -> str:
    x = float(v)
    ax = abs(x)
    if ax >= 100.0:
        return f'{x:.0f}'
    if ax >= 10.0:
        return f'{x:.1f}'
    if ax >= 1.0:
        return f'{x:.2f}'
    return f'{x:.3f}'.rstrip('0').rstrip('.')


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    neg_x = np.exp(arr[~pos])
    out[~pos] = neg_x / (1.0 + neg_x)
    return out


def _normalize_mode(mode: Any) -> str:
    s = str(mode or '').strip().lower()
    if s in {'rank', 'ranking', 'pairwise'}:
        return 'rank'
    if s in {'graph', 'grf', 'knn_graph', 'knn', 'laplacian', 'manifold'}:
        return 'graph'
    return 'gaussian'


def _normalize_model_type(model_type: Any) -> str:
    s = str(model_type or '').strip().lower()
    if s in {'piecewise', 'piecewise_linear', 'mixture', 'mixture_of_linear'}:
        return 'piecewise_linear'
    if s in {'residual', 'residual_gp', 'smooth_residual', 'gp_residual'}:
        return 'residual'
    return 'bayes_linear'


def _normalize_piecewise_aggregator(aggregator: Any) -> str:
    s = str(aggregator or '').strip().lower()
    if s in {'mean', 'avg', 'average'}:
        return 'mean'
    if s in {'softmax', 'smoothmax'}:
        return 'softmax'
    return 'max'


@dataclass
class CollectionCache:
    dataset_root: str
    collection_id: str
    ids: List[str]
    embeddings: np.ndarray
    id_to_index: Dict[str, int]
    feature_space: str
    semantic_method: str
    clip_dim: int
    dino_dim: int
    clip_scale: float
    dino_scale: float
    graph_laplacian: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    graph_prior_precision: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    graph_prior_cov: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    graph_prior_cov_diag: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    graph_knn_k: int = 0
    graph_similarity_scale: float = 0.0
    piecewise_basis_dirs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))


@dataclass
class AxisBayesState:
    axis_id: str
    collection_id: str
    dataset_root: str
    q: str
    axis_name: str
    model_type: str
    ids: List[str]
    image_embeddings: np.ndarray
    id_to_index: Dict[str, int]
    w0: np.ndarray
    pos_prompt_ensemble: List[str]
    neg_prompt_ensemble: List[str]
    prompt_ensemble: List[str]
    prompt_source: str
    prompt_provider: str
    mode: str
    feature_space: str
    semantic_method: str
    clip_dim: int
    dino_dim: int
    clip_scale: float
    dino_scale: float
    alpha: float
    dino_alpha: float
    bias_alpha: float
    sigma2: float
    rank_eta: float = 0.25
    rank_anchor_k: int = 6
    rank_anchor_delta: float = 0.12
    rank_max_pairs: int = 1600
    max_moves: int = 4
    graph_knn_k: int = 0
    graph_lambda_smooth: float = 0.0
    graph_lambda_prior: float = 0.0
    graph_jitter: float = 0.0
    graph_similarity_scale: float = 0.0
    b0: float = 0.0
    z0_all: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    z0_sorted: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    alpha_vec: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    alpha_inv_vec: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    move_order: List[str] = field(default_factory=list)
    move_targets: Dict[str, float] = field(default_factory=dict)
    move_weights: Dict[str, float] = field(default_factory=dict)
    undefined_order: List[str] = field(default_factory=list)
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    y: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    A_inv: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    mu: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    b: float = 0.0
    pair_i: List[int] = field(default_factory=list)
    pair_j: List[int] = field(default_factory=list)
    pair_y: List[float] = field(default_factory=list)
    pair_w: List[float] = field(default_factory=list)
    pair_D: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    pair_W_diag: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    pair_M_inv: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    graph_prior_cov: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    graph_prior_cov_diag: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    graph_posterior_mean: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    graph_posterior_cov_diag: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    graph_obs_indices: List[int] = field(default_factory=list)
    graph_obs_var: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    graph_obs_S_inv: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    created_at: float = field(default_factory=time.time)
    piecewise_num_experts: int = 1
    piecewise_use_gating: bool = False
    piecewise_aggregator: str = 'max'
    piecewise_clip_scale: float = 1.0
    piecewise_dino_scale: float = 1.0
    pairwise_from_scalar_margin: float = 0.08
    piecewise_prior_strength: float = 8.0
    piecewise_diversity_strength: float = 0.1
    piecewise_l2_reg: float = 0.02
    piecewise_learning_rate: float = 0.05
    piecewise_max_refine_steps: int = 120
    piecewise_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    piecewise_prior_vector: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    piecewise_expert_weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    piecewise_gate_weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    piecewise_gate_bias: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    piecewise_scores_all: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    piecewise_expert_scores_all: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    piecewise_uncertainty_all: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    piecewise_pair_i: List[int] = field(default_factory=list)
    piecewise_pair_j: List[int] = field(default_factory=list)
    piecewise_pair_w: List[float] = field(default_factory=list)
    piecewise_last_fit_signature: str = ''
    residual_alpha: float = 1.0
    residual_beta: float = 0.0
    residual_lambda: float = 0.12
    residual_sigma_y: float = 0.06
    residual_lengthscale_multiplier: float = 1.0
    residual_lengthscale: float = 1.0
    residual_jitter: float = 1e-6
    residual_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    residual_external_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    residual_external_targets: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    residual_external_weights: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    residual_posterior_mean: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    residual_posterior_std: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    residual_last_fit_signature: str = ''


class BaseAxisScorer:
    """Backend-neutral scorer API used by AxisBayesEngine."""

    model_type = 'bayes_linear'

    def initialize_state(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> None:
        return None

    def fit_from_feedback(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> None:
        raise NotImplementedError

    def score_embeddings(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> np.ndarray:
        raise NotImplementedError

    def get_uncertainty(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> np.ndarray:
        raise NotImplementedError

    def export_state(self, state: AxisBayesState) -> Dict[str, Any]:
        return {}

    def import_state(self, state: AxisBayesState, payload: Dict[str, Any]) -> None:
        return None


class BayesianLinearAxisScorer(BaseAxisScorer):
    """Compatibility shim over the existing gaussian / rank / graph axis engine."""

    model_type = 'bayes_linear'

    def fit_from_feedback(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> None:
        engine._refresh_posterior_legacy(state)

    def score_embeddings(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> np.ndarray:
        return engine._legacy_projection_values(state)

    def get_uncertainty(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> np.ndarray:
        return engine._predict_std_legacy(state)

    def export_state(self, state: AxisBayesState) -> Dict[str, Any]:
        return {
            'mu': np.asarray(state.mu, dtype=np.float32).tolist(),
            'b': float(state.b),
            'X': np.asarray(state.X, dtype=np.float32).tolist(),
            'y': np.asarray(state.y, dtype=np.float32).tolist(),
            'A_inv': np.asarray(state.A_inv, dtype=np.float32).tolist(),
            'pair_i': [int(v) for v in state.pair_i],
            'pair_j': [int(v) for v in state.pair_j],
            'pair_y': [float(v) for v in state.pair_y],
            'pair_w': [float(v) for v in state.pair_w],
            'pair_D': np.asarray(state.pair_D, dtype=np.float32).tolist(),
            'pair_W_diag': np.asarray(state.pair_W_diag, dtype=np.float32).tolist(),
            'pair_M_inv': np.asarray(state.pair_M_inv, dtype=np.float32).tolist(),
            'graph_prior_cov': np.asarray(state.graph_prior_cov, dtype=np.float32).tolist(),
            'graph_prior_cov_diag': np.asarray(state.graph_prior_cov_diag, dtype=np.float32).tolist(),
            'graph_posterior_mean': np.asarray(state.graph_posterior_mean, dtype=np.float32).tolist(),
            'graph_posterior_cov_diag': np.asarray(state.graph_posterior_cov_diag, dtype=np.float32).tolist(),
            'graph_obs_indices': [int(v) for v in state.graph_obs_indices],
            'graph_obs_var': np.asarray(state.graph_obs_var, dtype=np.float32).tolist(),
            'graph_obs_S_inv': np.asarray(state.graph_obs_S_inv, dtype=np.float32).tolist(),
        }

    def import_state(self, state: AxisBayesState, payload: Dict[str, Any]) -> None:
        state.mu = np.asarray(payload.get('mu') or [], dtype=np.float32)
        state.b = float(payload.get('b') or 0.0)
        state.X = np.asarray(payload.get('X') or [], dtype=np.float32)
        state.y = np.asarray(payload.get('y') or [], dtype=np.float32)
        state.A_inv = np.asarray(payload.get('A_inv') or [], dtype=np.float32)
        state.pair_i = [int(v) for v in (payload.get('pair_i') or [])]
        state.pair_j = [int(v) for v in (payload.get('pair_j') or [])]
        state.pair_y = [float(v) for v in (payload.get('pair_y') or [])]
        state.pair_w = [float(v) for v in (payload.get('pair_w') or [])]
        state.pair_D = np.asarray(payload.get('pair_D') or [], dtype=np.float32)
        state.pair_W_diag = np.asarray(payload.get('pair_W_diag') or [], dtype=np.float32)
        state.pair_M_inv = np.asarray(payload.get('pair_M_inv') or [], dtype=np.float32)
        state.graph_prior_cov = np.asarray(payload.get('graph_prior_cov') or [], dtype=np.float32)
        state.graph_prior_cov_diag = np.asarray(payload.get('graph_prior_cov_diag') or [], dtype=np.float32)
        state.graph_posterior_mean = np.asarray(payload.get('graph_posterior_mean') or [], dtype=np.float32)
        state.graph_posterior_cov_diag = np.asarray(payload.get('graph_posterior_cov_diag') or [], dtype=np.float32)
        state.graph_obs_indices = [int(v) for v in (payload.get('graph_obs_indices') or [])]
        state.graph_obs_var = np.asarray(payload.get('graph_obs_var') or [], dtype=np.float32)
        state.graph_obs_S_inv = np.asarray(payload.get('graph_obs_S_inv') or [], dtype=np.float32)


class PiecewiseLinearAxisScorer(BaseAxisScorer):
    """Small mixture-of-linear ranker trained from pairwise constraints derived from scalar moves."""

    model_type = 'piecewise_linear'
    _smoothmax_temperature = 0.25
    _weak_regression_strength = 0.05
    _gate_l2_reg = 0.01

    def initialize_state(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> None:
        features = engine._build_piecewise_feature_matrix(state)
        prior = engine._build_piecewise_prior_vector(state)
        state.piecewise_features = np.asarray(features, dtype=np.float32)
        state.piecewise_prior_vector = np.asarray(prior, dtype=np.float32)
        if state.piecewise_expert_weights.size > 0:
            return
        basis = engine._collection_piecewise_basis(state, state.piecewise_features)
        expert_count = max(1, int(state.piecewise_num_experts))
        prior_unit = _normalize_vec(prior)
        experts: List[np.ndarray] = [prior_unit]
        for idx in range(1, expert_count):
            if basis.size > 0:
                basis_idx = min(idx - 1, basis.shape[0] - 1)
                direction = basis[basis_idx]
            else:
                direction = np.zeros_like(prior, dtype=np.float32)
                if idx - 1 < direction.shape[0]:
                    direction[idx - 1] = 1.0
            direction = np.asarray(direction, dtype=np.float32) - (float(np.dot(direction, prior_unit)) * prior_unit)
            if float(np.linalg.norm(direction)) <= 1e-6:
                direction = np.zeros_like(prior_unit, dtype=np.float32)
                axis_idx = min(idx, max(0, direction.shape[0] - 1))
                if direction.size > 0:
                    direction[axis_idx] = 1.0
                direction = direction - (float(np.dot(direction, prior_unit)) * prior_unit)
            sign = -1.0 if idx % 2 == 0 else 1.0
            mixed = _normalize_vec((0.15 * prior_unit) + (sign * _normalize_vec(direction)))
            experts.append(np.asarray(mixed, dtype=np.float32))
        state.piecewise_expert_weights = np.vstack(experts).astype(np.float32)
        if state.piecewise_use_gating:
            state.piecewise_gate_weights = np.zeros_like(state.piecewise_expert_weights, dtype=np.float32)
            state.piecewise_gate_bias = np.zeros((expert_count,), dtype=np.float32)
        else:
            state.piecewise_gate_weights = np.zeros((0, 0), dtype=np.float32)
            state.piecewise_gate_bias = np.zeros((0,), dtype=np.float32)
        state.piecewise_scores_all, state.piecewise_expert_scores_all, state.piecewise_uncertainty_all = (
            self._numpy_forward(state)
        )

    def fit_from_feedback(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> None:
        self.initialize_state(engine, state)
        signature = engine._piecewise_fit_signature(state)
        if signature == state.piecewise_last_fit_signature and state.piecewise_scores_all.size == len(state.ids):
            return

        winners, losers, pair_weights = engine._build_piecewise_pairs(state)
        state.piecewise_pair_i = [int(v) for v in winners.tolist()]
        state.piecewise_pair_j = [int(v) for v in losers.tolist()]
        state.piecewise_pair_w = [float(v) for v in pair_weights.tolist()]

        if winners.size == 0:
            state.piecewise_scores_all, state.piecewise_expert_scores_all, state.piecewise_uncertainty_all = (
                self._numpy_forward(state)
            )
            state.piecewise_last_fit_signature = signature
            return

        undefined_ids = engine._undefined_id_set(state)
        moved_ids = [
            image_id for image_id in state.move_order
            if image_id in state.id_to_index and image_id not in undefined_ids
        ]
        fit_indices = sorted(set([int(v) for v in winners.tolist()] + [int(v) for v in losers.tolist()] + [
            int(state.id_to_index[image_id]) for image_id in moved_ids
        ]))
        fit_lookup = {global_idx: local_idx for local_idx, global_idx in enumerate(fit_indices)}
        X_fit = torch.tensor(state.piecewise_features[fit_indices], dtype=torch.float32)
        winner_local = torch.tensor([fit_lookup[int(v)] for v in winners.tolist()], dtype=torch.long)
        loser_local = torch.tensor([fit_lookup[int(v)] for v in losers.tolist()], dtype=torch.long)
        pair_weight_t = torch.tensor(pair_weights, dtype=torch.float32)
        prior_t = torch.tensor(state.piecewise_prior_vector, dtype=torch.float32)
        expert_w = torch.nn.Parameter(torch.tensor(state.piecewise_expert_weights, dtype=torch.float32))
        params: List[torch.nn.Parameter] = [expert_w]
        gate_w = None
        gate_b = None
        if state.piecewise_use_gating:
            gate_w = torch.nn.Parameter(torch.tensor(state.piecewise_gate_weights, dtype=torch.float32))
            gate_b = torch.nn.Parameter(torch.tensor(state.piecewise_gate_bias, dtype=torch.float32))
            params.extend([gate_w, gate_b])
        optimizer = torch.optim.Adam(params, lr=float(max(1e-4, state.piecewise_learning_rate)))

        label_idx = [fit_lookup[int(state.id_to_index[image_id])] for image_id in moved_ids]
        target_raw = np.asarray(
            [engine._quantile_from_sorted(state.z0_sorted, state.move_targets[image_id]) for image_id in moved_ids],
            dtype=np.float32,
        )
        label_idx_t = torch.tensor(label_idx, dtype=torch.long) if label_idx else None
        target_raw_t = torch.tensor(target_raw, dtype=torch.float32) if target_raw.size > 0 else None
        label_weight_t = torch.tensor(
            [float(max(1e-4, state.move_weights.get(image_id, 1.0))) for image_id in moved_ids],
            dtype=torch.float32,
        ) if label_idx else None

        for _ in range(max(1, int(state.piecewise_max_refine_steps))):
            optimizer.zero_grad()
            score_fit, expert_score_fit, gate_fit = self._torch_forward(
                X_fit,
                expert_w,
                gate_w,
                gate_b,
                aggregator=state.piecewise_aggregator,
                use_gating=state.piecewise_use_gating,
                train_mode=True,
            )
            rank_margin = score_fit[winner_local] - score_fit[loser_local]
            rank_loss = (pair_weight_t * F.softplus(-rank_margin)).mean()
            prior_loss = float(max(0.0, state.piecewise_prior_strength)) * torch.sum((expert_w[0] - prior_t) ** 2)
            l2_loss = float(max(0.0, state.piecewise_l2_reg)) * torch.sum(expert_w ** 2)
            diversity_loss = self._diversity_penalty(expert_w) * float(max(0.0, state.piecewise_diversity_strength))
            gate_loss = torch.tensor(0.0, dtype=torch.float32)
            if gate_w is not None and gate_b is not None:
                gate_loss = self._gate_l2_reg * (torch.sum(gate_w ** 2) + torch.sum(gate_b ** 2))
            weak_loss = torch.tensor(0.0, dtype=torch.float32)
            if label_idx_t is not None and target_raw_t is not None and label_weight_t is not None and label_idx_t.numel() > 0:
                weak_loss = self._weak_regression_strength * torch.mean(
                    label_weight_t * F.smooth_l1_loss(score_fit[label_idx_t], target_raw_t, reduction='none')
                )
            loss = rank_loss + prior_loss + l2_loss + diversity_loss + gate_loss + weak_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for expert_idx in range(expert_w.shape[0]):
                    if expert_idx == 0 and torch.dot(expert_w[expert_idx], prior_t) < 0:
                        expert_w[expert_idx].mul_(-1.0)
                    norm = torch.linalg.norm(expert_w[expert_idx]).clamp_min(1e-6)
                    expert_w[expert_idx].div_(norm)

        state.piecewise_expert_weights = expert_w.detach().cpu().numpy().astype(np.float32)
        if gate_w is not None and gate_b is not None:
            state.piecewise_gate_weights = gate_w.detach().cpu().numpy().astype(np.float32)
            state.piecewise_gate_bias = gate_b.detach().cpu().numpy().astype(np.float32)
        state.piecewise_scores_all, state.piecewise_expert_scores_all, state.piecewise_uncertainty_all = (
            self._numpy_forward(state)
        )
        state.piecewise_last_fit_signature = signature

    def score_embeddings(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> np.ndarray:
        self.fit_from_feedback(engine, state)
        return np.asarray(state.piecewise_scores_all, dtype=np.float32)

    def get_uncertainty(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> np.ndarray:
        self.fit_from_feedback(engine, state)
        return np.asarray(state.piecewise_uncertainty_all, dtype=np.float32)

    def export_state(self, state: AxisBayesState) -> Dict[str, Any]:
        return {
            'piecewise_features': np.asarray(state.piecewise_features, dtype=np.float32).tolist(),
            'piecewise_prior_vector': np.asarray(state.piecewise_prior_vector, dtype=np.float32).tolist(),
            'piecewise_expert_weights': np.asarray(state.piecewise_expert_weights, dtype=np.float32).tolist(),
            'piecewise_gate_weights': np.asarray(state.piecewise_gate_weights, dtype=np.float32).tolist(),
            'piecewise_gate_bias': np.asarray(state.piecewise_gate_bias, dtype=np.float32).tolist(),
            'piecewise_scores_all': np.asarray(state.piecewise_scores_all, dtype=np.float32).tolist(),
            'piecewise_expert_scores_all': np.asarray(state.piecewise_expert_scores_all, dtype=np.float32).tolist(),
            'piecewise_uncertainty_all': np.asarray(state.piecewise_uncertainty_all, dtype=np.float32).tolist(),
            'piecewise_pair_i': [int(v) for v in state.piecewise_pair_i],
            'piecewise_pair_j': [int(v) for v in state.piecewise_pair_j],
            'piecewise_pair_w': [float(v) for v in state.piecewise_pair_w],
            'piecewise_last_fit_signature': str(state.piecewise_last_fit_signature or ''),
        }

    def import_state(self, state: AxisBayesState, payload: Dict[str, Any]) -> None:
        state.piecewise_features = np.asarray(payload.get('piecewise_features') or [], dtype=np.float32)
        state.piecewise_prior_vector = np.asarray(payload.get('piecewise_prior_vector') or [], dtype=np.float32)
        state.piecewise_expert_weights = np.asarray(payload.get('piecewise_expert_weights') or [], dtype=np.float32)
        state.piecewise_gate_weights = np.asarray(payload.get('piecewise_gate_weights') or [], dtype=np.float32)
        state.piecewise_gate_bias = np.asarray(payload.get('piecewise_gate_bias') or [], dtype=np.float32)
        state.piecewise_scores_all = np.asarray(payload.get('piecewise_scores_all') or [], dtype=np.float32)
        state.piecewise_expert_scores_all = np.asarray(payload.get('piecewise_expert_scores_all') or [], dtype=np.float32)
        state.piecewise_uncertainty_all = np.asarray(payload.get('piecewise_uncertainty_all') or [], dtype=np.float32)
        state.piecewise_pair_i = [int(v) for v in (payload.get('piecewise_pair_i') or [])]
        state.piecewise_pair_j = [int(v) for v in (payload.get('piecewise_pair_j') or [])]
        state.piecewise_pair_w = [float(v) for v in (payload.get('piecewise_pair_w') or [])]
        state.piecewise_last_fit_signature = str(payload.get('piecewise_last_fit_signature') or '')

    def _diversity_penalty(self, expert_w: torch.Tensor) -> torch.Tensor:
        if expert_w.shape[0] <= 1:
            return torch.tensor(0.0, dtype=torch.float32, device=expert_w.device)
        normed = F.normalize(expert_w, dim=1)
        gram = normed @ normed.T
        off_diag = gram - torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        return torch.mean(off_diag ** 2)

    def _torch_forward(
        self,
        features: torch.Tensor,
        expert_w: torch.Tensor,
        gate_w: Optional[torch.Tensor],
        gate_b: Optional[torch.Tensor],
        aggregator: str,
        use_gating: bool,
        train_mode: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        expert_scores = features @ expert_w.T
        if use_gating and gate_w is not None and gate_b is not None and gate_w.numel() > 0:
            gates = torch.softmax((features @ gate_w.T) + gate_b, dim=1)
            scores = torch.sum(gates * expert_scores, dim=1)
            return scores, expert_scores, gates
        agg = _normalize_piecewise_aggregator(aggregator)
        if agg == 'mean' or expert_scores.shape[1] <= 1:
            scores = torch.mean(expert_scores, dim=1)
            return scores, expert_scores, None
        if agg == 'softmax':
            weights = torch.softmax(expert_scores, dim=1)
            scores = torch.sum(weights * expert_scores, dim=1)
            return scores, expert_scores, weights
        if train_mode:
            scores = self._smoothmax_temperature * torch.logsumexp(expert_scores / self._smoothmax_temperature, dim=1)
        else:
            scores = torch.max(expert_scores, dim=1).values
        return scores, expert_scores, None

    def _numpy_forward(self, state: AxisBayesState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.asarray(state.piecewise_features, dtype=np.float32)
        if features.size == 0 or state.piecewise_expert_weights.size == 0:
            empty = np.zeros((len(state.ids),), dtype=np.float32)
            return empty, np.zeros((len(state.ids), 0), dtype=np.float32), empty
        expert_scores = features @ state.piecewise_expert_weights.T
        if state.piecewise_use_gating and state.piecewise_gate_weights.size > 0 and state.piecewise_gate_bias.size > 0:
            logits = (features @ state.piecewise_gate_weights.T) + state.piecewise_gate_bias[None, :]
            logits = logits - np.max(logits, axis=1, keepdims=True)
            gates = np.exp(logits)
            gates = gates / np.maximum(np.sum(gates, axis=1, keepdims=True), 1e-8)
            scores = np.sum(gates * expert_scores, axis=1).astype(np.float32)
            uncertainty = np.sqrt(np.maximum(np.sum(gates * ((expert_scores - scores[:, None]) ** 2), axis=1), 1e-8)).astype(np.float32)
            return scores, expert_scores.astype(np.float32), uncertainty
        agg = _normalize_piecewise_aggregator(state.piecewise_aggregator)
        if agg == 'mean' or expert_scores.shape[1] <= 1:
            scores = np.mean(expert_scores, axis=1).astype(np.float32)
        elif agg == 'softmax':
            logits = expert_scores - np.max(expert_scores, axis=1, keepdims=True)
            weights = np.exp(logits)
            weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-8)
            scores = np.sum(weights * expert_scores, axis=1).astype(np.float32)
        else:
            scores = np.max(expert_scores, axis=1).astype(np.float32)
        uncertainty = np.std(expert_scores, axis=1).astype(np.float32)
        return scores, expert_scores.astype(np.float32), uncertainty


class ResidualAxisScorer(BaseAxisScorer):
    """Global CLIP prior plus a small smooth GP residual over fused image features."""

    model_type = 'residual'

    def initialize_state(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> None:
        state.residual_features = engine._build_residual_feature_matrix(state)
        if state.residual_lengthscale <= 1e-6:
            state.residual_lengthscale = engine._estimate_residual_lengthscale(state.residual_features, state)
        engine._refresh_residual_prior(state)
        if state.residual_posterior_mean.size != len(state.ids):
            state.residual_posterior_mean = np.asarray(state.z0_all, dtype=np.float32).copy()
        if state.residual_posterior_std.size != len(state.ids):
            base_std = float(max(1e-6, state.residual_lambda))
            state.residual_posterior_std = np.full((len(state.ids),), base_std, dtype=np.float32)

    def fit_from_feedback(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> None:
        self.initialize_state(engine, state)
        signature = engine._residual_fit_signature(state)
        if signature == state.residual_last_fit_signature and state.residual_posterior_mean.size == len(state.ids):
            return

        base = np.asarray(state.z0_all, dtype=np.float32)
        support_features, residual_targets, support_weights = engine._build_residual_support_data(state)
        n = int(len(state.ids))
        lambda_sq = float(max(1e-8, state.residual_lambda * state.residual_lambda))
        if support_features.size == 0 or residual_targets.size == 0:
            state.residual_posterior_mean = base.astype(np.float32)
            state.residual_posterior_std = np.full((n,), float(np.sqrt(lambda_sq)), dtype=np.float32)
            state.residual_last_fit_signature = signature
            return
        if support_features.ndim != 2 or support_features.shape[1] != state.residual_features.shape[1]:
            raise ValueError(
                f'Residual support feature dim mismatch: support={support_features.shape} target={state.residual_features.shape}'
            )

        noise_var = (
            (float(max(1e-6, state.residual_sigma_y)) ** 2)
            / np.maximum(np.asarray(support_weights, dtype=np.float64), 1e-3)
        )
        K_mm = engine._residual_kernel(
            np.asarray(support_features, dtype=np.float32),
            np.asarray(support_features, dtype=np.float32),
            state.residual_lengthscale,
            state.residual_lambda,
        ).astype(np.float64)
        K_obs = K_mm + np.diag(noise_var + float(max(1e-10, state.residual_jitter)))
        K_nm = engine._residual_kernel(
            np.asarray(state.residual_features, dtype=np.float32),
            np.asarray(support_features, dtype=np.float32),
            state.residual_lengthscale,
            state.residual_lambda,
        ).astype(np.float64)
        targets = np.asarray(residual_targets, dtype=np.float64).reshape(-1)

        try:
            chol = np.linalg.cholesky(K_obs)
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, targets))
            v = np.linalg.solve(chol, K_nm.T)
            quad = np.sum(v * v, axis=0)
        except Exception:
            inv = np.linalg.pinv(K_obs)
            alpha = inv @ targets
            quad = np.sum((K_nm @ inv) * K_nm, axis=1)

        mean = base.astype(np.float64) + (K_nm @ alpha)
        var = lambda_sq - quad
        state.residual_posterior_mean = np.asarray(mean, dtype=np.float32)
        state.residual_posterior_std = np.sqrt(np.maximum(var, float(max(1e-10, state.residual_jitter)))).astype(np.float32)
        state.residual_last_fit_signature = signature

    def score_embeddings(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> np.ndarray:
        self.fit_from_feedback(engine, state)
        return np.asarray(state.residual_posterior_mean, dtype=np.float32)

    def get_uncertainty(self, engine: 'AxisBayesEngine', state: AxisBayesState) -> np.ndarray:
        self.fit_from_feedback(engine, state)
        return np.asarray(state.residual_posterior_std, dtype=np.float32)

    def export_state(self, state: AxisBayesState) -> Dict[str, Any]:
        return {
            'residual_external_features': np.asarray(state.residual_external_features, dtype=np.float32).tolist(),
            'residual_external_targets': np.asarray(state.residual_external_targets, dtype=np.float32).tolist(),
            'residual_external_weights': np.asarray(state.residual_external_weights, dtype=np.float32).tolist(),
        }

    def import_state(self, state: AxisBayesState, payload: Dict[str, Any]) -> None:
        state.residual_external_features = np.asarray(payload.get('residual_external_features') or [], dtype=np.float32)
        state.residual_external_targets = np.asarray(payload.get('residual_external_targets') or [], dtype=np.float32)
        state.residual_external_weights = np.asarray(payload.get('residual_external_weights') or [], dtype=np.float32)


class AxisBayesEngine:
    def __init__(
        self,
        model_type: str = 'bayes_linear',
        mode: str = 'gaussian',
        feature_space: str = 'clip',
        semantic_method: str = AXIS_BAYES_SEMANTIC_METHOD,
        clip_weight: float = 0.7,
        dino_weight: float = 0.3,
        piecewise_num_experts: int = 2,
        piecewise_use_gating: bool = False,
        piecewise_aggregator: str = 'max',
        piecewise_clip_scale: float = 1.0,
        piecewise_dino_scale: float = 1.0,
        pairwise_from_scalar_margin: float = 0.08,
        piecewise_prior_strength: float = 8.0,
        piecewise_expert_diversity_strength: float = 0.1,
        piecewise_l2_reg: float = 0.02,
        piecewise_learning_rate: float = 0.05,
        piecewise_max_refine_steps: int = 120,
        alpha: float = 96.0,
        dino_alpha: float = 220.0,
        bias_alpha: float = 16.0,
        sigma2: float = 0.04,
        graph_knn_k: int = 16,
        graph_lambda_smooth: float = 6.0,
        graph_lambda_prior: float = 1.0,
        graph_jitter: float = 1e-6,
        move_trust: float = 1.0,
        move_mag_gain: float = 0.0,
        rank_eta: float = 0.25,
        rank_anchor_k: int = 6,
        rank_anchor_delta: float = 0.12,
        rank_max_pairs: int = 1600,
        residual_alpha: float = AXIS_RESIDUAL_ALPHA,
        residual_beta: float = AXIS_RESIDUAL_BETA,
        residual_lambda: float = AXIS_RESIDUAL_LAMBDA,
        residual_sigma_y: float = AXIS_RESIDUAL_SIGMA_Y,
        residual_lengthscale_multiplier: float = AXIS_RESIDUAL_LENGTHSCALE_MULTIPLIER,
        residual_jitter: float = AXIS_RESIDUAL_JITTER,
        hotspot_boundary: float = 50.0,
        hotspot_tau: float = 18.0,
        hotspot_k: int = 10,
        exemplar_k: int = 4,
        max_moves: int = 0,
        llm_engine: Optional[Any] = None,
        axis_bounds_text_source: str = AXIS_BUILDER_AXIS_BOUNDS_TEXT_SOURCE,
        use_llm_prompt_ensemble: bool = AXIS_BUILDER_USE_LLM_PROMPT_ENSEMBLE,
        llm_prompt_count: int = AXIS_BUILDER_LLM_PROMPT_COUNT,
    ):
        self.model_type = _normalize_model_type(model_type)
        self.mode = _normalize_mode(mode)
        fs = str(feature_space or '').strip().lower()
        self.feature_space = 'clip_dino' if fs in {'clip_dino', 'clip+dino', 'dino_clip'} else 'clip'
        self.semantic_method = _normalize_semantic_method(semantic_method)
        c_w = float(max(1e-6, clip_weight))
        d_w = float(max(1e-6, dino_weight))
        w_sum = c_w + d_w
        self.clip_weight = float(c_w / w_sum)
        self.dino_weight = float(d_w / w_sum)
        self.clip_scale = float(np.sqrt(self.clip_weight))
        self.dino_scale = float(np.sqrt(self.dino_weight))
        self.alpha = float(alpha)
        self.dino_alpha = float(max(1e-6, dino_alpha))
        self.bias_alpha = float(max(1e-6, float(bias_alpha)))
        self.sigma2 = float(sigma2)
        self.graph_knn_k = max(1, int(graph_knn_k))
        self.graph_lambda_smooth = float(max(0.0, float(graph_lambda_smooth)))
        self.graph_lambda_prior = float(max(1e-8, float(graph_lambda_prior)))
        self.graph_jitter = float(max(1e-10, float(graph_jitter)))
        self.move_trust = float(max(1e-3, float(move_trust)))
        self.move_mag_gain = float(max(0.0, float(move_mag_gain)))
        self.rank_eta = float(max(1e-4, float(rank_eta)))
        self.rank_anchor_k = max(1, int(rank_anchor_k))
        self.rank_anchor_delta = float(max(0.01, min(0.45, float(rank_anchor_delta))))
        self.rank_max_pairs = max(16, int(rank_max_pairs))
        self.residual_alpha = float(residual_alpha)
        self.residual_beta = float(residual_beta)
        self.residual_lambda = float(max(1e-6, float(residual_lambda)))
        self.residual_sigma_y = float(max(1e-6, float(residual_sigma_y)))
        self.residual_lengthscale_multiplier = float(max(1e-3, float(residual_lengthscale_multiplier)))
        self.residual_jitter = float(max(1e-10, float(residual_jitter)))
        self.piecewise_num_experts = max(1, int(piecewise_num_experts))
        self.piecewise_use_gating = bool(piecewise_use_gating)
        self.piecewise_aggregator = _normalize_piecewise_aggregator(piecewise_aggregator)
        self.piecewise_clip_scale = float(max(1e-6, float(piecewise_clip_scale)))
        self.piecewise_dino_scale = float(max(0.0, float(piecewise_dino_scale)))
        self.pairwise_from_scalar_margin = float(max(0.0, min(0.95, float(pairwise_from_scalar_margin))))
        self.piecewise_prior_strength = float(max(0.0, float(piecewise_prior_strength)))
        self.piecewise_expert_diversity_strength = float(max(0.0, float(piecewise_expert_diversity_strength)))
        self.piecewise_l2_reg = float(max(0.0, float(piecewise_l2_reg)))
        self.piecewise_learning_rate = float(max(1e-4, float(piecewise_learning_rate)))
        self.piecewise_max_refine_steps = max(1, int(piecewise_max_refine_steps))
        self.hotspot_boundary = float(hotspot_boundary)
        self.hotspot_tau = float(hotspot_tau)
        self.hotspot_k = int(hotspot_k)
        self.exemplar_k = int(exemplar_k)
        self.max_moves = max(0, int(max_moves))
        self.llm_engine = llm_engine
        src = str(axis_bounds_text_source or '').strip().lower()
        self.axis_bounds_text_source = 'llm' if src == 'llm' else 'template'
        self.use_llm_prompt_ensemble = bool(use_llm_prompt_ensemble)
        self.llm_prompt_count = max(2, int(llm_prompt_count))
        self._collections: Dict[str, CollectionCache] = {}
        self._axes: Dict[str, AxisBayesState] = {}
        self._text_extractors: Dict[str, Any] = {}
        self._scorers: Dict[str, BaseAxisScorer] = {
            'bayes_linear': BayesianLinearAxisScorer(),
            'piecewise_linear': PiecewiseLinearAxisScorer(),
            'residual': ResidualAxisScorer(),
        }

    def _log(self, msg: str, *args):
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = f'{msg} {args}'
        print(f'[axis-bayes] {msg}')

    def _get_text_extractor(self, semantic_method: Optional[str] = None):
        method = _normalize_semantic_method(semantic_method or self.semantic_method)
        extractor = self._text_extractors.get(method)
        if extractor is None:
            self._log('loading semantic text extractor method=%s', method)
            extractor = build_multimodal_extractor(method)
            if extractor is None:
                raise RuntimeError(f'No text extractor available for semantic method {method}')
            self._text_extractors[method] = extractor
        return extractor

    def _build_fixed_prompt_ensemble(self, q: str) -> tuple[List[str], List[str]]:
        query = re.sub(r'\s+', ' ', str(q or '').strip())
        if not query:
            return [], []
        pos_prompts = [
            f'an image with very strong presence of {query}',
            f'an image with high presence of {query}',
            f'an image with a lot of {query}',
        ]
        neg_prompts = [
            f'an image with very weak presence of {query}',
            f'an image with low presence of {query}',
            f'an image with very little {query}',
        ]
        pos_out: List[str] = []
        neg_out: List[str] = []
        pos_seen = set()
        neg_seen = set()
        for prompt in pos_prompts:
            key = prompt.lower()
            if key in pos_seen:
                continue
            pos_seen.add(key)
            pos_out.append(prompt)
        for prompt in neg_prompts:
            key = prompt.lower()
            if key in neg_seen:
                continue
            neg_seen.add(key)
            neg_out.append(prompt)
        return pos_out, neg_out

    def _clean_prompt_list(self, values: Any) -> List[str]:
        out: List[str] = []
        seen = set()
        if not isinstance(values, (list, tuple)):
            return out
        for raw in values:
            text = re.sub(r'\s+', ' ', str(raw or '').strip())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    def _embed_prompt_lists(
        self,
        pos_prompts: Any,
        neg_prompts: Any,
        *,
        semantic_method: str,
        source: str,
        provider: str,
    ) -> tuple[np.ndarray, List[str], List[str], Dict[str, str]]:
        pos_list = self._clean_prompt_list(pos_prompts)
        neg_list = self._clean_prompt_list(neg_prompts)
        if len(pos_list) == 0 or len(neg_list) == 0:
            raise ValueError('Both positive and negative prompt lists must contain at least one prompt')
        ext = self._get_text_extractor(semantic_method)
        pos_vecs = [_normalize_vec(ext.extract_text_embedding(prompt)) for prompt in pos_list]
        neg_vecs = [_normalize_vec(ext.extract_text_embedding(prompt)) for prompt in neg_list]
        pos_avg = np.mean(np.vstack(pos_vecs), axis=0, dtype=np.float32)
        neg_avg = np.mean(np.vstack(neg_vecs), axis=0, dtype=np.float32)
        w0 = _normalize_vec(pos_avg - neg_avg)
        return w0, pos_list, neg_list, {'source': str(source or 'unknown'), 'provider': str(provider or 'unknown')}

    def build_prompt_ensemble(self, q: str) -> tuple[List[str], List[str], Dict[str, str]]:
        query = re.sub(r'\s+', ' ', str(q or '').strip())
        if not query:
            return [], [], {'source': 'none', 'provider': 'none'}
        use_llm = (self.axis_bounds_text_source == 'llm') and self.use_llm_prompt_ensemble
        if not use_llm:
            pos_prompts, neg_prompts = self._build_fixed_prompt_ensemble(query)
            self._log(
                'using fixed prompt ensemble q="%s" pos_count=%d neg_count=%d',
                query,
                len(pos_prompts),
                len(neg_prompts),
            )
            return pos_prompts, neg_prompts, {'source': 'fixed_template', 'provider': 'fixed_template'}
        if self.llm_engine is None:
            raise RuntimeError('axis bounds text source is llm but no llm_engine is configured')
        result = self.llm_engine.generate_axis_prompt_ensemble(
            attribute=query,
            n_prompts=self.llm_prompt_count,
        )
        pos_prompts = [
            str(v).strip()
            for v in (result.get('pos_prompts') or [])
            if str(v).strip()
        ]
        neg_prompts = [
            str(v).strip()
            for v in (result.get('neg_prompts') or [])
            if str(v).strip()
        ]
        if len(pos_prompts) < 2 or len(neg_prompts) < 2:
            pos_prompts, neg_prompts = self._build_fixed_prompt_ensemble(query)
            self._log(
                'llm prompt ensemble invalid q="%s"; falling back to fixed template pos_count=%d neg_count=%d error=%s',
                query,
                len(pos_prompts),
                len(neg_prompts),
                result.get('error') or 'invalid prompt lists',
            )
            return pos_prompts, neg_prompts, {'source': 'fixed_template_fallback', 'provider': 'fixed_template'}
        provider = str(result.get('provider') or 'huggingface_local')
        self._log(
            'using llm prompt ensemble q="%s" pos_count=%d neg_count=%d provider=%s',
            query,
            len(pos_prompts),
            len(neg_prompts),
            provider,
        )
        return pos_prompts, neg_prompts, {'source': 'llm', 'provider': provider}

    def _embed_prompt_ensemble(self, q: str, semantic_method: str) -> tuple[np.ndarray, List[str], List[str], Dict[str, str]]:
        pos_prompts, neg_prompts, prompt_meta = self.build_prompt_ensemble(q)
        return self._embed_prompt_lists(
            pos_prompts,
            neg_prompts,
            semantic_method=semantic_method,
            source=prompt_meta.get('source') or 'unknown',
            provider=prompt_meta.get('provider') or 'unknown',
        )

    def _quantile_from_sorted(self, z_sorted: np.ndarray, p01: float) -> float:
        arr = np.asarray(z_sorted, dtype=np.float32).reshape(-1)
        n = int(arr.size)
        if n == 0:
            return 0.0
        if n == 1:
            return float(arr[0])
        p = float(max(0.0, min(1.0, float(p01))))
        pos = p * float(n - 1)
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))
        if lo == hi:
            return float(arr[lo])
        mix = float(pos - lo)
        return float((arr[lo] * (1.0 - mix)) + (arr[hi] * mix))

    def _get_collection(self, dataset_root: str, collection_id: str) -> CollectionCache:
        dataset_key = str(Path(dataset_root).resolve())
        key = f'{dataset_key}|{self.feature_space}|cw={self.clip_scale:.6f}|dw={self.dino_scale:.6f}'
        cached = self._collections.get(key)
        if cached is not None:
            return cached

        self._log('loading collection dataset=%s collection_id=%s', dataset_key, collection_id)
        engine = ImageGalleryEngine(dataset_key)
        entries = engine.list_images()
        if len(entries) == 0:
            raise ValueError('No images found in collection')
        semantic_method = None
        semantic_embs = None
        for candidate in (_normalize_semantic_method(self.semantic_method), 'clip'):
            semantic_embs = engine._load_embeddings_only(entries, method=candidate)
            if semantic_embs is not None:
                semantic_method = candidate
                break
        if semantic_embs is None or semantic_method is None:
            raise ValueError(
                f'Semantic embeddings not available for collection. Tried '
                f'{[_normalize_semantic_method(self.semantic_method), "clip"]}'
            )
        if semantic_method != _normalize_semantic_method(self.semantic_method):
            self._log('semantic cache fallback preferred=%s actual=%s', self.semantic_method, semantic_method)
        X_clip = _normalize_rows(np.asarray(semantic_embs, dtype=np.float32))
        clip_dim = int(X_clip.shape[1])
        dino_dim = 0
        X = X_clip
        if self.feature_space == 'clip_dino':
            dino_embs = engine._load_embeddings_only(entries, method='dino')
            if dino_embs is None:
                dino_cache = engine._cache_dir() / 'embeddings_dino.npz'
                raise ValueError(
                    f'DINO embeddings not available for collection "{dataset_key}" while '
                    f'AXIS_BAYES_FEATURE_SPACE=clip_dino; expected cache at {dino_cache}'
                )
            X_dino = _normalize_rows(np.asarray(dino_embs, dtype=np.float32))
            if int(X_dino.shape[0]) != int(X_clip.shape[0]):
                raise ValueError(f'DINO embedding count mismatch: clip={X_clip.shape[0]} dino={X_dino.shape[0]}')
            dino_dim = int(X_dino.shape[1])
            X = np.concatenate(
                [
                    self.clip_scale * X_clip,
                    self.dino_scale * X_dino,
                ],
                axis=1,
            ).astype(np.float32)
            self._log('feature fusion enabled clip_dim=%d dino_dim=%d clip_w=%.3f dino_w=%.3f',
                      clip_dim, dino_dim, self.clip_weight, self.dino_weight)
        ids = [str(e.id) for e in entries]
        id_to_index = {image_id: i for i, image_id in enumerate(ids)}
        cached = CollectionCache(
            dataset_root=dataset_key,
            collection_id=str(collection_id or Path(dataset_root).name),
            ids=ids,
            embeddings=X,
            id_to_index=id_to_index,
            feature_space=self.feature_space,
            semantic_method=semantic_method,
            clip_dim=clip_dim,
            dino_dim=dino_dim,
            clip_scale=self.clip_scale,
            dino_scale=self.dino_scale,
        )
        self._collections[key] = cached
        return cached

    def _scorer_for_model_type(self, model_type: str) -> BaseAxisScorer:
        normalized = _normalize_model_type(model_type)
        scorer = self._scorers.get(normalized)
        if scorer is None:
            raise ValueError(f'Unknown axis model type: {model_type}')
        return scorer

    def _scorer_for_state(self, state: AxisBayesState) -> BaseAxisScorer:
        return self._scorer_for_model_type(state.model_type)

    def _fit_state(self, state: AxisBayesState) -> None:
        self._scorer_for_state(state).fit_from_feedback(self, state)

    def _score_state(self, state: AxisBayesState) -> np.ndarray:
        return self._scorer_for_state(state).score_embeddings(self, state)

    def _uncertainty_state(self, state: AxisBayesState) -> np.ndarray:
        return self._scorer_for_state(state).get_uncertainty(self, state)

    def _undefined_id_set(self, state: AxisBayesState) -> set[str]:
        return {
            str(image_id)
            for image_id in (state.undefined_order or [])
            if str(image_id) in state.id_to_index
        }

    def _defined_indices(self, state: AxisBayesState) -> List[int]:
        undefined_ids = self._undefined_id_set(state)
        return [
            idx for idx, image_id in enumerate(state.ids)
            if image_id not in undefined_ids
        ]

    def _remove_image_feedback(self, state: AxisBayesState, image_key: str) -> None:
        image_key = str(image_key or '').strip()
        if not image_key:
            return
        state.move_order = [image_id for image_id in state.move_order if image_id != image_key]
        state.move_targets.pop(image_key, None)
        state.move_weights.pop(image_key, None)
        if state.model_type == 'piecewise_linear':
            state.piecewise_last_fit_signature = ''
        if state.model_type == 'residual':
            state.residual_last_fit_signature = ''
        if state.model_type == 'bayes_linear' and _normalize_mode(state.mode) == 'rank' and len(state.pair_i) > 0:
            image_idx = state.id_to_index.get(image_key)
            if image_idx is not None:
                keep = [
                    pos for pos, (idx_i, idx_j) in enumerate(zip(state.pair_i, state.pair_j))
                    if int(idx_i) != int(image_idx) and int(idx_j) != int(image_idx)
                ]
                state.pair_i = [state.pair_i[pos] for pos in keep]
                state.pair_j = [state.pair_j[pos] for pos in keep]
                state.pair_y = [state.pair_y[pos] for pos in keep]
                state.pair_w = [state.pair_w[pos] for pos in keep]

    def _set_image_undefined(self, state: AxisBayesState, image_key: str) -> None:
        image_key = str(image_key or '').strip()
        if not image_key:
            return
        self._remove_image_feedback(state, image_key)
        if image_key not in state.undefined_order:
            state.undefined_order.append(image_key)

    def _clear_image_undefined(self, state: AxisBayesState, image_key: str) -> None:
        image_key = str(image_key or '').strip()
        if not image_key:
            return
        if image_key in state.undefined_order:
            state.undefined_order = [image_id for image_id in state.undefined_order if image_id != image_key]

    def _build_piecewise_feature_matrix(self, state: AxisBayesState) -> np.ndarray:
        X = np.asarray(state.image_embeddings, dtype=np.float32)
        if X.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if state.feature_space == 'clip_dino' and int(state.dino_dim) > 0:
            clip_raw = X[:, : state.clip_dim] / max(1e-8, float(state.clip_scale))
            dino_raw = X[:, state.clip_dim :] / max(1e-8, float(state.dino_scale))
            return np.concatenate(
                [
                    float(state.piecewise_clip_scale) * clip_raw,
                    float(state.piecewise_dino_scale) * dino_raw,
                ],
                axis=1,
            ).astype(np.float32)
        return (float(state.piecewise_clip_scale) * X[:, : state.clip_dim]).astype(np.float32)

    def _build_piecewise_prior_vector(self, state: AxisBayesState) -> np.ndarray:
        if state.feature_space == 'clip_dino' and int(state.dino_dim) > 0:
            return np.concatenate(
                [
                    float(state.piecewise_clip_scale) * np.asarray(state.w0[: state.clip_dim], dtype=np.float32),
                    np.zeros((state.dino_dim,), dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)
        return (float(state.piecewise_clip_scale) * np.asarray(state.w0[: state.clip_dim], dtype=np.float32)).astype(np.float32)

    def _build_residual_feature_matrix(self, state: AxisBayesState) -> np.ndarray:
        X = np.asarray(state.image_embeddings, dtype=np.float32)
        if X.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if state.feature_space == 'clip_dino':
            clip_raw = X[:, : state.clip_dim] / max(1e-8, float(state.clip_scale))
        else:
            clip_raw = X[:, : state.clip_dim]
        if state.feature_space == 'clip_dino' and int(state.dino_dim) > 0:
            dino_raw = X[:, state.clip_dim :] / max(1e-8, float(state.dino_scale))
            fused = np.concatenate([clip_raw, dino_raw], axis=1).astype(np.float32)
            return _normalize_rows(fused)
        return _normalize_rows(clip_raw.astype(np.float32))

    def _build_residual_prior_scores(self, state: AxisBayesState) -> np.ndarray:
        X = np.asarray(state.image_embeddings, dtype=np.float32)
        if X.size == 0 or int(state.clip_dim) <= 0:
            return np.zeros((len(state.ids),), dtype=np.float32)
        if state.feature_space == 'clip_dino':
            clip_raw = X[:, : state.clip_dim] / max(1e-8, float(state.clip_scale))
        else:
            clip_raw = X[:, : state.clip_dim]
        raw_prior = clip_raw @ np.asarray(state.w0[: state.clip_dim], dtype=np.float32)
        stable_prior = _rank_percentile_01(np.asarray(raw_prior, dtype=np.float32))
        return (
            (float(state.residual_alpha) * stable_prior) + float(state.residual_beta)
        ).astype(np.float32)

    def _refresh_residual_prior(self, state: AxisBayesState) -> None:
        state.z0_all = self._build_residual_prior_scores(state)
        state.z0_sorted = np.sort(np.asarray(state.z0_all, dtype=np.float32)).astype(np.float32)

    def _estimate_residual_lengthscale(self, features: np.ndarray, state: AxisBayesState) -> float:
        arr = np.asarray(features, dtype=np.float32)
        n = int(arr.shape[0]) if arr.ndim == 2 else 0
        if n <= 1:
            return max(0.25, float(state.residual_lengthscale_multiplier))
        if n > 96:
            picks = np.linspace(0, n - 1, 96, dtype=np.int64)
            arr = arr[picks]
        diff = arr[:, None, :] - arr[None, :, :]
        dist2 = np.sum(diff * diff, axis=2, dtype=np.float32)
        tri = dist2[np.triu_indices(dist2.shape[0], k=1)]
        valid = tri[np.isfinite(tri) & (tri > 1e-8)]
        base = float(np.sqrt(np.median(valid))) if valid.size > 0 else 1.0
        return float(max(1e-3, float(state.residual_lengthscale_multiplier) * base))

    def _residual_kernel(
        self,
        X_a: np.ndarray,
        X_b: np.ndarray,
        lengthscale: float,
        amplitude: float,
    ) -> np.ndarray:
        a = np.asarray(X_a, dtype=np.float32)
        b = np.asarray(X_b, dtype=np.float32)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            return np.zeros((a.shape[0] if a.ndim == 2 else 0, b.shape[0] if b.ndim == 2 else 0), dtype=np.float32)
        ls = float(max(1e-6, lengthscale))
        amp_sq = float(max(1e-8, amplitude * amplitude))
        a_sq = np.sum(a * a, axis=1, keepdims=True)
        b_sq = np.sum(b * b, axis=1, keepdims=True).T
        dist2 = np.maximum(a_sq + b_sq - (2.0 * (a @ b.T)), 0.0)
        return (amp_sq * np.exp(-0.5 * dist2 / (ls * ls))).astype(np.float32)

    def _build_residual_support_data(self, state: AxisBayesState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if state.residual_features.size == 0:
            state.residual_features = self._build_residual_feature_matrix(state)
        if state.z0_all.size != len(state.ids):
            self._refresh_residual_prior(state)
        features_parts: List[np.ndarray] = []
        target_parts: List[np.ndarray] = []
        weight_parts: List[np.ndarray] = []
        if state.residual_external_features.size > 0 and state.residual_external_targets.size > 0:
            features_parts.append(np.asarray(state.residual_external_features, dtype=np.float32))
            target_parts.append(np.asarray(state.residual_external_targets, dtype=np.float32).reshape(-1))
            ext_weights = np.asarray(state.residual_external_weights, dtype=np.float32).reshape(-1)
            if ext_weights.size != state.residual_external_targets.size:
                ext_weights = np.ones((state.residual_external_targets.size,), dtype=np.float32)
            weight_parts.append(np.maximum(ext_weights, 1e-3))

        undefined_ids = self._undefined_id_set(state)
        local_ids = [
            image_id for image_id in state.move_order
            if image_id in state.id_to_index and image_id not in undefined_ids
        ]
        if local_ids:
            local_idx = np.asarray([int(state.id_to_index[image_id]) for image_id in local_ids], dtype=np.int64)
            local_features = np.asarray(state.residual_features[local_idx], dtype=np.float32)
            local_targets = np.asarray(
                [float(state.move_targets[image_id]) for image_id in local_ids],
                dtype=np.float32,
            ) - np.asarray(state.z0_all[local_idx], dtype=np.float32)
            local_weights = np.asarray(
                [float(max(1e-3, state.move_weights.get(image_id, 1.0))) for image_id in local_ids],
                dtype=np.float32,
            )
            features_parts.append(local_features)
            target_parts.append(local_targets)
            weight_parts.append(local_weights)

        if not features_parts:
            return (
                np.zeros((0, state.residual_features.shape[1] if state.residual_features.ndim == 2 else 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        return (
            np.concatenate(features_parts, axis=0).astype(np.float32),
            np.concatenate(target_parts, axis=0).astype(np.float32),
            np.concatenate(weight_parts, axis=0).astype(np.float32),
        )

    def _residual_fit_signature(self, state: AxisBayesState) -> str:
        payload = {
            'move_order': [str(v) for v in state.move_order],
            'move_targets': {str(k): round(float(v), 6) for k, v in sorted(state.move_targets.items())},
            'move_weights': {str(k): round(float(v), 6) for k, v in sorted(state.move_weights.items())},
            'undefined_order': [str(v) for v in state.undefined_order],
            'external_count': int(state.residual_external_targets.size),
            'lengthscale': round(float(state.residual_lengthscale), 6),
            'lambda': round(float(state.residual_lambda), 6),
            'sigma_y': round(float(state.residual_sigma_y), 6),
        }
        return json.dumps(payload, sort_keys=True)

    def _collection_piecewise_basis(self, state: AxisBayesState, features: np.ndarray) -> np.ndarray:
        coll = self._collections.get(
            f'{state.dataset_root}|{state.feature_space}|cw={self.clip_scale:.6f}|dw={self.dino_scale:.6f}'
        )
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 1:
            basis = np.zeros((0, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.float32)
        else:
            sample = arr
            if arr.shape[0] > 256:
                step = max(1, arr.shape[0] // 256)
                sample = arr[::step][:256]
            sample = sample - np.mean(sample, axis=0, keepdims=True)
            try:
                _, _, vt = np.linalg.svd(sample, full_matrices=False)
                basis = np.asarray(vt[: max(0, self.piecewise_num_experts - 1)], dtype=np.float32)
            except Exception:
                basis = np.zeros((0, arr.shape[1]), dtype=np.float32)
        if coll is not None:
            coll.piecewise_basis_dirs = np.asarray(basis, dtype=np.float32)
        return np.asarray(basis, dtype=np.float32)

    def _piecewise_fit_signature(self, state: AxisBayesState) -> str:
        payload = {
            'move_order': [str(v) for v in state.move_order],
            'move_targets': {str(k): round(float(v), 6) for k, v in sorted(state.move_targets.items())},
            'move_weights': {str(k): round(float(v), 6) for k, v in sorted(state.move_weights.items())},
            'undefined_order': [str(v) for v in state.undefined_order],
            'experts': int(state.piecewise_num_experts),
            'use_gating': bool(state.piecewise_use_gating),
            'aggregator': str(state.piecewise_aggregator),
            'margin': round(float(state.pairwise_from_scalar_margin), 6),
            'prior_strength': round(float(state.piecewise_prior_strength), 6),
            'diversity_strength': round(float(state.piecewise_diversity_strength), 6),
            'l2_reg': round(float(state.piecewise_l2_reg), 6),
            'lr': round(float(state.piecewise_learning_rate), 6),
            'steps': int(state.piecewise_max_refine_steps),
        }
        return json.dumps(payload, sort_keys=True)

    def _build_piecewise_pairs(self, state: AxisBayesState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        undefined_ids = self._undefined_id_set(state)
        valid_move_ids = [
            image_id for image_id in state.move_order
            if image_id in state.id_to_index and image_id not in undefined_ids
        ]
        if len(valid_move_ids) == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
            )
        pair_map: Dict[Tuple[int, int], float] = {}
        margin = float(max(0.0, state.pairwise_from_scalar_margin))
        label_rows = [
            (int(state.id_to_index[image_id]), float(state.move_targets[image_id]), float(max(1e-4, state.move_weights.get(image_id, 1.0))))
            for image_id in valid_move_ids
        ]
        for idx_a in range(len(label_rows)):
            image_a, value_a, weight_a = label_rows[idx_a]
            for idx_b in range(idx_a + 1, len(label_rows)):
                image_b, value_b, weight_b = label_rows[idx_b]
                delta = value_a - value_b
                if abs(delta) <= margin:
                    continue
                if delta > 0:
                    winner, loser = image_a, image_b
                else:
                    winner, loser = image_b, image_a
                pair_map[(winner, loser)] = max(pair_map.get((winner, loser), 0.0), (abs(delta) + 1e-3) * np.sqrt(weight_a * weight_b))

        prior_scores = _rank_percentile_01(state.z0_all)
        all_indices = np.asarray(self._defined_indices(state), dtype=np.int64)
        defined_prior_scores = prior_scores[all_indices] if all_indices.size > 0 else np.zeros((0,), dtype=np.float32)
        for image_idx, target, move_weight in label_rows:
            candidate_mask = all_indices != int(image_idx)
            candidate_indices = all_indices[candidate_mask]
            candidate_scores = defined_prior_scores[candidate_mask]
            lower = candidate_indices[
                (candidate_scores < target)
                & (candidate_scores >= max(0.0, target - state.rank_anchor_delta))
            ]
            upper = candidate_indices[
                (candidate_scores > target)
                & (candidate_scores <= min(1.0, target + state.rank_anchor_delta))
            ]
            if lower.size == 0:
                lower = candidate_indices[candidate_scores < target]
            if upper.size == 0:
                upper = candidate_indices[candidate_scores > target]
            if lower.size > 0:
                lower = lower[np.argsort(np.abs(prior_scores[lower] - target), kind='mergesort')[: state.rank_anchor_k]]
                for loser in lower.tolist():
                    key = (int(image_idx), int(loser))
                    pair_map[key] = max(pair_map.get(key, 0.0), 0.35 * float(move_weight))
            if upper.size > 0:
                upper = upper[np.argsort(np.abs(prior_scores[upper] - target), kind='mergesort')[: state.rank_anchor_k]]
                for winner in upper.tolist():
                    key = (int(winner), int(image_idx))
                    pair_map[key] = max(pair_map.get(key, 0.0), 0.35 * float(move_weight))

        if len(pair_map) == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
            )
        ordered = sorted(pair_map.items(), key=lambda item: (-float(item[1]), int(item[0][0]), int(item[0][1])))
        ordered = ordered[: int(max(16, state.rank_max_pairs))]
        winners = np.asarray([item[0][0] for item in ordered], dtype=np.int64)
        losers = np.asarray([item[0][1] for item in ordered], dtype=np.int64)
        weights = np.asarray([item[1] for item in ordered], dtype=np.float32)
        return winners, losers, weights

    def _ensure_graph_prior(self, coll: CollectionCache) -> CollectionCache:
        if coll.graph_prior_cov.size > 0 and coll.graph_knn_k == self.graph_knn_k:
            return coll

        X = np.asarray(coll.embeddings, dtype=np.float32)
        n = int(X.shape[0])
        if n <= 0:
            raise ValueError('Cannot build graph prior for empty collection')
        if n == 1:
            q0 = np.asarray([[self.graph_lambda_prior + self.graph_jitter]], dtype=np.float64)
            k0 = np.asarray([[1.0 / q0[0, 0]]], dtype=np.float32)
            coll.graph_laplacian = np.zeros((1, 1), dtype=np.float32)
            coll.graph_prior_precision = q0.astype(np.float32)
            coll.graph_prior_cov = k0
            coll.graph_prior_cov_diag = np.asarray([float(k0[0, 0])], dtype=np.float32)
            coll.graph_knn_k = 0
            coll.graph_similarity_scale = 1.0
            return coll

        k = min(max(1, int(self.graph_knn_k)), n - 1)
        sim = np.asarray(X @ X.T, dtype=np.float32)
        np.fill_diagonal(sim, -np.inf)

        kth = max(0, k - 1)
        nbr_idx = np.argpartition(-sim, kth=kth, axis=1)[:, :k]
        row_idx = np.arange(n, dtype=np.int64)[:, None]
        nbr_sim = np.asarray(sim[row_idx, nbr_idx], dtype=np.float32)
        nbr_dist = np.clip(1.0 - np.clip(nbr_sim, -1.0, 1.0), 0.0, 2.0).astype(np.float32)
        scale = float(np.mean(nbr_dist)) if nbr_dist.size > 0 else 1.0
        scale = max(scale, 1e-3)
        nbr_w = np.exp(-nbr_dist / scale).astype(np.float32)

        W = np.zeros((n, n), dtype=np.float32)
        W[row_idx, nbr_idx] = nbr_w
        W = np.maximum(W, W.T).astype(np.float32)
        np.fill_diagonal(W, 0.0)

        degree = np.sum(W, axis=1).astype(np.float32)
        L = (np.diag(degree) - W).astype(np.float32)
        q0 = (
            (self.graph_lambda_smooth * L.astype(np.float64))
            + ((self.graph_lambda_prior + self.graph_jitter) * np.eye(n, dtype=np.float64))
        )
        try:
            k0 = np.linalg.inv(q0).astype(np.float32)
        except Exception:
            k0 = np.linalg.pinv(q0).astype(np.float32)

        coll.graph_laplacian = L
        coll.graph_prior_precision = q0.astype(np.float32)
        coll.graph_prior_cov = k0
        coll.graph_prior_cov_diag = np.diag(k0).astype(np.float32)
        coll.graph_knn_k = int(k)
        coll.graph_similarity_scale = float(scale)
        self._log(
            'graph prior cached collection=%s n=%d knn_k=%d lambda_smooth=%.4f lambda_prior=%.4f',
            coll.collection_id,
            n,
            k,
            self.graph_lambda_smooth,
            self.graph_lambda_prior,
        )
        return coll

    def _legacy_projection_values(self, state: AxisBayesState) -> np.ndarray:
        mode = _normalize_mode(state.mode)
        if mode == 'graph':
            if state.graph_posterior_mean.size == len(state.ids):
                return np.asarray(state.graph_posterior_mean, dtype=np.float32)
            return np.asarray(state.z0_all, dtype=np.float32)
        return np.asarray((state.image_embeddings @ state.mu) + float(state.b), dtype=np.float32)

    def _append_rank_pairs_from_move(self, state: AxisBayesState, image_idx: int, target_p01: float):
        if int(state.image_embeddings.shape[0]) <= 1:
            return
        target = float(max(0.0, min(1.0, target_p01)))
        z = self._legacy_projection_values(state)
        score01 = _rank_percentile_01(z)
        old_p = float(score01[int(image_idx)])

        k = int(max(1, state.rank_anchor_k))
        delta = float(max(0.01, min(0.45, state.rank_anchor_delta)))
        max_cross = int(max(2, 2 * k))
        move_mag = abs(target - old_p)
        moved_id = state.ids[int(image_idx)] if int(image_idx) < len(state.ids) else None
        base_move_weight = float(max(0.25, state.move_weights.get(str(moved_id), self.move_trust)))
        pair_weight = float(max(0.25, base_move_weight * (1.0 + (2.0 * move_mag))))

        all_idx = np.asarray(self._defined_indices(state), dtype=np.int32)
        not_self = all_idx[all_idx != int(image_idx)]
        below_band = not_self[(score01[not_self] < target) & (score01[not_self] >= (target - delta))]
        above_band = not_self[(score01[not_self] > target) & (score01[not_self] <= (target + delta))]

        def _pick(ids: np.ndarray, want_k: int, side: str) -> List[int]:
            if ids.size == 0:
                pool = not_self[score01[not_self] < target] if side == 'below' else not_self[score01[not_self] > target]
                ids = pool
            if ids.size == 0:
                return []
            order = np.argsort(np.abs(score01[ids] - target), kind='mergesort')
            picked = ids[order[: min(want_k, ids.size)]]
            return [int(v) for v in picked.tolist()]

        below = _pick(below_band, k, 'below')
        above = _pick(above_band, k, 'above')

        pair_i: List[int] = []
        pair_j: List[int] = []
        pair_y: List[float] = []
        pair_w: List[float] = []

        for j in below:
            pair_i.append(int(image_idx))
            pair_j.append(int(j))
            pair_y.append(1.0)
            pair_w.append(pair_weight)
        for j in above:
            pair_i.append(int(image_idx))
            pair_j.append(int(j))
            pair_y.append(-1.0)
            pair_w.append(pair_weight)

        lo = min(old_p, target)
        hi = max(old_p, target)
        crossed = not_self[(score01[not_self] >= lo) & (score01[not_self] <= hi)]
        if crossed.size > 0:
            if crossed.size > max_cross:
                step = max(1, int(np.floor(crossed.size / max_cross)))
                crossed = crossed[::step][:max_cross]
            crossed_y = 1.0 if target >= old_p else -1.0
            for j in crossed.tolist():
                pair_i.append(int(image_idx))
                pair_j.append(int(j))
                pair_y.append(float(crossed_y))
                pair_w.append(float(max(0.25, 0.9 * pair_weight)))

        if len(pair_i) == 0:
            return

        state.pair_i.extend(pair_i)
        state.pair_j.extend(pair_j)
        state.pair_y.extend(pair_y)
        state.pair_w.extend(pair_w)
        overflow = len(state.pair_i) - int(max(16, state.rank_max_pairs))
        if overflow > 0:
            state.pair_i = state.pair_i[overflow:]
            state.pair_j = state.pair_j[overflow:]
            state.pair_y = state.pair_y[overflow:]
            state.pair_w = state.pair_w[overflow:]

    def _solve_hinv_vec(self, alpha_inv_vec: np.ndarray, D: np.ndarray, w_diag: np.ndarray, vec: np.ndarray) -> np.ndarray:
        a_inv = np.asarray(alpha_inv_vec, dtype=np.float32).reshape(-1)
        if D.size == 0:
            return (a_inv * np.asarray(vec, dtype=np.float32)).astype(np.float32)
        D_Ainv = (D * a_inv[None, :]).astype(np.float32)
        DDt = (D_Ainv @ D.T).astype(np.float32)
        inv_w = (1.0 / np.maximum(np.asarray(w_diag, dtype=np.float32), 1e-8)).astype(np.float32)
        M = np.diag(inv_w) + DDt
        rhs = D @ (a_inv * np.asarray(vec, dtype=np.float32))
        try:
            sol = np.linalg.solve(M + (1e-6 * np.eye(M.shape[0], dtype=np.float32)), rhs)
        except Exception:
            sol = np.linalg.pinv(M + (1e-6 * np.eye(M.shape[0], dtype=np.float32))) @ rhs
        out = (a_inv * vec) - (a_inv * (D.T @ sol))
        return np.asarray(out, dtype=np.float32)

    def _rank_loss(
        self,
        w: np.ndarray,
        w0: np.ndarray,
        D: np.ndarray,
        weights: np.ndarray,
        eta: float,
        alpha_vec: np.ndarray,
    ) -> float:
        diff = np.asarray(w, dtype=np.float32) - np.asarray(w0, dtype=np.float32)
        u = (D @ np.asarray(w, dtype=np.float32)) / float(max(1e-6, eta))
        logistic = np.logaddexp(0.0, -u).astype(np.float32)
        reg = 0.5 * float(np.sum(np.asarray(alpha_vec, dtype=np.float32) * diff * diff))
        return float(reg + float(np.sum(weights * logistic)))

    def _refresh_posterior_gaussian(self, state: AxisBayesState):
        d = int(state.image_embeddings.shape[1])
        alpha_inv = state.alpha_inv_vec if state.alpha_inv_vec.shape[0] == d else np.full((d,), 1.0 / max(1e-8, self.alpha), dtype=np.float32)
        valid_move_ids = [
            image_id for image_id in state.move_order
            if image_id in state.id_to_index and image_id not in self._undefined_id_set(state)
        ]
        if len(valid_move_ids) == 0:
            state.X = np.zeros((0, d), dtype=np.float32)
            state.y = np.zeros((0,), dtype=np.float32)
            state.A_inv = np.zeros((0, 0), dtype=np.float32)
            state.mu = np.asarray(state.w0, dtype=np.float32).copy()
            state.b = float(state.b0)
            state.pair_D = np.zeros((0, d), dtype=np.float32)
            state.pair_W_diag = np.zeros((0,), dtype=np.float32)
            state.pair_M_inv = np.zeros((0, 0), dtype=np.float32)
            return

        move_idx = [state.id_to_index[image_id] for image_id in valid_move_ids]
        X = state.image_embeddings[move_idx, :].astype(np.float32)
        y = np.asarray(
            [self._quantile_from_sorted(state.z0_sorted, state.move_targets[image_id]) for image_id in valid_move_ids],
            dtype=np.float32,
        )
        obs_weights = np.asarray(
            [max(1e-4, float(state.move_weights.get(image_id, 1.0))) for image_id in valid_move_ids],
            dtype=np.float32,
        )
        obs_var = (self.sigma2 / np.maximum(obs_weights, 1e-4)).astype(np.float32)
        m = int(X.shape[0])
        XS = (X * alpha_inv[None, :]).astype(np.float32)
        XXt = XS @ X.T
        ones = np.ones((m,), dtype=np.float32)
        ones_outer = np.outer(ones, ones).astype(np.float32)
        A = (
            np.diag(obs_var.astype(np.float32))
            + XXt
            + ((1.0 / self.bias_alpha) * ones_outer)
        )
        A_inv = np.linalg.inv(A + (1e-6 * np.eye(m, dtype=np.float32)))
        r = y - (X @ state.w0) - float(state.b0)
        mu = state.w0 + (alpha_inv * (X.T @ (A_inv @ r)))
        b = float(state.b0) + float((1.0 / self.bias_alpha) * (ones @ (A_inv @ r)))
        if float(np.dot(mu, state.w0)) < 0.0:
            mu = -mu
            b = -b
        mu_norm = float(np.linalg.norm(mu))
        if mu_norm > 1e-8:
            mu = (mu / mu_norm).astype(np.float32)
            b = float(b / mu_norm)
        else:
            mu = _normalize_vec(state.w0)
            b = float(state.b0)

        state.X = X
        state.y = y
        state.A_inv = np.asarray(A_inv, dtype=np.float32)
        state.mu = np.asarray(mu, dtype=np.float32)
        state.b = float(b)
        state.pair_D = np.zeros((0, d), dtype=np.float32)
        state.pair_W_diag = np.zeros((0,), dtype=np.float32)
        state.pair_M_inv = np.zeros((0, 0), dtype=np.float32)

    def _refresh_posterior_rank(self, state: AxisBayesState):
        d = int(state.image_embeddings.shape[1])
        if len(state.pair_i) == 0:
            state.X = np.zeros((0, d), dtype=np.float32)
            state.y = np.zeros((0,), dtype=np.float32)
            state.A_inv = np.zeros((0, 0), dtype=np.float32)
            state.mu = np.asarray(state.w0, dtype=np.float32).copy()
            state.b = 0.0
            state.pair_D = np.zeros((0, d), dtype=np.float32)
            state.pair_W_diag = np.zeros((0,), dtype=np.float32)
            state.pair_M_inv = np.zeros((0, 0), dtype=np.float32)
            return

        idx_i = np.asarray(state.pair_i, dtype=np.int64)
        idx_j = np.asarray(state.pair_j, dtype=np.int64)
        y = np.asarray(state.pair_y, dtype=np.float32)
        weights = np.asarray(state.pair_w, dtype=np.float32)
        Xi = state.image_embeddings[idx_i, :]
        Xj = state.image_embeddings[idx_j, :]
        Delta = Xi - Xj
        D = (y[:, None] * Delta).astype(np.float32)
        eta = float(max(1e-6, state.rank_eta))

        w = np.asarray(state.mu if state.mu.size == d else state.w0, dtype=np.float32).copy()
        alpha_vec = state.alpha_vec if state.alpha_vec.shape[0] == d else np.full((d,), float(max(1e-6, self.alpha)), dtype=np.float32)
        alpha_inv = (1.0 / np.maximum(alpha_vec, 1e-8)).astype(np.float32)

        for _ in range(40):
            u = (D @ w) / eta
            sig_neg = _sigmoid(-u)
            grad = (alpha_vec * (w - state.w0)) - ((D.T @ (weights * sig_neg)) / eta)
            gnorm = float(np.linalg.norm(grad))
            if not np.isfinite(gnorm) or gnorm <= 1e-5:
                break

            p = _sigmoid(u)
            hdiag = np.maximum((weights * p * (1.0 - p)) / max(1e-8, eta ** 2), 1e-8).astype(np.float32)
            step = self._solve_hinv_vec(alpha_inv, D, hdiag, grad)
            step_dot = float(np.dot(grad, step))
            if not np.isfinite(step_dot) or step_dot <= 0.0:
                step = np.asarray(grad, dtype=np.float32) / np.maximum(alpha_vec, 1e-8)
                step_dot = float(np.dot(grad, step))
            if not np.isfinite(step_dot) or step_dot <= 0.0:
                break

            base_loss = self._rank_loss(w, state.w0, D, weights, eta, alpha_vec)
            t = 1.0
            accepted = False
            while t >= 1e-4:
                cand = np.asarray(w - (t * step), dtype=np.float32)
                cand_loss = self._rank_loss(cand, state.w0, D, weights, eta, alpha_vec)
                if np.isfinite(cand_loss) and cand_loss <= (base_loss - (1e-4 * t * step_dot)):
                    w = cand
                    accepted = True
                    break
                t *= 0.5
            if not accepted:
                w = np.asarray(w - (0.05 * step), dtype=np.float32)

        if float(np.dot(w, state.w0)) < 0.0:
            w = -w
        w_norm = float(np.linalg.norm(w))
        if w_norm > 1e-8:
            w = (w / w_norm).astype(np.float32)
        else:
            w = np.asarray(state.w0, dtype=np.float32).copy()

        u = (D @ w) / eta
        p = _sigmoid(u)
        w_diag = np.maximum((weights * p * (1.0 - p)) / max(1e-8, eta ** 2), 1e-8).astype(np.float32)
        D_Ainv = (D * alpha_inv[None, :]).astype(np.float32)
        DDt = (D_Ainv @ D.T).astype(np.float32)
        inv_w = (1.0 / w_diag).astype(np.float32)
        M = np.diag(inv_w) + DDt
        try:
            M_inv = np.linalg.inv(M + (1e-6 * np.eye(M.shape[0], dtype=np.float32)))
        except Exception:
            M_inv = np.linalg.pinv(M + (1e-6 * np.eye(M.shape[0], dtype=np.float32)))

        state.X = np.zeros((0, d), dtype=np.float32)
        state.y = np.zeros((0,), dtype=np.float32)
        state.A_inv = np.zeros((0, 0), dtype=np.float32)
        state.mu = np.asarray(w, dtype=np.float32)
        state.b = 0.0
        state.pair_D = np.asarray(D, dtype=np.float32)
        state.pair_W_diag = np.asarray(w_diag, dtype=np.float32)
        state.pair_M_inv = np.asarray(M_inv, dtype=np.float32)

    def _refresh_posterior_graph(self, state: AxisBayesState):
        n = int(len(state.ids))
        state.X = np.zeros((0, 0), dtype=np.float32)
        state.A_inv = np.zeros((0, 0), dtype=np.float32)
        state.mu = np.asarray(state.w0, dtype=np.float32).copy()
        state.b = float(state.b0)
        state.pair_D = np.zeros((0, 0), dtype=np.float32)
        state.pair_W_diag = np.zeros((0,), dtype=np.float32)
        state.pair_M_inv = np.zeros((0, 0), dtype=np.float32)

        if state.graph_prior_cov.size == 0:
            state.y = np.zeros((0,), dtype=np.float32)
            state.graph_posterior_mean = np.asarray(state.z0_all, dtype=np.float32).copy()
            state.graph_posterior_cov_diag = np.full((n,), 1.0 / max(1e-8, state.graph_lambda_prior), dtype=np.float32)
            state.graph_obs_indices = []
            state.graph_obs_var = np.zeros((0,), dtype=np.float32)
            state.graph_obs_S_inv = np.zeros((0, 0), dtype=np.float32)
            return

        valid_move_ids = [
            image_id for image_id in state.move_order
            if image_id in state.id_to_index and image_id not in self._undefined_id_set(state)
        ]
        if len(valid_move_ids) == 0:
            state.y = np.zeros((0,), dtype=np.float32)
            state.graph_posterior_mean = np.asarray(state.z0_all, dtype=np.float32).copy()
            state.graph_posterior_cov_diag = np.asarray(state.graph_prior_cov_diag, dtype=np.float32).copy()
            state.graph_obs_indices = []
            state.graph_obs_var = np.zeros((0,), dtype=np.float32)
            state.graph_obs_S_inv = np.zeros((0, 0), dtype=np.float32)
            return

        obs_idx = np.asarray([state.id_to_index[image_id] for image_id in valid_move_ids], dtype=np.int64)
        y = np.asarray(
            [self._quantile_from_sorted(state.z0_sorted, state.move_targets[image_id]) for image_id in valid_move_ids],
            dtype=np.float32,
        )
        obs_weights = np.asarray(
            [max(1e-4, float(state.move_weights.get(image_id, 1.0))) for image_id in valid_move_ids],
            dtype=np.float32,
        )
        obs_var = (self.sigma2 / np.maximum(obs_weights, 1e-4)).astype(np.float32)

        K0 = np.asarray(state.graph_prior_cov, dtype=np.float32)
        K_xo = K0[:, obs_idx]
        K_oo = K0[np.ix_(obs_idx, obs_idx)]
        S = np.asarray(K_oo + np.diag(obs_var), dtype=np.float64)
        jitter = self.graph_jitter * np.eye(S.shape[0], dtype=np.float64)
        try:
            S_inv = np.linalg.inv(S + jitter).astype(np.float32)
        except Exception:
            S_inv = np.linalg.pinv(S + jitter).astype(np.float32)

        residual = np.asarray(y - state.z0_all[obs_idx], dtype=np.float32)
        gain = np.asarray(S_inv @ residual, dtype=np.float32)
        post_mean = np.asarray(state.z0_all + (K_xo @ gain), dtype=np.float32)

        tmp = np.asarray(S_inv @ K_xo.T, dtype=np.float32)
        quad = np.sum(K_xo * tmp.T, axis=1).astype(np.float32)
        post_var = np.asarray(state.graph_prior_cov_diag - quad, dtype=np.float32)

        state.y = y
        state.graph_posterior_mean = post_mean
        state.graph_posterior_cov_diag = np.maximum(post_var, 1e-8).astype(np.float32)
        state.graph_obs_indices = [int(v) for v in obs_idx.tolist()]
        state.graph_obs_var = obs_var
        state.graph_obs_S_inv = S_inv

    def _refresh_posterior_legacy(self, state: AxisBayesState):
        mode = _normalize_mode(state.mode)
        if mode == 'graph':
            self._refresh_posterior_graph(state)
            return
        if mode == 'rank':
            self._refresh_posterior_rank(state)
            return
        self._refresh_posterior_gaussian(state)

    def _predict_std_gaussian(self, state: AxisBayesState) -> np.ndarray:
        X_all = state.image_embeddings
        d = int(X_all.shape[1])
        alpha_inv = state.alpha_inv_vec if state.alpha_inv_vec.shape[0] == d else np.full((d,), 1.0 / max(1e-8, self.alpha), dtype=np.float32)
        n = int(X_all.shape[0])
        prior_x = np.sum((X_all * X_all) * alpha_inv[None, :], axis=1).astype(np.float32)
        if state.X.size == 0 or state.A_inv.size == 0:
            var = self.sigma2 + prior_x + (1.0 / self.bias_alpha)
            return np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

        XS = (state.X * alpha_inv[None, :]).astype(np.float32)
        U = XS @ X_all.T  # m x n
        ones = np.ones((state.X.shape[0], 1), dtype=np.float32)
        V = U + ((1.0 / self.bias_alpha) * ones)
        AV = state.A_inv @ V  # m x n
        quad = np.sum(V * AV, axis=0).astype(np.float32)
        var = self.sigma2 + prior_x + (1.0 / self.bias_alpha) - quad
        return np.sqrt(np.maximum(var, 1e-8)).astype(np.float32).reshape(n)

    def _predict_std_rank(self, state: AxisBayesState) -> np.ndarray:
        X_all = state.image_embeddings
        d = int(X_all.shape[1])
        alpha_inv = state.alpha_inv_vec if state.alpha_inv_vec.shape[0] == d else np.full((d,), 1.0 / max(1e-8, self.alpha), dtype=np.float32)
        n = int(X_all.shape[0])
        prior_x = np.sum((X_all * X_all) * alpha_inv[None, :], axis=1).astype(np.float32)
        if state.pair_D.size == 0 or state.pair_M_inv.size == 0:
            var = prior_x
            return np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

        V = (state.pair_D * alpha_inv[None, :]) @ X_all.T  # m x n
        MV = state.pair_M_inv @ V
        quad = np.sum(V * MV, axis=0)
        var = prior_x - quad
        return np.sqrt(np.maximum(var, 1e-8)).astype(np.float32).reshape(n)

    def _predict_std_graph(self, state: AxisBayesState) -> np.ndarray:
        n = int(len(state.ids))
        if state.graph_posterior_cov_diag.size == n:
            return np.sqrt(np.maximum(state.graph_posterior_cov_diag, 1e-8)).astype(np.float32)
        if state.graph_prior_cov_diag.size == n:
            return np.sqrt(np.maximum(state.graph_prior_cov_diag, 1e-8)).astype(np.float32)
        return np.zeros((n,), dtype=np.float32)

    def _predict_std_legacy(self, state: AxisBayesState) -> np.ndarray:
        mode = _normalize_mode(state.mode)
        if mode == 'graph':
            return self._predict_std_graph(state)
        if mode == 'rank':
            return self._predict_std_rank(state)
        return self._predict_std_gaussian(state)

    def _select_diverse_ids(
        self,
        embeddings: np.ndarray,
        candidate_indices: List[int],
        priority: np.ndarray,
        k: int,
    ) -> List[int]:
        if len(candidate_indices) == 0 or k <= 0:
            return []
        if len(candidate_indices) <= k:
            return list(candidate_indices)

        pr = np.asarray(priority, dtype=np.float32).reshape(-1)
        pr = pr - float(np.min(pr))
        denom = float(np.max(pr)) if pr.size > 0 else 0.0
        if denom > 1e-8:
            pr = pr / denom
        else:
            pr = np.ones_like(pr, dtype=np.float32)

        priority_by_idx = {
            int(candidate_indices[i]): float(pr[i])
            for i in range(min(len(candidate_indices), int(pr.size)))
        }

        selected = [candidate_indices[int(np.argmax(pr))]]
        remaining = [idx for idx in candidate_indices if idx not in selected]

        while len(selected) < k and len(remaining) > 0:
            best_idx = remaining[0]
            best_score = -1.0
            for idx in remaining:
                min_dist = min(_cosine_distance(embeddings[idx], embeddings[sidx]) for sidx in selected)
                score = (0.72 * float(priority_by_idx.get(int(idx), 0.0))) + (0.28 * float(min_dist))
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            remaining = [idx for idx in remaining if idx != best_idx]
        return selected

    def _build_decile_exemplars(self, state: AxisBayesState, score01: np.ndarray, std: np.ndarray) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        ids = state.ids
        X = state.image_embeddings
        defined_idx = np.asarray(self._defined_indices(state), dtype=np.int64)
        if defined_idx.size == 0:
            return [
                {
                    'bin_index': int(bin_idx),
                    'score_range': [int(bin_idx * 10), int((bin_idx + 1) * 10)],
                    'center_score_0_100': float((bin_idx * 10) + 5),
                    'count': 0,
                    'exemplars': [],
                }
                for bin_idx in range(10)
            ]
        local_score01 = _rank_percentile_01(score01[defined_idx])
        local_score100 = local_score01 * 100.0
        local_lookup = {int(defined_idx[pos]): int(pos) for pos in range(int(defined_idx.shape[0]))}

        for bin_idx in range(10):
            lo = bin_idx * 0.1
            hi = 1.0 if bin_idx == 9 else ((bin_idx + 1) * 0.1)
            center = lo + 0.05
            in_bin_local = np.where((local_score01 >= lo) & ((local_score01 < hi) if bin_idx < 9 else (local_score01 <= hi)))[0]
            in_bin = defined_idx[in_bin_local]
            in_bin_set = {int(idx) for idx in in_bin.tolist()}
            candidate_order = defined_idx[np.argsort(np.abs(local_score01 - center), kind='mergesort')]
            candidate_indices = [int(idx) for idx in candidate_order.tolist() if int(idx) in in_bin_set][:50]
            if len(candidate_indices) == 0:
                candidate_indices = [int(idx) for idx in candidate_order[:50].tolist()]
            priority = 1.0 / (1.0 + np.abs(np.asarray([
                local_score01[local_lookup[int(idx)]]
                for idx in candidate_indices
            ], dtype=np.float32) - center))
            selected_idx = self._select_diverse_ids(
                embeddings=X,
                candidate_indices=candidate_indices,
                priority=np.asarray(priority, dtype=np.float32),
                k=self.exemplar_k,
            )
            exemplars = [
                {
                    'id': ids[idx],
                    'score_0_100': float(local_score100[local_lookup[int(idx)]]),
                    'std': float(std[idx]),
                }
                for idx in selected_idx
            ]
            out.append({
                'bin_index': int(bin_idx),
                'score_range': [int(bin_idx * 10), int((bin_idx + 1) * 10)],
                'center_score_0_100': float((bin_idx * 10) + 5),
                'count': int(len(in_bin_local)),
                'exemplars': exemplars,
            })
        return out

    def _build_hotspots(self, state: AxisBayesState, score01: np.ndarray, std: np.ndarray) -> List[Dict[str, Any]]:
        ids = state.ids
        X = state.image_embeddings
        defined_idx = np.asarray(self._defined_indices(state), dtype=np.int64)
        if defined_idx.size == 0:
            return []
        local_score01 = _rank_percentile_01(score01[defined_idx])
        local_score100 = local_score01 * 100.0
        local_std = std[defined_idx]
        priority = local_std * np.exp(-np.abs(local_score100 - self.hotspot_boundary) / max(1e-6, self.hotspot_tau))
        order = np.argsort(-priority, kind='mergesort')
        candidate_indices = [int(defined_idx[idx]) for idx in order[: max(32, self.hotspot_k * 6)].tolist()]
        local_lookup = {int(defined_idx[pos]): int(pos) for pos in range(int(defined_idx.shape[0]))}
        selected_idx = self._select_diverse_ids(
            embeddings=X,
            candidate_indices=candidate_indices,
            priority=np.asarray([priority[local_lookup[int(idx)]] for idx in candidate_indices], dtype=np.float32),
            k=self.hotspot_k,
        )
        out = []
        for idx in selected_idx:
            out.append({
                'id': ids[idx],
                'score_0_100': float(local_score100[local_lookup[int(idx)]]),
                'std': float(std[idx]),
                'priority': float(priority[local_lookup[int(idx)]]),
            })
        return out

    def _state_payload(self, state: AxisBayesState) -> Dict[str, Any]:
        self._fit_state(state)
        z = self._score_state(state)
        score01 = _rank_percentile_01(z)
        score100 = (score01 * 100.0).astype(np.float32)
        std = self._uncertainty_state(state)
        z_min = float(np.min(z)) if z.size > 0 else 0.0
        z_max = float(np.max(z)) if z.size > 0 else 0.0
        z_span = z_max - z_min

        if z.size == 0 or z_span <= 1e-8:
            coords = {state.ids[i]: 0.5 for i in range(len(state.ids))}
        else:
            coords = {
                state.ids[i]: float((z[i] - z_min) / z_span)
                for i in range(len(state.ids))
            }
        deciles = self._build_decile_exemplars(state=state, score01=score01, std=std)
        hotspots = self._build_hotspots(state=state, score01=score01, std=std)

        move_payload = []
        for image_id in state.move_order:
            idx = state.id_to_index[image_id]
            move_payload.append({
                'image_id': image_id,
                'target_score_0_100': float(state.move_targets[image_id] * 100.0),
                'current_score_0_100': float(score100[idx]),
                'move_weight': float(state.move_weights.get(image_id, 1.0)),
            })

        mode_name = _normalize_mode(state.mode)
        if state.model_type == 'piecewise_linear':
            scoring_method = 'piecewise_linear_ranker'
        elif state.model_type == 'residual':
            scoring_method = 'bayesian_smooth_residual'
        elif mode_name == 'graph':
            scoring_method = 'bayesian_graph_field'
        elif mode_name == 'rank':
            scoring_method = 'bayesian_pairwise_rank'
        else:
            scoring_method = 'bayesian_ridge_refinement'

        return {
            'axis_id': state.axis_id,
            'collection_id': state.collection_id,
            'dataset_root': state.dataset_root,
            'q': state.q,
            'mode': mode_name,
            'model_type': str(state.model_type),
            'w0_summary': {
                'prompt_count': int(len(state.prompt_ensemble)),
                'pos_prompt_count': int(len(state.pos_prompt_ensemble)),
                'neg_prompt_count': int(len(state.neg_prompt_ensemble)),
                'pos_prompts': list(state.pos_prompt_ensemble),
                'neg_prompts': list(state.neg_prompt_ensemble),
                'prompts': list(state.prompt_ensemble),
                'prompt_source': state.prompt_source,
                'prompt_provider': state.prompt_provider,
                'embedding_dim': int(state.w0.shape[0]),
                'prior_bias': float(state.b0),
                'posterior_bias': float(state.b),
                'prior_projection_min': float(np.min(state.z0_all)) if state.z0_all.size > 0 else 0.0,
                'prior_projection_max': float(np.max(state.z0_all)) if state.z0_all.size > 0 else 0.0,
                'posterior_projection_min': float(np.min(z)) if z.size > 0 else 0.0,
                'posterior_projection_max': float(np.max(z)) if z.size > 0 else 0.0,
                'feature_space': str(state.feature_space),
                'semantic_method': str(state.semantic_method),
                'clip_dim': int(state.clip_dim),
                'dino_dim': int(state.dino_dim),
                'clip_scale': float(state.clip_scale),
                'dino_scale': float(state.dino_scale),
                'piecewise_clip_scale': float(state.piecewise_clip_scale),
                'piecewise_dino_scale': float(state.piecewise_dino_scale),
                'piecewise_num_experts': int(state.piecewise_num_experts),
                'piecewise_use_gating': bool(state.piecewise_use_gating),
                'piecewise_aggregator': str(state.piecewise_aggregator),
                'residual_alpha': float(state.residual_alpha),
                'residual_beta': float(state.residual_beta),
                'residual_lambda': float(state.residual_lambda),
                'residual_sigma_y': float(state.residual_sigma_y),
                'residual_lengthscale': float(state.residual_lengthscale),
                'graph_knn_k': int(state.graph_knn_k),
                'graph_lambda_smooth': float(state.graph_lambda_smooth),
                'graph_lambda_prior': float(state.graph_lambda_prior),
                'graph_similarity_scale': float(state.graph_similarity_scale),
            },
            'alpha': float(state.alpha),
            'dino_alpha': float(state.dino_alpha),
            'bias_alpha': float(state.bias_alpha),
            'sigma2': float(state.sigma2),
            'residual_alpha': float(state.residual_alpha),
            'residual_beta': float(state.residual_beta),
            'residual_lambda': float(state.residual_lambda),
            'residual_sigma_y': float(state.residual_sigma_y),
            'residual_lengthscale_multiplier': float(state.residual_lengthscale_multiplier),
            'residual_lengthscale': float(state.residual_lengthscale),
            'residual_jitter': float(state.residual_jitter),
            'piecewise_num_experts': int(state.piecewise_num_experts),
            'piecewise_use_gating': bool(state.piecewise_use_gating),
            'piecewise_aggregator': str(state.piecewise_aggregator),
            'pairwise_from_scalar_margin': float(state.pairwise_from_scalar_margin),
            'graph_knn_k': int(state.graph_knn_k),
            'graph_lambda_smooth': float(state.graph_lambda_smooth),
            'graph_lambda_prior': float(state.graph_lambda_prior),
            'rank_eta': float(state.rank_eta),
            'pair_count': int(len(state.pair_i)),
            'max_moves': int(state.max_moves),
            'move_count': int(len(state.move_order)),
            'undefined_count': int(len(state.undefined_order)),
            'undefined_ids': list(state.undefined_order),
            'moves': move_payload,
            'ids': list(state.ids),
            'projection_values': z.astype(np.float32).tolist(),
            'projection_min': z_min,
            'projection_max': z_max,
            'scores': score100.astype(np.float32).tolist(),
            'std': std.astype(np.float32).tolist(),
            'decile_exemplars': deciles,
            'hotspots': hotspots,
            'axis': {
                'id': state.axis_id,
                'name': state.axis_name,
                'coords': coords,
                'group': 'bayes',
                'attribute_type': 'continuous',
                'scoring_method': scoring_method,
                'model_type': str(state.model_type),
                'feature_space': str(state.feature_space),
                'semantic_method': str(state.semantic_method),
                'labels': [
                    _format_axis_value(z_min),
                    _format_axis_value(z_min + (0.5 * z_span)),
                    _format_axis_value(z_max),
                ],
                'label_positions': [0.0, 0.5, 1.0],
            },
        }

    def create_axis(
        self,
        collection_id: str,
        dataset_root: str,
        q: str,
        mode: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        query = re.sub(r'\s+', ' ', str(q or '').strip())
        if not query:
            raise ValueError('Missing dimension query')
        coll = self._get_collection(dataset_root=dataset_root, collection_id=collection_id)
        model_name = _normalize_model_type(model_type if model_type is not None else self.model_type)
        mode_name = _normalize_mode(mode if mode is not None else self.mode)
        if model_name == 'piecewise_linear':
            mode_name = 'rank'
        if mode_name == 'graph':
            coll = self._ensure_graph_prior(coll)
        w0_clip, pos_prompts, neg_prompts, prompt_meta = self._embed_prompt_ensemble(
            query,
            semantic_method=coll.semantic_method,
        )
        if int(w0_clip.shape[0]) != int(coll.clip_dim):
            raise ValueError(f'Semantic text/image dim mismatch: text={w0_clip.shape[0]} image_semantic={coll.clip_dim}')
        if coll.dino_dim > 0:
            w0 = np.concatenate(
                [
                    w0_clip.astype(np.float32),
                    np.zeros((coll.dino_dim,), dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)
        else:
            w0 = w0_clip.astype(np.float32)
        if model_name == 'residual':
            if coll.feature_space == 'clip_dino':
                clip_raw = coll.embeddings[:, : coll.clip_dim] / max(1e-8, float(coll.clip_scale))
            else:
                clip_raw = coll.embeddings[:, : coll.clip_dim]
            z0_all = (
                (float(self.residual_alpha) * _rank_percentile_01(clip_raw @ w0[: coll.clip_dim]))
                + float(self.residual_beta)
            ).astype(np.float32)
        else:
            z0_all = (coll.embeddings @ w0).astype(np.float32)
        z0_sorted = np.sort(z0_all).astype(np.float32)
        alpha_vec = np.concatenate(
            [
                np.full((coll.clip_dim,), float(self.alpha), dtype=np.float32),
                np.full((coll.dino_dim,), float(self.dino_alpha), dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        alpha_inv_vec = (1.0 / np.maximum(alpha_vec, 1e-8)).astype(np.float32)
        axis_id = f'axis:bayes:{_slugify(query)}:{uuid.uuid4().hex[:8]}'
        axis_name = query[:1].upper() + query[1:] if query else 'Axis'
        state = AxisBayesState(
            axis_id=axis_id,
            collection_id=str(collection_id or Path(dataset_root).name),
            dataset_root=str(Path(dataset_root).resolve()),
            q=query,
            axis_name=axis_name,
            model_type=model_name,
            ids=list(coll.ids),
            image_embeddings=coll.embeddings,
            id_to_index=coll.id_to_index,
            w0=w0,
            pos_prompt_ensemble=list(pos_prompts),
            neg_prompt_ensemble=list(neg_prompts),
            prompt_ensemble=list(pos_prompts) + list(neg_prompts),
            prompt_source=str(prompt_meta.get('source') or 'unknown'),
            prompt_provider=str(prompt_meta.get('provider') or 'unknown'),
            mode=mode_name,
            feature_space=coll.feature_space,
            semantic_method=coll.semantic_method,
            clip_dim=int(coll.clip_dim),
            dino_dim=int(coll.dino_dim),
            clip_scale=float(coll.clip_scale),
            dino_scale=float(coll.dino_scale),
            alpha=self.alpha,
            dino_alpha=self.dino_alpha,
            bias_alpha=self.bias_alpha,
            sigma2=self.sigma2,
            rank_eta=self.rank_eta,
            rank_anchor_k=self.rank_anchor_k,
            rank_anchor_delta=self.rank_anchor_delta,
            rank_max_pairs=self.rank_max_pairs,
            max_moves=self.max_moves,
            graph_knn_k=int(coll.graph_knn_k),
            graph_lambda_smooth=float(self.graph_lambda_smooth),
            graph_lambda_prior=float(self.graph_lambda_prior),
            graph_jitter=float(self.graph_jitter),
            graph_similarity_scale=float(coll.graph_similarity_scale),
            b0=0.0,
            z0_all=z0_all,
            z0_sorted=z0_sorted,
            alpha_vec=alpha_vec,
            alpha_inv_vec=alpha_inv_vec,
            graph_prior_cov=np.asarray(coll.graph_prior_cov, dtype=np.float32),
            graph_prior_cov_diag=np.asarray(coll.graph_prior_cov_diag, dtype=np.float32),
            piecewise_num_experts=int(self.piecewise_num_experts),
            piecewise_use_gating=bool(self.piecewise_use_gating),
            piecewise_aggregator=str(self.piecewise_aggregator),
            piecewise_clip_scale=float(self.piecewise_clip_scale),
            piecewise_dino_scale=float(self.piecewise_dino_scale),
            pairwise_from_scalar_margin=float(self.pairwise_from_scalar_margin),
            piecewise_prior_strength=float(self.piecewise_prior_strength),
            piecewise_diversity_strength=float(self.piecewise_expert_diversity_strength),
            piecewise_l2_reg=float(self.piecewise_l2_reg),
            piecewise_learning_rate=float(self.piecewise_learning_rate),
            piecewise_max_refine_steps=int(self.piecewise_max_refine_steps),
            residual_alpha=float(self.residual_alpha),
            residual_beta=float(self.residual_beta),
            residual_lambda=float(self.residual_lambda),
            residual_sigma_y=float(self.residual_sigma_y),
            residual_lengthscale_multiplier=float(self.residual_lengthscale_multiplier),
            residual_lengthscale=0.0,
            residual_jitter=float(self.residual_jitter),
        )
        self._scorer_for_state(state).initialize_state(self, state)
        self._axes[axis_id] = state
        self._log(
            'axis created axis_id=%s collection=%s q="%s" mode=%s model_type=%s',
            axis_id,
            collection_id,
            query,
            mode_name,
            model_name,
        )
        return self._state_payload(state)

    def _rebuild_rank_pairs_from_feedback(self, state: AxisBayesState) -> None:
        state.pair_i = []
        state.pair_j = []
        state.pair_y = []
        state.pair_w = []
        state.pair_D = np.zeros((0, state.image_embeddings.shape[1]), dtype=np.float32)
        state.pair_W_diag = np.zeros((0,), dtype=np.float32)
        state.pair_M_inv = np.zeros((0, 0), dtype=np.float32)
        for image_id in list(state.move_order):
            if image_id not in state.id_to_index:
                continue
            if image_id in self._undefined_id_set(state):
                continue
            self._append_rank_pairs_from_move(
                state=state,
                image_idx=int(state.id_to_index[image_id]),
                target_p01=float(state.move_targets.get(image_id, 0.5)),
            )

    def _reset_state_for_new_prior(self, state: AxisBayesState) -> None:
        n = int(len(state.ids))
        d = int(state.image_embeddings.shape[1]) if state.image_embeddings.ndim == 2 else 0
        state.b0 = 0.0
        if state.model_type == 'residual':
            self._refresh_residual_prior(state)
        else:
            state.z0_all = np.asarray(state.image_embeddings @ state.w0, dtype=np.float32)
        state.z0_sorted = np.sort(state.z0_all).astype(np.float32)
        state.X = np.zeros((0, d), dtype=np.float32)
        state.y = np.zeros((0,), dtype=np.float32)
        state.A_inv = np.zeros((0, 0), dtype=np.float32)
        state.mu = np.asarray(state.w0, dtype=np.float32).copy()
        state.b = 0.0
        state.pair_D = np.zeros((0, d), dtype=np.float32)
        state.pair_W_diag = np.zeros((0,), dtype=np.float32)
        state.pair_M_inv = np.zeros((0, 0), dtype=np.float32)
        state.graph_posterior_mean = np.asarray(state.z0_all, dtype=np.float32).copy()
        if state.graph_prior_cov_diag.size == n:
            state.graph_posterior_cov_diag = np.asarray(state.graph_prior_cov_diag, dtype=np.float32).copy()
        else:
            state.graph_posterior_cov_diag = np.zeros((n,), dtype=np.float32)
        state.graph_obs_indices = []
        state.graph_obs_var = np.zeros((0,), dtype=np.float32)
        state.graph_obs_S_inv = np.zeros((0, 0), dtype=np.float32)
        if state.model_type == 'residual':
            state.residual_features = self._build_residual_feature_matrix(state)
            state.residual_external_features = np.zeros((0, 0), dtype=np.float32)
            state.residual_external_targets = np.zeros((0,), dtype=np.float32)
            state.residual_external_weights = np.zeros((0,), dtype=np.float32)
            state.residual_posterior_mean = np.asarray(state.z0_all, dtype=np.float32).copy()
            state.residual_posterior_std = np.full((n,), float(max(1e-6, state.residual_lambda)), dtype=np.float32)
            state.residual_last_fit_signature = ''
            return
        if state.model_type == 'piecewise_linear':
            state.piecewise_features = self._build_piecewise_feature_matrix(state)
            state.piecewise_prior_vector = self._build_piecewise_prior_vector(state)
            state.piecewise_expert_weights = np.zeros((0, 0), dtype=np.float32)
            state.piecewise_gate_weights = np.zeros((0, 0), dtype=np.float32)
            state.piecewise_gate_bias = np.zeros((0,), dtype=np.float32)
            state.piecewise_scores_all = np.zeros((n,), dtype=np.float32)
            state.piecewise_expert_scores_all = np.zeros((n, 0), dtype=np.float32)
            state.piecewise_uncertainty_all = np.zeros((n,), dtype=np.float32)
            state.piecewise_pair_i = []
            state.piecewise_pair_j = []
            state.piecewise_pair_w = []
            state.piecewise_last_fit_signature = ''
            return
        if _normalize_mode(state.mode) == 'rank':
            self._rebuild_rank_pairs_from_feedback(state)
        else:
            state.pair_i = []
            state.pair_j = []
            state.pair_y = []
            state.pair_w = []

    def update_axis_prompts(
        self,
        axis_id: str,
        pos_prompts: Any,
        neg_prompts: Any,
    ) -> Dict[str, Any]:
        state = self._axes.get(str(axis_id or '').strip())
        if state is None:
            raise KeyError(f'Unknown axis_id: {axis_id}')
        w0_clip, pos_list, neg_list, prompt_meta = self._embed_prompt_lists(
            pos_prompts,
            neg_prompts,
            semantic_method=state.semantic_method,
            source='manual_edit',
            provider='user_edit',
        )
        if int(w0_clip.shape[0]) != int(state.clip_dim):
            raise ValueError(f'Semantic text/image dim mismatch: text={w0_clip.shape[0]} image_semantic={state.clip_dim}')
        if int(state.dino_dim) > 0:
            w0 = np.concatenate(
                [
                    np.asarray(w0_clip, dtype=np.float32),
                    np.zeros((state.dino_dim,), dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)
        else:
            w0 = np.asarray(w0_clip, dtype=np.float32)
        state.w0 = np.asarray(w0, dtype=np.float32)
        state.pos_prompt_ensemble = list(pos_list)
        state.neg_prompt_ensemble = list(neg_list)
        state.prompt_ensemble = list(pos_list) + list(neg_list)
        state.prompt_source = str(prompt_meta.get('source') or 'manual_edit')
        state.prompt_provider = str(prompt_meta.get('provider') or 'user_edit')
        self._reset_state_for_new_prior(state)
        self._log(
            'axis prompt update axis_id=%s pos_count=%d neg_count=%d model_type=%s mode=%s',
            state.axis_id,
            len(state.pos_prompt_ensemble),
            len(state.neg_prompt_ensemble),
            state.model_type,
            state.mode,
        )
        return self._state_payload(state)

    def move_axis(self, axis_id: str, image_id: str, new_score_0_100: float, move_type: Optional[str] = None) -> Dict[str, Any]:
        state = self._axes.get(str(axis_id or '').strip())
        if state is None:
            raise KeyError(f'Unknown axis_id: {axis_id}')
        image_key = str(image_id or '').strip()
        if image_key not in state.id_to_index:
            base = Path(image_key).name
            if base:
                matches = [candidate for candidate in state.ids if Path(str(candidate)).name == base]
                if len(matches) == 1:
                    image_key = str(matches[0])
        if image_key not in state.id_to_index:
            raise KeyError(f'Unknown image_id: {image_key}')

        move_kind = str(move_type or 'score').strip().lower()
        if move_kind in {'undefined', 'exclude'}:
            self._set_image_undefined(state, image_key)
            self._log(
                'axis move axis_id=%s image_id=%s move_type=undefined undefined_count=%d model_type=%s',
                state.axis_id,
                image_key,
                len(state.undefined_order),
                state.model_type,
            )
            return self._state_payload(state)

        self._clear_image_undefined(state, image_key)
        target = float(max(0.0, min(100.0, float(new_score_0_100)))) / 100.0
        z_now = self._score_state(state)
        score_now = _rank_percentile_01(z_now)
        current_score = float(score_now[state.id_to_index[image_key]])
        move_mag = abs(target - current_score)
        move_weight = float(max(1e-3, self.move_trust + (self.move_mag_gain * move_mag)))
        if image_key not in state.move_targets:
            if int(state.max_moves) > 0 and len(state.move_order) >= int(state.max_moves):
                raise ValueError(f'Maximum number of moves reached ({state.max_moves})')
            state.move_order.append(image_key)
        state.move_targets[image_key] = target
        state.move_weights[image_key] = move_weight
        if state.model_type == 'bayes_linear' and _normalize_mode(state.mode) == 'rank':
            self._fit_state(state)
            self._append_rank_pairs_from_move(
                state=state,
                image_idx=int(state.id_to_index[image_key]),
                target_p01=target,
            )
        if state.model_type == 'piecewise_linear':
            state.piecewise_last_fit_signature = ''
        self._log(
            'axis move axis_id=%s image_id=%s target=%.1f current=%.1f weight=%.2f move_count=%d model_type=%s',
            state.axis_id,
            image_key,
            target * 100.0,
            current_score * 100.0,
            move_weight,
            len(state.move_order),
            state.model_type,
        )
        return self._state_payload(state)

    def _serialize_state(self, state: AxisBayesState, fit: bool = True) -> Dict[str, Any]:
        """Serialize an in-memory axis state, optionally without refitting first."""
        if fit:
            self._fit_state(state)
        scorer = self._scorer_for_state(state)
        return {
            'format_version': 2,
            'axis_id': state.axis_id,
            'collection_id': state.collection_id,
            'dataset_root': state.dataset_root,
            'q': state.q,
            'axis_name': state.axis_name,
            'model_type': state.model_type,
            'mode': state.mode,
            'ids': list(state.ids),
            'id_to_index': {str(k): int(v) for k, v in state.id_to_index.items()},
            'image_embeddings': np.asarray(state.image_embeddings, dtype=np.float32).tolist(),
            'w0': np.asarray(state.w0, dtype=np.float32).tolist(),
            'pos_prompt_ensemble': list(state.pos_prompt_ensemble),
            'neg_prompt_ensemble': list(state.neg_prompt_ensemble),
            'prompt_ensemble': list(state.prompt_ensemble),
            'prompt_source': str(state.prompt_source),
            'prompt_provider': str(state.prompt_provider),
            'feature_space': str(state.feature_space),
            'semantic_method': str(state.semantic_method),
            'clip_dim': int(state.clip_dim),
            'dino_dim': int(state.dino_dim),
            'clip_scale': float(state.clip_scale),
            'dino_scale': float(state.dino_scale),
            'alpha': float(state.alpha),
            'dino_alpha': float(state.dino_alpha),
            'bias_alpha': float(state.bias_alpha),
            'sigma2': float(state.sigma2),
            'rank_eta': float(state.rank_eta),
            'rank_anchor_k': int(state.rank_anchor_k),
            'rank_anchor_delta': float(state.rank_anchor_delta),
            'rank_max_pairs': int(state.rank_max_pairs),
            'max_moves': int(state.max_moves),
            'graph_knn_k': int(state.graph_knn_k),
            'graph_lambda_smooth': float(state.graph_lambda_smooth),
            'graph_lambda_prior': float(state.graph_lambda_prior),
            'graph_jitter': float(state.graph_jitter),
            'graph_similarity_scale': float(state.graph_similarity_scale),
            'b0': float(state.b0),
            'z0_all': np.asarray(state.z0_all, dtype=np.float32).tolist(),
            'z0_sorted': np.asarray(state.z0_sorted, dtype=np.float32).tolist(),
            'alpha_vec': np.asarray(state.alpha_vec, dtype=np.float32).tolist(),
            'alpha_inv_vec': np.asarray(state.alpha_inv_vec, dtype=np.float32).tolist(),
            'move_order': list(state.move_order),
            'move_targets': {str(k): float(v) for k, v in state.move_targets.items()},
            'move_weights': {str(k): float(v) for k, v in state.move_weights.items()},
            'undefined_order': list(state.undefined_order),
            'piecewise_num_experts': int(state.piecewise_num_experts),
            'piecewise_use_gating': bool(state.piecewise_use_gating),
            'piecewise_aggregator': str(state.piecewise_aggregator),
            'piecewise_clip_scale': float(state.piecewise_clip_scale),
            'piecewise_dino_scale': float(state.piecewise_dino_scale),
            'pairwise_from_scalar_margin': float(state.pairwise_from_scalar_margin),
            'piecewise_prior_strength': float(state.piecewise_prior_strength),
            'piecewise_diversity_strength': float(state.piecewise_diversity_strength),
            'piecewise_l2_reg': float(state.piecewise_l2_reg),
            'piecewise_learning_rate': float(state.piecewise_learning_rate),
            'piecewise_max_refine_steps': int(state.piecewise_max_refine_steps),
            'residual_alpha': float(state.residual_alpha),
            'residual_beta': float(state.residual_beta),
            'residual_lambda': float(state.residual_lambda),
            'residual_sigma_y': float(state.residual_sigma_y),
            'residual_lengthscale_multiplier': float(state.residual_lengthscale_multiplier),
            'residual_lengthscale': float(state.residual_lengthscale),
            'residual_jitter': float(state.residual_jitter),
            'scorer_state': scorer.export_state(state),
        }

    def serialize_portable_axis(self, axis_id: str) -> Dict[str, Any]:
        """Serialize only the transferable refined axis state for reuse on other datasets.

        For linear and piecewise models, this stores the learned scorer parameters and
        the feature-space / prompt metadata needed to reapply the axis to another
        collection. Graph-mode axes remain collection-bound because their posterior
        lives on dataset nodes rather than in a transferable parametric form.
        """
        state = self._axes.get(str(axis_id or '').strip())
        if state is None:
            raise KeyError(f'Unknown axis_id: {axis_id}')

        if state.model_type == 'bayes_linear' and state.mode == 'graph':
            payload = self._serialize_state(state, fit=True)
            payload['artifact_scope'] = 'collection_bound'
            return payload

        self._fit_state(state)
        base = {
            'format_version': 2,
            'artifact_scope': 'portable',
            'axis_id': state.axis_id,
            'q': state.q,
            'axis_name': state.axis_name,
            'model_type': state.model_type,
            'mode': state.mode,
            'w0': np.asarray(state.w0, dtype=np.float32).tolist(),
            'pos_prompt_ensemble': list(state.pos_prompt_ensemble),
            'neg_prompt_ensemble': list(state.neg_prompt_ensemble),
            'prompt_ensemble': list(state.prompt_ensemble),
            'prompt_source': str(state.prompt_source),
            'prompt_provider': str(state.prompt_provider),
            'feature_space': str(state.feature_space),
            'clip_dim': int(state.clip_dim),
            'dino_dim': int(state.dino_dim),
            'clip_scale': float(state.clip_scale),
            'dino_scale': float(state.dino_scale),
            'alpha': float(state.alpha),
            'dino_alpha': float(state.dino_alpha),
            'bias_alpha': float(state.bias_alpha),
            'sigma2': float(state.sigma2),
            'rank_eta': float(state.rank_eta),
            'rank_anchor_k': int(state.rank_anchor_k),
            'rank_anchor_delta': float(state.rank_anchor_delta),
            'rank_max_pairs': int(state.rank_max_pairs),
            'max_moves': int(state.max_moves),
            'graph_knn_k': int(state.graph_knn_k),
            'graph_lambda_smooth': float(state.graph_lambda_smooth),
            'graph_lambda_prior': float(state.graph_lambda_prior),
            'graph_jitter': float(state.graph_jitter),
            'graph_similarity_scale': float(state.graph_similarity_scale),
            'piecewise_num_experts': int(state.piecewise_num_experts),
            'piecewise_use_gating': bool(state.piecewise_use_gating),
            'piecewise_aggregator': str(state.piecewise_aggregator),
            'piecewise_clip_scale': float(state.piecewise_clip_scale),
            'piecewise_dino_scale': float(state.piecewise_dino_scale),
            'pairwise_from_scalar_margin': float(state.pairwise_from_scalar_margin),
            'piecewise_prior_strength': float(state.piecewise_prior_strength),
            'piecewise_diversity_strength': float(state.piecewise_diversity_strength),
            'piecewise_l2_reg': float(state.piecewise_l2_reg),
            'piecewise_learning_rate': float(state.piecewise_learning_rate),
            'piecewise_max_refine_steps': int(state.piecewise_max_refine_steps),
            'residual_alpha': float(state.residual_alpha),
            'residual_beta': float(state.residual_beta),
            'residual_lambda': float(state.residual_lambda),
            'residual_sigma_y': float(state.residual_sigma_y),
            'residual_lengthscale_multiplier': float(state.residual_lengthscale_multiplier),
            'residual_lengthscale': float(state.residual_lengthscale),
            'residual_jitter': float(state.residual_jitter),
        }

        if state.model_type == 'piecewise_linear':
            base['scorer_state'] = {
                'piecewise_expert_weights': np.asarray(state.piecewise_expert_weights, dtype=np.float32).tolist(),
                'piecewise_gate_weights': np.asarray(state.piecewise_gate_weights, dtype=np.float32).tolist(),
                'piecewise_gate_bias': np.asarray(state.piecewise_gate_bias, dtype=np.float32).tolist(),
            }
        elif state.model_type == 'residual':
            support_features, support_targets, support_weights = self._build_residual_support_data(state)
            base['scorer_state'] = {
                'residual_external_features': np.asarray(support_features, dtype=np.float32).tolist(),
                'residual_external_targets': np.asarray(support_targets, dtype=np.float32).tolist(),
                'residual_external_weights': np.asarray(support_weights, dtype=np.float32).tolist(),
            }
        else:
            base['scorer_state'] = {
                'mu': np.asarray(state.mu, dtype=np.float32).tolist(),
                'b': float(state.b),
            }
        return base

    def serialize_axis(self, axis_id: str) -> Dict[str, Any]:
        """Serialize an in-memory axis state, including scorer-specific parameters."""
        state = self._axes.get(str(axis_id or '').strip())
        if state is None:
            raise KeyError(f'Unknown axis_id: {axis_id}')
        return self._serialize_state(state, fit=True)

    def deserialize_axis(self, payload: Dict[str, Any]) -> AxisBayesState:
        """Load a serialized axis state back into memory. Missing newer fields fall back safely."""
        if not isinstance(payload, dict):
            raise ValueError('Serialized axis payload must be a dict')
        model_type = _normalize_model_type(payload.get('model_type') or 'bayes_linear')
        ids = [str(v) for v in (payload.get('ids') or [])]
        id_to_index = {str(k): int(v) for k, v in (payload.get('id_to_index') or {}).items()}
        if not id_to_index and ids:
            id_to_index = {image_id: idx for idx, image_id in enumerate(ids)}
        state = AxisBayesState(
            axis_id=str(payload.get('axis_id') or f'axis:restored:{uuid.uuid4().hex[:8]}'),
            collection_id=str(payload.get('collection_id') or ''),
            dataset_root=str(payload.get('dataset_root') or ''),
            q=str(payload.get('q') or ''),
            axis_name=str(payload.get('axis_name') or payload.get('q') or 'Axis'),
            model_type=model_type,
            ids=ids,
            image_embeddings=np.asarray(payload.get('image_embeddings') or [], dtype=np.float32),
            id_to_index=id_to_index,
            w0=np.asarray(payload.get('w0') or [], dtype=np.float32),
            pos_prompt_ensemble=[str(v) for v in (payload.get('pos_prompt_ensemble') or [])],
            neg_prompt_ensemble=[str(v) for v in (payload.get('neg_prompt_ensemble') or [])],
            prompt_ensemble=[str(v) for v in (payload.get('prompt_ensemble') or [])],
            prompt_source=str(payload.get('prompt_source') or 'unknown'),
            prompt_provider=str(payload.get('prompt_provider') or 'unknown'),
            mode=_normalize_mode(payload.get('mode') or self.mode),
            feature_space=str(payload.get('feature_space') or self.feature_space),
            semantic_method=_normalize_semantic_method(payload.get('semantic_method') or self.semantic_method),
            clip_dim=int(payload.get('clip_dim') or 0),
            dino_dim=int(payload.get('dino_dim') or 0),
            clip_scale=float(payload.get('clip_scale') or self.clip_scale),
            dino_scale=float(payload.get('dino_scale') or self.dino_scale),
            alpha=float(payload.get('alpha') or self.alpha),
            dino_alpha=float(payload.get('dino_alpha') or self.dino_alpha),
            bias_alpha=float(payload.get('bias_alpha') or self.bias_alpha),
            sigma2=float(payload.get('sigma2') or self.sigma2),
            rank_eta=float(payload.get('rank_eta') or self.rank_eta),
            rank_anchor_k=int(payload.get('rank_anchor_k') or self.rank_anchor_k),
            rank_anchor_delta=float(payload.get('rank_anchor_delta') or self.rank_anchor_delta),
            rank_max_pairs=int(payload.get('rank_max_pairs') or self.rank_max_pairs),
            max_moves=int(payload.get('max_moves') or self.max_moves),
            graph_knn_k=int(payload.get('graph_knn_k') or self.graph_knn_k),
            graph_lambda_smooth=float(payload.get('graph_lambda_smooth') or self.graph_lambda_smooth),
            graph_lambda_prior=float(payload.get('graph_lambda_prior') or self.graph_lambda_prior),
            graph_jitter=float(payload.get('graph_jitter') or self.graph_jitter),
            graph_similarity_scale=float(payload.get('graph_similarity_scale') or 0.0),
            b0=float(payload.get('b0') or 0.0),
            z0_all=np.asarray(payload.get('z0_all') or [], dtype=np.float32),
            z0_sorted=np.asarray(payload.get('z0_sorted') or [], dtype=np.float32),
            alpha_vec=np.asarray(payload.get('alpha_vec') or [], dtype=np.float32),
            alpha_inv_vec=np.asarray(payload.get('alpha_inv_vec') or [], dtype=np.float32),
            move_order=[str(v) for v in (payload.get('move_order') or [])],
            move_targets={str(k): float(v) for k, v in (payload.get('move_targets') or {}).items()},
            move_weights={str(k): float(v) for k, v in (payload.get('move_weights') or {}).items()},
            undefined_order=[str(v) for v in (payload.get('undefined_order') or [])],
            piecewise_num_experts=int(payload.get('piecewise_num_experts') or self.piecewise_num_experts),
            piecewise_use_gating=bool(payload.get('piecewise_use_gating') if 'piecewise_use_gating' in payload else self.piecewise_use_gating),
            piecewise_aggregator=str(payload.get('piecewise_aggregator') or self.piecewise_aggregator),
            piecewise_clip_scale=float(payload.get('piecewise_clip_scale') or self.piecewise_clip_scale),
            piecewise_dino_scale=float(payload.get('piecewise_dino_scale') or self.piecewise_dino_scale),
            pairwise_from_scalar_margin=float(payload.get('pairwise_from_scalar_margin') or self.pairwise_from_scalar_margin),
            piecewise_prior_strength=float(payload.get('piecewise_prior_strength') or self.piecewise_prior_strength),
            piecewise_diversity_strength=float(payload.get('piecewise_diversity_strength') or self.piecewise_expert_diversity_strength),
            piecewise_l2_reg=float(payload.get('piecewise_l2_reg') or self.piecewise_l2_reg),
            piecewise_learning_rate=float(payload.get('piecewise_learning_rate') or self.piecewise_learning_rate),
            piecewise_max_refine_steps=int(payload.get('piecewise_max_refine_steps') or self.piecewise_max_refine_steps),
            residual_alpha=float(payload.get('residual_alpha') or self.residual_alpha),
            residual_beta=float(payload.get('residual_beta') or self.residual_beta),
            residual_lambda=float(payload.get('residual_lambda') or self.residual_lambda),
            residual_sigma_y=float(payload.get('residual_sigma_y') or self.residual_sigma_y),
            residual_lengthscale_multiplier=float(
                payload.get('residual_lengthscale_multiplier') or self.residual_lengthscale_multiplier
            ),
            residual_lengthscale=float(payload.get('residual_lengthscale') or 0.0),
            residual_jitter=float(payload.get('residual_jitter') or self.residual_jitter),
        )
        scorer = self._scorer_for_state(state)
        scorer.import_state(state, payload.get('scorer_state') or {})
        self._axes[state.axis_id] = state
        return state

    def project_serialized_axis(
        self,
        payload: Dict[str, Any],
        collection_id: str,
        dataset_root: str,
        axis_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Project a saved refined artifact onto a target collection.

        If the target collection matches the original one exactly, restore the full
        saved state. Otherwise, transfer the learned scorer parameters to the new
        collection and treat them as the starting model for future refinement.
        """
        if not isinstance(payload, dict):
            raise ValueError('Serialized axis payload must be a dict')

        target_root = str(Path(dataset_root).resolve())
        target_collection_id = str(collection_id or Path(target_root).name)
        coll = self._get_collection(dataset_root=target_root, collection_id=target_collection_id)

        source_root = str(payload.get('dataset_root') or '').strip()
        source_ids = [str(v) for v in (payload.get('ids') or [])]
        same_collection = (
            source_root
            and Path(source_root).resolve() == Path(target_root).resolve()
            and source_ids == list(coll.ids)
        )

        requested_name = str(axis_name or payload.get('axis_name') or payload.get('q') or 'Axis').strip() or 'Axis'
        scorer_state = payload.get('scorer_state') or {}
        model_type = _normalize_model_type(payload.get('model_type') or self.model_type)
        mode_name = _normalize_mode(payload.get('mode') or self.mode)

        if same_collection:
            restored = self.deserialize_axis(payload)
            old_axis_id = str(restored.axis_id)
            self._axes.pop(old_axis_id, None)
            restored.axis_id = f'axis:bayes:{_slugify(restored.q or requested_name)}:{uuid.uuid4().hex[:8]}'
            restored.collection_id = target_collection_id
            restored.dataset_root = target_root
            restored.axis_name = requested_name
            self._axes[restored.axis_id] = restored
            return self._serialize_state(restored, fit=False)

        if model_type == 'bayes_linear' and mode_name == 'graph':
            raise ValueError('Graph artifacts can only be restored on their original collection')

        w0 = np.asarray(payload.get('w0') or [], dtype=np.float32)
        if w0.size == 0:
            raise ValueError('Serialized axis payload is missing w0')
        if int(w0.shape[0]) != int(coll.embeddings.shape[1]):
            raise ValueError(
                f'Artifact embedding dim mismatch: artifact={w0.shape[0]} target={coll.embeddings.shape[1]}'
            )

        alpha = float(payload.get('alpha') or self.alpha)
        dino_alpha = float(payload.get('dino_alpha') or self.dino_alpha)
        alpha_vec = np.concatenate(
            [
                np.full((coll.clip_dim,), alpha, dtype=np.float32),
                np.full((coll.dino_dim,), dino_alpha, dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        alpha_inv_vec = (1.0 / np.maximum(alpha_vec, 1e-8)).astype(np.float32)

        state = AxisBayesState(
            axis_id=f'axis:bayes:{_slugify(str(payload.get("q") or requested_name))}:{uuid.uuid4().hex[:8]}',
            collection_id=target_collection_id,
            dataset_root=target_root,
            q=str(payload.get('q') or requested_name),
            axis_name=requested_name,
            model_type=model_type,
            ids=list(coll.ids),
            image_embeddings=np.asarray(coll.embeddings, dtype=np.float32),
            id_to_index=dict(coll.id_to_index),
            w0=np.asarray(w0, dtype=np.float32).copy(),
            pos_prompt_ensemble=[str(v) for v in (payload.get('pos_prompt_ensemble') or [])],
            neg_prompt_ensemble=[str(v) for v in (payload.get('neg_prompt_ensemble') or [])],
            prompt_ensemble=[str(v) for v in (payload.get('prompt_ensemble') or [])],
            prompt_source=str(payload.get('prompt_source') or 'unknown'),
            prompt_provider=str(payload.get('prompt_provider') or 'unknown'),
            mode=mode_name,
            feature_space=str(payload.get('feature_space') or coll.feature_space),
            semantic_method=_normalize_semantic_method(payload.get('semantic_method') or coll.semantic_method),
            clip_dim=int(payload.get('clip_dim') or coll.clip_dim),
            dino_dim=int(payload.get('dino_dim') or coll.dino_dim),
            clip_scale=float(payload.get('clip_scale') or coll.clip_scale),
            dino_scale=float(payload.get('dino_scale') or coll.dino_scale),
            alpha=alpha,
            dino_alpha=dino_alpha,
            bias_alpha=float(payload.get('bias_alpha') or self.bias_alpha),
            sigma2=float(payload.get('sigma2') or self.sigma2),
            rank_eta=float(payload.get('rank_eta') or self.rank_eta),
            rank_anchor_k=int(payload.get('rank_anchor_k') or self.rank_anchor_k),
            rank_anchor_delta=float(payload.get('rank_anchor_delta') or self.rank_anchor_delta),
            rank_max_pairs=int(payload.get('rank_max_pairs') or self.rank_max_pairs),
            max_moves=int(payload.get('max_moves') or self.max_moves),
            graph_knn_k=int(payload.get('graph_knn_k') or coll.graph_knn_k or self.graph_knn_k),
            graph_lambda_smooth=float(payload.get('graph_lambda_smooth') or self.graph_lambda_smooth),
            graph_lambda_prior=float(payload.get('graph_lambda_prior') or self.graph_lambda_prior),
            graph_jitter=float(payload.get('graph_jitter') or self.graph_jitter),
            graph_similarity_scale=float(payload.get('graph_similarity_scale') or coll.graph_similarity_scale or 0.0),
            b0=float(payload.get('b0') or 0.0),
            z0_all=np.zeros((len(coll.ids),), dtype=np.float32),
            z0_sorted=np.zeros((len(coll.ids),), dtype=np.float32),
            alpha_vec=alpha_vec,
            alpha_inv_vec=alpha_inv_vec,
            move_order=[],
            move_targets={},
            move_weights={},
            undefined_order=[],
            graph_prior_cov=np.asarray(coll.graph_prior_cov, dtype=np.float32),
            graph_prior_cov_diag=np.asarray(coll.graph_prior_cov_diag, dtype=np.float32),
            piecewise_num_experts=int(payload.get('piecewise_num_experts') or self.piecewise_num_experts),
            piecewise_use_gating=bool(
                payload.get('piecewise_use_gating')
                if 'piecewise_use_gating' in payload else self.piecewise_use_gating
            ),
            piecewise_aggregator=str(payload.get('piecewise_aggregator') or self.piecewise_aggregator),
            piecewise_clip_scale=float(payload.get('piecewise_clip_scale') or self.piecewise_clip_scale),
            piecewise_dino_scale=float(payload.get('piecewise_dino_scale') or self.piecewise_dino_scale),
            pairwise_from_scalar_margin=float(payload.get('pairwise_from_scalar_margin') or self.pairwise_from_scalar_margin),
            piecewise_prior_strength=float(payload.get('piecewise_prior_strength') or self.piecewise_prior_strength),
            piecewise_diversity_strength=float(payload.get('piecewise_diversity_strength') or self.piecewise_expert_diversity_strength),
            piecewise_l2_reg=float(payload.get('piecewise_l2_reg') or self.piecewise_l2_reg),
            piecewise_learning_rate=float(payload.get('piecewise_learning_rate') or self.piecewise_learning_rate),
            piecewise_max_refine_steps=int(payload.get('piecewise_max_refine_steps') or self.piecewise_max_refine_steps),
            residual_alpha=float(payload.get('residual_alpha') or self.residual_alpha),
            residual_beta=float(payload.get('residual_beta') or self.residual_beta),
            residual_lambda=float(payload.get('residual_lambda') or self.residual_lambda),
            residual_sigma_y=float(payload.get('residual_sigma_y') or self.residual_sigma_y),
            residual_lengthscale_multiplier=float(
                payload.get('residual_lengthscale_multiplier') or self.residual_lengthscale_multiplier
            ),
            residual_lengthscale=float(payload.get('residual_lengthscale') or 0.0),
            residual_jitter=float(payload.get('residual_jitter') or self.residual_jitter),
        )

        if model_type == 'piecewise_linear':
            piecewise_scorer = self._scorers['piecewise_linear']
            state.piecewise_features = self._build_piecewise_feature_matrix(state)
            expert_w = np.asarray(scorer_state.get('piecewise_expert_weights') or [], dtype=np.float32)
            if expert_w.size == 0:
                raise ValueError('Serialized piecewise artifact is missing expert weights')
            if expert_w.ndim != 2 or expert_w.shape[1] != state.piecewise_features.shape[1]:
                raise ValueError(
                    f'Piecewise artifact feature dim mismatch: artifact={expert_w.shape} target={state.piecewise_features.shape}'
                )
            state.piecewise_expert_weights = expert_w
            state.piecewise_gate_weights = np.asarray(scorer_state.get('piecewise_gate_weights') or [], dtype=np.float32)
            state.piecewise_gate_bias = np.asarray(scorer_state.get('piecewise_gate_bias') or [], dtype=np.float32)
            state.piecewise_prior_vector = np.asarray(expert_w[0], dtype=np.float32).copy()
            (
                state.piecewise_scores_all,
                state.piecewise_expert_scores_all,
                state.piecewise_uncertainty_all,
            ) = piecewise_scorer._numpy_forward(state)
            state.piecewise_pair_i = []
            state.piecewise_pair_j = []
            state.piecewise_pair_w = []
            state.piecewise_last_fit_signature = ''
            state.z0_all = np.asarray(state.piecewise_scores_all, dtype=np.float32).copy()
            state.z0_sorted = np.sort(state.z0_all).astype(np.float32)
        elif model_type == 'residual':
            residual_scorer = self._scorers['residual']
            residual_scorer.import_state(state, scorer_state)
            residual_scorer.initialize_state(self, state)
            state.residual_last_fit_signature = ''
        else:
            mu = np.asarray(scorer_state.get('mu') or [], dtype=np.float32)
            if mu.size == 0:
                raise ValueError('Serialized linear artifact is missing posterior weights')
            if int(mu.shape[0]) != int(coll.embeddings.shape[1]):
                raise ValueError(
                    f'Linear artifact dim mismatch: artifact={mu.shape[0]} target={coll.embeddings.shape[1]}'
                )
            b = float(scorer_state.get('b') or 0.0)
            state.w0 = np.asarray(mu, dtype=np.float32).copy()
            state.b0 = b
            state.mu = np.asarray(mu, dtype=np.float32).copy()
            state.b = b
            state.X = np.zeros((0, coll.embeddings.shape[1]), dtype=np.float32)
            state.y = np.zeros((0,), dtype=np.float32)
            state.A_inv = np.zeros((0, 0), dtype=np.float32)
            state.pair_i = []
            state.pair_j = []
            state.pair_y = []
            state.pair_w = []
            state.pair_D = np.zeros((0, coll.embeddings.shape[1]), dtype=np.float32)
            state.pair_W_diag = np.zeros((0,), dtype=np.float32)
            state.pair_M_inv = np.zeros((0, 0), dtype=np.float32)
            state.graph_obs_indices = []
            state.graph_obs_var = np.zeros((0,), dtype=np.float32)
            state.graph_obs_S_inv = np.zeros((0, 0), dtype=np.float32)
            state.z0_all = np.asarray((state.image_embeddings @ state.mu) + float(state.b0), dtype=np.float32)
            state.z0_sorted = np.sort(state.z0_all).astype(np.float32)

        self._axes[state.axis_id] = state
        return self._serialize_state(state, fit=False)
