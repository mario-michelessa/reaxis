#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from .constants import (
        ZERO_SHOT_HISTOGRAM_BINS,
        ZERO_SHOT_SOFTMAX_TEMPERATURE,
        ZERO_SHOT_VALUE_PROMPT_TEMPLATE,
        ZERO_SHOT_VALUE_PROMPT_WITH_CONTEXT_TEMPLATE,
    )
    from .embeddings import DEFAULT_SEMANTIC_EMBED_METHOD, build_multimodal_extractor
except ImportError:
    from constants import (
        ZERO_SHOT_HISTOGRAM_BINS,
        ZERO_SHOT_SOFTMAX_TEMPERATURE,
        ZERO_SHOT_VALUE_PROMPT_TEMPLATE,
        ZERO_SHOT_VALUE_PROMPT_WITH_CONTEXT_TEMPLATE,
    )
    from embeddings import DEFAULT_SEMANTIC_EMBED_METHOD, build_multimodal_extractor


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / denom


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    den = np.sum(ex, axis=axis, keepdims=True) + 1e-8
    return ex / den


def _minmax_01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    if span <= 1e-8:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return ((arr - lo) / span).astype(np.float32)


def _rank_spread_01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
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
        # Keep equal-valued samples together and assign the same average rank.
        while j < n and abs(float(sorted_vals[j]) - float(sorted_vals[i])) <= 1e-12:
            j += 1
        mid_rank = (i + (j - 1)) * 0.5
        spread_val = float(mid_rank) / float(n - 1)
        out[order[i:j]] = spread_val
        i = j
    return out


def slugify(text: str) -> str:
    s = re.sub(r'[^a-z0-9]+', '-', str(text or '').strip().lower()).strip('-')
    return s or 'attribute'


class ZeroShotAttributeRegressor:
    """Maps image embeddings to a continuous [0,1] attribute score.

    Method:
    - Build VLM text embeddings for ordered attribute values (low -> high).
    - Compute image-to-value similarity.
    - Convert similarities to probabilities via softmax.
    - Score = expected position over ordered values.
    """

    def __init__(
        self,
        temperature: float = ZERO_SHOT_SOFTMAX_TEMPERATURE,
        text_method: str = DEFAULT_SEMANTIC_EMBED_METHOD,
    ):
        self.temperature = float(temperature)
        self.text_method = str(text_method or DEFAULT_SEMANTIC_EMBED_METHOD)
        self._text_extractor: Optional[Any] = None

    def _get_text_extractor(self):
        if self._text_extractor is None:
            self._text_extractor = build_multimodal_extractor(self.text_method)
            if self._text_extractor is None:
                raise RuntimeError(f'Unsupported text embedding method: {self.text_method}')
        return self._text_extractor

    def _embed_text(self, text: str) -> np.ndarray:
        ext = self._get_text_extractor()
        vec = ext.extract_text_embedding(text)
        return np.asarray(vec, dtype=np.float32)

    def _value_prompts(self, attribute: str, value: str, context: str) -> List[str]:
        attr = str(attribute or '').strip()
        val = str(value or '').strip()
        ctx = str(context or '').strip()
        prompts = []
        if ctx:
            prompts.append(ZERO_SHOT_VALUE_PROMPT_WITH_CONTEXT_TEMPLATE.format(context=ctx, attribute=attr, value=val))
        prompts.append(ZERO_SHOT_VALUE_PROMPT_TEMPLATE.format(attribute=attr, value=val))
        prompts.append(f'{val} {attr}'.strip())
        return [p for p in prompts if p]

    def _score_histogram(self, scores: np.ndarray, histogram_bins: int):
        bins = max(2, int(histogram_bins))
        edges = np.linspace(0.0, 1.0, num=bins + 1, dtype=np.float32)
        hist_counts, _ = np.histogram(np.asarray(scores, dtype=np.float32), bins=edges)
        centers = (edges[:-1] + edges[1:]) * 0.5
        return {
            'centers': centers.astype(np.float32).tolist(),
            'counts': [int(c) for c in hist_counts.tolist()],
        }

    def encode_ordered_values(self, attribute: str, values: List[str], context: str = '') -> np.ndarray:
        if not values:
            raise ValueError('values must be non-empty')
        rows = []
        for value in values:
            prompt_vecs = []
            for prompt in self._value_prompts(attribute=attribute, value=value, context=context):
                prompt_vecs.append(self._embed_text(prompt))
            if not prompt_vecs:
                raise RuntimeError(f'Unable to encode value "{value}"')
            v = np.mean(np.vstack(prompt_vecs), axis=0, dtype=np.float32)
            rows.append(v)
        mat = np.vstack(rows).astype(np.float32)
        return _normalize_rows(mat)

    def score_discrete(
        self,
        image_embeddings: np.ndarray,
        attribute: str,
        categories: List[str],
        attribute_type: str = 'ordinal',
        context: str = '',
        histogram_bins: int = ZERO_SHOT_HISTOGRAM_BINS,
        include_probabilities: bool = False,
    ) -> Dict[str, Any]:
        if image_embeddings is None:
            raise ValueError('image_embeddings is required')
        if len(categories) < 2:
            raise ValueError('categories must contain at least 2 labels')

        X = np.asarray(image_embeddings, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError('image_embeddings must be NxD')
        attr_type = str(attribute_type or '').strip().lower()
        if attr_type not in {'categorical', 'ordinal'}:
            raise ValueError('attribute_type must be categorical or ordinal')

        Xn = _normalize_rows(X)
        Tn = self.encode_ordered_values(attribute=attribute, values=categories, context=context)
        logits = (Xn @ Tn.T) * float(self.temperature)
        probs = _softmax(logits, axis=1)

        k = len(categories)
        positions = np.linspace(0.0, 1.0, num=k, dtype=np.float32)
        pred_idx = np.argmax(probs, axis=1)
        pred_labels = [categories[int(i)] for i in pred_idx]
        pred_counts = [int(np.sum(pred_idx == i)) for i in range(k)]

        if attr_type == 'categorical':
            denom = max(1, k - 1)
            raw_scores = (pred_idx.astype(np.float32) / float(denom)).astype(np.float32)
            scores = raw_scores.astype(np.float32)
            minmax_scores = _minmax_01(raw_scores)
            score_transform = {
                'method': 'argmax_category_index',
                'raw_min': float(np.min(raw_scores)) if raw_scores.size > 0 else 0.0,
                'raw_max': float(np.max(raw_scores)) if raw_scores.size > 0 else 0.0,
            }
        else:
            raw_scores = (probs * positions[None, :]).sum(axis=1).astype(np.float32)
            minmax_scores = _minmax_01(raw_scores)
            scores = _rank_spread_01(minmax_scores).astype(np.float32)
            score_transform = {
                'method': 'rank_spread_after_minmax_expected_position',
                'raw_min': float(np.min(raw_scores)) if raw_scores.size > 0 else 0.0,
                'raw_max': float(np.max(raw_scores)) if raw_scores.size > 0 else 0.0,
            }

        out: Dict[str, Any] = {
            'attribute': attribute,
            'attribute_type': attr_type,
            'values': list(categories),
            'value_positions': positions.tolist(),
            'scores': scores.astype(np.float32).tolist(),
            'scores_raw': raw_scores.astype(np.float32).tolist(),
            'scores_minmax': minmax_scores.astype(np.float32).tolist(),
            'score_transform': score_transform,
            'predicted_values': pred_labels,
            'value_histogram': [
                {'value': categories[i], 'count': int(pred_counts[i])}
                for i in range(k)
            ],
            'score_histogram': self._score_histogram(scores, histogram_bins=histogram_bins),
            'scoring_method': 'zero_shot_classification',
        }
        if include_probabilities:
            out['value_probabilities'] = probs.astype(np.float32).tolist()
        return out

    def score_continuous_anchors(
        self,
        image_embeddings: np.ndarray,
        attribute: str,
        low_anchor: str,
        high_anchor: str,
        context: str = '',
        histogram_bins: int = ZERO_SHOT_HISTOGRAM_BINS,
    ) -> Dict[str, Any]:
        if image_embeddings is None:
            raise ValueError('image_embeddings is required')
        low = str(low_anchor or '').strip()
        high = str(high_anchor or '').strip()
        if not low or not high:
            raise ValueError('low_anchor and high_anchor are required')
        if low.lower() == high.lower():
            raise ValueError('low_anchor and high_anchor must be different')

        X = np.asarray(image_embeddings, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError('image_embeddings must be NxD')
        Xn = _normalize_rows(X)

        low_mat = self.encode_ordered_values(attribute=attribute, values=[low], context=context)
        high_mat = self.encode_ordered_values(attribute=attribute, values=[high], context=context)
        low_vec = low_mat[0]
        high_vec = high_mat[0]

        direction = np.asarray(high_vec - low_vec, dtype=np.float32)
        direction_norm = float(np.linalg.norm(direction)) + 1e-8
        direction = direction / direction_norm

        raw_scores = (Xn @ direction).astype(np.float32)
        minmax_scores = _minmax_01(raw_scores)
        scores = _rank_spread_01(minmax_scores).astype(np.float32)

        sim_low = (Xn @ low_vec).astype(np.float32)
        sim_high = (Xn @ high_vec).astype(np.float32)
        pred_idx = (sim_high >= sim_low).astype(np.int32)
        values = [low, high]
        pred_labels = [values[int(i)] for i in pred_idx]
        pred_counts = [int(np.sum(pred_idx == i)) for i in range(2)]

        return {
            'attribute': attribute,
            'attribute_type': 'continuous',
            'values': values,
            'value_positions': [0.0, 1.0],
            'scores': scores.astype(np.float32).tolist(),
            'scores_raw': raw_scores.astype(np.float32).tolist(),
            'scores_minmax': minmax_scores.astype(np.float32).tolist(),
            'score_transform': {
                'method': 'rank_spread_after_minmax_clip_direction_projection',
                'raw_min': float(np.min(raw_scores)) if raw_scores.size > 0 else 0.0,
                'raw_max': float(np.max(raw_scores)) if raw_scores.size > 0 else 0.0,
            },
            'predicted_values': pred_labels,
            'value_histogram': [
                {'value': values[i], 'count': int(pred_counts[i])}
                for i in range(2)
            ],
            'score_histogram': self._score_histogram(scores, histogram_bins=histogram_bins),
            'scoring_method': 'clip_direction_projection',
            'anchors': {
                'low': low,
                'high': high,
            },
        }

    def score(
        self,
        image_embeddings: np.ndarray,
        attribute: str,
        values: List[str],
        context: str = '',
        histogram_bins: int = ZERO_SHOT_HISTOGRAM_BINS,
        include_probabilities: bool = False,
    ) -> Dict[str, Any]:
        # Backward-compatible alias for ordinal scoring.
        return self.score_discrete(
            image_embeddings=image_embeddings,
            attribute=attribute,
            categories=values,
            attribute_type='ordinal',
            context=context,
            histogram_bins=histogram_bins,
            include_probabilities=include_probabilities,
        )
