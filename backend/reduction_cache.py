#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable, List, Sequence

try:
    from .embeddings import normalize_multimodal_method
except ImportError:
    from embeddings import normalize_multimodal_method


SUPPORTED_REDUCTION_METHODS = ('pca', 'umap', 'tsne')
DEFAULT_REDUCTION_METHOD = 'pca'
DEFAULT_REDUCTION_SPEC = 'all'


def normalize_reduction_method(method: str) -> str:
    raw = str(method or DEFAULT_REDUCTION_METHOD).strip().lower()
    if raw == 't-sne':
        raw = 'tsne'
    if raw not in SUPPORTED_REDUCTION_METHODS:
        raise ValueError(
            f'Unsupported reduction method `{method}`. Expected one of {SUPPORTED_REDUCTION_METHODS}.'
        )
    return raw


def parse_reduction_methods(raw: str | Sequence[str] | None) -> List[str]:
    if raw is None:
        return list(SUPPORTED_REDUCTION_METHODS)
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(',') if part.strip()]
    else:
        tokens = [str(part).strip() for part in raw if str(part).strip()]
    if not tokens:
        return list(SUPPORTED_REDUCTION_METHODS)
    if any(token.lower() == DEFAULT_REDUCTION_SPEC for token in tokens):
        return list(SUPPORTED_REDUCTION_METHODS)

    ordered: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = normalize_reduction_method(token)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def coords_cache_filename(embed_method: str, reduction: str = DEFAULT_REDUCTION_METHOD) -> str:
    reduction_method = normalize_reduction_method(reduction)
    cache_method = normalize_multimodal_method(embed_method)
    return f'coords_{reduction_method}2d_{cache_method}.npz'
