#!/usr/bin/env python3
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

import grid as grid_utils

# Create a fixed random reference at import time and project it to 2D via PCA.
# We will align all subsequent 2D projections (PCA/UMAP) to this reference
# using an orthogonal Procrustes rotation to stabilize orientation.
_REF_ROWS = 8192
_REF_DIM = 64
_rng = np.random.default_rng(12345)
_REF_X = _rng.standard_normal((_REF_ROWS, _REF_DIM)).astype(np.float32)


def _pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]
    return (Xc @ comps.T)


_REF_2D = _pca_2d(_REF_X)  # shape (_REF_ROWS, 2)


def _orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return A rotated to best match B (both Nx2), rotation only (no scaling/translation).

    We center both A and B before solving R that minimizes ||(A - Ā) R - (B - B̄)||_F.
    R is obtained from SVD of M = (A - Ā)^T (B - B̄) as R = U V^T.
    """
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 2:
        return A
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    M = Ac.T @ Bc
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return Ac @ R


def reduce_to_2d(embeddings: np.ndarray, method: str = "pca", random_state: int = 42) -> np.ndarray:
    coords: np.ndarray
    # Degenerate guard: no features
    if embeddings.size == 0 or embeddings.ndim != 2 or embeddings.shape[1] == 0:
        n = embeddings.shape[0] if embeddings.ndim >= 1 else 0
        return np.zeros((n, 2), dtype=np.float32)
    if method == "umap":
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(embeddings)
        except Exception:
            coords = _pca_2d(embeddings)
    else:
        coords = _pca_2d(embeddings)

    # Align to common reference via orthogonal Procrustes (rotation-only)
    N = coords.shape[0]
    if N > 0:
        # Ensure we have enough reference rows; if not, tile the reference
        if N <= _REF_2D.shape[0]:
            ref = _REF_2D[:N, :]
        else:
            reps = int(np.ceil(N / _REF_2D.shape[0]))
            ref = np.vstack([_REF_2D] * reps)[:N, :]
        coords = _orthogonal_procrustes(coords.astype(np.float32), ref.astype(np.float32))

    # Normalize to [0,1] after alignment for consistent plotting
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    coords01 = (coords - mins) / denom
    return coords01.astype(np.float32)


def _pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]
    return (Xc @ comps.T)


def pack_to_grid(coords01: np.ndarray, n_layer: int = 64, n_tile: int = 8,
                 filter_fn: Optional[Callable[[int, Dict], bool]] = None) -> Tuple[np.ndarray, int]:
    if n_layer is None or n_layer <= 0:
        count = max(1, coords01.shape[0])
        n_layer = int(np.ceil(np.sqrt(2.0 * count)))
        n_layer = max(n_layer, 6)

    x = coords01[:, 0]
    y = coords01[:, 1]
    layout = {"x": x, "y": y}
    metadata = {"dummy": np.zeros(len(x))}

    params = {"n_layer": n_layer, "n_tile": n_tile}
    if filter_fn is not None:
        params["filter"] = filter_fn

    tiles = grid_utils.grid(metadata, layout, params)

    N = coords01.shape[0]
    desired = np.full((N, 2), -1, dtype=np.int32)
    for _, _, tile in grid_utils.enumerate_tiles(tiles):
        for idx, gi in enumerate(tile["gi"]):
            desired[gi, 0] = int(tile["ci"][idx])
            desired[gi, 1] = int(tile["cj"][idx])

    taken = np.zeros((n_layer, n_layer), dtype=bool)
    final = np.zeros_like(coords01, dtype=np.float32)

    def is_free(ci, cj):
        if ci < 0 or cj < 0 or ci >= n_layer or cj >= n_layer:
            return False
        return not taken[ci, cj]

    def find_nearest_free(ci, cj, max_r=5):
        if is_free(ci, cj):
            return ci, cj
        for r in range(1, max_r + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if abs(di) != r and abs(dj) != r:
                        continue
                    ni, nj = ci + di, cj + dj
                    if is_free(ni, nj):
                        return ni, nj
        for ni in range(n_layer):
            for nj in range(n_layer):
                if not taken[ni, nj]:
                    return ni, nj
        return None, None

    order = np.argsort((desired[:, 0] + desired[:, 1] * n_layer))
    for idx in order:
        ci, cj = int(desired[idx, 0]), int(desired[idx, 1])
        ni, nj = find_nearest_free(ci, cj)
        if ni is None:
            ni, nj = 0, 0
        taken[ni, nj] = True
        final[idx, 0] = (ni + 0.5) / n_layer
        final[idx, 1] = (nj + 0.5) / n_layer

    return final, n_layer
