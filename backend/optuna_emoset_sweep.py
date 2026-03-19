#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError as exc:
    raise RuntimeError('optuna is required. Install with: pip install optuna') from exc

try:
    from .gallery_backend import ImageGalleryEngine
except ImportError:
    from gallery_backend import ImageGalleryEngine


# =============================
# Tunables (no argparse needed)
# =============================
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / 'data' / 'datasets' / 'EmoSet'
CSV_LOG_PATH: Optional[Path] = None

CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'  # or local path
CLIP_LOCAL_FILES_ONLY = False

AXIS_BAYES_MODE = 'rank'  # 'gaussian' | 'rank' | 'graph'
AXIS_BAYES_FEATURE_SPACE = 'clip_dino'  # 'clip' | 'clip_dino'
AXIS_BAYES_GRAPH_KNN_K = 16
AXIS_BAYES_GRAPH_LAMBDA_SMOOTH = 6.0
AXIS_BAYES_GRAPH_LAMBDA_PRIOR = 1.0
AXIS_BAYES_GRAPH_JITTER = 1e-6
AXIS_BAYES_RANK_ETA = 0.25
AXIS_BAYES_RANK_ANCHOR_K = 6
AXIS_BAYES_RANK_ANCHOR_DELTA = 0.12
AXIS_BAYES_RANK_MAX_PAIRS = 100

MOVES_PER_EMOTION = 3
MOVE_SELECTION_SEED = 123
N_TRIALS = 200
OPTUNA_SAMPLER_SEED = 2026

POS_PROMPT_TEMPLATE = 'an image that strongly conveys {emotion}'
NEG_PROMPT_COMMON = 'an emotionally neutral image with no clear emotion'
POS_PROMPTS_OVERRIDE: Dict[str, str] = {}
NEG_PROMPTS_OVERRIDE: Dict[str, str] = {}
OPTUNA_OBJECTIVE_NAME = 'final_macro_auroc'

# Rank sweep bounds are priors, not guarantees. If the best trial lands on a
# boundary, widen that parameter and rerun rather than assuming convergence.
RANK_SWEEP_ALPHA_MIN = 3e-3
RANK_SWEEP_ALPHA_MAX = 3.0
RANK_SWEEP_ETA_MIN = 2e-2
RANK_SWEEP_ETA_MAX = 2.0
RANK_SWEEP_ANCHOR_K_MIN = 2
RANK_SWEEP_ANCHOR_K_MAX = 10
RANK_SWEEP_ANCHOR_DELTA_MIN = 0.02
RANK_SWEEP_ANCHOR_DELTA_MAX = 0.40
RANK_SWEEP_MAX_PAIRS_MIN = 32


def utc_now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class CSVAppender:
    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._init = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open('r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
            if list(existing_header) == self.fieldnames:
                self._init = True
            else:
                with self.path.open('r', newline='', encoding='utf-8') as f:
                    old_rows = list(csv.DictReader(f))
                with self.path.open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                    for row in old_rows:
                        writer.writerow({k: row.get(k, '') for k in self.fieldnames})
                    f.flush()
                self._init = True

    def append(self, row: Dict[str, object]) -> None:
        safe = {k: row.get(k, '') for k in self.fieldnames}
        with self.path.open('a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._init:
                w.writeheader()
                self._init = True
            w.writerow(safe)
            f.flush()


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / n).astype(np.float32)


def l2_normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if n <= 1e-8:
        raise ValueError('Cannot normalize zero vector')
    return (x / n).astype(np.float32)


def normalize_mode(mode: str) -> str:
    s = str(mode).strip().lower()
    if s == 'graph':
        return 'graph'
    if s == 'rank':
        return 'rank'
    return 'gaussian'


def default_csv_log_path(mode: str) -> Path:
    mode_name = normalize_mode(mode)
    return REPO_ROOT / 'backend' / 'experiments' / f'emoset_optuna_{mode_name}.csv'


def quantile_from_sorted(z_sorted: np.ndarray, p01: float) -> float:
    z = np.asarray(z_sorted, dtype=np.float32).reshape(-1)
    n = int(z.size)
    if n == 0:
        return 0.0
    if n == 1:
        return float(z[0])
    p = float(np.clip(float(p01), 0.0, 1.0))
    pos = p * float(n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(z[lo])
    t = float(pos - lo)
    return float((1.0 - t) * z[lo] + t * z[hi])


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / den)


def average_ranks(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(arr, kind='mergesort')
    sorted_vals = arr[order]
    ranks = np.zeros((n,), dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(float(sorted_vals[j]) - float(sorted_vals[i])) <= 1e-12:
            j += 1
        avg_rank = 0.5 * float(i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if s.size != y.size or s.size == 0:
        return np.nan
    mask = np.isfinite(s) & np.isfinite(y)
    if not np.any(mask):
        return np.nan
    s = s[mask]
    y = y[mask] > 0.5
    n_pos = int(np.sum(y))
    n_neg = int(y.size - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return np.nan
    ranks = average_ranks(s)
    pos_rank_sum = float(np.sum(ranks[y]))
    auc = (pos_rank_sum - (0.5 * n_pos * (n_pos + 1))) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def rank_percentile_01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.asarray([0.5], dtype=np.float32)
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    neg_x = np.exp(arr[~pos])
    out[~pos] = neg_x / (1.0 + neg_x)
    return out


def gaussian_update(
    X_all: np.ndarray,
    w0: np.ndarray,
    z0_sorted: np.ndarray,
    moves: List[Tuple[int, float]],
    alpha_vec: np.ndarray,
    bias_alpha: float,
    sigma2: float,
    b0: float = 0.0,
) -> Tuple[np.ndarray, float]:
    w0 = np.asarray(w0, dtype=np.float32)
    alpha_vec = np.asarray(alpha_vec, dtype=np.float32)
    alpha_inv = 1.0 / np.maximum(alpha_vec, 1e-8)

    if len(moves) == 0:
        return w0.copy(), float(b0)

    idx = np.asarray([int(i) for i, _ in moves], dtype=np.int64)
    p01 = [float(p) for _, p in moves]
    X = X_all[idx, :].astype(np.float32)
    y_raw = np.asarray([quantile_from_sorted(z0_sorted, p) for p in p01], dtype=np.float32)

    m = int(X.shape[0])
    ones = np.ones((m,), dtype=np.float32)
    XS = (X * alpha_inv[None, :]).astype(np.float32)
    A = (
        (float(sigma2) * np.eye(m, dtype=np.float32))
        + (XS @ X.T)
        + ((1.0 / float(bias_alpha)) * np.outer(ones, ones).astype(np.float32))
    )
    A_inv = np.linalg.inv(A + (1e-6 * np.eye(m, dtype=np.float32)))

    r = y_raw - (X @ w0) - float(b0)
    mu = w0 + (alpha_inv * (X.T @ (A_inv @ r)))
    b = float(b0) + float((1.0 / float(bias_alpha)) * (ones @ (A_inv @ r)))

    if float(np.dot(mu, w0)) < 0.0:
        mu = -mu
        b = -b
    n = float(np.linalg.norm(mu))
    if n > 1e-8:
        mu = (mu / n).astype(np.float32)
        b = float(b / n)
    else:
        mu = w0.copy()
        b = float(b0)
    return mu, b


def build_graph_prior(
    X_all: np.ndarray,
    knn_k: int,
    lambda_smooth: float,
    lambda_prior: float,
    jitter: float,
) -> Dict[str, np.ndarray]:
    X = np.asarray(X_all, dtype=np.float32)
    n = int(X.shape[0])
    if n <= 0:
        raise RuntimeError('Cannot build graph prior for empty feature matrix')
    if n == 1:
        q0 = np.asarray([[float(lambda_prior) + float(jitter)]], dtype=np.float64)
        k0 = np.asarray([[1.0 / q0[0, 0]]], dtype=np.float32)
        return {
            'K0': k0,
            'diag': np.asarray([float(k0[0, 0])], dtype=np.float32),
            'knn_k': 0,
            'scale': 1.0,
        }

    k = min(max(1, int(knn_k)), n - 1)
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
        (float(lambda_smooth) * L.astype(np.float64))
        + ((float(lambda_prior) + float(jitter)) * np.eye(n, dtype=np.float64))
    )
    try:
        k0 = np.linalg.inv(q0).astype(np.float32)
    except Exception:
        k0 = np.linalg.pinv(q0).astype(np.float32)
    return {
        'K0': k0,
        'diag': np.diag(k0).astype(np.float32),
        'knn_k': int(k),
        'scale': float(scale),
    }


def graph_update_with_uncertainty(
    z0: np.ndarray,
    z0_sorted: np.ndarray,
    prior_cov: np.ndarray,
    prior_cov_diag: np.ndarray,
    moves: List[Tuple[int, float]],
    sigma2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    z0 = np.asarray(z0, dtype=np.float32).reshape(-1)
    prior_cov = np.asarray(prior_cov, dtype=np.float32)
    prior_cov_diag = np.asarray(prior_cov_diag, dtype=np.float32).reshape(-1)
    n = int(z0.size)

    if len(moves) == 0:
        std = np.sqrt(np.maximum(prior_cov_diag, 1e-8)).astype(np.float32)
        return z0.copy(), std

    obs_idx = np.asarray([int(i) for i, _ in moves], dtype=np.int64)
    target_p01 = [float(p) for _, p in moves]
    y_raw = np.asarray([quantile_from_sorted(z0_sorted, p) for p in target_p01], dtype=np.float32)
    obs_var = np.full((len(moves),), float(sigma2), dtype=np.float32)

    K_xo = prior_cov[:, obs_idx]
    K_oo = prior_cov[np.ix_(obs_idx, obs_idx)]
    S = np.asarray(K_oo + np.diag(obs_var), dtype=np.float64)
    try:
        S_inv = np.linalg.inv(S + (1e-6 * np.eye(S.shape[0], dtype=np.float64))).astype(np.float32)
    except Exception:
        S_inv = np.linalg.pinv(S + (1e-6 * np.eye(S.shape[0], dtype=np.float64))).astype(np.float32)

    residual = np.asarray(y_raw - z0[obs_idx], dtype=np.float32)
    gain = np.asarray(S_inv @ residual, dtype=np.float32)
    post_mean = np.asarray(z0 + (K_xo @ gain), dtype=np.float32)

    tmp = np.asarray(S_inv @ K_xo.T, dtype=np.float32)
    quad = np.sum(K_xo * tmp.T, axis=1).astype(np.float32)
    post_var = np.asarray(prior_cov_diag - quad, dtype=np.float32)
    std = np.sqrt(np.maximum(post_var, 1e-8)).astype(np.float32).reshape(n)
    return post_mean, std


def solve_symmetric_system(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    A = np.asarray(matrix, dtype=np.float32)
    B = np.asarray(rhs, dtype=np.float32)
    if A.size == 0:
        return B.copy()
    eye = np.eye(A.shape[0], dtype=np.float32)
    for jitter in (1e-6, 1e-5, 1e-4):
        try:
            L = np.linalg.cholesky(A + (jitter * eye))
            y = np.linalg.solve(L, B)
            return np.linalg.solve(L.T, y).astype(np.float32)
        except Exception:
            continue
    try:
        return np.linalg.solve(A + (1e-4 * eye), B).astype(np.float32)
    except Exception:
        return (np.linalg.pinv(A + (1e-4 * eye)) @ B).astype(np.float32)


def solve_hinv_vec(
    alpha_inv_vec: np.ndarray,
    D: np.ndarray,
    D_Ainv: np.ndarray,
    gram: np.ndarray,
    w_diag: np.ndarray,
    vec: np.ndarray,
) -> np.ndarray:
    a_inv = np.asarray(alpha_inv_vec, dtype=np.float32).reshape(-1)
    if D.size == 0:
        return (a_inv * np.asarray(vec, dtype=np.float32)).astype(np.float32)
    inv_w = (1.0 / np.maximum(np.asarray(w_diag, dtype=np.float32), 1e-8)).astype(np.float32)
    M = np.diag(inv_w) + np.asarray(gram, dtype=np.float32)
    rhs = np.asarray(D_Ainv @ np.asarray(vec, dtype=np.float32), dtype=np.float32)
    sol = solve_symmetric_system(M, rhs)
    return (np.asarray(a_inv * vec, dtype=np.float32) - np.asarray(D_Ainv.T @ sol, dtype=np.float32)).astype(np.float32)


def rank_loss(
    w: np.ndarray,
    w0: np.ndarray,
    D: np.ndarray,
    weights: np.ndarray,
    eta: float,
    alpha_vec: np.ndarray,
) -> float:
    u = (D @ np.asarray(w, dtype=np.float32)) / float(max(1e-6, eta))
    logistic = np.logaddexp(0.0, -u).astype(np.float32)
    diff = np.asarray(w, dtype=np.float32) - np.asarray(w0, dtype=np.float32)
    reg = 0.5 * float(np.sum(np.asarray(alpha_vec, dtype=np.float32) * diff * diff))
    return float(reg + float(np.sum(np.asarray(weights, dtype=np.float32) * logistic)))


def refresh_rank_posterior(
    X_all: np.ndarray,
    w0: np.ndarray,
    alpha_vec: np.ndarray,
    pair_i: List[int],
    pair_j: List[int],
    pair_y: List[float],
    pair_w: List[float],
    rank_eta: float,
    initial_w: Optional[np.ndarray] = None,
    compute_pair_matrix: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X_all, dtype=np.float32)
    d = int(X.shape[1])
    if len(pair_i) == 0:
        return (
            np.asarray(w0, dtype=np.float32).copy(),
            np.zeros((0, d), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    idx_i = np.asarray(pair_i, dtype=np.int64)
    idx_j = np.asarray(pair_j, dtype=np.int64)
    y = np.asarray(pair_y, dtype=np.float32)
    weights = np.asarray(pair_w, dtype=np.float32)
    Delta = (X[idx_i, :] - X[idx_j, :]).astype(np.float32)
    D = (y[:, None] * Delta).astype(np.float32)
    eta = float(max(1e-6, rank_eta))
    alpha_vec = np.maximum(np.asarray(alpha_vec, dtype=np.float32), 1e-8)
    alpha_inv = (1.0 / alpha_vec).astype(np.float32)
    D_Ainv = (D * alpha_inv[None, :]).astype(np.float32)
    gram = (D_Ainv @ D.T).astype(np.float32)

    if initial_w is not None and int(np.asarray(initial_w).size) == d:
        w = np.asarray(initial_w, dtype=np.float32).copy()
    else:
        w = np.asarray(w0, dtype=np.float32).copy()
    for _ in range(40):
        u = (D @ w) / eta
        sig_neg = sigmoid(-u)
        grad = (alpha_vec * (w - w0)) - ((D.T @ (weights * sig_neg)) / eta)
        gnorm = float(np.linalg.norm(grad))
        if not np.isfinite(gnorm) or gnorm <= 1e-5:
            break

        p = sigmoid(u)
        hdiag = np.maximum((weights * p * (1.0 - p)) / max(1e-8, eta ** 2), 1e-8).astype(np.float32)
        step = solve_hinv_vec(alpha_inv, D, D_Ainv, gram, hdiag, grad)
        step_dot = float(np.dot(grad, step))
        if not np.isfinite(step_dot) or step_dot <= 0.0:
            step = np.asarray(grad, dtype=np.float32) / np.maximum(alpha_vec, 1e-8)
            step_dot = float(np.dot(grad, step))
        if not np.isfinite(step_dot) or step_dot <= 0.0:
            break

        base = rank_loss(w, w0, D, weights, eta, alpha_vec)
        t = 1.0
        accepted = False
        while t >= 1e-4:
            cand = np.asarray(w - (t * step), dtype=np.float32)
            cand_loss = rank_loss(cand, w0, D, weights, eta, alpha_vec)
            if np.isfinite(cand_loss) and cand_loss <= (base - (1e-4 * t * step_dot)):
                w = cand
                accepted = True
                break
            t *= 0.5
        if not accepted:
            w = np.asarray(w - (0.05 * step), dtype=np.float32)

    if float(np.dot(w, w0)) < 0.0:
        w = -w
    nrm = float(np.linalg.norm(w))
    if nrm > 1e-8:
        w = (w / nrm).astype(np.float32)
    else:
        w = np.asarray(w0, dtype=np.float32).copy()

    if not compute_pair_matrix:
        return (
            w,
            np.asarray(D, dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    u = (D @ w) / eta
    p = sigmoid(u)
    w_diag = np.maximum((weights * p * (1.0 - p)) / max(1e-8, eta ** 2), 1e-8).astype(np.float32)
    inv_w = (1.0 / w_diag).astype(np.float32)
    M = np.diag(inv_w) + gram
    return w, np.asarray(D, dtype=np.float32), w_diag, np.asarray(M, dtype=np.float32)


@dataclass
class RankSweepState:
    mu: np.ndarray
    z: np.ndarray
    std: np.ndarray
    pair_i: List[int]
    pair_j: List[int]
    pair_y: List[float]
    pair_w: List[float]
    pair_D: np.ndarray
    pair_M: np.ndarray


def initialize_rank_sweep_state(
    X_all: np.ndarray,
    w0: np.ndarray,
    prior_std: np.ndarray,
) -> RankSweepState:
    X = np.asarray(X_all, dtype=np.float32)
    mu = np.asarray(w0, dtype=np.float32).copy()
    z = np.asarray(X @ mu, dtype=np.float32)
    std = np.asarray(prior_std, dtype=np.float32).copy()
    return RankSweepState(
        mu=mu,
        z=z,
        std=std,
        pair_i=[],
        pair_j=[],
        pair_y=[],
        pair_w=[],
        pair_D=np.zeros((0, X.shape[1]), dtype=np.float32),
        pair_M=np.zeros((0, 0), dtype=np.float32),
    )


def advance_rank_sweep_state(
    X_all: np.ndarray,
    w0: np.ndarray,
    alpha_vec: np.ndarray,
    alpha_inv: np.ndarray,
    prior_x: np.ndarray,
    state: RankSweepState,
    image_idx: int,
    target_p01: float,
    rank_eta: float,
    rank_anchor_k: int,
    rank_anchor_delta: float,
    rank_max_pairs: int,
    compute_std: bool = True,
) -> RankSweepState:
    X = np.asarray(X_all, dtype=np.float32)
    append_rank_pairs_from_move(
        z_now=state.z,
        image_idx=int(image_idx),
        target_p01=float(target_p01),
        rank_anchor_k=int(rank_anchor_k),
        rank_anchor_delta=float(rank_anchor_delta),
        rank_max_pairs=int(rank_max_pairs),
        pair_i=state.pair_i,
        pair_j=state.pair_j,
        pair_y=state.pair_y,
        pair_w=state.pair_w,
    )
    mu, pair_D, _, pair_M = refresh_rank_posterior(
        X_all=X,
        w0=w0,
        alpha_vec=alpha_vec,
        pair_i=state.pair_i,
        pair_j=state.pair_j,
        pair_y=state.pair_y,
        pair_w=state.pair_w,
        rank_eta=float(rank_eta),
        initial_w=state.mu,
        compute_pair_matrix=compute_std,
    )
    z = np.asarray(X @ mu, dtype=np.float32)
    if not compute_std:
        std = np.zeros((0,), dtype=np.float32)
    elif pair_D.size == 0 or pair_M.size == 0:
        std = np.sqrt(np.maximum(prior_x, 1e-8)).astype(np.float32)
    else:
        V = (pair_D * alpha_inv[None, :]) @ X.T
        MV = solve_symmetric_system(pair_M, V)
        quad = np.sum(V * MV, axis=0)
        var = prior_x - quad
        std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    state.mu = mu
    state.z = z
    state.std = std
    state.pair_D = pair_D
    state.pair_M = pair_M
    return state


def append_rank_pairs_from_move(
    z_now: np.ndarray,
    image_idx: int,
    target_p01: float,
    rank_anchor_k: int,
    rank_anchor_delta: float,
    rank_max_pairs: int,
    pair_i: List[int],
    pair_j: List[int],
    pair_y: List[float],
    pair_w: List[float],
) -> None:
    score01 = rank_percentile_01(z_now)
    target = float(max(0.0, min(1.0, target_p01)))
    old_p = float(score01[int(image_idx)])
    k = int(max(1, rank_anchor_k))
    delta = float(max(0.01, min(0.45, rank_anchor_delta)))
    max_cross = int(max(2, 2 * k))
    move_mag = abs(target - old_p)
    weight = float(max(0.25, 1.0 + (2.0 * move_mag)))

    n = int(len(score01))
    idx_all = np.arange(n, dtype=np.int64)
    not_self = idx_all[idx_all != int(image_idx)]
    below_band = not_self[(score01[not_self] < target) & (score01[not_self] >= (target - delta))]
    above_band = not_self[(score01[not_self] > target) & (score01[not_self] <= (target + delta))]

    def pick(cands: np.ndarray, side: str) -> List[int]:
        ids = cands
        if ids.size == 0:
            pool = not_self[score01[not_self] < target] if side == 'below' else not_self[score01[not_self] > target]
            ids = pool
        if ids.size == 0:
            return []
        order = np.argsort(np.abs(score01[ids] - target), kind='mergesort')
        return [int(v) for v in ids[order[: min(k, ids.size)]].tolist()]

    for j in pick(below_band, 'below'):
        pair_i.append(int(image_idx))
        pair_j.append(int(j))
        pair_y.append(1.0)
        pair_w.append(weight)
    for j in pick(above_band, 'above'):
        pair_i.append(int(image_idx))
        pair_j.append(int(j))
        pair_y.append(-1.0)
        pair_w.append(weight)

    lo = min(old_p, target)
    hi = max(old_p, target)
    crossed = not_self[(score01[not_self] >= lo) & (score01[not_self] <= hi)]
    if crossed.size > 0:
        if crossed.size > max_cross:
            step = max(1, int(np.floor(crossed.size / max_cross)))
            crossed = crossed[::step][:max_cross]
        signed = 1.0 if target >= old_p else -1.0
        for j in crossed.tolist():
            pair_i.append(int(image_idx))
            pair_j.append(int(j))
            pair_y.append(float(signed))
            pair_w.append(float(max(0.15, 0.75 * weight)))

    overflow = len(pair_i) - int(max(16, rank_max_pairs))
    if overflow > 0:
        del pair_i[:overflow]
        del pair_j[:overflow]
        del pair_y[:overflow]
        del pair_w[:overflow]


def rank_update_with_uncertainty(
    X_all: np.ndarray,
    w0: np.ndarray,
    moves: List[Tuple[int, float]],
    alpha_vec: np.ndarray,
    rank_eta: float,
    rank_anchor_k: int,
    rank_anchor_delta: float,
    rank_max_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X_all, dtype=np.float32)
    w0 = np.asarray(w0, dtype=np.float32)
    alpha_vec = np.maximum(np.asarray(alpha_vec, dtype=np.float32), 1e-8)
    alpha_inv = (1.0 / alpha_vec).astype(np.float32)

    mu = w0.copy()
    pair_i: List[int] = []
    pair_j: List[int] = []
    pair_y: List[float] = []
    pair_w: List[float] = []
    pair_D = np.zeros((0, X.shape[1]), dtype=np.float32)
    pair_M = np.zeros((0, 0), dtype=np.float32)

    for image_idx, target_p01 in moves:
        z_now = np.asarray(X @ mu, dtype=np.float32)
        append_rank_pairs_from_move(
            z_now=z_now,
            image_idx=int(image_idx),
            target_p01=float(target_p01),
            rank_anchor_k=int(rank_anchor_k),
            rank_anchor_delta=float(rank_anchor_delta),
            rank_max_pairs=int(rank_max_pairs),
            pair_i=pair_i,
            pair_j=pair_j,
            pair_y=pair_y,
            pair_w=pair_w,
        )
        mu, pair_D, _, pair_M = refresh_rank_posterior(
            X_all=X,
            w0=w0,
            alpha_vec=alpha_vec,
            pair_i=pair_i,
            pair_j=pair_j,
            pair_y=pair_y,
            pair_w=pair_w,
            rank_eta=float(rank_eta),
            initial_w=mu,
            compute_pair_matrix=True,
        )

    z = np.asarray(X @ mu, dtype=np.float32)
    prior_x = np.sum((X * X) * alpha_inv[None, :], axis=1).astype(np.float32)
    if pair_D.size == 0 or pair_M.size == 0:
        return z, np.sqrt(np.maximum(prior_x, 1e-8)).astype(np.float32)
    V = (pair_D * alpha_inv[None, :]) @ X.T
    MV = solve_symmetric_system(pair_M, V)
    quad = np.sum(V * MV, axis=0)
    var = prior_x - quad
    return z, np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)


@dataclass
class LoadedData:
    ids: List[str]
    emotions: List[str]
    labels: np.ndarray
    id_to_idx: Dict[str, int]
    id_to_emotion: Dict[str, str]
    Y: np.ndarray
    X_clip: np.ndarray
    X_dino: np.ndarray


def load_emoset_data(dataset_root: Path) -> LoadedData:
    if not dataset_root.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_root}')

    engine = ImageGalleryEngine(str(dataset_root))
    entries = engine.list_images()
    if len(entries) == 0:
        raise RuntimeError(f'No images found under {dataset_root}')

    ids = [str(e.id) for e in entries]
    id_to_idx = {img_id: i for i, img_id in enumerate(ids)}

    metadata_path = dataset_root / 'metadata.csv'
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        meta.columns = [str(c).strip() for c in meta.columns]
        image_col = 'image' if 'image' in meta.columns else meta.columns[0]
        emotion_col = 'emotion' if 'emotion' in meta.columns else None
        if emotion_col is None:
            for c in meta.columns:
                if c.lower().strip() == 'emotion':
                    emotion_col = c
                    break
        if emotion_col is None:
            raise RuntimeError(f'Could not find emotion column in {metadata_path}')
        meta[image_col] = meta[image_col].astype(str).str.strip()
        meta[emotion_col] = meta[emotion_col].astype(str).str.strip().str.lower()
        id_to_emotion = dict(zip(meta[image_col], meta[emotion_col]))
    else:
        id_to_emotion = {img_id: img_id.split('_', 1)[0].strip().lower() for img_id in ids}

    labels = np.asarray([id_to_emotion.get(img_id, img_id.split('_', 1)[0].lower()) for img_id in ids])
    emotions = sorted(pd.unique(labels).tolist())
    Y = np.stack([(labels == emo).astype(np.float32) for emo in emotions], axis=1)

    clip = engine._load_embeddings_only(entries, method='clip')
    dino = engine._load_embeddings_only(entries, method='dino')
    if clip is None or dino is None:
        raise RuntimeError('Expected both clip and dino caches for EmoSet')

    X_clip = l2_normalize_rows(np.asarray(clip, dtype=np.float32))
    X_dino = l2_normalize_rows(np.asarray(dino, dtype=np.float32))
    if X_clip.shape[0] != len(ids) or X_dino.shape[0] != len(ids):
        raise RuntimeError('Embedding/image count mismatch')

    return LoadedData(
        ids=ids,
        emotions=emotions,
        labels=labels,
        id_to_idx=id_to_idx,
        id_to_emotion=id_to_emotion,
        Y=Y,
        X_clip=X_clip,
        X_dino=X_dino,
    )


def build_text_encoder(model_name: str, local_only: bool):
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True, local_files_only=local_only).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model.eval()

    def encode(text: str) -> np.ndarray:
        with torch.no_grad():
            payload = tokenizer([text], return_tensors='pt', padding=True, truncation=True)
            payload = {k: v.to(device) for k, v in payload.items()}
            feats = model.get_text_features(**payload)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.squeeze(0).detach().cpu().numpy().astype(np.float32)

    return encode


def build_move_catalog(data: LoadedData, k_per_emotion: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    ids_by_emotion = {
        emo: [img_id for img_id, y in zip(data.ids, data.labels) if y == emo]
        for emo in data.emotions
    }
    for emo in data.emotions:
        if len(ids_by_emotion[emo]) < k_per_emotion:
            raise RuntimeError(f'Not enough images for emotion={emo}')
    sampled = {
        emo: rng.choice(ids_by_emotion[emo], size=k_per_emotion, replace=False).tolist()
        for emo in data.emotions
    }
    catalog: List[str] = []
    for r in range(k_per_emotion):
        for emo in data.emotions:
            catalog.append(sampled[emo][r])
    return catalog


def run_experiment(
    data: LoadedData,
    w_clip_text: Dict[str, np.ndarray],
    move_catalog: List[str],
    mode: str,
    feature_space: str,
    clip_weight: float,
    dino_weight: float,
    alpha: float,
    dino_alpha: float,
    bias_alpha: float,
    sigma2: float,
    graph_knn_k: int,
    graph_lambda_smooth: float,
    graph_lambda_prior: float,
    graph_jitter: float,
    rank_eta: float,
    rank_anchor_k: int,
    rank_anchor_delta: float,
    rank_max_pairs: int,
) -> Dict[str, object]:
    mode_name = normalize_mode(mode)
    fs = str(feature_space).strip().lower()
    if fs != 'clip_dino' and fs != 'clip':
        raise ValueError(f'Unsupported feature_space={feature_space}')

    if fs == 'clip_dino':
        cw = float(max(1e-6, clip_weight))
        dw = float(max(1e-6, dino_weight))
        s = cw + dw
        cw = cw / s
        dw = dw / s
        X = np.concatenate(
            [np.sqrt(cw, dtype=np.float32) * data.X_clip, np.sqrt(dw, dtype=np.float32) * data.X_dino],
            axis=1,
        ).astype(np.float32)
        alpha_vec = np.concatenate(
            [
                np.full((data.X_clip.shape[1],), float(alpha), dtype=np.float32),
                np.full((data.X_dino.shape[1],), float(dino_alpha), dtype=np.float32),
            ],
            axis=0,
        )
    else:
        X = data.X_clip.astype(np.float32)
        alpha_vec = np.full((data.X_clip.shape[1],), float(alpha), dtype=np.float32)

    priors: Dict[str, Dict[str, np.ndarray]] = {}
    dino_dim = int(data.X_dino.shape[1])
    for emo in data.emotions:
        if fs == 'clip_dino':
            w0 = l2_normalize_vec(np.concatenate([w_clip_text[emo], np.zeros((dino_dim,), dtype=np.float32)]))
        else:
            w0 = w_clip_text[emo].copy()
        z0 = (X @ w0).astype(np.float32)
        priors[emo] = {'w0': w0, 'z0': z0, 'z0_sorted': np.sort(z0).astype(np.float32)}

    graph_prior = None
    if mode_name == 'graph':
        graph_prior = build_graph_prior(
            X_all=X,
            knn_k=int(graph_knn_k),
            lambda_smooth=float(graph_lambda_smooth),
            lambda_prior=float(graph_lambda_prior),
            jitter=float(graph_jitter),
        )
    rank_states: Dict[str, RankSweepState] = {}
    rank_prior_x = np.zeros((0,), dtype=np.float32)
    alpha_inv = np.zeros((0,), dtype=np.float32)
    if mode_name == 'rank':
        alpha_inv = (1.0 / np.maximum(alpha_vec, 1e-8)).astype(np.float32)
        rank_prior_x = np.sum((X * X) * alpha_inv[None, :], axis=1).astype(np.float32)
        rank_prior_std = np.sqrt(np.maximum(rank_prior_x, 1e-8)).astype(np.float32)
        for emo in data.emotions:
            rank_states[emo] = initialize_rank_sweep_state(
                X_all=X,
                w0=priors[emo]['w0'],
                prior_std=rank_prior_std,
            )

    corr_mats: List[np.ndarray] = []
    auroc_mats: List[np.ndarray] = []
    diag_sum: List[float] = []
    macro_auroc: List[float] = []
    mean_uncertainty_global: List[float] = []
    n_steps = len(move_catalog) + 1
    for step in range(n_steps):
        axis_scores: Dict[str, np.ndarray] = {}
        axis_std: Dict[str, np.ndarray] = {}
        need_step_uncertainty = step == (n_steps - 1)
        if mode_name == 'rank':
            if step > 0:
                moved_id = move_catalog[step - 1]
                moved_idx = data.id_to_idx[moved_id]
                moved_gt = data.id_to_emotion.get(moved_id, moved_id.split('_', 1)[0].lower())
                for axis_emo in data.emotions:
                    rank_states[axis_emo] = advance_rank_sweep_state(
                        X_all=X,
                        w0=priors[axis_emo]['w0'],
                        alpha_vec=alpha_vec,
                        alpha_inv=alpha_inv,
                        prior_x=rank_prior_x,
                        state=rank_states[axis_emo],
                        image_idx=int(moved_idx),
                        target_p01=1.0 if moved_gt == axis_emo else 0.0,
                        rank_eta=float(rank_eta),
                        rank_anchor_k=int(rank_anchor_k),
                        rank_anchor_delta=float(rank_anchor_delta),
                        rank_max_pairs=int(rank_max_pairs),
                        compute_std=need_step_uncertainty,
                    )
            for axis_emo in data.emotions:
                axis_scores[axis_emo] = rank_states[axis_emo].z
                axis_std[axis_emo] = rank_states[axis_emo].std if need_step_uncertainty else np.zeros((0,), dtype=np.float32)
        else:
            moved_ids = move_catalog[:step]
            for axis_emo in data.emotions:
                moves: List[Tuple[int, float]] = []
                for img_id in moved_ids:
                    idx = data.id_to_idx[img_id]
                    gt = data.id_to_emotion.get(img_id, img_id.split('_', 1)[0].lower())
                    moves.append((idx, 1.0 if gt == axis_emo else 0.0))

                if mode_name == 'graph':
                    z, std = graph_update_with_uncertainty(
                        z0=priors[axis_emo]['z0'],
                        z0_sorted=priors[axis_emo]['z0_sorted'],
                        prior_cov=graph_prior['K0'],
                        prior_cov_diag=graph_prior['diag'],
                        moves=moves,
                        sigma2=float(sigma2),
                    )
                else:
                    mu, b = gaussian_update(
                        X_all=X,
                        w0=priors[axis_emo]['w0'],
                        z0_sorted=priors[axis_emo]['z0_sorted'],
                        moves=moves,
                        alpha_vec=alpha_vec,
                        bias_alpha=float(bias_alpha),
                        sigma2=float(sigma2),
                        b0=0.0,
                    )
                    z = np.asarray((X @ mu) + float(b), dtype=np.float32)
                    std = np.full((X.shape[0],), np.nan, dtype=np.float32)
                axis_scores[axis_emo] = z
                axis_std[axis_emo] = std

        C_rows = []
        for axis_emo in data.emotions:
            z = axis_scores[axis_emo]
            C_rows.append([pearson_corr(z, data.Y[:, j]) for j in range(data.Y.shape[1])])
        C = np.asarray(C_rows, dtype=np.float32)
        corr_mats.append(C)
        diag_sum.append(float(np.trace(C)))
        A_rows = []
        for axis_emo in data.emotions:
            z = axis_scores[axis_emo]
            A_rows.append([binary_auroc(z, data.Y[:, j]) for j in range(data.Y.shape[1])])
        A = np.asarray(A_rows, dtype=np.float32)
        auroc_mats.append(A)
        macro_auroc.append(float(np.nanmean(np.diag(A))))
        if mode_name != 'rank' or need_step_uncertainty:
            step_axis_mean = [float(np.nanmean(axis_std[emo])) for emo in data.emotions]
            mean_uncertainty_global.append(float(np.nanmean(step_axis_mean)))
        else:
            mean_uncertainty_global.append(float('nan'))

    diag_arr = np.asarray(diag_sum, dtype=np.float32)
    macro_auroc_arr = np.asarray(macro_auroc, dtype=np.float32)
    return {
        'corr_mats': corr_mats,
        'auroc_mats': auroc_mats,
        'diag_sum': diag_arr,
        'final_diag_sum': float(diag_arr[-1]),
        'best_step_diag_sum': float(np.max(diag_arr)),
        'auc_diag_sum': float(np.mean(diag_arr)),
        'macro_auroc': macro_auroc_arr,
        'final_macro_auroc': float(macro_auroc_arr[-1]),
        'best_step_macro_auroc': float(np.nanmax(macro_auroc_arr)),
        'auc_macro_auroc': float(np.nanmean(macro_auroc_arr)),
        'mean_uncertainty_global': np.asarray(mean_uncertainty_global, dtype=np.float32),
    }


def main() -> None:
    mode_name = normalize_mode(AXIS_BAYES_MODE)
    if mode_name not in {'gaussian', 'rank', 'graph'}:
        raise RuntimeError(f'Unsupported AXIS_BAYES_MODE={AXIS_BAYES_MODE}')
    csv_log_path = Path(CSV_LOG_PATH) if CSV_LOG_PATH is not None else default_csv_log_path(mode_name)

    run_id = uuid.uuid4().hex[:12]
    print(f'[optuna-sweep] run_id={run_id} mode={mode_name}')

    data = load_emoset_data(DATASET_ROOT)
    print(f'[optuna-sweep] loaded n_images={len(data.ids)} emotions={data.emotions}')

    encode_text = build_text_encoder(CLIP_MODEL_NAME, CLIP_LOCAL_FILES_ONLY)
    w_clip_text: Dict[str, np.ndarray] = {}
    for emo in data.emotions:
        pos_prompt = POS_PROMPTS_OVERRIDE.get(emo, POS_PROMPT_TEMPLATE.format(emotion=emo))
        neg_prompt = NEG_PROMPTS_OVERRIDE.get(emo, NEG_PROMPT_COMMON)
        pos = l2_normalize_vec(encode_text(pos_prompt))
        neg = l2_normalize_vec(encode_text(neg_prompt))
        w_clip_text[emo] = l2_normalize_vec(pos - neg)
    print('[optuna-sweep] built text directions')

    move_catalog = build_move_catalog(data, MOVES_PER_EMOTION, MOVE_SELECTION_SEED)
    print(f'[optuna-sweep] move_catalog size={len(move_catalog)}')

    fields = [
        'ts_utc',
        'run_id',
        'row_type',
        'trial_number',
        'trial_state',
        'objective_name',
        'value',
        'mode',
        'feature_space',
        'clip_weight',
        'dino_weight',
        'alpha',
        'dino_alpha',
        'bias_alpha',
        'sigma2',
        'rank_eta',
        'rank_anchor_k',
        'rank_anchor_delta',
        'rank_max_pairs',
        'graph_knn_k',
        'graph_lambda_smooth',
        'graph_lambda_prior',
        'graph_jitter',
        'n_steps',
        'final_macro_auroc',
        'best_step_macro_auroc',
        'auc_macro_auroc',
        'final_diag_sum',
        'best_step_diag_sum',
        'auc_diag_sum',
        'final_mean_uncertainty',
        'macro_auroc_json',
        'diag_sum_json',
        'error',
    ]
    logger = CSVAppender(csv_log_path, fields)

    trial_results: List[Tuple[int, float, Dict[str, object]]] = []
    best_trial_metrics: Dict[int, Dict[str, object]] = {}
    rank_max_pairs_hi = max(RANK_SWEEP_MAX_PAIRS_MIN, 16 * max(2, int(min(128, 6 * len(move_catalog))) // 16))

    def objective(trial: optuna.Trial) -> float:
        clip_weight = trial.suggest_float('clip_weight', 0.1, 0.90)
        dino_weight = 1.0 - clip_weight
        alpha = float('nan')
        dino_alpha = float('nan')
        bias_alpha = float('nan')
        rank_eta = float('nan')
        rank_anchor_k = int(AXIS_BAYES_RANK_ANCHOR_K)
        rank_anchor_delta = float('nan')
        rank_max_pairs = int(AXIS_BAYES_RANK_MAX_PAIRS)
        graph_knn_k = int(AXIS_BAYES_GRAPH_KNN_K)
        graph_lambda_smooth = float(AXIS_BAYES_GRAPH_LAMBDA_SMOOTH)
        graph_lambda_prior = float(AXIS_BAYES_GRAPH_LAMBDA_PRIOR)
        graph_jitter = float(AXIS_BAYES_GRAPH_JITTER)
        sigma2 = float('nan')

        if mode_name == 'graph':
            sigma2 = trial.suggest_float('sigma2', 0.0005, 0.05, log=True)
            graph_knn_k = trial.suggest_int('graph_knn_k', 4, 32)
            graph_lambda_smooth = trial.suggest_float('graph_lambda_smooth', 0.1, 40.0, log=True)
            graph_lambda_prior = trial.suggest_float('graph_lambda_prior', 0.05, 8.0, log=True)
        elif mode_name == 'rank':
            alpha = trial.suggest_float('alpha', RANK_SWEEP_ALPHA_MIN, RANK_SWEEP_ALPHA_MAX, log=True)
            dino_alpha = trial.suggest_float('dino_alpha', RANK_SWEEP_ALPHA_MIN, RANK_SWEEP_ALPHA_MAX, log=True)
            rank_eta = trial.suggest_float('rank_eta', RANK_SWEEP_ETA_MIN, RANK_SWEEP_ETA_MAX, log=True)
            rank_anchor_k = trial.suggest_int('rank_anchor_k', RANK_SWEEP_ANCHOR_K_MIN, RANK_SWEEP_ANCHOR_K_MAX)
            rank_anchor_delta = trial.suggest_float('rank_anchor_delta', RANK_SWEEP_ANCHOR_DELTA_MIN, RANK_SWEEP_ANCHOR_DELTA_MAX)
            rank_max_pairs = trial.suggest_int('rank_max_pairs', RANK_SWEEP_MAX_PAIRS_MIN, rank_max_pairs_hi, step=16)
        else:
            alpha = trial.suggest_float('alpha', 0.01, 1.0, log=True)
            dino_alpha = trial.suggest_float('dino_alpha', 0.01, 1.0, log=True)
            bias_alpha = trial.suggest_float('bias_alpha', 0.02, 4.0, log=True)
            sigma2 = trial.suggest_float('sigma2', 0.0005, 0.01, log=True)

        try:
            res = run_experiment(
                data=data,
                w_clip_text=w_clip_text,
                move_catalog=move_catalog,
                mode=mode_name,
                feature_space=AXIS_BAYES_FEATURE_SPACE,
                clip_weight=clip_weight,
                dino_weight=dino_weight,
                alpha=alpha,
                dino_alpha=dino_alpha,
                bias_alpha=bias_alpha,
                sigma2=sigma2,
                graph_knn_k=graph_knn_k,
                graph_lambda_smooth=graph_lambda_smooth,
                graph_lambda_prior=graph_lambda_prior,
                graph_jitter=graph_jitter,
                rank_eta=rank_eta,
                rank_anchor_k=rank_anchor_k,
                rank_anchor_delta=rank_anchor_delta,
                rank_max_pairs=rank_max_pairs,
            )
            val = float(res[OPTUNA_OBJECTIVE_NAME])
            best_trial_metrics[int(trial.number)] = {
                'final_macro_auroc': float(res['final_macro_auroc']),
                'best_step_macro_auroc': float(res['best_step_macro_auroc']),
                'auc_macro_auroc': float(res['auc_macro_auroc']),
                'final_diag_sum': float(res['final_diag_sum']),
                'best_step_diag_sum': float(res['best_step_diag_sum']),
                'auc_diag_sum': float(res['auc_diag_sum']),
                'final_mean_uncertainty': float(res['mean_uncertainty_global'][-1]),
                'macro_auroc_json': json.dumps(res['macro_auroc'].tolist(), ensure_ascii=True),
                'diag_sum_json': json.dumps(res['diag_sum'].tolist(), ensure_ascii=True),
            }
            trial_results.append(
                (
                    int(trial.number),
                    val,
                    {
                        'mode': mode_name,
                        'clip_weight': float(clip_weight),
                        'dino_weight': float(dino_weight),
                        'alpha': float(alpha),
                        'dino_alpha': float(dino_alpha),
                        'bias_alpha': float(bias_alpha),
                        'sigma2': float(sigma2),
                        'rank_eta': float(rank_eta),
                        'rank_anchor_k': int(rank_anchor_k),
                        'rank_anchor_delta': float(rank_anchor_delta),
                        'rank_max_pairs': int(rank_max_pairs),
                        'graph_knn_k': int(graph_knn_k) if mode_name == 'graph' else '',
                        'graph_lambda_smooth': float(graph_lambda_smooth) if mode_name == 'graph' else '',
                        'graph_lambda_prior': float(graph_lambda_prior) if mode_name == 'graph' else '',
                    },
                ),
            )
            logger.append(
                {
                    'ts_utc': utc_now_iso(),
                    'run_id': run_id,
                    'row_type': 'trial',
                    'trial_number': int(trial.number),
                    'trial_state': 'ok',
                    'objective_name': OPTUNA_OBJECTIVE_NAME,
                    'value': val,
                    'mode': mode_name,
                    'feature_space': AXIS_BAYES_FEATURE_SPACE,
                    'clip_weight': float(clip_weight),
                    'dino_weight': float(dino_weight),
                    'alpha': float(alpha) if np.isfinite(alpha) else '',
                    'dino_alpha': float(dino_alpha) if np.isfinite(dino_alpha) else '',
                    'bias_alpha': float(bias_alpha) if np.isfinite(bias_alpha) else '',
                    'sigma2': float(sigma2) if np.isfinite(sigma2) else '',
                    'rank_eta': float(rank_eta) if np.isfinite(rank_eta) else '',
                    'rank_anchor_k': int(rank_anchor_k) if mode_name == 'rank' else '',
                    'rank_anchor_delta': float(rank_anchor_delta) if np.isfinite(rank_anchor_delta) else '',
                    'rank_max_pairs': int(rank_max_pairs) if mode_name == 'rank' else '',
                    'graph_knn_k': int(graph_knn_k) if mode_name == 'graph' else '',
                    'graph_lambda_smooth': float(graph_lambda_smooth) if mode_name == 'graph' else '',
                    'graph_lambda_prior': float(graph_lambda_prior) if mode_name == 'graph' else '',
                    'graph_jitter': float(graph_jitter) if mode_name == 'graph' else '',
                    'n_steps': int(len(move_catalog) + 1),
                    'final_macro_auroc': float(res['final_macro_auroc']),
                    'best_step_macro_auroc': float(res['best_step_macro_auroc']),
                    'auc_macro_auroc': float(res['auc_macro_auroc']),
                    'final_diag_sum': float(res['final_diag_sum']),
                    'best_step_diag_sum': float(res['best_step_diag_sum']),
                    'auc_diag_sum': float(res['auc_diag_sum']),
                    'final_mean_uncertainty': float(res['mean_uncertainty_global'][-1]),
                    'macro_auroc_json': json.dumps(res['macro_auroc'].tolist(), ensure_ascii=True),
                    'diag_sum_json': json.dumps(res['diag_sum'].tolist(), ensure_ascii=True),
                    'error': '',
                },
            )
            return val
        except Exception as exc:
            logger.append(
                {
                    'ts_utc': utc_now_iso(),
                    'run_id': run_id,
                    'row_type': 'trial',
                    'trial_number': int(trial.number),
                    'trial_state': 'error',
                    'objective_name': OPTUNA_OBJECTIVE_NAME,
                    'value': '',
                    'mode': mode_name,
                    'feature_space': AXIS_BAYES_FEATURE_SPACE,
                    'clip_weight': float(clip_weight),
                    'dino_weight': float(dino_weight),
                    'alpha': float(alpha) if np.isfinite(alpha) else '',
                    'dino_alpha': float(dino_alpha) if np.isfinite(dino_alpha) else '',
                    'bias_alpha': float(bias_alpha) if np.isfinite(bias_alpha) else '',
                    'sigma2': float(sigma2) if np.isfinite(sigma2) else '',
                    'rank_eta': float(rank_eta) if np.isfinite(rank_eta) else '',
                    'rank_anchor_k': int(rank_anchor_k) if mode_name == 'rank' else '',
                    'rank_anchor_delta': float(rank_anchor_delta) if np.isfinite(rank_anchor_delta) else '',
                    'rank_max_pairs': int(rank_max_pairs) if mode_name == 'rank' else '',
                    'graph_knn_k': int(graph_knn_k) if mode_name == 'graph' else '',
                    'graph_lambda_smooth': float(graph_lambda_smooth) if mode_name == 'graph' else '',
                    'graph_lambda_prior': float(graph_lambda_prior) if mode_name == 'graph' else '',
                    'graph_jitter': float(graph_jitter) if mode_name == 'graph' else '',
                    'n_steps': int(len(move_catalog) + 1),
                    'final_macro_auroc': '',
                    'best_step_macro_auroc': '',
                    'auc_macro_auroc': '',
                    'final_diag_sum': '',
                    'best_step_diag_sum': '',
                    'auc_diag_sum': '',
                    'final_mean_uncertainty': '',
                    'macro_auroc_json': '',
                    'diag_sum_json': '',
                    'error': f'{type(exc).__name__}: {exc}'[:500],
                },
            )
            raise

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SAMPLER_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=int(N_TRIALS), n_jobs=1, show_progress_bar=True)

    best_val = float(study.best_value)
    best_params = dict(study.best_params)
    best_metrics = best_trial_metrics.get(int(study.best_trial.number), {})
    logger.append(
        {
            'ts_utc': utc_now_iso(),
            'run_id': run_id,
            'row_type': 'best_summary',
            'trial_number': int(study.best_trial.number),
            'trial_state': 'ok',
            'objective_name': OPTUNA_OBJECTIVE_NAME,
            'value': best_val,
            'mode': mode_name,
            'feature_space': AXIS_BAYES_FEATURE_SPACE,
            'clip_weight': float(best_params.get('clip_weight', 'nan')),
            'dino_weight': float(1.0 - float(best_params.get('clip_weight', 0.0))),
            'alpha': float(best_params.get('alpha', 'nan')) if 'alpha' in best_params else '',
            'dino_alpha': float(best_params.get('dino_alpha', 'nan')) if 'dino_alpha' in best_params else '',
            'bias_alpha': float(best_params.get('bias_alpha', 'nan')) if 'bias_alpha' in best_params else '',
            'sigma2': float(best_params.get('sigma2', 'nan')) if 'sigma2' in best_params else '',
            'rank_eta': float(best_params.get('rank_eta', 'nan')) if 'rank_eta' in best_params else '',
            'rank_anchor_k': int(best_params.get('rank_anchor_k', AXIS_BAYES_RANK_ANCHOR_K)) if 'rank_anchor_k' in best_params else '',
            'rank_anchor_delta': float(best_params.get('rank_anchor_delta', 'nan')) if 'rank_anchor_delta' in best_params else '',
            'rank_max_pairs': int(best_params.get('rank_max_pairs', AXIS_BAYES_RANK_MAX_PAIRS)) if 'rank_max_pairs' in best_params else '',
            'graph_knn_k': int(best_params.get('graph_knn_k', AXIS_BAYES_GRAPH_KNN_K)) if 'graph_knn_k' in best_params else '',
            'graph_lambda_smooth': float(best_params.get('graph_lambda_smooth', AXIS_BAYES_GRAPH_LAMBDA_SMOOTH)) if 'graph_lambda_smooth' in best_params else '',
            'graph_lambda_prior': float(best_params.get('graph_lambda_prior', AXIS_BAYES_GRAPH_LAMBDA_PRIOR)) if 'graph_lambda_prior' in best_params else '',
            'graph_jitter': float(AXIS_BAYES_GRAPH_JITTER) if mode_name == 'graph' else '',
            'n_steps': int(len(move_catalog) + 1),
            'final_macro_auroc': float(best_metrics.get('final_macro_auroc', best_val)),
            'best_step_macro_auroc': float(best_metrics.get('best_step_macro_auroc', best_val)),
            'auc_macro_auroc': float(best_metrics.get('auc_macro_auroc', best_val)),
            'final_diag_sum': best_metrics.get('final_diag_sum', ''),
            'best_step_diag_sum': best_metrics.get('best_step_diag_sum', ''),
            'auc_diag_sum': best_metrics.get('auc_diag_sum', ''),
            'final_mean_uncertainty': best_metrics.get('final_mean_uncertainty', ''),
            'macro_auroc_json': best_metrics.get('macro_auroc_json', ''),
            'diag_sum_json': best_metrics.get('diag_sum_json', ''),
            'error': '',
        },
    )

    print(f'[optuna-sweep] objective={OPTUNA_OBJECTIVE_NAME} best_value={best_val:.6f}')
    print(f'[optuna-sweep] best_params={best_params}')
    print(f'[optuna-sweep] appended results to {csv_log_path}')


if __name__ == '__main__':
    main()
