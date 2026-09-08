from __future__ import annotations

import numpy as np

from .data import OrdinalTask
from .metrics import rank_percentile_01, spearman_corr


LAMBDA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
BETA_GRID = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
TARGET_MODE_GROUND_TRUTH = 'ground_truth_normalized'
TARGET_MODE_PRIOR_QUANTILE = 'prior_quantile'
TARGET_MODES = (TARGET_MODE_GROUND_TRUTH, TARGET_MODE_PRIOR_QUANTILE)


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    scale = float(np.std(arr))
    if scale <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.asarray((arr - float(np.mean(arr))) / scale, dtype=np.float32)


def validate_target_mode(target_mode: str) -> str:
    mode = str(target_mode or TARGET_MODE_GROUND_TRUTH).strip().lower()
    if mode not in TARGET_MODES:
        raise ValueError(f'Unknown target_mode {target_mode!r}; expected one of {TARGET_MODES}')
    return mode


def prior_quantile(prior_scores: np.ndarray, p01: float) -> float:
    values = np.asarray(prior_scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return float('nan')
    p = float(max(0.0, min(1.0, float(p01))))
    return float(np.quantile(values, p))


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size != bb.size or aa.size == 0:
        return float('nan')
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return float('nan')
    return float(np.dot(aa, bb) / denom)


def pairwise_agreement(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if s.size != y.size or s.size < 2:
        return float('nan')
    ii, jj = np.triu_indices(s.size, k=1)
    dy = y[ii] - y[jj]
    valid = np.abs(dy) > 1e-8
    if not np.any(valid):
        return float('nan')
    ds = s[ii] - s[jj]
    signed = ds[valid] * dy[valid]
    return float((np.sum(signed > 0.0) + (0.5 * np.sum(np.abs(signed) <= 1e-12))) / np.sum(valid))


def labeled_rank_mae(scores: np.ndarray, labels: np.ndarray, labeled_idx: list[int]) -> float:
    if not labeled_idx:
        return float('nan')
    idx = np.asarray(labeled_idx, dtype=np.int64)
    ranks = rank_percentile_01(np.asarray(scores, dtype=np.float32))
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    return float(np.mean(np.abs(ranks[idx] - y[idx])))


def _distinct_count(values: np.ndarray) -> int:
    return int(np.unique(np.round(np.asarray(values, dtype=np.float64), decimals=8)).size)


def _solve_prior_ridge(X: np.ndarray, y: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
    xx = np.asarray(X, dtype=np.float32)
    yy = np.asarray(y, dtype=np.float32).reshape(-1)
    p = np.asarray(prior, dtype=np.float32).reshape(-1)
    if xx.shape[0] == 0:
        return p.copy()
    rhs = yy - (xx @ p)
    gram = (xx @ xx.T) + (float(lam) * np.eye(xx.shape[0], dtype=np.float32))
    coef = np.linalg.solve(gram, rhs)
    return np.asarray(p + (xx.T @ coef), dtype=np.float32)


def _solve_zero_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xx = np.asarray(X, dtype=np.float32)
    yy = np.asarray(y, dtype=np.float32).reshape(-1)
    if xx.shape[0] == 0:
        return np.zeros((xx.shape[1] if xx.ndim == 2 else 0,), dtype=np.float32)
    gram = (xx @ xx.T) + (float(lam) * np.eye(xx.shape[0], dtype=np.float32))
    coef = np.linalg.solve(gram, yy)
    return np.asarray(xx.T @ coef, dtype=np.float32)


def _choose_conservative_lambda(candidate_scores: dict[float, float]) -> float:
    finite = {k: v for k, v in candidate_scores.items() if np.isfinite(v)}
    if not finite:
        return float(max(LAMBDA_GRID))
    best = max(finite.values())
    threshold = best - 0.01
    eligible = [lam for lam, score in finite.items() if score >= threshold]
    return float(max(eligible))


def _fit_affine_calibration(s0: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ss = np.asarray(s0, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if ss.size == 0:
        return 0.0, 0.5
    if ss.size < 2 or float(np.std(ss)) <= 1e-8:
        return 0.0, float(np.mean(yy))
    A = np.column_stack([ss, np.ones_like(ss)])
    reg = np.diag([1e-3, 1e-8])
    sol = np.linalg.solve((A.T @ A) + reg, A.T @ yy)
    return float(sol[0]), float(sol[1])


class _BaseFixedReaxisSession:
    def __init__(self, task: OrdinalTask, direction: np.ndarray, target_mode: str = TARGET_MODE_GROUND_TRUTH):
        self.task = task
        self.target_mode = validate_target_mode(target_mode)
        self.d0 = np.asarray(direction, dtype=np.float32).reshape(-1)
        self.X = np.asarray(task.X, dtype=np.float32)
        self.x_mean = np.mean(self.X, axis=0, keepdims=True)
        self.Xc = np.asarray(self.X - self.x_mean, dtype=np.float32)
        self.s0 = np.asarray(self.X @ self.d0, dtype=np.float32)
        self.labeled_idx: list[int] = []
        self.targets: list[float] = []
        self.selected_lambda = float('nan')
        self.selected_beta = float('nan')
        self.mu = self.d0.copy()
        self._scores = self.s0.copy()
        self.last_feedback_before_transform = float('nan')
        self.last_bayes_y_used = float('nan')

    def transform_target(self, target01: float) -> float:
        if self.target_mode == TARGET_MODE_PRIOR_QUANTILE:
            return prior_quantile(self.s0, target01)
        return float(target01)

    def scores(self) -> np.ndarray:
        return np.asarray(self._scores, dtype=np.float32)

    def uncertainty(self) -> np.ndarray:
        return np.full((len(self.task.ids),), np.nan, dtype=np.float32)

    def debug_prior_scores(self) -> np.ndarray:
        return np.asarray(self.s0, dtype=np.float32)

    def debug_bayes_y_used(self) -> float:
        return float(self.last_bayes_y_used)

    def observe(self, image_idx: int, target01: float) -> None:
        bayes_y = self.transform_target(float(target01))
        self.labeled_idx.append(int(image_idx))
        self.targets.append(float(bayes_y))
        self.last_feedback_before_transform = float(target01)
        self.last_bayes_y_used = float(bayes_y)
        self._fit()

    def fixed_diagnostics(self, labels: np.ndarray) -> dict[str, float]:
        idx = np.asarray(self.labeled_idx, dtype=np.int64)
        labeled_scores = self._scores[idx] if idx.size else np.zeros((0,), dtype=np.float32)
        labeled_y = np.asarray(labels, dtype=np.float32)[idx] if idx.size else np.zeros((0,), dtype=np.float32)
        return {
            'selected_lambda': float(self.selected_lambda),
            'selected_beta': float(self.selected_beta),
            'score_preservation_spearman': spearman_corr(self.s0, self._scores),
            'mu_d0_cos': safe_cosine(self.mu, self.d0),
            'direction_relative_change': float(np.linalg.norm(self.mu - self.d0) / max(float(np.linalg.norm(self.d0)), 1e-12)),
            'labeled_pairwise_agreement': pairwise_agreement(labeled_scores, labeled_y),
            'labeled_mae': labeled_rank_mae(self._scores, labels, self.labeled_idx),
            'prior_spearman_b0': spearman_corr(self.s0, labels),
        }

    def _fit(self) -> None:
        raise NotImplementedError


class CenteredReaxisSession(_BaseFixedReaxisSession):
    def _loo_scores(self, lam: float, idx: np.ndarray, y: np.ndarray) -> np.ndarray:
        pred = np.zeros((idx.size,), dtype=np.float32)
        for pos in range(idx.size):
            keep = np.ones((idx.size,), dtype=bool)
            keep[pos] = False
            train_idx = idx[keep]
            y_train = y[keep]
            y_mean = float(np.mean(y_train))
            mu = _solve_prior_ridge(self.Xc[train_idx], y_train - y_mean, self.d0, lam)
            pred[pos] = float(y_mean + (self.Xc[idx[pos]] @ mu))
        return pred

    def _select_lambda(self, idx: np.ndarray, y: np.ndarray) -> float:
        if idx.size < 3 or _distinct_count(y) < 2:
            return float(max(LAMBDA_GRID))
        scores = {
            float(lam): pairwise_agreement(self._loo_scores(float(lam), idx, y), y)
            for lam in LAMBDA_GRID
        }
        return _choose_conservative_lambda(scores)

    def _fit(self) -> None:
        idx = np.asarray(self.labeled_idx, dtype=np.int64)
        y = np.asarray(self.targets, dtype=np.float32)
        if idx.size < 2 or _distinct_count(y) < 2:
            self.selected_lambda = float(max(LAMBDA_GRID))
            self.mu = self.d0.copy()
            self._scores = self.s0.copy()
            return
        self.selected_lambda = self._select_lambda(idx, y)
        y_mean = float(np.mean(y))
        self.mu = _solve_prior_ridge(self.Xc[idx], y - y_mean, self.d0, self.selected_lambda)
        self._scores = np.asarray(self.Xc @ self.mu, dtype=np.float32)


class CalibratedResidualReaxisSession(_BaseFixedReaxisSession):
    def __init__(self, task: OrdinalTask, direction: np.ndarray, target_mode: str = TARGET_MODE_GROUND_TRUTH):
        super().__init__(task, direction, target_mode=target_mode)
        self.delta = np.zeros_like(self.d0, dtype=np.float32)
        self.calibration_a = 1.0
        self.calibration_c = 0.0
        self.selected_beta = 0.0

    def _fit_delta(self, idx: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, float, np.ndarray]:
        a, c = _fit_affine_calibration(self.s0[idx], y)
        residual = y - ((a * self.s0[idx]) + c)
        delta = _solve_zero_ridge(self.Xc[idx], residual, lam)
        return a, c, delta

    def _loo_scores(self, lam: float, beta: float, idx: np.ndarray, y: np.ndarray) -> np.ndarray:
        pred = np.zeros((idx.size,), dtype=np.float32)
        s0_norm = zscore(self.s0)
        for pos in range(idx.size):
            keep = np.ones((idx.size,), dtype=bool)
            keep[pos] = False
            train_idx = idx[keep]
            y_train = y[keep]
            a, _, delta = self._fit_delta(train_idx, y_train, lam)
            raw = np.asarray((a * self.s0) + (self.Xc @ delta), dtype=np.float32)
            blended = ((1.0 - float(beta)) * s0_norm) + (float(beta) * zscore(raw))
            pred[pos] = float(blended[idx[pos]])
        return pred

    def _select_lambda_beta(self, idx: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        if idx.size < 3 or _distinct_count(y) < 2:
            return float(max(LAMBDA_GRID)), 0.0
        lambda_scores: dict[float, float] = {}
        lambda_betas: dict[float, float] = {}
        for lam in LAMBDA_GRID:
            beta_scores = {
                float(beta): pairwise_agreement(self._loo_scores(float(lam), float(beta), idx, y), y)
                for beta in BETA_GRID
            }
            finite = {k: v for k, v in beta_scores.items() if np.isfinite(v)}
            if not finite:
                lambda_scores[float(lam)] = float('nan')
                lambda_betas[float(lam)] = 0.0
                continue
            best = max(finite.values())
            eligible = [beta for beta, score in finite.items() if score >= (best - 0.01)]
            selected_beta = float(min(eligible))
            lambda_scores[float(lam)] = float(beta_scores[selected_beta])
            lambda_betas[float(lam)] = selected_beta
        selected_lambda = _choose_conservative_lambda(lambda_scores)
        return selected_lambda, float(lambda_betas.get(selected_lambda, 0.0))

    def _fit(self) -> None:
        idx = np.asarray(self.labeled_idx, dtype=np.int64)
        y = np.asarray(self.targets, dtype=np.float32)
        if idx.size < 2 or _distinct_count(y) < 2:
            self.selected_lambda = float(max(LAMBDA_GRID))
            self.selected_beta = 0.0
            self.delta = np.zeros_like(self.d0, dtype=np.float32)
            self.mu = self.d0.copy()
            self._scores = zscore(self.s0)
            return
        self.selected_lambda, self.selected_beta = self._select_lambda_beta(idx, y)
        self.calibration_a, self.calibration_c, self.delta = self._fit_delta(idx, y, self.selected_lambda)
        raw = np.asarray((self.calibration_a * self.s0) + (self.Xc @ self.delta), dtype=np.float32)
        self.mu = np.asarray(self.d0 + self.delta, dtype=np.float32)
        self._scores = np.asarray(
            ((1.0 - self.selected_beta) * zscore(self.s0)) + (self.selected_beta * zscore(raw)),
            dtype=np.float32,
        )
