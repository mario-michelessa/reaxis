from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression, Ridge
from sklearn.svm import LinearSVR

from .baselines import BaseSession, LabelOnlySession
from .data import OrdinalTask


class LabelMeanSession(LabelOnlySession):
    def _fit(self) -> None:
        if len(self.targets) == 0:
            value = 0.5
        else:
            value = float(np.mean(np.asarray(self.targets, dtype=np.float32)))
        self._scores = np.full((len(self.task.ids),), value, dtype=np.float32)


class OrdinaryLeastSquaresSession(LabelOnlySession):
    def _fit(self) -> None:
        if len(self.labeled_idx) < 2 or len(set(round(v, 8) for v in self.targets)) < 2:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        model = LinearRegression()
        model.fit(self.task.X[self.labeled_idx], np.asarray(self.targets, dtype=np.float32))
        self._scores = np.asarray(model.predict(self.task.X), dtype=np.float32)


class BayesianRidgeSession(LabelOnlySession):
    def _fit(self) -> None:
        if len(self.labeled_idx) < 2 or len(set(round(v, 8) for v in self.targets)) < 2:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        model = BayesianRidge()
        model.fit(self.task.X[self.labeled_idx], np.asarray(self.targets, dtype=np.float32))
        self._scores = np.asarray(model.predict(self.task.X), dtype=np.float32)


class ElasticNetSession(LabelOnlySession):
    def _fit(self) -> None:
        if len(self.labeled_idx) < 2 or len(set(round(v, 8) for v in self.targets)) < 2:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        model = ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000, random_state=0)
        model.fit(self.task.X[self.labeled_idx], np.asarray(self.targets, dtype=np.float32))
        self._scores = np.asarray(model.predict(self.task.X), dtype=np.float32)


class PCRRidgeSession(LabelOnlySession):
    def __init__(self, task: OrdinalTask):
        super().__init__(task)
        self._projected: np.ndarray | None = None

    def _features(self) -> np.ndarray:
        if self._projected is not None:
            return self._projected
        n_components = min(16, self.task.X.shape[0], self.task.X.shape[1])
        pca = PCA(n_components=n_components, random_state=0)
        self._projected = np.asarray(pca.fit_transform(self.task.X), dtype=np.float32)
        return self._projected

    def _fit(self) -> None:
        if len(self.labeled_idx) < 2 or len(set(round(v, 8) for v in self.targets)) < 2:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        features = self._features()
        model = Ridge(alpha=1.0)
        model.fit(features[self.labeled_idx], np.asarray(self.targets, dtype=np.float32))
        self._scores = np.asarray(model.predict(features), dtype=np.float32)


class LinearSVRSession(LabelOnlySession):
    def _fit(self) -> None:
        if len(self.labeled_idx) < 2 or len(set(round(v, 8) for v in self.targets)) < 2:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        model = LinearSVR(C=1.0, epsilon=0.05, dual='auto', max_iter=5000, random_state=0)
        model.fit(self.task.X[self.labeled_idx], np.asarray(self.targets, dtype=np.float32))
        self._scores = np.asarray(model.predict(self.task.X), dtype=np.float32)


class PriorAffineSession(BaseSession):
    def __init__(self, task: OrdinalTask, direction: np.ndarray):
        super().__init__(task)
        self.s0 = np.asarray(task.X @ np.asarray(direction, dtype=np.float32), dtype=np.float32)
        self.labeled_idx: list[int] = []
        self.targets: list[float] = []
        self._scores = self.s0.copy()

    def scores(self) -> np.ndarray:
        return self._scores

    def debug_prior_scores(self) -> np.ndarray:
        return self.s0

    def observe(self, image_idx: int, target01: float) -> None:
        self.labeled_idx.append(int(image_idx))
        self.targets.append(float(target01))
        if len(self.labeled_idx) < 2:
            self._scores = self.s0.copy()
            return
        x = self.s0[self.labeled_idx].reshape(-1, 1)
        y = np.asarray(self.targets, dtype=np.float32)
        model = Ridge(alpha=0.1, fit_intercept=True)
        model.fit(x, y)
        self._scores = np.asarray(model.predict(self.s0.reshape(-1, 1)), dtype=np.float32)


class PriorPairwiseReaxisSession(BaseSession):
    def __init__(self, task: OrdinalTask, direction: np.ndarray, lambda_prior: float = 0.03):
        super().__init__(task)
        self.d0 = np.asarray(direction, dtype=np.float64).reshape(-1)
        self.lambda_prior = float(lambda_prior)
        self.labeled_idx: list[int] = []
        self.targets: list[float] = []
        self.mu = self.d0.copy()
        self._scores = np.asarray(task.X @ self.mu, dtype=np.float32)

    def scores(self) -> np.ndarray:
        return self._scores

    def debug_prior_scores(self) -> np.ndarray:
        return np.asarray(self.task.X @ self.d0, dtype=np.float32)

    def observe(self, image_idx: int, target01: float) -> None:
        self.labeled_idx.append(int(image_idx))
        self.targets.append(float(target01))
        self._fit()

    def _pair_rows(self) -> np.ndarray:
        idx = list(self.labeled_idx)
        y = np.asarray(self.targets, dtype=np.float64)
        rows: list[np.ndarray] = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                diff_y = float(y[a] - y[b])
                if abs(diff_y) <= 1e-8:
                    continue
                sign = 1.0 if diff_y > 0.0 else -1.0
                rows.append(sign * (self.task.X[idx[a]] - self.task.X[idx[b]]))
        if not rows:
            return np.zeros((0, self.task.X.shape[1]), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def _fit(self) -> None:
        rows = self._pair_rows()
        if rows.shape[0] == 0:
            self.mu = self.d0.copy()
            self._scores = np.asarray(self.task.X @ self.mu, dtype=np.float32)
            return
        lam = float(self.lambda_prior)
        d0 = self.d0

        def objective(d: np.ndarray) -> tuple[float, np.ndarray]:
            margins = rows @ d
            loss = float(np.mean(np.logaddexp(0.0, -margins)))
            prior = 0.5 * lam * float(np.sum((d - d0) ** 2))
            weights = -expit(-margins) / float(rows.shape[0])
            grad = np.asarray(rows.T @ weights, dtype=np.float64) + lam * (d - d0)
            return loss + prior, grad

        result = minimize(
            fun=lambda d: objective(d)[0],
            x0=self.mu.astype(np.float64, copy=True),
            jac=lambda d: objective(d)[1],
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8},
        )
        if not result.success:
            raise RuntimeError(f'reaxis_pairwise optimization failed: {result.message}')
        self.mu = np.asarray(result.x, dtype=np.float64)
        self._scores = np.asarray(self.task.X @ self.mu, dtype=np.float32)
