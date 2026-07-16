from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC

try:
    from ..axis_bayes import AxisBayesEngine
except ImportError:
    from axis_bayes import AxisBayesEngine

from .config import REAXIS_AXISBAYES_PARAMS
from .data import OrdinalTask
from .fixed_reaxis import TARGET_MODE_GROUND_TRUTH, validate_target_mode


AXISBAYES_MODES = ('gaussian', 'rank')
REAXIS_METHOD_MODES = {
    'reaxis_random_gaussian': ('reaxis_random', 'gaussian'),
    'reaxis_active_gaussian': ('reaxis_active', 'gaussian'),
    'reaxis_random_rank': ('reaxis_random', 'rank'),
    'reaxis_active_rank': ('reaxis_active', 'rank'),
    'reaxis_quantile_gaussian': ('reaxis_random', 'gaussian'),
    'reaxis_log': ('reaxis_log', 'gaussian'),
}


METHOD_SUPPORTS_FEEDBACK = {
    'text_prior': False,
    'prompt_ladder': False,
    'ordinal_ridge': True,
    'rank_svm': True,
    'knn_ordinal': True,
    'kernel_ridge': True,
    'label_mean': True,
    'ols_linear': True,
    'bayesian_ridge': True,
    'elastic_net': True,
    'pcr_ridge': True,
    'reaxis_random': True,
    'reaxis_active': True,
    'reaxis_random_gaussian': True,
    'reaxis_active_gaussian': True,
    'reaxis_random_rank': True,
    'reaxis_active_rank': True,
    'reaxis_quantile_gaussian': True,
    'reaxis_log': True,
    'reaxis_pairwise': True,
    'prior_affine': True,
    'linear_svr': True,
    'reaxis_centered': True,
    'reaxis_calibrated_residual': True,
}


def validate_axisbayes_mode(axisbayes_mode: str) -> str:
    mode = str(axisbayes_mode or 'gaussian').strip().lower()
    if mode not in AXISBAYES_MODES:
        raise ValueError(f'Unknown axisbayes_mode {axisbayes_mode!r}; expected one of {AXISBAYES_MODES}')
    return mode


def is_axisbayes_reaxis_method(method: str) -> bool:
    name = str(method or '').strip()
    return name in {'reaxis_random', 'reaxis_active'} or name in REAXIS_METHOD_MODES


def base_reaxis_method(method: str) -> str:
    name = str(method or '').strip()
    return REAXIS_METHOD_MODES.get(name, (name, 'gaussian'))[0]


def axisbayes_mode_for_method(method: str, default_mode: str = 'gaussian') -> str:
    name = str(method or '').strip()
    if name in REAXIS_METHOD_MODES:
        return REAXIS_METHOD_MODES[name][1]
    return validate_axisbayes_mode(default_mode)


def mode_label_for_method(method: str, default_mode: str = 'gaussian') -> str:
    if is_axisbayes_reaxis_method(method):
        return axisbayes_mode_for_method(method, default_mode)
    if str(method or '').strip() == 'reaxis_pairwise':
        return 'pairwise_prior'
    if str(method or '').strip() in {'reaxis_centered', 'reaxis_calibrated_residual'}:
        return 'fixed'
    return 'baseline'


@dataclass
class BaseSession:
    task: OrdinalTask

    def scores(self) -> np.ndarray:
        raise NotImplementedError

    def uncertainty(self) -> np.ndarray:
        return np.full((len(self.task.ids),), np.nan, dtype=np.float32)

    def observe(self, image_idx: int, target01: float) -> None:
        _ = image_idx
        _ = target01

    def debug_bayes_y_used(self) -> float:
        return float('nan')

    def debug_prior_scores(self) -> np.ndarray:
        return np.zeros((len(self.task.ids),), dtype=np.float32)


class ClipTextEmbedder:
    def __init__(self) -> None:
        self.engine = AxisBayesEngine(
            model_type='bayes_linear',
            mode='rank',
            feature_space='clip',
            semantic_method='clip',
            norm=True,
            axis_bounds_text_source='template',
            use_llm_prompt_ensemble=False,
        )
        self._text_cache: dict[str, np.ndarray] = {}

    def direction(self, high_text: str, low_text: str) -> np.ndarray:
        w0, _, _, _ = self.engine._embed_prompt_lists(
            [high_text],
            [low_text],
            semantic_method='clip',
            norm=True,
            source='ordinal_study',
            provider='manual',
        )
        return np.asarray(w0, dtype=np.float32)

    def text_embedding(self, text: str) -> np.ndarray:
        key = str(text)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        extractor = self.engine._get_text_extractor('clip')
        vec = np.asarray(extractor.extract_text_embedding(key, normalize=True), dtype=np.float32)
        self._text_cache[key] = vec
        return vec


class TextPriorSession(BaseSession):
    def __init__(self, task: OrdinalTask, embedder: ClipTextEmbedder):
        super().__init__(task)
        direction = embedder.direction(task.high_text, task.low_text)
        self._scores = np.asarray(task.X @ direction, dtype=np.float32)

    def scores(self) -> np.ndarray:
        return self._scores


class PromptLadderSession(BaseSession):
    def __init__(self, task: OrdinalTask, embedder: ClipTextEmbedder, temperature: float = 30.0):
        super().__init__(task)
        text = np.vstack([embedder.text_embedding(prompt) for prompt in task.ladder]).astype(np.float32)
        sims = np.asarray(task.X @ text.T, dtype=np.float32)
        logits = float(temperature) * (sims - np.max(sims, axis=1, keepdims=True))
        weights = np.exp(logits)
        weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-8)
        levels = np.linspace(0.0, 1.0, len(task.ladder), dtype=np.float32)
        self._scores = np.asarray(weights @ levels, dtype=np.float32)

    def scores(self) -> np.ndarray:
        return self._scores


class LabelOnlySession(BaseSession):
    def __init__(self, task: OrdinalTask):
        super().__init__(task)
        self.labeled_idx: list[int] = []
        self.targets: list[float] = []
        self._scores = np.full((len(task.ids),), 0.5, dtype=np.float32)

    def scores(self) -> np.ndarray:
        return self._scores

    def observe(self, image_idx: int, target01: float) -> None:
        self.labeled_idx.append(int(image_idx))
        self.targets.append(float(target01))
        self._fit()

    def _fit(self) -> None:
        raise NotImplementedError


class OrdinalRidgeSession(LabelOnlySession):
    def _fit(self) -> None:
        if len(self.labeled_idx) == 0:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        model = Ridge(alpha=1.0)
        model.fit(self.task.X[self.labeled_idx], np.asarray(self.targets, dtype=np.float32))
        self._scores = np.asarray(model.predict(self.task.X), dtype=np.float32)


class RankSVMSession(LabelOnlySession):
    def _fit(self) -> None:
        if len(self.labeled_idx) < 2:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        rows = []
        labels = []
        idx = list(self.labeled_idx)
        y = np.asarray(self.targets, dtype=np.float32)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                diff_y = float(y[a] - y[b])
                if abs(diff_y) <= 1e-8:
                    continue
                diff_x = self.task.X[idx[a]] - self.task.X[idx[b]]
                label = 1 if diff_y > 0 else -1
                rows.extend([diff_x, -diff_x])
                labels.extend([label, -label])
        if not rows:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        model = LinearSVC(C=1.0, dual='auto', max_iter=5000)
        model.fit(np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int32))
        self._scores = np.asarray(self.task.X @ model.coef_.reshape(-1), dtype=np.float32)


class KNNOrdinalSession(LabelOnlySession):
    def _fit(self) -> None:
        if len(self.labeled_idx) == 0:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        x_lab = self.task.X[self.labeled_idx]
        y_lab = np.asarray(self.targets, dtype=np.float32)
        sims = np.asarray(self.task.X @ x_lab.T, dtype=np.float32)
        k = min(15, x_lab.shape[0])
        top = np.argsort(sims, axis=1, kind='mergesort')[:, -k:]
        top_sims = np.take_along_axis(sims, top, axis=1)
        logits = 12.0 * (top_sims - np.max(top_sims, axis=1, keepdims=True))
        weights = np.exp(logits)
        weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-8)
        self._scores = np.asarray(np.sum(weights * y_lab[top], axis=1), dtype=np.float32)


class KernelRidgeSession(LabelOnlySession):
    def _fit(self) -> None:
        if len(self.labeled_idx) == 0:
            self._scores = np.full((len(self.task.ids),), 0.5, dtype=np.float32)
            return
        gamma = 1.0 / float(max(1, self.task.X.shape[1]))
        model = KernelRidge(alpha=1.0, kernel='rbf', gamma=gamma)
        model.fit(self.task.X[self.labeled_idx], np.asarray(self.targets, dtype=np.float32))
        self._scores = np.asarray(model.predict(self.task.X), dtype=np.float32)


class ReaxisSession(BaseSession):
    def __init__(
        self,
        task: OrdinalTask,
        target_mode: str = TARGET_MODE_GROUND_TRUTH,
        axisbayes_mode: str = 'gaussian',
    ):
        super().__init__(task)
        self.target_mode = validate_target_mode(target_mode)
        self.axisbayes_mode = validate_axisbayes_mode(axisbayes_mode)
        self.last_bayes_y_used = float('nan')
        self.last_feedback_before_transform = float('nan')
        self.last_target_p01 = float('nan')
        self.last_gaussian_scalar_target = float('nan')
        params = {**REAXIS_AXISBAYES_PARAMS, 'mode': self.axisbayes_mode}
        self.engine = AxisBayesEngine(**params)
        self.payload = self.engine.create_axis(
            collection_id=task.dataset,
            dataset_root=str(task.dataset_root),
            q=task.query,
            mode=self.axisbayes_mode,
            model_type='bayes_linear',
        )
        self.axis_id = str(self.payload['axis_id'])
        self.payload = self.engine.update_axis_prompts(
            axis_id=self.axis_id,
            pos_prompts=[task.high_text],
            neg_prompts=[task.low_text],
        )

    def scores(self) -> np.ndarray:
        return np.asarray(self.payload.get('projection_values') or [], dtype=np.float32)

    def uncertainty(self) -> np.ndarray:
        return np.asarray(self.payload.get('std') or [], dtype=np.float32)

    @property
    def state(self):
        return self.engine._axes[self.axis_id]

    def debug_prior_scores(self) -> np.ndarray:
        w0 = np.asarray(self.state.w0, dtype=np.float32).reshape(-1)
        if w0.shape[0] == self.task.X.shape[1]:
            return np.asarray(self.task.X @ w0, dtype=np.float32)
        return np.asarray(self.state.z0_all, dtype=np.float32)

    def debug_bayes_y_used(self) -> float:
        return float(self.last_bayes_y_used)

    def transform_target(self, target01: float) -> float:
        return float(target01)

    def observe(self, image_idx: int, target01: float) -> None:
        image_id = self.task.ids[int(image_idx)]
        target_p01 = self.transform_target(float(target01))
        gaussian_target = float('nan')
        if self.axisbayes_mode == 'gaussian':
            gaussian_target = float(self.engine._quantile_from_sorted(self.state.z0_sorted, target_p01))
        self.last_feedback_before_transform = float(target01)
        self.last_target_p01 = float(target_p01)
        self.last_gaussian_scalar_target = gaussian_target
        self.last_bayes_y_used = gaussian_target if np.isfinite(gaussian_target) else float(target_p01)
        self.payload = self.engine.move_axis(
            axis_id=self.axis_id,
            image_id=image_id,
            new_score_0_100=float(target_p01) * 100.0,
            move_type='score',
        )


class LogReaxisSession(ReaxisSession):
    def transform_target(self, target01: float) -> float:
        value01 = float(np.clip(target01, 0.0, 1.0))
        lo = float(self.task.raw_min_value)
        hi = float(self.task.raw_max_value)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return value01
        raw = lo + value01 * (hi - lo)
        if lo > 0.0:
            denom = np.log(hi) - np.log(lo)
            if denom <= 0.0 or not np.isfinite(denom):
                return value01
            out = (np.log(max(raw, lo)) - np.log(lo)) / denom
        else:
            offset = 1.0 - lo
            low = lo + offset
            high = hi + offset
            shifted = raw + offset
            denom = np.log(high) - np.log(low)
            if denom <= 0.0 or not np.isfinite(denom):
                return value01
            out = (np.log(max(shifted, low)) - np.log(low)) / denom
        return float(np.clip(out, 0.0, 1.0))


def make_session(
    method: str,
    task: OrdinalTask,
    embedder: Optional[ClipTextEmbedder] = None,
    target_mode: str = TARGET_MODE_GROUND_TRUTH,
    axisbayes_mode: str = 'gaussian',
) -> BaseSession:
    if method == 'text_prior':
        if embedder is None:
            raise ValueError('text_prior requires a ClipTextEmbedder')
        return TextPriorSession(task, embedder)
    if method == 'prompt_ladder':
        if embedder is None:
            raise ValueError('prompt_ladder requires a ClipTextEmbedder')
        return PromptLadderSession(task, embedder)
    if method == 'ordinal_ridge':
        return OrdinalRidgeSession(task)
    if method == 'rank_svm':
        return RankSVMSession(task)
    if method == 'knn_ordinal':
        return KNNOrdinalSession(task)
    if method == 'kernel_ridge':
        return KernelRidgeSession(task)
    if method in {'label_mean', 'ols_linear', 'bayesian_ridge', 'elastic_net', 'pcr_ridge'}:
        from .extra_baselines import (
            BayesianRidgeSession,
            ElasticNetSession,
            LabelMeanSession,
            OrdinaryLeastSquaresSession,
            PCRRidgeSession,
        )
        if method == 'label_mean':
            return LabelMeanSession(task)
        if method == 'ols_linear':
            return OrdinaryLeastSquaresSession(task)
        if method == 'bayesian_ridge':
            return BayesianRidgeSession(task)
        if method == 'elastic_net':
            return ElasticNetSession(task)
        return PCRRidgeSession(task)
    if method == 'linear_svr':
        from .extra_baselines import LinearSVRSession
        return LinearSVRSession(task)
    if method in {'prior_affine', 'reaxis_pairwise'}:
        if embedder is None:
            raise ValueError(f'{method} requires a ClipTextEmbedder')
        from .extra_baselines import PriorAffineSession, PriorPairwiseReaxisSession
        direction = embedder.direction(task.high_text, task.low_text)
        if method == 'prior_affine':
            return PriorAffineSession(task, direction)
        return PriorPairwiseReaxisSession(task, direction)
    if is_axisbayes_reaxis_method(method):
        if method == 'reaxis_log':
            return LogReaxisSession(
                task,
                target_mode=target_mode,
                axisbayes_mode=axisbayes_mode_for_method(method, axisbayes_mode),
            )
        return ReaxisSession(
            task,
            target_mode=target_mode,
            axisbayes_mode=axisbayes_mode_for_method(method, axisbayes_mode),
        )
    if method in {'reaxis_centered', 'reaxis_calibrated_residual'}:
        if embedder is None:
            raise ValueError(f'{method} requires a ClipTextEmbedder')
        from .fixed_reaxis import CalibratedResidualReaxisSession, CenteredReaxisSession
        direction = embedder.direction(task.high_text, task.low_text)
        if method == 'reaxis_centered':
            return CenteredReaxisSession(task, direction, target_mode=target_mode)
        return CalibratedResidualReaxisSession(task, direction, target_mode=target_mode)
    raise ValueError(f'Unknown ordinal method: {method}')
