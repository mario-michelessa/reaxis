from __future__ import annotations

import csv
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from .baselines import ClipTextEmbedder
from .config import BUDGETS, EXPANDED_DATASETS, GLOBAL_SEED, OUTPUT_DIR, REPETITIONS
from .data import OrdinalTask, load_tasks
from .fixed_reaxis import labeled_rank_mae, pairwise_agreement, safe_cosine
from .gaussian_sweep_reports import write_gaussian_sweep_summaries
from .io import CSVAppender, utc_now_iso, write_json
from .metrics import compute_metrics, spearman_corr


BINARY_GAUSSIAN_ALPHA = 0.17392955514866384
BINARY_GAUSSIAN_SIGMA2 = 0.0023543092401027846
BINARY_GAUSSIAN_BIAS_ALPHA = 0.02885457255148477

DEFAULT_ALPHAS = (0.03, 0.1, BINARY_GAUSSIAN_ALPHA, 0.34361647953904884, 1.0, 3.0, 10.0, 30.0)
DEFAULT_SIGMA2S = (0.0005, 0.001, BINARY_GAUSSIAN_SIGMA2, 0.005, 0.01, 0.02, 0.04, 0.08)
DEFAULT_BIAS_ALPHAS = (BINARY_GAUSSIAN_BIAS_ALPHA, 0.1, 1.0, 16.0)
DEFAULT_SCORE_MODES = ('cosine', 'linear')
DEFAULT_POLICIES = ('random',)

METRIC_COLS = (
    'spearman', 'kendall_tau', 'mae_isotonic', 'qwk',
    'axis_r2', 'monotonicity_violations',
)

RESULT_FIELDS = (
    'ts_utc', 'run_id', 'dataset', 'axis_field', 'axis_name', 'axis_category',
    'method', 'policy', 'replicate', 'budget', 'label_count',
    'alpha', 'sigma2', 'bias_alpha', 'score_mode',
    'prior_spearman_b0', 'score_preservation_spearman',
    'mu_d0_cos', 'direction_relative_change', 'direction_norm', 'posterior_bias',
    'labeled_pairwise_agreement', 'labeled_mae',
    *METRIC_COLS,
)


@dataclass(frozen=True)
class GaussianConfig:
    alpha: float
    sigma2: float
    bias_alpha: float
    score_mode: str

    @property
    def method(self) -> str:
        return 'reaxis_gaussian_sweep'

    @property
    def key(self) -> tuple[float, float, float, str]:
        return (float(self.alpha), float(self.sigma2), float(self.bias_alpha), str(self.score_mode))


@dataclass
class PosteriorState:
    mu: np.ndarray
    bias: float
    scores: np.ndarray
    predictive_var: np.ndarray
    scalar_targets: np.ndarray


def stable_seed(*parts: object) -> int:
    text = '||'.join(str(part) for part in parts)
    return int(zlib.crc32(text.encode('utf-8')) & 0xFFFFFFFF)


def _validate_score_mode(score_mode: str) -> str:
    mode = str(score_mode or '').strip().lower()
    if mode not in {'cosine', 'linear'}:
        raise ValueError(f'Unknown score_mode {score_mode!r}; expected cosine or linear')
    return mode


def _cosine_scores(X: np.ndarray, direction: np.ndarray, bias: float = 0.0) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float32).reshape(-1)
    denom = max(float(np.linalg.norm(vec)), 1e-8)
    return np.asarray((np.asarray(X, dtype=np.float32) @ vec) / denom + float(bias), dtype=np.float32)


def _scores_for_mode(X: np.ndarray, mu: np.ndarray, bias: float, score_mode: str) -> np.ndarray:
    if score_mode == 'cosine':
        return _cosine_scores(X, mu, bias=bias)
    return np.asarray((np.asarray(X, dtype=np.float32) @ np.asarray(mu, dtype=np.float32)) + float(bias), dtype=np.float32)


def _quantile_from_sorted(z_sorted: np.ndarray, p01: float) -> float:
    arr = np.asarray(z_sorted, dtype=np.float32).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return 0.0
    if n == 1:
        return float(arr[0])
    p = float(np.clip(float(p01), 0.0, 1.0))
    pos = p * float(n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(arr[lo])
    mix = float(pos - lo)
    return float((arr[lo] * (1.0 - mix)) + (arr[hi] * mix))


def _gaussian_posterior(
    *,
    task: OrdinalTask,
    d0: np.ndarray,
    selected_idx: Sequence[int],
    config: GaussianConfig,
    prior_scores: np.ndarray,
) -> PosteriorState:
    X_all = np.asarray(task.X, dtype=np.float32)
    d = int(X_all.shape[1])
    idx = np.asarray(list(selected_idx), dtype=np.int64)
    d0_vec = np.asarray(d0, dtype=np.float32).reshape(-1)
    prior_sorted = np.sort(np.asarray(prior_scores, dtype=np.float32).reshape(-1))
    if idx.size == 0:
        scores = _scores_for_mode(X_all, d0_vec, 0.0, config.score_mode)
        prior_var = np.full((len(task.ids),), float(config.sigma2 + (1.0 / config.alpha) + (1.0 / config.bias_alpha)), dtype=np.float32)
        return PosteriorState(d0_vec.copy(), 0.0, scores, prior_var, np.zeros((0,), dtype=np.float32))

    X = X_all[idx, :]
    y01 = np.asarray(task.y01, dtype=np.float32)[idx]
    y = np.asarray([_quantile_from_sorted(prior_sorted, float(p)) for p in y01], dtype=np.float32)
    alpha_inv = np.full((d,), 1.0 / max(float(config.alpha), 1e-8), dtype=np.float32)
    obs_var = np.full((idx.size,), float(config.sigma2), dtype=np.float32)
    XS = X * alpha_inv[None, :]
    ones = np.ones((idx.size,), dtype=np.float32)
    A = np.diag(obs_var) + (XS @ X.T) + ((1.0 / float(config.bias_alpha)) * np.outer(ones, ones))
    A_inv = np.linalg.inv(A + (1e-6 * np.eye(idx.size, dtype=np.float32)))
    residual = y - (X @ d0_vec)
    mu = d0_vec + (alpha_inv * (X.T @ (A_inv @ residual)))
    bias = float((1.0 / float(config.bias_alpha)) * (ones @ (A_inv @ residual)))
    if float(np.dot(mu, d0_vec)) < 0.0:
        mu = -mu
        bias = -bias
    if float(np.linalg.norm(mu)) <= 1e-8:
        mu = d0_vec.copy()
        bias = 0.0
    scores = _scores_for_mode(X_all, mu, bias, config.score_mode)
    predictive_var = _predictive_variance(X_all, X, alpha_inv, A_inv, float(config.sigma2), float(config.bias_alpha))
    return PosteriorState(
        mu=np.asarray(mu, dtype=np.float32),
        bias=float(bias),
        scores=np.asarray(scores, dtype=np.float32),
        predictive_var=np.asarray(predictive_var, dtype=np.float32),
        scalar_targets=y,
    )


def _predictive_variance(
    X_all: np.ndarray,
    X_obs: np.ndarray,
    alpha_inv: np.ndarray,
    A_inv: np.ndarray,
    sigma2: float,
    bias_alpha: float,
) -> np.ndarray:
    prior_x = np.sum((np.asarray(X_all, dtype=np.float32) ** 2) * alpha_inv[None, :], axis=1)
    base = np.asarray(float(sigma2) + prior_x + (1.0 / float(bias_alpha)), dtype=np.float32)
    if X_obs.shape[0] == 0:
        return base
    U = (X_obs * alpha_inv[None, :]) @ X_all.T
    V = U + (1.0 / float(bias_alpha))
    correction = np.sum(V * (A_inv @ V), axis=0)
    return np.asarray(np.maximum(base - correction, 0.0), dtype=np.float32)


def _select_next(
    *,
    policy: str,
    state: PosteriorState,
    queried: np.ndarray,
    rng: np.random.Generator,
) -> int | None:
    available = np.flatnonzero(~queried)
    if available.size == 0:
        return None
    if policy == 'active':
        var = np.asarray(state.predictive_var, dtype=np.float32)
        if var.shape[0] == queried.shape[0] and np.any(np.isfinite(var[available])):
            masked = np.where(np.isfinite(var), var, -np.inf)
            masked[queried] = -np.inf
            spread = float(np.nanmax(masked[available]) - np.nanmin(masked[available]))
            if spread > 1e-9:
                return int(np.argmax(masked))
    return int(rng.choice(available))


def _read_completed(path: Path, run_id: str) -> set[tuple[str, str, str, int, int, float, float, float, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    completed: set[tuple[str, str, str, int, int, float, float, float, str]] = set()
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get('run_id', '')) != str(run_id):
                continue
            completed.add((
                str(row['dataset']),
                str(row['axis_field']),
                str(row['policy']),
                int(row['replicate']),
                int(row['budget']),
                float(row['alpha']),
                float(row['sigma2']),
                float(row['bias_alpha']),
                str(row['score_mode']),
            ))
    return completed


def _config_grid(
    alphas: Sequence[float],
    sigma2s: Sequence[float],
    bias_alphas: Sequence[float],
    score_modes: Sequence[str],
) -> list[GaussianConfig]:
    out = []
    seen = set()
    for alpha in alphas:
        for sigma2 in sigma2s:
            for bias_alpha in bias_alphas:
                for score_mode in score_modes:
                    config = GaussianConfig(
                        alpha=float(alpha),
                        sigma2=float(sigma2),
                        bias_alpha=float(bias_alpha),
                        score_mode=_validate_score_mode(score_mode),
                    )
                    if config.key not in seen:
                        out.append(config)
                        seen.add(config.key)
    return out


def _append_state_metrics(
    *,
    logger: CSVAppender,
    completed: set[tuple[str, str, str, int, int, float, float, float, str]],
    run_id: str,
    task: OrdinalTask,
    policy: str,
    replicate: int,
    budget: int,
    config: GaussianConfig,
    state: PosteriorState,
    d0: np.ndarray,
    prior_scores: np.ndarray,
    prior_spearman: float,
    selected: Sequence[int],
    metric_mode: str,
) -> None:
    key = (
        task.dataset, task.axis_field, policy, int(replicate), int(budget),
        float(config.alpha), float(config.sigma2), float(config.bias_alpha), str(config.score_mode),
    )
    if key in completed:
        return
    if metric_mode == 'spearman':
        metrics = {metric: float('nan') for metric in METRIC_COLS}
        metrics['spearman'] = spearman_corr(state.scores, task.y01)
    else:
        rng = np.random.default_rng(GLOBAL_SEED + stable_seed(task.task_id, policy, replicate, budget, *config.key))
        metrics, _ = compute_metrics(state.scores, task.y01, rng)
    selected_list = [int(v) for v in selected]
    logger.append({
        'ts_utc': utc_now_iso(),
        'run_id': run_id,
        'dataset': task.dataset,
        'axis_field': task.axis_field,
        'axis_name': task.axis_name,
        'axis_category': task.axis_category,
        'method': config.method,
        'policy': policy,
        'replicate': int(replicate),
        'budget': int(budget),
        'label_count': int(len(selected_list)),
        'alpha': float(config.alpha),
        'sigma2': float(config.sigma2),
        'bias_alpha': float(config.bias_alpha),
        'score_mode': str(config.score_mode),
        'prior_spearman_b0': float(prior_spearman),
        'score_preservation_spearman': spearman_corr(prior_scores, state.scores),
        'mu_d0_cos': safe_cosine(state.mu, d0),
        'direction_relative_change': float(np.linalg.norm(state.mu - d0) / max(float(np.linalg.norm(d0)), 1e-12)),
        'direction_norm': float(np.linalg.norm(state.mu)),
        'posterior_bias': float(state.bias),
        'labeled_pairwise_agreement': pairwise_agreement(state.scores[selected_list], task.y01[selected_list]) if selected_list else float('nan'),
        'labeled_mae': labeled_rank_mae(state.scores, task.y01, selected_list),
        **metrics,
    })
    completed.add(key)


def run_gaussian_sweep(
    *,
    run_id: str,
    output_dir: Path = OUTPUT_DIR,
    datasets: Sequence[str] = EXPANDED_DATASETS,
    budgets: Sequence[int] = BUDGETS,
    repetitions: int = REPETITIONS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    sigma2s: Sequence[float] = DEFAULT_SIGMA2S,
    bias_alphas: Sequence[float] = DEFAULT_BIAS_ALPHAS,
    score_modes: Sequence[str] = DEFAULT_SCORE_MODES,
    policies: Sequence[str] = DEFAULT_POLICIES,
    max_tasks: int = 0,
    comparison_dirs: Sequence[Path] = (),
    metric_mode: str = 'full',
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    budgets = sorted({int(v) for v in budgets})
    if not budgets or budgets[0] != 0:
        budgets = sorted({0, *budgets})
    policies = tuple(str(policy).strip().lower() for policy in policies if str(policy).strip())
    invalid = sorted(set(policies) - {'random', 'active'})
    if invalid:
        raise ValueError(f'Unknown selection policies: {invalid}')
    metric_mode = str(metric_mode or 'full').strip().lower()
    if metric_mode not in {'full', 'spearman'}:
        raise ValueError(f'Unknown metric_mode {metric_mode!r}; expected full or spearman')
    configs = _config_grid(alphas, sigma2s, bias_alphas, score_modes)
    if not configs:
        raise RuntimeError('No Gaussian sweep configurations selected')
    tasks = load_tasks(datasets)
    if int(max_tasks) > 0:
        tasks = tasks[: int(max_tasks)]
    if not tasks:
        raise RuntimeError('No ordinal tasks selected for Gaussian sweep')

    write_json(root / 'ordinal_gaussian_sweep_manifest.json', {
        'run_id': run_id,
        'created_utc': utc_now_iso(),
        'datasets': list(datasets),
        'budgets': budgets,
        'repetitions': int(repetitions),
        'alphas': [float(v) for v in alphas],
        'sigma2s': [float(v) for v in sigma2s],
        'bias_alphas': [float(v) for v in bias_alphas],
        'score_modes': list(score_modes),
        'policies': list(policies),
        'max_tasks': int(max_tasks),
        'metric_mode': metric_mode,
        'note': 'Gaussian AxisBayes scalar update, with current percentile-to-prior-quantile target mapping.',
    })

    results_path = root / 'ordinal_gaussian_sweep_results.csv'
    logger = CSVAppender(results_path, RESULT_FIELDS)
    completed = _read_completed(results_path, run_id)
    embedder = ClipTextEmbedder()
    max_budget = int(max(budgets))

    for task in tqdm(tasks, desc='gaussian sweep tasks'):
        d0 = np.asarray(embedder.direction(task.high_text, task.low_text), dtype=np.float32)
        prior_scores = _cosine_scores(task.X, d0)
        prior_spearman = spearman_corr(prior_scores, task.y01)
        for config in configs:
            for policy in policies:
                for replicate in range(int(repetitions)):
                    rng = np.random.default_rng(GLOBAL_SEED + stable_seed(task.task_id, policy, replicate, *config.key))
                    queried = np.zeros((len(task.ids),), dtype=bool)
                    selected: list[int] = []
                    state = _gaussian_posterior(
                        task=task,
                        d0=d0,
                        selected_idx=selected,
                        config=config,
                        prior_scores=prior_scores,
                    )
                    for step in range(max_budget + 1):
                        if step in budgets:
                            _append_state_metrics(
                                logger=logger,
                                completed=completed,
                                run_id=run_id,
                                task=task,
                                policy=policy,
                                replicate=replicate,
                                budget=step,
                                config=config,
                                state=state,
                                d0=d0,
                                prior_scores=prior_scores,
                                prior_spearman=prior_spearman,
                                selected=selected,
                                metric_mode=metric_mode,
                            )
                        if step >= max_budget:
                            break
                        idx = _select_next(policy=policy, state=state, queried=queried, rng=rng)
                        if idx is None:
                            break
                        queried[int(idx)] = True
                        selected.append(int(idx))
                        state = _gaussian_posterior(
                            task=task,
                            d0=d0,
                            selected_idx=selected,
                            config=config,
                            prior_scores=prior_scores,
                        )

    write_gaussian_sweep_summaries(root, run_id, comparison_dirs=comparison_dirs, metric_cols=METRIC_COLS)
    return root / 'ordinal_gaussian_sweep_report.md'
