#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError as exc:
    raise RuntimeError('optuna is required. Install with: pip install optuna') from exc

try:
    from .axis_bayes import AxisBayesEngine, CollectionCache
    from .gallery_backend import ImageGalleryEngine
except ImportError:
    from axis_bayes import AxisBayesEngine, CollectionCache
    from gallery_backend import ImageGalleryEngine


# =============================
# Sweep configuration
# =============================
REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_LOG_PATH = REPO_ROOT / 'backend' / 'experiments' / 'piecewise_multidataset_optuna.csv'

AXIS_MODEL_TYPE = 'piecewise_linear'
AXIS_FEATURE_SPACE = 'clip_dino'

N_TRIALS = 120
OPTUNA_SAMPLER_SEED = 2026
MOVES_PER_LABEL = 3
MOVE_SELECTION_SEED = 123

# The sweep intentionally uses fixed prompt overrides instead of the LLM prompt
# ensemble so the evaluation is deterministic and faster.
USE_LLM_PROMPT_ENSEMBLE = False


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset_root: Path
    field: str
    query_prefix: str
    min_count: int = 1
    max_labels: int = 0
    exclude_values: Tuple[str, ...] = ()
    weight: float = 1.0


@dataclass
class LoadedTask:
    name: str
    dataset_root: Path
    collection_id: str
    field: str
    ids: List[str]
    id_to_idx: Dict[str, int]
    id_to_label: Dict[str, str]
    class_names: List[str]
    labels: np.ndarray
    Y: np.ndarray
    X_clip: np.ndarray
    X_dino: np.ndarray
    move_catalog: List[str]
    queries_by_label: Dict[str, str]
    prompt_overrides: Dict[str, Tuple[List[str], List[str]]]
    weight: float


TASK_SPECS: List[TaskSpec] = [
    TaskSpec(
        name='emoset_emotion',
        dataset_root=REPO_ROOT / 'data' / 'datasets' / 'EmoSet',
        field='emotion',
        query_prefix='emotion',
        min_count=1,
        max_labels=0,
        exclude_values=(),
        weight=1.0,
    ),
    TaskSpec(
        name='paintings_wikiart_genre',
        dataset_root=REPO_ROOT / 'data' / 'datasets' / 'paintings_wikiart',
        field='genre',
        query_prefix='genre',
        min_count=30,
        max_labels=0,
        exclude_values=('Unknown Genre',),
        weight=1.0,
    ),
    TaskSpec(
        name='paintings_wikiart_style',
        dataset_root=REPO_ROOT / 'data' / 'datasets' / 'paintings_wikiart',
        field='style',
        query_prefix='style',
        min_count=20,
        max_labels=0,
        exclude_values=(),
        weight=1.0,
    ),
]


def utc_now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class CSVAppender:
    def __init__(self, path: Path, fieldnames: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._initialized = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open('r', newline='', encoding='utf-8') as f:
                existing_header = next(csv.reader(f), [])
            if list(existing_header) == self.fieldnames:
                self._initialized = True
            else:
                with self.path.open('r', newline='', encoding='utf-8') as f:
                    old_rows = list(csv.DictReader(f))
                with self.path.open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                    for row in old_rows:
                        writer.writerow({k: row.get(k, '') for k in self.fieldnames})
                self._initialized = True

    def append(self, row: Mapping[str, object]) -> None:
        safe = {k: row.get(k, '') for k in self.fieldnames}
        with self.path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(safe)
            f.flush()


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return (arr / norms).astype(np.float32)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if xa.size != ya.size or xa.size == 0:
        return np.nan
    xa = xa - xa.mean()
    ya = ya - ya.mean()
    denom = float(np.linalg.norm(xa) * np.linalg.norm(ya))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xa, ya) / denom)


def humanize_label(label: str) -> str:
    return str(label or '').strip().replace('_', ' ')


def resolve_image_column(meta: pd.DataFrame) -> str:
    if 'image' in meta.columns:
        return 'image'
    return str(meta.columns[0])


def resolve_field_column(meta: pd.DataFrame, field: str) -> str:
    target = str(field or '').strip().lower()
    for col in meta.columns:
        if str(col).strip().lower() == target:
            return str(col)
    raise RuntimeError(f'Could not find field={field!r} in metadata columns={list(meta.columns)}')


def build_prompt_override(prefix: str, label: str) -> Tuple[List[str], List[str]]:
    prefix_key = str(prefix or '').strip().lower()
    label_text = humanize_label(label)
    if prefix_key == 'emotion':
        pos = [
            f'an image that strongly conveys {label_text}',
            f'an image full of {label_text}',
            f'a scene expressing {label_text}',
        ]
        neg = [
            f'an image without {label_text}',
            'an emotionally neutral image',
            f'a calm scene not expressing {label_text}',
        ]
        return pos, neg
    if prefix_key == 'genre':
        pos = [
            f'a painting in the {label_text} genre',
            f'an artwork of genre {label_text}',
            f'an example of {label_text} painting',
        ]
        neg = [
            f'a painting not in the {label_text} genre',
            'an artwork of a different painting genre',
            f'a non-{label_text} painting',
        ]
        return pos, neg
    if prefix_key == 'style':
        pos = [
            f'a painting in the {label_text} style',
            f'an artwork in {label_text} style',
            f'an example of {label_text} painting style',
        ]
        neg = [
            f'a painting not in the {label_text} style',
            'an artwork in a different painting style',
            f'a non-{label_text} painting style',
        ]
        return pos, neg
    pos = [
        f'an image with very strong presence of {label_text}',
        f'an image with high presence of {label_text}',
        f'an image with a lot of {label_text}',
    ]
    neg = [
        f'an image with very weak presence of {label_text}',
        f'an image with low presence of {label_text}',
        f'an image with very little {label_text}',
    ]
    return pos, neg


def build_move_catalog(ids: Sequence[str], labels: Sequence[str], class_names: Sequence[str], k_per_label: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    ids_by_label: Dict[str, List[str]] = {cls: [] for cls in class_names}
    for image_id, label in zip(ids, labels):
        if label in ids_by_label:
            ids_by_label[label].append(str(image_id))
    for cls in class_names:
        if len(ids_by_label[cls]) < int(k_per_label):
            raise RuntimeError(f'Not enough images for label={cls!r}; need {k_per_label}, found {len(ids_by_label[cls])}')
    sampled = {
        cls: rng.choice(ids_by_label[cls], size=int(k_per_label), replace=False).tolist()
        for cls in class_names
    }
    catalog: List[str] = []
    for round_idx in range(int(k_per_label)):
        for cls in class_names:
            catalog.append(sampled[cls][round_idx])
    return catalog


def load_task(spec: TaskSpec, moves_per_label: int, move_seed: int) -> LoadedTask:
    dataset_root = Path(spec.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_root}')

    engine = ImageGalleryEngine(str(dataset_root))
    entries = engine.list_images()
    if len(entries) == 0:
        raise RuntimeError(f'No images found under {dataset_root}')

    metadata_path = dataset_root / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f'Metadata not found: {metadata_path}')
    meta = pd.read_csv(metadata_path)
    meta.columns = [str(c).strip() for c in meta.columns]
    image_col = resolve_image_column(meta)
    field_col = resolve_field_column(meta, spec.field)
    meta[image_col] = meta[image_col].astype(str).str.strip()
    meta[field_col] = meta[field_col].astype(str).str.strip()
    image_to_value = dict(zip(meta[image_col], meta[field_col]))

    ids_all = [str(entry.id) for entry in entries]
    clip = engine._load_embeddings_only(entries, method='clip')
    dino = engine._load_embeddings_only(entries, method='dino')
    if clip is None or dino is None:
        raise RuntimeError(f'Expected both clip and dino embeddings for {dataset_root}')
    X_clip_all = l2_normalize_rows(np.asarray(clip, dtype=np.float32))
    X_dino_all = l2_normalize_rows(np.asarray(dino, dtype=np.float32))
    if X_clip_all.shape[0] != len(ids_all) or X_dino_all.shape[0] != len(ids_all):
        raise RuntimeError(f'Embedding count mismatch for {dataset_root}')

    exclude = {str(v).strip().lower() for v in spec.exclude_values}
    kept_rows: List[Tuple[int, str]] = []
    for idx, image_id in enumerate(ids_all):
        raw_value = str(image_to_value.get(image_id, '')).strip()
        if not raw_value:
            continue
        if raw_value.lower() in exclude:
            continue
        kept_rows.append((idx, raw_value))
    if len(kept_rows) == 0:
        raise RuntimeError(f'No labeled examples kept for task={spec.name}')

    label_series = pd.Series([label for _, label in kept_rows], dtype='object')
    counts = label_series.value_counts()
    selected_labels = [str(label) for label, count in counts.items() if int(count) >= int(spec.min_count)]
    if int(spec.max_labels) > 0:
        selected_labels = selected_labels[: int(spec.max_labels)]
    if len(selected_labels) < 2:
        raise RuntimeError(f'Task={spec.name} retained fewer than two labels after filtering')
    selected_set = set(selected_labels)

    kept_indices = [idx for idx, label in kept_rows if label in selected_set]
    labels = [label for _, label in kept_rows if label in selected_set]
    ids = [ids_all[idx] for idx in kept_indices]
    X_clip = X_clip_all[np.asarray(kept_indices, dtype=np.int64)]
    X_dino = X_dino_all[np.asarray(kept_indices, dtype=np.int64)]
    id_to_idx = {image_id: i for i, image_id in enumerate(ids)}
    id_to_label = {image_id: label for image_id, label in zip(ids, labels)}
    Y = np.stack([(np.asarray(labels, dtype=object) == cls).astype(np.float32) for cls in selected_labels], axis=1)

    collection_id = f'{spec.name}'
    queries_by_label: Dict[str, str] = {}
    prompt_overrides: Dict[str, Tuple[List[str], List[str]]] = {}
    for label in selected_labels:
        query = f'{spec.query_prefix}: {humanize_label(label)}'
        queries_by_label[label] = query
        prompt_overrides[query] = build_prompt_override(spec.query_prefix, label)

    move_catalog = build_move_catalog(
        ids=ids,
        labels=labels,
        class_names=selected_labels,
        k_per_label=int(moves_per_label),
        seed=int(move_seed),
    )

    return LoadedTask(
        name=spec.name,
        dataset_root=dataset_root,
        collection_id=collection_id,
        field=spec.field,
        ids=ids,
        id_to_idx=id_to_idx,
        id_to_label=id_to_label,
        class_names=selected_labels,
        labels=np.asarray(labels, dtype=object),
        Y=Y.astype(np.float32),
        X_clip=X_clip.astype(np.float32),
        X_dino=X_dino.astype(np.float32),
        move_catalog=move_catalog,
        queries_by_label=queries_by_label,
        prompt_overrides=prompt_overrides,
        weight=float(spec.weight),
    )


class SweepAxisBayesEngine(AxisBayesEngine):
    _prompt_embed_cache: Dict[Tuple[str, str], Tuple[np.ndarray, List[str], List[str], Dict[str, str]]] = {}

    def __init__(
        self,
        tasks_by_collection: Mapping[str, LoadedTask],
        prompt_overrides: Mapping[str, Tuple[List[str], List[str]]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tasks_by_collection = dict(tasks_by_collection)
        self._sweep_prompt_overrides = {
            str(key).strip(): (list(value[0]), list(value[1]))
            for key, value in prompt_overrides.items()
        }

    def build_prompt_ensemble(self, q: str) -> tuple[List[str], List[str], Dict[str, str]]:
        query = ' '.join(str(q or '').strip().split())
        if query in self._sweep_prompt_overrides:
            pos, neg = self._sweep_prompt_overrides[query]
            return list(pos), list(neg), {'source': 'sweep_override', 'provider': 'fixed_template'}
        return super().build_prompt_ensemble(query)

    def _embed_prompt_ensemble(self, q: str) -> tuple[np.ndarray, List[str], List[str], Dict[str, str]]:
        query = ' '.join(str(q or '').strip().split())
        cache_key = (self.axis_bounds_text_source, query)
        cached = self._prompt_embed_cache.get(cache_key)
        if cached is not None:
            return (
                np.asarray(cached[0], dtype=np.float32).copy(),
                list(cached[1]),
                list(cached[2]),
                dict(cached[3]),
            )
        result = super()._embed_prompt_ensemble(query)
        self._prompt_embed_cache[cache_key] = (
            np.asarray(result[0], dtype=np.float32).copy(),
            list(result[1]),
            list(result[2]),
            dict(result[3]),
        )
        return result

    def _get_collection(self, dataset_root: str, collection_id: str) -> CollectionCache:
        collection_key = str(collection_id or '').strip()
        if collection_key not in self._tasks_by_collection:
            raise KeyError(f'Unknown preloaded collection_id={collection_key!r}')
        task = self._tasks_by_collection[collection_key]
        key = f'{collection_key}|{self.feature_space}|cw={self.clip_scale:.6f}|dw={self.dino_scale:.6f}'
        cached = self._collections.get(key)
        if cached is not None:
            return cached

        X_clip = np.asarray(task.X_clip, dtype=np.float32)
        X_dino = np.asarray(task.X_dino, dtype=np.float32)
        clip_dim = int(X_clip.shape[1])
        dino_dim = int(X_dino.shape[1]) if self.feature_space == 'clip_dino' else 0
        if self.feature_space == 'clip_dino':
            X = np.concatenate(
                [
                    self.clip_scale * X_clip,
                    self.dino_scale * X_dino,
                ],
                axis=1,
            ).astype(np.float32)
        else:
            X = X_clip.astype(np.float32)
        cached = CollectionCache(
            dataset_root=str(task.dataset_root.resolve()),
            collection_id=collection_key,
            ids=list(task.ids),
            embeddings=X,
            id_to_index=dict(task.id_to_idx),
            feature_space=self.feature_space,
            clip_dim=clip_dim,
            dino_dim=dino_dim,
            clip_scale=float(self.clip_scale),
            dino_scale=float(self.dino_scale),
        )
        self._collections[key] = cached
        return cached


def compute_task_metrics(engine: SweepAxisBayesEngine, task: LoadedTask) -> Dict[str, object]:
    axis_scores: List[np.ndarray] = []
    axis_uncertainties: List[np.ndarray] = []
    for class_name in task.class_names:
        query = task.queries_by_label[class_name]
        payload = engine.create_axis(
            collection_id=task.collection_id,
            dataset_root=str(task.dataset_root),
            q=query,
            mode='rank',
            model_type=AXIS_MODEL_TYPE,
        )
        axis_id = str(payload['axis_id'])
        for image_id in task.move_catalog:
            target = 100.0 if task.id_to_label[image_id] == class_name else 0.0
            payload = engine.move_axis(axis_id=axis_id, image_id=image_id, new_score_0_100=target)
        axis_scores.append(np.asarray(payload['projection_values'], dtype=np.float32))
        axis_uncertainties.append(np.asarray(payload['std'], dtype=np.float32))

    Z = np.stack(axis_scores, axis=1).astype(np.float32)
    C = np.asarray(
        [[pearson_corr(Z[:, axis_idx], task.Y[:, gt_idx]) for gt_idx in range(task.Y.shape[1])] for axis_idx in range(Z.shape[1])],
        dtype=np.float32,
    )
    diag_sum = float(np.trace(C))
    diag_mean = float(diag_sum / max(1, len(task.class_names)))
    pred_idx = np.argmax(Z, axis=1)
    gt_idx = np.argmax(task.Y, axis=1)
    top1_acc = float(np.mean(pred_idx == gt_idx))
    if len(axis_uncertainties) > 0:
        uncertainty_mean = float(np.nanmean(np.asarray(axis_uncertainties, dtype=np.float32)))
    else:
        uncertainty_mean = float('nan')
    return {
        'corr_matrix': C,
        'diag_sum': diag_sum,
        'diag_mean': diag_mean,
        'top1_acc': top1_acc,
        'uncertainty_mean': uncertainty_mean,
        'label_count': int(len(task.class_names)),
        'image_count': int(len(task.ids)),
    }


def evaluate_tasks(engine: SweepAxisBayesEngine, tasks: Sequence[LoadedTask]) -> Dict[str, object]:
    per_task: Dict[str, Dict[str, object]] = {}
    weighted_scores: List[float] = []
    weights: List[float] = []
    for task in tasks:
        metrics = compute_task_metrics(engine, task)
        per_task[task.name] = metrics
        weighted_scores.append(float(metrics['diag_mean']) * float(task.weight))
        weights.append(float(task.weight))
    denom = float(np.sum(weights)) if len(weights) > 0 else 1.0
    objective = float(np.sum(weighted_scores) / max(1e-8, denom))
    return {
        'objective': objective,
        'per_task': per_task,
    }


def sample_params(trial: optuna.Trial) -> Dict[str, object]:
    clip_weight = trial.suggest_float('clip_weight', 0.15, 0.9)
    dino_weight = 1.0 - clip_weight
    params = {
        'clip_weight': float(clip_weight),
        'dino_weight': float(dino_weight),
        'piecewise_num_experts': int(trial.suggest_int('piecewise_num_experts', 1, 4)),
        'piecewise_use_gating': bool(trial.suggest_categorical('piecewise_use_gating', [False, True])),
        'piecewise_aggregator': str(trial.suggest_categorical('piecewise_aggregator', ['max', 'mean', 'softmax'])),
        'piecewise_clip_scale': float(trial.suggest_float('piecewise_clip_scale', 0.2, 2.5, log=True)),
        'piecewise_dino_scale': float(trial.suggest_float('piecewise_dino_scale', 0.2, 2.5, log=True)),
        'pairwise_from_scalar_margin': float(trial.suggest_float('pairwise_from_scalar_margin', 0.0, 0.35)),
        'piecewise_prior_strength': float(trial.suggest_float('piecewise_prior_strength', 0.02, 40.0, log=True)),
        'piecewise_expert_diversity_strength': float(trial.suggest_float('piecewise_expert_diversity_strength', 0.0, 1.5)),
        'piecewise_l2_reg': float(trial.suggest_float('piecewise_l2_reg', 1e-5, 0.5, log=True)),
        'piecewise_learning_rate': float(trial.suggest_float('piecewise_learning_rate', 1e-3, 0.2, log=True)),
        'piecewise_max_refine_steps': int(trial.suggest_int('piecewise_max_refine_steps', 24, 180, step=12)),
    }
    return params


def build_engine(tasks: Sequence[LoadedTask], params: Mapping[str, object]) -> SweepAxisBayesEngine:
    prompt_overrides: Dict[str, Tuple[List[str], List[str]]] = {}
    tasks_by_collection: Dict[str, LoadedTask] = {}
    for task in tasks:
        tasks_by_collection[task.collection_id] = task
        prompt_overrides.update(task.prompt_overrides)
    return SweepAxisBayesEngine(
        tasks_by_collection=tasks_by_collection,
        prompt_overrides=prompt_overrides,
        model_type=AXIS_MODEL_TYPE,
        mode='rank',
        feature_space=AXIS_FEATURE_SPACE,
        clip_weight=float(params['clip_weight']),
        dino_weight=float(params['dino_weight']),
        piecewise_num_experts=int(params['piecewise_num_experts']),
        piecewise_use_gating=bool(params['piecewise_use_gating']),
        piecewise_aggregator=str(params['piecewise_aggregator']),
        piecewise_clip_scale=float(params['piecewise_clip_scale']),
        piecewise_dino_scale=float(params['piecewise_dino_scale']),
        pairwise_from_scalar_margin=float(params['pairwise_from_scalar_margin']),
        piecewise_prior_strength=float(params['piecewise_prior_strength']),
        piecewise_expert_diversity_strength=float(params['piecewise_expert_diversity_strength']),
        piecewise_l2_reg=float(params['piecewise_l2_reg']),
        piecewise_learning_rate=float(params['piecewise_learning_rate']),
        piecewise_max_refine_steps=int(params['piecewise_max_refine_steps']),
        max_moves=0,
        axis_bounds_text_source='template',
        use_llm_prompt_ensemble=USE_LLM_PROMPT_ENSEMBLE,
    )


def json_compact(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(',', ':'), sort_keys=True)


def main() -> None:
    run_id = uuid.uuid4().hex[:12]
    print(f'[piecewise-sweep] run_id={run_id}')
    print(f'[piecewise-sweep] loading tasks={len(TASK_SPECS)}')
    tasks = [load_task(spec, moves_per_label=MOVES_PER_LABEL, move_seed=MOVE_SELECTION_SEED) for spec in TASK_SPECS]
    for task in tasks:
        print(
            f'[piecewise-sweep] task={task.name} images={len(task.ids)} '
            f'labels={len(task.class_names)} field={task.field} dataset={task.dataset_root.name}'
        )

    fields = [
        'ts_utc',
        'run_id',
        'row_type',
        'trial_number',
        'trial_state',
        'value',
        'model_type',
        'feature_space',
        'clip_weight',
        'dino_weight',
        'piecewise_num_experts',
        'piecewise_use_gating',
        'piecewise_aggregator',
        'piecewise_clip_scale',
        'piecewise_dino_scale',
        'pairwise_from_scalar_margin',
        'piecewise_prior_strength',
        'piecewise_expert_diversity_strength',
        'piecewise_l2_reg',
        'piecewise_learning_rate',
        'piecewise_max_refine_steps',
        'task_count',
        'objective',
        'per_task_diag_mean_json',
        'per_task_diag_sum_json',
        'per_task_top1_acc_json',
        'per_task_uncertainty_json',
        'per_task_label_count_json',
        'error',
    ]
    logger = CSVAppender(CSV_LOG_PATH, fields)

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        try:
            engine = build_engine(tasks, params)
            result = evaluate_tasks(engine, tasks)
            per_task = result['per_task']
            val = float(result['objective'])
            logger.append(
                {
                    'ts_utc': utc_now_iso(),
                    'run_id': run_id,
                    'row_type': 'trial',
                    'trial_number': int(trial.number),
                    'trial_state': 'ok',
                    'value': val,
                    'model_type': AXIS_MODEL_TYPE,
                    'feature_space': AXIS_FEATURE_SPACE,
                    'clip_weight': float(params['clip_weight']),
                    'dino_weight': float(params['dino_weight']),
                    'piecewise_num_experts': int(params['piecewise_num_experts']),
                    'piecewise_use_gating': bool(params['piecewise_use_gating']),
                    'piecewise_aggregator': str(params['piecewise_aggregator']),
                    'piecewise_clip_scale': float(params['piecewise_clip_scale']),
                    'piecewise_dino_scale': float(params['piecewise_dino_scale']),
                    'pairwise_from_scalar_margin': float(params['pairwise_from_scalar_margin']),
                    'piecewise_prior_strength': float(params['piecewise_prior_strength']),
                    'piecewise_expert_diversity_strength': float(params['piecewise_expert_diversity_strength']),
                    'piecewise_l2_reg': float(params['piecewise_l2_reg']),
                    'piecewise_learning_rate': float(params['piecewise_learning_rate']),
                    'piecewise_max_refine_steps': int(params['piecewise_max_refine_steps']),
                    'task_count': int(len(tasks)),
                    'objective': val,
                    'per_task_diag_mean_json': json_compact({k: float(v['diag_mean']) for k, v in per_task.items()}),
                    'per_task_diag_sum_json': json_compact({k: float(v['diag_sum']) for k, v in per_task.items()}),
                    'per_task_top1_acc_json': json_compact({k: float(v['top1_acc']) for k, v in per_task.items()}),
                    'per_task_uncertainty_json': json_compact({k: float(v['uncertainty_mean']) for k, v in per_task.items()}),
                    'per_task_label_count_json': json_compact({k: int(v['label_count']) for k, v in per_task.items()}),
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
                    'value': '',
                    'model_type': AXIS_MODEL_TYPE,
                    'feature_space': AXIS_FEATURE_SPACE,
                    'clip_weight': float(params['clip_weight']),
                    'dino_weight': float(params['dino_weight']),
                    'piecewise_num_experts': int(params['piecewise_num_experts']),
                    'piecewise_use_gating': bool(params['piecewise_use_gating']),
                    'piecewise_aggregator': str(params['piecewise_aggregator']),
                    'piecewise_clip_scale': float(params['piecewise_clip_scale']),
                    'piecewise_dino_scale': float(params['piecewise_dino_scale']),
                    'pairwise_from_scalar_margin': float(params['pairwise_from_scalar_margin']),
                    'piecewise_prior_strength': float(params['piecewise_prior_strength']),
                    'piecewise_expert_diversity_strength': float(params['piecewise_expert_diversity_strength']),
                    'piecewise_l2_reg': float(params['piecewise_l2_reg']),
                    'piecewise_learning_rate': float(params['piecewise_learning_rate']),
                    'piecewise_max_refine_steps': int(params['piecewise_max_refine_steps']),
                    'task_count': int(len(tasks)),
                    'objective': '',
                    'per_task_diag_mean_json': '',
                    'per_task_diag_sum_json': '',
                    'per_task_top1_acc_json': '',
                    'per_task_uncertainty_json': '',
                    'per_task_label_count_json': '',
                    'error': f'{type(exc).__name__}: {exc}'[:1000],
                },
            )
            raise

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SAMPLER_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=int(N_TRIALS), n_jobs=1, show_progress_bar=True)

    best_trial = study.best_trial
    logger.append(
        {
            'ts_utc': utc_now_iso(),
            'run_id': run_id,
            'row_type': 'best_summary',
            'trial_number': int(best_trial.number),
            'trial_state': 'ok',
            'value': float(best_trial.value),
            'model_type': AXIS_MODEL_TYPE,
            'feature_space': AXIS_FEATURE_SPACE,
            'clip_weight': float(best_trial.params.get('clip_weight', 'nan')),
            'dino_weight': float(1.0 - float(best_trial.params.get('clip_weight', 0.0))),
            'piecewise_num_experts': int(best_trial.params.get('piecewise_num_experts', 0)),
            'piecewise_use_gating': bool(best_trial.params.get('piecewise_use_gating', False)),
            'piecewise_aggregator': str(best_trial.params.get('piecewise_aggregator', '')),
            'piecewise_clip_scale': float(best_trial.params.get('piecewise_clip_scale', 'nan')),
            'piecewise_dino_scale': float(best_trial.params.get('piecewise_dino_scale', 'nan')),
            'pairwise_from_scalar_margin': float(best_trial.params.get('pairwise_from_scalar_margin', 'nan')),
            'piecewise_prior_strength': float(best_trial.params.get('piecewise_prior_strength', 'nan')),
            'piecewise_expert_diversity_strength': float(best_trial.params.get('piecewise_expert_diversity_strength', 'nan')),
            'piecewise_l2_reg': float(best_trial.params.get('piecewise_l2_reg', 'nan')),
            'piecewise_learning_rate': float(best_trial.params.get('piecewise_learning_rate', 'nan')),
            'piecewise_max_refine_steps': int(best_trial.params.get('piecewise_max_refine_steps', 0)),
            'task_count': int(len(tasks)),
            'objective': float(best_trial.value),
            'per_task_diag_mean_json': '',
            'per_task_diag_sum_json': '',
            'per_task_top1_acc_json': '',
            'per_task_uncertainty_json': '',
            'per_task_label_count_json': '',
            'error': '',
        },
    )
    print(f'[piecewise-sweep] best_value={float(best_trial.value):.6f}')
    print(f'[piecewise-sweep] best_params={best_trial.params}')
    print(f'[piecewise-sweep] appended results to {CSV_LOG_PATH}')


if __name__ == '__main__':
    main()
