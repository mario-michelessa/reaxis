#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from .gallery_backend import ImageGalleryEngine
    from .runtime_config import OUTPUT_ROOT
except ImportError:
    from gallery_backend import ImageGalleryEngine
    from runtime_config import OUTPUT_ROOT


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D embedding matrix, got shape={arr.shape}')
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return (arr / norms).astype(np.float32)


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        raise ValueError('Cannot normalize zero vector')
    return (v / n).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_x = np.exp(arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


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


def _quantile_from_sorted(z_sorted: np.ndarray, p01: float) -> float:
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


def _spearman_from_scores(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return float('nan')
    ra = _rank_percentile_01(a).astype(np.float32)
    rb = _rank_percentile_01(b).astype(np.float32)
    ra = ra - float(np.mean(ra))
    rb = rb - float(np.mean(rb))
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(ra, rb) / den)


def _topk_jaccard_from_scores(a: np.ndarray, b: np.ndarray, frac: float = 0.10) -> float:
    if a.shape != b.shape or a.size == 0:
        return float('nan')
    n = int(a.size)
    k = max(1, int(round(frac * n)))
    ia = set(np.argsort(a)[-k:].tolist())
    ib = set(np.argsort(b)[-k:].tolist())
    inter = len(ia.intersection(ib))
    union = len(ia.union(ib))
    if union == 0:
        return 0.0
    return float(inter / union)


def _parse_move_template(raw: str) -> List[Tuple[float, float]]:
    # Format: "10:80,25:70,75:30,90:20"
    out: List[Tuple[float, float]] = []
    for chunk in str(raw or '').split(','):
        c = chunk.strip()
        if not c:
            continue
        if ':' not in c:
            raise ValueError(f'Invalid move template chunk "{c}" (expected src:dst)')
        src_s, dst_s = c.split(':', 1)
        src = float(src_s.strip()) / 100.0
        dst = float(dst_s.strip()) / 100.0
        out.append((max(0.0, min(1.0, src)), max(0.0, min(1.0, dst))))
    if len(out) == 0:
        raise ValueError('Move template parsed to empty list')
    return out


def _pick_id_near_percentile(ids: List[str], score01: np.ndarray, p01: float, used: set[str]) -> str:
    n = len(ids)
    order = np.argsort(score01, kind='mergesort')
    target = int(round(max(0.0, min(1.0, p01)) * max(0, n - 1)))
    for radius in range(0, n):
        lo = max(0, target - radius)
        hi = min(n - 1, target + radius)
        for pos in (lo, hi):
            image_id = ids[int(order[pos])]
            if image_id not in used:
                return image_id
    return ids[int(order[target])]


def _utc_now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class CSVRowLogger:
    def __init__(self, csv_path: Path, fieldnames: Sequence[str]):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._initialized = self.path.exists() and self.path.stat().st_size > 0

    def append(self, row: Dict[str, Any]):
        safe = {k: row.get(k, '') for k in self.fieldnames}
        with self.path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(safe)
            f.flush()


def _batched(seq: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(seq), bs):
        yield seq[i:i + bs]


def _load_rgb_batch(paths: Sequence[str]) -> List[Image.Image]:
    out: List[Image.Image] = []
    for p in paths:
        with Image.open(p) as im:
            out.append(im.convert('RGB').copy())
    return out


@dataclass
class RepresentationResult:
    name: str
    ids: List[str]
    embeddings: np.ndarray
    w0: np.ndarray
    alpha_vec: np.ndarray
    feature_space: str
    info: Dict[str, Any]


class BaseRepAdapter:
    def __init__(self, name: str):
        self.name = name

    def encode_images(self, ids: List[str], paths: List[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def encode_texts(self, texts: List[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError


class CachedClipAdapter(BaseRepAdapter):
    def __init__(self, dataset_path: str, method: str, text_model_id: str):
        super().__init__(name=f'cache:{method}')
        self.dataset_path = str(dataset_path)
        self.method = str(method)
        self.text_model_id = str(text_model_id)
        self._clip_model = None
        self._clip_tokenizer = None
        self._torch = None
        self._device = None

    def encode_images(self, ids: List[str], paths: List[str], batch_size: int) -> np.ndarray:
        engine = ImageGalleryEngine(self.dataset_path)
        entries_all = engine.list_images()
        by_id = {str(e.id): i for i, e in enumerate(entries_all)}
        embs_all = engine._load_embeddings_only(entries_all, method=self.method)
        if embs_all is None:
            raise ValueError(f'Cached embeddings not found for method="{self.method}"')
        subset_indices: List[int] = []
        for image_id in ids:
            if image_id not in by_id:
                raise ValueError(f'Image id "{image_id}" not found in dataset for cache adapter')
            subset_indices.append(int(by_id[image_id]))
        return _normalize_rows(np.asarray(embs_all, dtype=np.float32)[np.asarray(subset_indices, dtype=np.int64)])

    def _ensure_text_model(self):
        if self._clip_model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPTokenizer
        self._torch = torch
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._clip_model = CLIPModel.from_pretrained(self.text_model_id).to(self._device)
        self._clip_model.eval()
        self._clip_tokenizer = CLIPTokenizer.from_pretrained(self.text_model_id)

    def encode_texts(self, texts: List[str], batch_size: int) -> np.ndarray:
        self._ensure_text_model()
        torch = self._torch
        feats_all: List[np.ndarray] = []
        for batch in _batched(texts, batch_size):
            tokenized = self._clip_tokenizer(list(batch), return_tensors='pt', padding=True, truncation=True)
            tokenized = {k: v.to(self._device) for k, v in tokenized.items()}
            with torch.no_grad():
                feats = self._clip_model.get_text_features(**tokenized)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            feats_all.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.vstack(feats_all)


class HFClipAdapter(BaseRepAdapter):
    def __init__(self, model_id: str):
        super().__init__(name=f'hf_clip:{model_id}')
        self.model_id = str(model_id)
        self._torch = None
        self._model = None
        self._processor = None
        self._device = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self._torch = torch
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = CLIPModel.from_pretrained(self.model_id).to(self._device)
        self._model.eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_id)

    def encode_images(self, ids: List[str], paths: List[str], batch_size: int) -> np.ndarray:
        self._ensure()
        torch = self._torch
        chunks: List[np.ndarray] = []
        for path_batch in _batched(paths, batch_size):
            images = _load_rgb_batch(path_batch)
            inputs = self._processor(images=images, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(self._device)
            with torch.no_grad():
                feats = self._model.get_image_features(pixel_values=pixel_values)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            chunks.append(feats.detach().cpu().numpy().astype(np.float32))
        return _normalize_rows(np.vstack(chunks))

    def encode_texts(self, texts: List[str], batch_size: int) -> np.ndarray:
        self._ensure()
        torch = self._torch
        chunks: List[np.ndarray] = []
        for txt_batch in _batched(texts, batch_size):
            inputs = self._processor(text=list(txt_batch), return_tensors='pt', padding=True, truncation=True)
            payload = {k: v.to(self._device) for k, v in inputs.items() if k in {'input_ids', 'attention_mask'}}
            with torch.no_grad():
                feats = self._model.get_text_features(**payload)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            chunks.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.vstack(chunks)


class HFSiglipAdapter(BaseRepAdapter):
    def __init__(self, model_id: str):
        super().__init__(name=f'hf_siglip:{model_id}')
        self.model_id = str(model_id)
        self._torch = None
        self._model = None
        self._processor = None
        self._device = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, SiglipModel
        self._torch = torch
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = SiglipModel.from_pretrained(self.model_id).to(self._device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def encode_images(self, ids: List[str], paths: List[str], batch_size: int) -> np.ndarray:
        self._ensure()
        torch = self._torch
        chunks: List[np.ndarray] = []
        for path_batch in _batched(paths, batch_size):
            images = _load_rgb_batch(path_batch)
            inputs = self._processor(images=images, return_tensors='pt')
            payload = {k: v.to(self._device) for k, v in inputs.items() if k == 'pixel_values'}
            with torch.no_grad():
                feats = self._model.get_image_features(**payload)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            chunks.append(feats.detach().cpu().numpy().astype(np.float32))
        return _normalize_rows(np.vstack(chunks))

    def encode_texts(self, texts: List[str], batch_size: int) -> np.ndarray:
        self._ensure()
        torch = self._torch
        chunks: List[np.ndarray] = []
        for txt_batch in _batched(texts, batch_size):
            inputs = self._processor(text=list(txt_batch), return_tensors='pt', padding=True, truncation=True)
            payload = {k: v.to(self._device) for k, v in inputs.items() if k in {'input_ids', 'attention_mask'}}
            with torch.no_grad():
                feats = self._model.get_text_features(**payload)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            chunks.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.vstack(chunks)


class OpenCLIPAdapter(BaseRepAdapter):
    def __init__(self, model_name: str, pretrained: str):
        super().__init__(name=f'openclip:{model_name}:{pretrained}')
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self._torch = None
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        import open_clip
        self._torch = torch
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        model = model.to(self._device)
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def encode_images(self, ids: List[str], paths: List[str], batch_size: int) -> np.ndarray:
        self._ensure()
        torch = self._torch
        chunks: List[np.ndarray] = []
        for path_batch in _batched(paths, batch_size):
            imgs = _load_rgb_batch(path_batch)
            tensors = [self._preprocess(im) for im in imgs]
            batch = torch.stack(tensors, dim=0).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_image(batch)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            chunks.append(feats.detach().cpu().numpy().astype(np.float32))
        return _normalize_rows(np.vstack(chunks))

    def encode_texts(self, texts: List[str], batch_size: int) -> np.ndarray:
        self._ensure()
        torch = self._torch
        chunks: List[np.ndarray] = []
        for txt_batch in _batched(texts, batch_size):
            tokens = self._tokenizer(list(txt_batch)).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            chunks.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.vstack(chunks)


def _parse_rep_adapter(spec: str, dataset_path: str, cache_text_model: str) -> BaseRepAdapter:
    parts = str(spec).split(':')
    kind = parts[0].strip().lower()
    if kind == 'cache':
        if len(parts) != 2:
            raise ValueError(f'Invalid cache spec "{spec}", expected cache:<method>')
        return CachedClipAdapter(dataset_path=dataset_path, method=parts[1].strip(), text_model_id=cache_text_model)
    if kind == 'hf_clip':
        model_id = ':'.join(parts[1:]).strip()
        if not model_id:
            raise ValueError(f'Invalid hf_clip spec "{spec}"')
        return HFClipAdapter(model_id=model_id)
    if kind == 'hf_siglip':
        model_id = ':'.join(parts[1:]).strip()
        if not model_id:
            raise ValueError(f'Invalid hf_siglip spec "{spec}"')
        return HFSiglipAdapter(model_id=model_id)
    if kind == 'openclip':
        if len(parts) < 3:
            raise ValueError(f'Invalid openclip spec "{spec}", expected openclip:<model_name>:<pretrained>')
        model_name = parts[1].strip()
        pretrained = ':'.join(parts[2:]).strip()
        if not model_name or not pretrained:
            raise ValueError(f'Invalid openclip spec "{spec}"')
        return OpenCLIPAdapter(model_name=model_name, pretrained=pretrained)
    raise ValueError(f'Unknown representation spec "{spec}"')


@dataclass
class SimulatorConfig:
    mode: str
    alpha: float
    dino_alpha: float
    bias_alpha: float
    sigma2: float
    rank_eta: float
    rank_anchor_k: int
    rank_anchor_delta: float
    rank_max_pairs: int


class AxisSimulator:
    def __init__(self, X: np.ndarray, w0: np.ndarray, config: SimulatorConfig, alpha_vec: Optional[np.ndarray] = None):
        self.X = _normalize_rows(X)
        self.n, self.d = self.X.shape
        self.w0 = _normalize_vec(w0)
        if self.w0.shape[0] != self.d:
            raise ValueError(f'w0 dim mismatch: w0={self.w0.shape[0]} embeddings={self.d}')
        if alpha_vec is None:
            self.alpha_vec = np.full((self.d,), float(max(1e-6, config.alpha)), dtype=np.float32)
        else:
            av = np.asarray(alpha_vec, dtype=np.float32).reshape(-1)
            if av.shape[0] != self.d:
                raise ValueError(f'alpha_vec dim mismatch: alpha_vec={av.shape[0]} embeddings={self.d}')
            self.alpha_vec = np.maximum(av, 1e-6).astype(np.float32)
        self.alpha_inv = (1.0 / self.alpha_vec).astype(np.float32)
        self.cfg = config
        self.mode = 'rank' if str(config.mode).strip().lower() == 'rank' else 'gaussian'

        self.z0 = (self.X @ self.w0).astype(np.float32)
        self.z0_sorted = np.sort(self.z0).astype(np.float32)

        self.move_order: List[int] = []
        self.move_targets: Dict[int, float] = {}

        self.mu = self.w0.copy()
        self.b = 0.0
        self.b0 = 0.0

        self.Xm = np.zeros((0, self.d), dtype=np.float32)
        self.y_raw = np.zeros((0,), dtype=np.float32)
        self.A_inv = np.zeros((0, 0), dtype=np.float32)

        self.pair_i: List[int] = []
        self.pair_j: List[int] = []
        self.pair_y: List[float] = []
        self.pair_w: List[float] = []
        self.pair_D = np.zeros((0, self.d), dtype=np.float32)
        self.pair_W_diag = np.zeros((0,), dtype=np.float32)
        self.pair_M_inv = np.zeros((0, 0), dtype=np.float32)

    def _refresh_gaussian(self):
        if len(self.move_order) == 0:
            self.mu = self.w0.copy()
            self.b = float(self.b0)
            self.Xm = np.zeros((0, self.d), dtype=np.float32)
            self.y_raw = np.zeros((0,), dtype=np.float32)
            self.A_inv = np.zeros((0, 0), dtype=np.float32)
            return

        idx = np.asarray(self.move_order, dtype=np.int64)
        X = self.X[idx, :]
        y = np.asarray([_quantile_from_sorted(self.z0_sorted, self.move_targets[i]) for i in self.move_order], dtype=np.float32)
        m = int(X.shape[0])
        XS = (X * self.alpha_inv[None, :]).astype(np.float32)
        XXt = XS @ X.T
        ones = np.ones((m,), dtype=np.float32)
        A = (
            (self.cfg.sigma2 * np.eye(m, dtype=np.float32))
            + XXt
            + ((1.0 / self.cfg.bias_alpha) * np.outer(ones, ones).astype(np.float32))
        )
        A_inv = np.linalg.inv(A + (1e-6 * np.eye(m, dtype=np.float32)))
        r = y - (X @ self.w0) - float(self.b0)
        mu = self.w0 + (self.alpha_inv * (X.T @ (A_inv @ r)))
        b = float(self.b0) + float((1.0 / self.cfg.bias_alpha) * (ones @ (A_inv @ r)))
        if float(np.dot(mu, self.w0)) < 0.0:
            mu = -mu
            b = -b
        nrm = float(np.linalg.norm(mu))
        if nrm > 1e-8:
            mu = (mu / nrm).astype(np.float32)
            b = float(b / nrm)
        else:
            mu = self.w0.copy()
            b = float(self.b0)
        self.mu = mu
        self.b = b
        self.Xm = X
        self.y_raw = y
        self.A_inv = A_inv.astype(np.float32)

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
        return ((a_inv * vec) - (a_inv * (D.T @ sol))).astype(np.float32)

    def _rank_loss(self, w: np.ndarray, D: np.ndarray, weights: np.ndarray, alpha_vec: np.ndarray) -> float:
        eta = float(max(1e-6, self.cfg.rank_eta))
        u = (D @ np.asarray(w, dtype=np.float32)) / eta
        logistic = np.logaddexp(0.0, -u).astype(np.float32)
        diff = np.asarray(w, dtype=np.float32) - self.w0
        reg = 0.5 * float(np.sum(np.asarray(alpha_vec, dtype=np.float32) * diff * diff))
        return float(reg + float(np.sum(weights * logistic)))

    def _refresh_rank(self):
        if len(self.pair_i) == 0:
            self.mu = self.w0.copy()
            self.b = 0.0
            self.pair_D = np.zeros((0, self.d), dtype=np.float32)
            self.pair_W_diag = np.zeros((0,), dtype=np.float32)
            self.pair_M_inv = np.zeros((0, 0), dtype=np.float32)
            return

        idx_i = np.asarray(self.pair_i, dtype=np.int64)
        idx_j = np.asarray(self.pair_j, dtype=np.int64)
        y = np.asarray(self.pair_y, dtype=np.float32)
        weights = np.asarray(self.pair_w, dtype=np.float32)
        Delta = (self.X[idx_i, :] - self.X[idx_j, :]).astype(np.float32)
        D = (y[:, None] * Delta).astype(np.float32)
        eta = float(max(1e-6, self.cfg.rank_eta))
        alpha_vec = self.alpha_vec
        alpha_inv = self.alpha_inv

        w = np.asarray(self.mu if self.mu.shape[0] == self.d else self.w0, dtype=np.float32).copy()
        for _ in range(40):
            u = (D @ w) / eta
            sig_neg = _sigmoid(-u)
            grad = (alpha_vec * (w - self.w0)) - ((D.T @ (weights * sig_neg)) / eta)
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

            base_loss = self._rank_loss(w, D, weights, alpha_vec)
            t = 1.0
            accepted = False
            while t >= 1e-4:
                cand = np.asarray(w - (t * step), dtype=np.float32)
                cand_loss = self._rank_loss(cand, D, weights, alpha_vec)
                if np.isfinite(cand_loss) and cand_loss <= (base_loss - (1e-4 * t * step_dot)):
                    w = cand
                    accepted = True
                    break
                t *= 0.5
            if not accepted:
                w = np.asarray(w - (0.05 * step), dtype=np.float32)

        if float(np.dot(w, self.w0)) < 0.0:
            w = -w
        nrm = float(np.linalg.norm(w))
        if nrm > 1e-8:
            w = (w / nrm).astype(np.float32)
        else:
            w = self.w0.copy()
        self.mu = w
        self.b = 0.0

        u = (D @ self.mu) / eta
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
        self.pair_D = D
        self.pair_W_diag = w_diag
        self.pair_M_inv = np.asarray(M_inv, dtype=np.float32)

    def _refresh(self):
        if self.mode == 'rank':
            self._refresh_rank()
        else:
            self._refresh_gaussian()

    def _append_pairs_from_move(self, image_idx: int, target_p01: float):
        if self.n <= 1:
            return
        z = self.projection_values()
        score01 = _rank_percentile_01(z)
        target = float(max(0.0, min(1.0, target_p01)))
        old_p = float(score01[int(image_idx)])
        k = int(max(1, self.cfg.rank_anchor_k))
        delta = float(max(0.01, min(0.45, self.cfg.rank_anchor_delta)))
        max_cross = int(max(2, 2 * k))
        move_mag = abs(target - old_p)
        pair_weight = float(max(0.25, 1.0 + (2.0 * move_mag)))

        idx_all = np.arange(self.n, dtype=np.int64)
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
            chosen = ids[order[: min(k, ids.size)]]
            return [int(v) for v in chosen.tolist()]

        below = pick(below_band, 'below')
        above = pick(above_band, 'above')

        for j in below:
            self.pair_i.append(int(image_idx))
            self.pair_j.append(int(j))
            self.pair_y.append(1.0)
            self.pair_w.append(pair_weight)
        for j in above:
            self.pair_i.append(int(image_idx))
            self.pair_j.append(int(j))
            self.pair_y.append(-1.0)
            self.pair_w.append(pair_weight)

        lo = min(old_p, target)
        hi = max(old_p, target)
        crossed = not_self[(score01[not_self] >= lo) & (score01[not_self] <= hi)]
        if crossed.size > 0:
            if crossed.size > max_cross:
                step = max(1, int(np.floor(crossed.size / max_cross)))
                crossed = crossed[::step][:max_cross]
            signed = 1.0 if target >= old_p else -1.0
            for j in crossed.tolist():
                self.pair_i.append(int(image_idx))
                self.pair_j.append(int(j))
                self.pair_y.append(float(signed))
                self.pair_w.append(float(max(0.15, 0.75 * pair_weight)))

        overflow = len(self.pair_i) - int(max(16, self.cfg.rank_max_pairs))
        if overflow > 0:
            self.pair_i = self.pair_i[overflow:]
            self.pair_j = self.pair_j[overflow:]
            self.pair_y = self.pair_y[overflow:]
            self.pair_w = self.pair_w[overflow:]

    def add_move(self, image_idx: int, target_pct_0_100: float):
        idx = int(image_idx)
        target = float(max(0.0, min(100.0, float(target_pct_0_100)))) / 100.0
        if idx not in self.move_targets:
            self.move_order.append(idx)
        self.move_targets[idx] = target

        if self.mode == 'rank':
            self._refresh()
            self._append_pairs_from_move(image_idx=idx, target_p01=target)
        self._refresh()

    def projection_values(self) -> np.ndarray:
        return ((self.X @ self.mu) + float(self.b)).astype(np.float32)

    def score01(self) -> np.ndarray:
        return _rank_percentile_01(self.projection_values()).astype(np.float32)

    def score100(self) -> np.ndarray:
        return (100.0 * self.score01()).astype(np.float32)

    def std(self) -> np.ndarray:
        if self.mode == 'rank':
            prior_x = np.sum((self.X * self.X) * self.alpha_inv[None, :], axis=1).astype(np.float32)
            if self.pair_D.size == 0 or self.pair_M_inv.size == 0:
                var = prior_x
                return np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
            V = (self.pair_D * self.alpha_inv[None, :]) @ self.X.T
            MV = self.pair_M_inv @ V
            quad = np.sum(V * MV, axis=0)
            var = prior_x - quad
            return np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

        n = self.n
        prior_x = np.sum((self.X * self.X) * self.alpha_inv[None, :], axis=1).astype(np.float32)
        if self.Xm.size == 0 or self.A_inv.size == 0:
            var = self.cfg.sigma2 + prior_x + (1.0 / self.cfg.bias_alpha)
            return np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        XmS = (self.Xm * self.alpha_inv[None, :]).astype(np.float32)
        U = XmS @ self.X.T
        ones = np.ones((self.Xm.shape[0], 1), dtype=np.float32)
        V = U + ((1.0 / self.cfg.bias_alpha) * ones)
        AV = self.A_inv @ V
        quad = np.sum(V * AV, axis=0)
        var = self.cfg.sigma2 + prior_x + (1.0 / self.cfg.bias_alpha) - quad
        return np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

    @property
    def pair_count(self) -> int:
        return int(len(self.pair_i))


def _build_w0_from_prompts(rep: BaseRepAdapter, pos_prompts: List[str], neg_prompts: List[str], batch_size: int) -> np.ndarray:
    if len(pos_prompts) < 1 or len(neg_prompts) < 1:
        raise ValueError('Need at least one positive and one negative prompt')
    pos = rep.encode_texts(pos_prompts, batch_size=batch_size)
    neg = rep.encode_texts(neg_prompts, batch_size=batch_size)
    pos = _normalize_rows(pos)
    neg = _normalize_rows(neg)
    pos_avg = np.mean(pos, axis=0, dtype=np.float32)
    neg_avg = np.mean(neg, axis=0, dtype=np.float32)
    return _normalize_vec(pos_avg - neg_avg)


def _load_entries(dataset_path: str, max_images: int) -> Tuple[List[str], List[str]]:
    engine = ImageGalleryEngine(dataset_path)
    entries = engine.list_images()
    if len(entries) == 0:
        raise ValueError(f'No images found in dataset: {dataset_path}')
    if max_images > 0:
        entries = entries[: int(max_images)]
    ids = [str(e.id) for e in entries]
    paths = [str(e.path) for e in entries]
    return ids, paths


def _load_cached_dino_embeddings(dataset_path: str, ids: List[str]) -> np.ndarray:
    engine = ImageGalleryEngine(dataset_path)
    entries_all = engine.list_images()
    by_id = {str(e.id): i for i, e in enumerate(entries_all)}
    embs_all = engine._load_embeddings_only(entries_all, method='dino')
    if embs_all is None:
        raise ValueError('Cached DINO embeddings not found for this dataset')
    subset_indices: List[int] = []
    for image_id in ids:
        if image_id not in by_id:
            raise ValueError(f'Image id "{image_id}" not found while loading DINO cache')
        subset_indices.append(int(by_id[image_id]))
    embs = np.asarray(embs_all, dtype=np.float32)[np.asarray(subset_indices, dtype=np.int64)]
    return _normalize_rows(embs)


def _build_fused_features(
    X_vlm: np.ndarray,
    w0_vlm: np.ndarray,
    X_dino: np.ndarray,
    clip_weight: float,
    dino_weight: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    c_w = float(max(1e-6, clip_weight))
    d_w = float(max(1e-6, dino_weight))
    w_sum = c_w + d_w
    clip_scale = float(math.sqrt(c_w / w_sum))
    dino_scale = float(math.sqrt(d_w / w_sum))
    X_fused = np.concatenate(
        [clip_scale * _normalize_rows(X_vlm), dino_scale * _normalize_rows(X_dino)],
        axis=1,
    ).astype(np.float32)
    dino_zeros = np.zeros((X_dino.shape[1],), dtype=np.float32)
    w0_fused = _normalize_vec(np.concatenate([_normalize_vec(w0_vlm), dino_zeros], axis=0))
    return X_fused, w0_fused, clip_scale, dino_scale


def _load_move_plan(
    moves_json: str,
    move_template: str,
    ref_ids: List[str],
    ref_score01: np.ndarray,
) -> List[Dict[str, Any]]:
    if moves_json:
        p = Path(moves_json)
        if not p.exists():
            raise FileNotFoundError(f'moves_json not found: {moves_json}')
        payload = json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(payload, list):
            raise ValueError('moves_json must contain a JSON list')
        out: List[Dict[str, Any]] = []
        for i, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f'Invalid move at index {i}: expected object')
            image_id = str(row.get('image_id') or row.get('id') or '').strip()
            target = float(row.get('target_pct', row.get('new_score_0_100', row.get('target'))))
            if not image_id:
                raise ValueError(f'Missing image_id at move index {i}')
            out.append({'image_id': image_id, 'target_pct': max(0.0, min(100.0, target))})
        if len(out) == 0:
            raise ValueError('moves_json is empty')
        return out

    template = _parse_move_template(move_template)
    used: set[str] = set()
    out = []
    for src_p, dst_p in template:
        image_id = _pick_id_near_percentile(ref_ids, ref_score01, src_p, used)
        used.add(image_id)
        out.append({
            'image_id': image_id,
            'source_pct': float(src_p * 100.0),
            'target_pct': float(dst_p * 100.0),
        })
    return out


def _default_pos_neg(q: str) -> Tuple[List[str], List[str]]:
    query = str(q or '').strip()
    if not query:
        raise ValueError('q must be non-empty')
    pos = [
        f'a photo with strong {query}',
        f'an image where {query} is clearly visible',
        f'a realistic photo emphasizing {query}',
    ]
    neg = [
        f'a photo with no visible {query}',
        f'an image where {query} is absent',
        f'a realistic photo with minimal {query}',
    ]
    return pos, neg


def run(args: argparse.Namespace):
    run_id = uuid.uuid4().hex[:12]
    dataset_path = str(Path(args.dataset).resolve())
    mode = 'rank' if str(args.mode).strip().lower() == 'rank' else 'gaussian'
    feature_space = 'vlm_dino' if str(args.feature_space).strip().lower() in {'vlm_dino', 'clip_dino', 'vlm+dino'} else 'vlm'
    print(f'[sim] run_id={run_id} dataset={dataset_path} mode={mode} feature_space={feature_space}')

    ids, paths = _load_entries(dataset_path=dataset_path, max_images=int(args.max_images))
    n_images = len(ids)
    id_to_idx = {image_id: i for i, image_id in enumerate(ids)}
    print(f'[sim] loaded images n={n_images}')

    clip_weight = float(max(1e-6, float(args.clip_weight)))
    dino_weight = float(max(1e-6, float(args.dino_weight)))
    weights_sum = clip_weight + dino_weight
    clip_weight = float(clip_weight / weights_sum)
    dino_weight = float(dino_weight / weights_sum)

    dino_embeddings: Optional[np.ndarray] = None
    dino_dim = 0
    if feature_space == 'vlm_dino':
        print('[sim] loading cached dino embeddings for feature fusion')
        dino_embeddings = _load_cached_dino_embeddings(dataset_path, ids)
        if int(dino_embeddings.shape[0]) != n_images:
            raise ValueError(f'DINO embedding count mismatch: {dino_embeddings.shape[0]} != {n_images}')
        dino_dim = int(dino_embeddings.shape[1])
        print(f'[sim] loaded dino dim={dino_dim} for fusion')

    pos_prompts = [str(x).strip() for x in (args.pos or []) if str(x).strip()]
    neg_prompts = [str(x).strip() for x in (args.neg or []) if str(x).strip()]
    if len(pos_prompts) == 0 or len(neg_prompts) == 0:
        pos_prompts, neg_prompts = _default_pos_neg(args.q)
    print(f'[sim] prompts pos={len(pos_prompts)} neg={len(neg_prompts)}')

    rep_specs = list(args.repr or [])
    if len(rep_specs) == 0:
        rep_specs = [
            'cache:clip',
            'hf_clip:openai/clip-vit-base-patch32',
            'hf_siglip:google/siglip-base-patch16-224',
        ]
    print(f'[sim] representations={rep_specs}')

    fields = [
        'ts_utc',
        'run_id',
        'dataset',
        'row_type',
        'status',
        'mode',
        'feature_space',
        'representation',
        'representation_b',
        'step',
        'move_index',
        'move_image_id',
        'move_target_pct',
        'n_images',
        'dim',
        'pair_count',
        'move_count',
        'spearman_vs_prior',
        'top10_jaccard_vs_prior',
        'spearman_vs_ref',
        'top10_jaccard_vs_ref',
        'mean_std',
        'p95_std',
        'score_shift_l2',
        'alpha',
        'dino_alpha',
        'clip_weight',
        'dino_weight',
        'vlm_dim',
        'dino_dim',
        'error',
    ]
    logger = CSVRowLogger(Path(args.log_csv), fields)

    cfg = SimulatorConfig(
        mode=mode,
        alpha=float(args.alpha),
        dino_alpha=float(args.dino_alpha),
        bias_alpha=float(args.bias_alpha),
        sigma2=float(args.sigma2),
        rank_eta=float(args.rank_eta),
        rank_anchor_k=int(args.rank_anchor_k),
        rank_anchor_delta=float(args.rank_anchor_delta),
        rank_max_pairs=int(args.rank_max_pairs),
    )

    rep_results: List[RepresentationResult] = []
    for spec in rep_specs:
        t0 = time.time()
        print(f'[sim] loading representation: {spec}')
        try:
            adapter = _parse_rep_adapter(
                spec=spec,
                dataset_path=dataset_path,
                cache_text_model=str(args.cache_text_model),
            )
            X = adapter.encode_images(ids=ids, paths=paths, batch_size=int(args.batch_size))
            if X.shape[0] != n_images:
                raise ValueError(f'Embedding count mismatch for {adapter.name}: {X.shape[0]} != {n_images}')
            w0 = _build_w0_from_prompts(adapter, pos_prompts, neg_prompts, batch_size=int(args.batch_size))
            if w0.shape[0] != X.shape[1]:
                raise ValueError(
                    f'Embedding dim mismatch for {adapter.name}: image_dim={X.shape[1]} text_dim={w0.shape[0]}',
                )
            X_use = _normalize_rows(X)
            w0_use = _normalize_vec(w0)
            alpha_vec = np.full((int(X_use.shape[1]),), float(max(1e-6, cfg.alpha)), dtype=np.float32)
            vlm_dim = int(X_use.shape[1])
            dino_dim_rep = 0
            clip_scale = 1.0
            dino_scale = 0.0
            if feature_space == 'vlm_dino':
                if dino_embeddings is None:
                    raise ValueError('feature_space=vlm_dino but DINO embeddings are missing')
                X_use, w0_use, clip_scale, dino_scale = _build_fused_features(
                    X_vlm=X_use,
                    w0_vlm=w0_use,
                    X_dino=dino_embeddings,
                    clip_weight=clip_weight,
                    dino_weight=dino_weight,
                )
                dino_dim_rep = int(dino_embeddings.shape[1])
                alpha_vec = np.concatenate(
                    [
                        np.full((vlm_dim,), float(max(1e-6, cfg.alpha)), dtype=np.float32),
                        np.full((dino_dim_rep,), float(max(1e-6, cfg.dino_alpha)), dtype=np.float32),
                    ],
                    axis=0,
                ).astype(np.float32)
            rep_results.append(
                RepresentationResult(
                    name=adapter.name,
                    ids=list(ids),
                    embeddings=X_use,
                    w0=w0_use,
                    alpha_vec=alpha_vec,
                    feature_space=feature_space,
                    info={
                        'load_sec': float(time.time() - t0),
                        'dim': int(X_use.shape[1]),
                        'vlm_dim': int(vlm_dim),
                        'dino_dim': int(dino_dim_rep),
                        'clip_scale': float(clip_scale),
                        'dino_scale': float(dino_scale),
                    },
                ),
            )
            print(
                f'[sim] loaded {adapter.name} dim={X_use.shape[1]} '
                f'(vlm_dim={vlm_dim} dino_dim={dino_dim_rep} clip_scale={clip_scale:.3f} dino_scale={dino_scale:.3f}) '
                f'in {time.time() - t0:.2f}s',
            )
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
            print(f'[sim] representation failed {spec}: {err}')
            logger.append({
                'ts_utc': _utc_now_iso(),
                'run_id': run_id,
                'dataset': dataset_path,
                'row_type': 'representation',
                'status': 'error',
                'mode': mode,
                'feature_space': feature_space,
                'representation': spec,
                'representation_b': '',
                'step': -1,
                'move_index': -1,
                'move_image_id': '',
                'move_target_pct': '',
                'n_images': n_images,
                'dim': '',
                'pair_count': '',
                'move_count': '',
                'spearman_vs_prior': '',
                'top10_jaccard_vs_prior': '',
                'spearman_vs_ref': '',
                'top10_jaccard_vs_ref': '',
                'mean_std': '',
                'p95_std': '',
                'score_shift_l2': '',
                'alpha': float(cfg.alpha),
                'dino_alpha': float(cfg.dino_alpha),
                'clip_weight': float(clip_weight if feature_space == 'vlm_dino' else 1.0),
                'dino_weight': float(dino_weight if feature_space == 'vlm_dino' else 0.0),
                'vlm_dim': '',
                'dino_dim': '',
                'error': err[:500],
            })

    if len(rep_results) == 0:
        raise RuntimeError('No representation loaded successfully')

    ref = rep_results[0]
    ref_sim = AxisSimulator(ref.embeddings, ref.w0, cfg, alpha_vec=ref.alpha_vec)
    ref_prior = ref_sim.score01()
    move_plan = _load_move_plan(
        moves_json=str(args.moves_json or '').strip(),
        move_template=str(args.move_template),
        ref_ids=ids,
        ref_score01=ref_prior,
    )
    print(f'[sim] move_plan ({len(move_plan)}): {[{"id":m["image_id"], "target":m["target_pct"]} for m in move_plan]}')

    ref_step_scores: List[np.ndarray] = []
    final_scores: Dict[str, np.ndarray] = {}

    for ridx, rep in enumerate(rep_results):
        print(f'[sim] simulate representation={rep.name}')
        sim = AxisSimulator(rep.embeddings, rep.w0, cfg, alpha_vec=rep.alpha_vec)
        prior_scores = sim.score100()
        step_scores: List[np.ndarray] = []

        def append_step_row(step_idx: int, move_idx: int, move_image_id: str, move_target_pct: Optional[float]):
            scores = sim.score100()
            std = sim.std()
            step_scores.append(scores.copy())
            if ridx == 0:
                ref_scores = scores
            else:
                ref_scores = ref_step_scores[step_idx] if step_idx < len(ref_step_scores) else np.array([], dtype=np.float32)
            row = {
                'ts_utc': _utc_now_iso(),
                'run_id': run_id,
                'dataset': dataset_path,
                'row_type': 'step',
                'status': 'ok',
                'mode': mode,
                'feature_space': feature_space,
                'representation': rep.name,
                'representation_b': '',
                'step': step_idx,
                'move_index': move_idx,
                'move_image_id': move_image_id,
                'move_target_pct': '' if move_target_pct is None else float(move_target_pct),
                'n_images': n_images,
                'dim': int(rep.embeddings.shape[1]),
                'pair_count': int(sim.pair_count),
                'move_count': int(len(sim.move_order)),
                'spearman_vs_prior': float(_spearman_from_scores(scores, prior_scores)),
                'top10_jaccard_vs_prior': float(_topk_jaccard_from_scores(scores, prior_scores, frac=0.10)),
                'spearman_vs_ref': float(_spearman_from_scores(scores, ref_scores)) if ref_scores.size == scores.size else '',
                'top10_jaccard_vs_ref': float(_topk_jaccard_from_scores(scores, ref_scores, frac=0.10)) if ref_scores.size == scores.size else '',
                'mean_std': float(np.mean(std)) if std.size > 0 else '',
                'p95_std': float(np.percentile(std, 95)) if std.size > 0 else '',
                'score_shift_l2': float(np.linalg.norm(scores - prior_scores) / math.sqrt(max(1, scores.size))),
                'alpha': float(cfg.alpha),
                'dino_alpha': float(cfg.dino_alpha),
                'clip_weight': float(clip_weight if feature_space == 'vlm_dino' else 1.0),
                'dino_weight': float(dino_weight if feature_space == 'vlm_dino' else 0.0),
                'vlm_dim': int(rep.info.get('vlm_dim', rep.embeddings.shape[1])),
                'dino_dim': int(rep.info.get('dino_dim', 0)),
                'error': '',
            }
            logger.append(row)

        append_step_row(step_idx=0, move_idx=-1, move_image_id='', move_target_pct=None)

        for m_idx, move in enumerate(move_plan):
            image_id = str(move['image_id'])
            target_pct = float(move['target_pct'])
            if image_id not in id_to_idx:
                raise KeyError(f'Move image id not found in dataset: {image_id}')
            sim.add_move(image_idx=id_to_idx[image_id], target_pct_0_100=target_pct)
            append_step_row(step_idx=m_idx + 1, move_idx=m_idx, move_image_id=image_id, move_target_pct=target_pct)

        if ridx == 0:
            ref_step_scores = [s.copy() for s in step_scores]
        final_scores[rep.name] = step_scores[-1].copy()

    rep_names = [rep.name for rep in rep_results]
    for i in range(len(rep_names)):
        for j in range(i + 1, len(rep_names)):
            a = rep_names[i]
            b = rep_names[j]
            sa = final_scores[a]
            sb = final_scores[b]
            logger.append({
                'ts_utc': _utc_now_iso(),
                'run_id': run_id,
                'dataset': dataset_path,
                'row_type': 'pairwise_final',
                'status': 'ok',
                'mode': mode,
                'feature_space': feature_space,
                'representation': a,
                'representation_b': b,
                'step': -1,
                'move_index': -1,
                'move_image_id': '',
                'move_target_pct': '',
                'n_images': n_images,
                'dim': '',
                'pair_count': '',
                'move_count': len(move_plan),
                'spearman_vs_prior': '',
                'top10_jaccard_vs_prior': '',
                'spearman_vs_ref': float(_spearman_from_scores(sa, sb)),
                'top10_jaccard_vs_ref': float(_topk_jaccard_from_scores(sa, sb, frac=0.10)),
                'mean_std': '',
                'p95_std': '',
                'score_shift_l2': float(np.linalg.norm(sa - sb) / math.sqrt(max(1, sa.size))),
                'alpha': float(cfg.alpha),
                'dino_alpha': float(cfg.dino_alpha),
                'clip_weight': float(clip_weight if feature_space == 'vlm_dino' else 1.0),
                'dino_weight': float(dino_weight if feature_space == 'vlm_dino' else 0.0),
                'vlm_dim': '',
                'dino_dim': '',
                'error': '',
            })

    print(f'[sim] complete run_id={run_id} rows appended to {args.log_csv}')


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Simulate axis Bayesian refinement across representation spaces and append metrics to one CSV.',
    )
    p.add_argument('--dataset', type=str, required=True, help='Dataset path (folder with images and optional .cache).')
    p.add_argument('--q', type=str, default='attribute', help='Axis query label (for default prompt generation).')
    p.add_argument('--pos', action='append', default=[], help='Positive prompt (repeatable).')
    p.add_argument('--neg', action='append', default=[], help='Negative prompt (repeatable).')
    p.add_argument('--repr', action='append', default=[], help='Representation spec. '
                                                               'Supported: cache:<method>, '
                                                               'hf_clip:<model_id>, '
                                                               'hf_siglip:<model_id>, '
                                                               'openclip:<model_name>:<pretrained>')
    p.add_argument('--cache-text-model', type=str, default='openai/clip-vit-base-patch32',
                   help='Text model id used with cache:<method> specs.')
    p.add_argument('--mode', type=str, default='rank', choices=['gaussian', 'rank'], help='Simulation update mode.')
    p.add_argument('--feature-space', type=str, default='vlm', choices=['vlm', 'vlm_dino'],
                   help='Feature space used in simulation: VLM only or VLM fused with cached DINO.')
    p.add_argument('--clip-weight', type=float, default=0.70,
                   help='Relative weight for the VLM block when --feature-space=vlm_dino.')
    p.add_argument('--dino-weight', type=float, default=0.30,
                   help='Relative weight for the DINO block when --feature-space=vlm_dino.')
    p.add_argument('--alpha', type=float, default=96.0, help='Weight prior precision alpha.')
    p.add_argument('--dino-alpha', type=float, default=220.0,
                   help='Prior precision for DINO dimensions in fused mode.')
    p.add_argument('--bias-alpha', type=float, default=1.0, help='Bias prior precision (gaussian mode).')
    p.add_argument('--sigma2', type=float, default=0.04, help='Observation noise variance (gaussian mode).')
    p.add_argument('--rank-eta', type=float, default=0.25, help='Preference temperature eta (rank mode).')
    p.add_argument('--rank-anchor-k', type=int, default=6, help='Anchors below/above target per move (rank mode).')
    p.add_argument('--rank-anchor-delta', type=float, default=0.12, help='Anchor percentile window radius (rank mode).')
    p.add_argument('--rank-max-pairs', type=int, default=1600, help='Maximum stored pairwise constraints.')
    p.add_argument('--moves-json', type=str, default='', help='Optional JSON list of fixed moves: [{"image_id","target_pct"}].')
    p.add_argument('--move-template', type=str, default='10:80,25:70,75:30,90:20',
                   help='Fallback template as src:dst percent pairs (used if --moves-json is empty).')
    p.add_argument('--max-images', type=int, default=1500, help='Maximum images to include (0 = all).')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size for model encoding.')
    p.add_argument('--log-csv', type=str, default=str(OUTPUT_ROOT / 'bayes_repr_compare.csv'),
                   help='Append-only CSV path for all rows.')
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
