#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def split_emb(emb, n_parts=4):
    """Split embeddings into n_parts parts."""
    size = emb.shape[2]
    part_size = size // n_parts
    parts = np.zeros((emb.shape[0], emb.shape[1], n_parts, n_parts), dtype=emb.dtype)
    for i in range(n_parts):
        for j in range(n_parts):
            emb_slice = emb[:, :, i*part_size:(i+1)*part_size, j*part_size:(j+1)*part_size]
            parts[:, :, i, j] = np.mean(emb_slice, axis=(2, 3))
    return parts

@dataclass
class ImageEntry:
    id: str
    path: str
    class_name: str


class CLIPEmbeddingExtractor:
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        import torch
        from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(
            clip_model_name,
            use_safetensors=True,
            local_files_only=False,
        ).to(self.device)
        self.processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.available = True

    def extract_image_embedding(self, image: Image.Image):
        if not self.available:
            raise RuntimeError("CLIPExtractor unavailable")
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.squeeze(0).detach().cpu().numpy().astype("float32")

    def extract_text_embedding(self, text: str):
        if not self.available or self.tokenizer is None:
            raise RuntimeError("CLIPExtractor unavailable for text")
        torch = self.torch
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.squeeze(0).detach().cpu().numpy().astype("float32")


class DINOEmbeddingExtractor:
    def __init__(self):
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.model.eval().to(self.device)
        self.preprocess = self._make_preprocess(224)
        for p in self.model.parameters():
            p.requires_grad = False
        self.available = True

    def _make_preprocess(self, size: int):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def extract_image_embedding(self, image: Image.Image):
        if not self.available:
            raise RuntimeError("DINOExtractor unavailable")
        torch = self.torch
        img_t = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model.forward_features(img_t) if hasattr(self.model, 'forward_features') else self.model(img_t)
            if isinstance(out, dict) and 'x_norm_clstoken' in out:
                feats = out['x_norm_clstoken']
            elif isinstance(out, dict) and 'pool' in out:
                feats = out['pool']
            elif isinstance(out, (list, tuple)):
                feats = out[0]
            else:
                feats = out
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.squeeze(0).detach().cpu().numpy().astype('float32')


class EmbeddingEngine:
    """Handles image discovery and embedding estimation/caching."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

    def list_images(self) -> List[ImageEntry]:
        entries: List[ImageEntry] = []
        # Only process files in the immediate subdirectories (no recursion)
        for file_path in os.listdir(self.dataset_path):
            file_path = self.dataset_path / file_path
            if file_path.is_file():
                ext = Path(file_path).suffix.lower()
                basename = Path(file_path).name
                if ext in SUPPORTED_FORMATS:
                    entries.append(ImageEntry(id=basename, path=str(file_path), class_name=''))
        return sorted(entries, key=lambda e: e.id)
    
    # def list_images(self) -> List[ImageEntry]:
    #     entries: List[ImageEntry] = []
    #     for root, _, files in os.walk(self.dataset_path):
    #         for fname in files:
    #             ext = Path(fname).suffix.lower()
    #             if ext in SUPPORTED_FORMATS:
    #                 fpath = Path(root) / fname
    #                 class_name = Path(root).name
    #                 img_id = f"{class_name}/{fname}"
    #                 entries.append(ImageEntry(id=img_id, path=str(fpath), class_name=class_name))
    #     return sorted(entries, key=lambda e: e.id)

    def estimate_embeddings(self, images: Sequence[ImageEntry],
                            method: str = "color_rgb",
                            resize: Tuple[int, int] = (32, 32)) -> np.ndarray:
        method_l = method.lower()
        if method_l in {"clip", "clip-vit", "clip32"}:
            extractor = CLIPEmbeddingExtractor()
            if getattr(extractor, 'available', False):
                return self._extract_with_extractor(images, extractor, fallback_dim=512)
            method_l = "color_rgb"
        elif method_l in {"dino", "dino-vit"}:
            vecs = self._extract_with_local_dino(images)
            if vecs is not None:
                return vecs
            extractor = DINOEmbeddingExtractor()
            if getattr(extractor, 'available', False):
                return self._extract_with_extractor(images, extractor, fallback_dim=384)
            method_l = "color_rgb"
        elif method_l in {"sd", "dift", "dift_sd", "diftsd"}:
            vecs = self._extract_with_local_dift(images)
            if vecs is not None:
                print(vecs.shape)
                return vecs
            method_l = "color_rgb"
        elif method_l in {"color_hsv", "hsv"}:
            return self._extract_color_hsv(images, resize=resize)
        elif method_l in {"color_lch", "lch", "hcl", "color_lab", "lab"}:
            return self._extract_color_lch(images, resize=resize)
        
        embs = []
        for e in images:
            with Image.open(e.path) as img:
                img = img.convert('RGB')
                img = img.resize(resize, Image.Resampling.LANCZOS)
                arr = np.asarray(img).astype(np.float32) / 255.0
                embs.append(arr.reshape(-1))
        return np.vstack(embs)

    @staticmethod
    def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
        # rgb in [0,1], shape (..., 3)
        # sRGB inverse companding
        def inv_compand(c):
            return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        r = inv_compand(rgb[..., 0])
        g = inv_compand(rgb[..., 1])
        b = inv_compand(rgb[..., 2])
        # sRGB to XYZ (D65)
        X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        return np.stack([X, Y, Z], axis=-1)

    @staticmethod
    def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
        # D65 reference white
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        x = xyz[..., 0] / Xn
        y = xyz[..., 1] / Yn
        z = xyz[..., 2] / Zn
        eps = 216 / 24389
        kappa = 24389 / 27
        def f(t):
            return np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)
        fx = f(x)
        fy = f(y)
        fz = f(z)
        L = (116 * fy) - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        return np.stack([L, a, b], axis=-1)

    def _extract_color_lch(self, images: Sequence[ImageEntry], resize: Tuple[int, int] = (64, 64)) -> np.ndarray:
        vecs = []
        for e in images:
            try:
                with Image.open(e.path) as img:
                    arr = np.asarray(img.convert('RGB').resize(resize, Image.Resampling.LANCZOS)).astype(np.float32) / 255.0
            except Exception:
                arr = np.zeros((resize[1], resize[0], 3), dtype=np.float32)
            lab = self._xyz_to_lab(self._rgb_to_xyz(arr))  # (...,3)
            # Average in a/b chroma plane
            a_mean = float(np.mean(lab[..., 1]))
            b_mean = float(np.mean(lab[..., 2]))
            vecs.append([a_mean, b_mean])
        return np.array(vecs, dtype=np.float32)

    def _extract_color_hsv(self, images: Sequence[ImageEntry], resize: Tuple[int, int] = (64, 64)) -> np.ndarray:
        vecs = []
        for e in images:
            try:
                with Image.open(e.path) as img:
                    im = img.convert('HSV').resize(resize, Image.Resampling.LANCZOS)
                    hsv = np.asarray(im).astype(np.float32)
            except Exception:
                hsv = np.zeros((resize[1], resize[0], 3), dtype=np.float32)
            # PIL HSV: H,S,V in 0..255
            H = hsv[..., 0] * (360.0 / 255.0)
            S = hsv[..., 1] / 255.0
            ang = np.deg2rad(H)
            cx = float(np.mean(np.cos(ang) * S))
            sx = float(np.mean(np.sin(ang) * S))
            vecs.append([cx, sx])
        return np.array(vecs, dtype=np.float32)

    def _extract_with_local_dift(self, images: Sequence[ImageEntry], n_parts: Optional[int]=3) -> Optional[np.ndarray]:
        """Extract DIFT features by calling dift_sd.create_feature on file list, then pool.

        Expects dift_sd.create_feature(filelist, prompt) to return (ft, imglist) where
        ft is a torch tensor shaped [N, C, H, W]. We global-average-pool to [N, C] on n_parts x n_parts regions,
        and L2-normalize per vector.
        """
        import torch
        from dift_sd import create_feature, SDFeaturizer  # type: ignore
        filelist = [str(e.path) for e in images]
        # Use an empty or neutral prompt; dift_sd.create_feature should decide exact usage
        prompt = ''
        extractor = SDFeaturizer()
        ft = create_feature(extractor, filelist, prompt)
        print("DIFT feature shape:", ft.shape)
        ft =  ft.detach().cpu().numpy().astype('float32')
        parts = split_emb(ft, n_parts=n_parts)
        print("DIFT split parts shape:", parts.shape)
        return parts

    def _extract_with_local_dino(self, images: Sequence[ImageEntry]) -> Optional[np.ndarray]:
        from dino import DINOFeaturizer  # type: ignore
        import torch
        from torchvision import transforms
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feat = DINOFeaturizer()
        tfm = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        vecs = []
        for e in images:
            with Image.open(e.path) as im:
                x = tfm(im.convert('RGB')).unsqueeze(0).to(device)
                feats = feat.forward(x, block_index=0)
                v = feats.mean(dim=(2, 3)).squeeze(0)
                v = v / (v.norm() + 1e-8)
                vecs.append(v.detach().cpu().numpy().astype('float32'))
        return np.vstack(vecs)


    def _extract_with_extractor(self, images: Sequence[ImageEntry], extractor, fallback_dim: int) -> np.ndarray:
        embs = []
        for e in images:
            with Image.open(e.path) as img:
                img = img.convert('RGB')
                vec = extractor.extract_image_embedding(img)
                embs.append(vec)
        return np.vstack(embs)

    def _cache_dir(self) -> Path:
        d = self.dataset_path / '.cache'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_embeddings_only(self, entries: List[ImageEntry], method: str) -> Optional[np.ndarray]:
        cache = self._cache_dir() / f'embeddings_{method.lower()}.npz'
        if not cache.exists():
            return None
        data = np.load(cache, allow_pickle=False)
        # Minimal only-embeddings format
        if 'embeddings' in data.files and 'paths' not in data.files:
            arr = data['embeddings']
            if arr.ndim == 2 and arr.shape[0] == len(entries):
                return arr
            return None
        # Full format: compare ids (class/filename) only
        if {'paths', 'embeddings'}.issubset(set(data.files)):
            current_ids = np.array([e.id for e in entries])
            cached_ids = np.array([f"{Path(p).parent.name}/{Path(p).name}" for p in data['paths']])
            if len(cached_ids) == len(current_ids) and np.all(cached_ids == current_ids):
                embs = data['embeddings']
                if embs.ndim == 2 and embs.shape[0] == len(entries):
                    return embs
        return None

    def compute_and_cache_embeddings(self, entries: List[ImageEntry], method: str) -> np.ndarray:
        embs = self.estimate_embeddings(entries, method=method)
        import numpy as np
        cache = self._cache_dir() / f'embeddings_{method.lower()}.npz'
        paths = np.array([e.path for e in entries])
        mtimes = np.array([int(Path(p).stat().st_mtime) if Path(p).exists() else 0 for p in paths], dtype=np.int64)
        np.savez_compressed(cache, paths=paths, mtimes=mtimes, embeddings=embs)
        return embs

    def text_embedding(self, method: str, text: str) -> Optional[np.ndarray]:
        m = (method or '').lower()
        if m in { 'clip', 'clip-vit', 'clip32' }:
            extractor = CLIPEmbeddingExtractor()
            if getattr(extractor, 'available', False):
                vec = extractor.extract_text_embedding(text)
                return vec
        return None

