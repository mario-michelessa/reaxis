#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    from .import_curated_dataset import DEFAULT_REDUCTION, parse_methods, precompute_dataset
    from .embeddings import normalize_multimodal_method
except ImportError:
    from import_curated_dataset import DEFAULT_REDUCTION, parse_methods, precompute_dataset
    from embeddings import normalize_multimodal_method


DEFAULT_METHODS = ["siglip2"]
DATASETS_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"


def resolve_datasets(raw_datasets: List[str]) -> List[Path]:
    if raw_datasets:
        return [Path(dataset).resolve() for dataset in raw_datasets]
    return sorted(path.resolve() for path in DATASETS_DIR.iterdir() if path.is_dir())


def missing_methods(dataset: Path, methods: List[str]) -> List[str]:
    cache_dir = dataset / ".cache"
    missing: List[str] = []
    for method in methods:
        cache_method = normalize_multimodal_method(method)
        emb_cache = cache_dir / f"embeddings_{cache_method}.npz"
        coords_cache = cache_dir / f"coords_pca2d_{cache_method}.npz"
        if not emb_cache.exists() or not coords_cache.exists():
            missing.append(cache_method)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute embedding and PCA caches for existing prepared datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Prepared dataset directories. Defaults to every directory under data/datasets.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated embedding methods to precompute.",
    )
    parser.add_argument(
        "--reduction",
        default=DEFAULT_REDUCTION,
        help="2D reduction method for cached coordinates (default: pca).",
    )
    parser.add_argument(
        "--include-complete",
        action="store_true",
        help="Process datasets even when all requested caches already exist.",
    )
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    datasets = resolve_datasets(list(args.datasets))
    if not datasets:
        raise RuntimeError(f"No datasets found under {DATASETS_DIR}")

    queued: List[Path] = []
    for dataset in datasets:
        if not dataset.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {dataset}")
        missing = missing_methods(dataset, methods)
        if missing or args.include_complete:
            if missing:
                print(f"Queue: {dataset} missing={missing}")
            else:
                print(f"Queue: {dataset} already complete for {methods}")
            queued.append(dataset)
            continue
        print(f"Skip: {dataset} already has {methods}")

    if not queued:
        print("Nothing to do.")
        return

    for dataset in queued:
        print(f"Dataset: {dataset}")
        precompute_dataset(dataset_root=dataset, methods=methods, reduction=args.reduction)
    print("All done.")


if __name__ == "__main__":
    main()
