#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    from .import_curated_dataset import DEFAULT_REDUCTION, parse_methods, precompute_dataset
    from .embeddings import embedding_cache_filename, normalize_multimodal_method
    from .reduction_cache import coords_cache_filename, parse_reduction_methods
    from .runtime_config import DATASETS_ROOT
except ImportError:
    from import_curated_dataset import DEFAULT_REDUCTION, parse_methods, precompute_dataset
    from embeddings import embedding_cache_filename, normalize_multimodal_method
    from reduction_cache import coords_cache_filename, parse_reduction_methods
    from runtime_config import DATASETS_ROOT


DEFAULT_METHODS = ["siglip2"]
DEFAULT_RAW_METHODS = ["siglip2", "clip"]
DATASETS_DIR = DATASETS_ROOT


def resolve_datasets(raw_datasets: List[str]) -> List[Path]:
    if raw_datasets:
        return [Path(dataset).resolve() for dataset in raw_datasets]
    return sorted(path.resolve() for path in DATASETS_DIR.iterdir() if path.is_dir())


def missing_methods(dataset: Path, methods: List[str], reduction: str, *, normalize: bool, include_coords: bool) -> List[str]:
    cache_dir = dataset / ".cache"
    reduction_methods = parse_reduction_methods(reduction) if include_coords else []
    missing: List[str] = []
    for method in methods:
        cache_method = normalize_multimodal_method(method)
        emb_cache = cache_dir / embedding_cache_filename(cache_method, normalize=normalize)
        if not emb_cache.exists():
            label = f"{cache_method}_raw" if not normalize and cache_method in {"clip", "siglip2"} else cache_method
            missing.append(label)
            continue
        if include_coords:
            for reduction_method in reduction_methods:
                coords_cache = cache_dir / coords_cache_filename(cache_method, reduction=reduction_method)
                if not coords_cache.exists():
                    missing.append(f"{cache_method}:{reduction_method}")
    return missing


def validate_raw_methods(methods: List[str]) -> None:
    invalid = sorted({
        normalize_multimodal_method(method)
        for method in methods
        if normalize_multimodal_method(method) not in {"clip", "siglip2"}
    })
    if invalid:
        raise ValueError(
            f"--raw only supports semantic methods clip/siglip2; got {invalid}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute embedding and 2D reduction caches for existing prepared datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Prepared dataset directories. Defaults to every directory under data/datasets.",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated embedding methods to precompute. Defaults to siglip2, or siglip2+clip when --raw is set.",
    )
    parser.add_argument(
        "--reduction",
        default=DEFAULT_REDUCTION,
        help="Comma-separated 2D reduction methods for cached coordinates: pca, umap, tsne, or all.",
    )
    parser.add_argument(
        "--include-complete",
        action="store_true",
        help="Process datasets even when all requested caches already exist.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Precompute non-normalized semantic caches (.cache/embeddings_{clip|siglip2}_raw.npz) and skip PCA coords.",
    )
    args = parser.parse_args()

    default_methods = DEFAULT_RAW_METHODS if args.raw else DEFAULT_METHODS
    methods = parse_methods(args.methods or ",".join(default_methods))
    if args.raw:
        validate_raw_methods(methods)
    datasets = resolve_datasets(list(args.datasets))
    if not datasets:
        raise RuntimeError(f"No datasets found under {DATASETS_DIR}")
    include_coords = not bool(args.raw)

    queued: List[Path] = []
    for dataset in datasets:
        if not dataset.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {dataset}")
        missing = missing_methods(
            dataset,
            methods,
            str(args.reduction or DEFAULT_REDUCTION),
            normalize=not args.raw,
            include_coords=include_coords,
        )
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
        precompute_dataset(
            dataset_root=dataset,
            methods=methods,
            reduction=args.reduction,
            normalize=not args.raw,
            include_coords=include_coords,
        )
    print("All done.")


if __name__ == "__main__":
    main()
