#!/usr/bin/env python3

from __future__ import annotations
from typing import List
from pathlib import Path
from embeddings import EmbeddingEngine


# DEFAULT_METHODS = ["avg", "clip", "dino", "dift_sd"]
DEFAULT_METHODS = ["dift_sd"]
DATASETS_DIR = Path(__file__).parent.parent / "data" / "datasets"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "embeddings"

def precompute(dataset: str, methods: List[str]) -> None:
    engine = EmbeddingEngine(dataset)
    entries = engine.list_images()
    if not entries:
        print(f"No images found under: {dataset}")
        return
    print(f"Found {len(entries)} images. Precomputing embeddings for: {methods}")
    ok = []
    fail = []
    for m in methods:
        print(f"-> {m} ...", end="", flush=True)
        try:
            embs = engine.estimate_embeddings(entries, method=m)
            # Persist via the same cache path the server expects
            cache = engine._cache_dir() / f"embeddings_{m.lower()}.npz"
            import numpy as np
            paths = np.array([e.path for e in entries])
            mtimes = np.array([int(Path(p).stat().st_mtime) if Path(p).exists() else 0 for p in paths], dtype=np.int64)
            np.savez_compressed(cache, paths=paths, mtimes=mtimes, embeddings=embs)
            print(" done")
            ok.append(m)
        except Exception as e:
            print(f" failed: {e}")
            fail.append((m, str(e)))

    print()
    print(f"Completed. OK={ok}")
    if fail:
        print("Failures:")
        for m, err in fail:
            print(f"  - {m}: {err}")


def main():
    for dataset in DATASETS_DIR.iterdir():
        if not dataset.is_dir():
            continue
        print(f"Dataset: {dataset.name}")
        precompute(dataset, DEFAULT_METHODS)
    print("All done.")
if __name__ == "__main__":
    main()
