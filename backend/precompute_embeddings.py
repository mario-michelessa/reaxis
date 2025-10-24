#!/usr/bin/env python3

from __future__ import annotations
from typing import List
from pathlib import Path
from embeddings import EmbeddingEngine


# DEFAULT_METHODS = ["color_rgb","color_hsv", "color_lch"]
# DEFAULT_METHODS = ["dift_sd", ]
DEFAULT_METHODS = ["color_rgb","color_hsv", "color_lch", "clip", "dino", "dift_sd", ]
# DEFAULT_METHODS = ["dift_sd"]
DATASETS_DIR = Path(__file__).parent.parent / "data" / "datasets"

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
        embs = engine.estimate_embeddings(entries, method=m)
        # Persist via the same cache path the server expects
        import numpy as np
        paths = np.array([e.path for e in entries])
        mtimes = np.array([int(Path(p).stat().st_mtime) if Path(p).exists() else 0 for p in paths], dtype=np.int64)
        
        if m == 'dift_sd':
            n_parts = embs.shape[2]
            for i in range(n_parts):
                for j in range(n_parts):
                    cache = engine._cache_dir() / f"embeddings_{m.lower()}_part{i}{j}.npz"
                    np.savez_compressed(cache, paths=paths, mtimes=mtimes, embeddings=embs[:,:,i,j])
        else:
            cache = engine._cache_dir() / f"embeddings_{m.lower()}.npz"
            np.savez_compressed(cache, paths=paths, mtimes=mtimes, embeddings=embs)
        print(" done")
        ok.append(m)

    print()
    print(f"Completed. OK={ok}")
    if fail:
        print("Failures:")
        for m, err in fail:
            print(f"  - {m}: {err}")


def main():
    # for dataset in DATASETS_DIR.iterdir():
    for dataset in ['../data/datasets/EmoSet', '../data/datasets/VIS30KGUI']:
        dataset = Path(dataset)
        # if dataset.name in ['VIS30K', 'ImageNet_R', 'ImageNet']:
        #     print(f"Skipping dataset: {dataset}")
        #     continue
        if not dataset.is_dir():
            continue
        print(f"Dataset: {dataset.name}")
        try :
            precompute(dataset, DEFAULT_METHODS)
        except Exception as e:
            print(f"Error processing dataset {dataset.name}: {e}")
    print("All done.")

if __name__ == "__main__":
    main()
