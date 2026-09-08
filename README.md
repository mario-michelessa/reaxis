# Reaxis

Reaxis allows to define and refine semantic axes over image collections. 
The Svelte interface displays an image projection; the Flask backend serves prepared datasets, metadata, embedding layouts, language-model suggestions, and Bayesian axis updates.

## Architecture

```text
frontend/                    Svelte 4 + Vite interface
backend/                     Flask API, embeddings, AxisBayes, dataset importers
backend/ordinal_study/       ordinal modeling and evaluation pipeline
scripts/                     small experiment launchers
tests/                       backend unit tests
data/README.md               dataset acquisition and preparation contract
```

## Installation

```bash
conda env create -f environment.yml
conda activate scaledit
npm --prefix frontend ci
cp .env.example .env
```

## Dataset Preparation

Obtain a source dataset, prepare a flat UI dataset under `REAXIS_DATASETS_ROOT`, and precompute at least the CLIP embedding and PCA coordinate caches.

Four curated sources can be downloaded directly:

```bash
python -m backend.import_curated_dataset --list-presets
python -m backend.import_curated_dataset \
  --preset imagenette_160 \
  --name Imagenette1500 \
  --limit 1500 \
  --methods clip \
  --reduction pca
```

The ordinal datasets are manual downloads:

```bash
python -m backend.import_ordinal_datasets --only utkface koniq10k
python -m backend.import_expanded_ordinal_datasets --only scut aadb lamem oasis house
```

Generate or repair the normalized layout caches and the raw CLIP axis cache with:

```bash
python -m backend.precompute_embeddings /path/to/prepared-dataset \
  --methods clip,siglip2 \
  --reduction pca,umap
python -m backend.precompute_embeddings /path/to/prepared-dataset \
  --methods clip \
  --raw
```

Run the Interface

```bash
conda activate scaledit
make dev
```

Ordinal Modeling Study

Main entrypoint:

```bash
python -m backend.run_ordinal_modeling_study --help
```
