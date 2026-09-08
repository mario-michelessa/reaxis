# Reaxis

Reaxis allows to define and refine semantic axes over image collections. 
The Svelte interface displays an image projection; the Flask backend serves prepared datasets, metadata, embedding layouts, language-model suggestions, and Bayesian axis updates.

## Architecture

```text
frontend/                    Svelte 4 + Vite interface
backend/                     Flask API, embeddings, AxisBayes, dataset importers
backend/ordinal_study/       ordinal modeling and evaluation pipeline
postprocessing/user_study/  offline user-study analysis and figure source
scripts/                     small experiment launchers
tests/                       backend unit tests
data/README.md               dataset acquisition and preparation contract
outputs/                     generated results; ignored
```

Runtime and analysis are deliberately separated. Nothing under `postprocessing/` is imported by the live backend.

## Installation

```bash
conda env create -f environment.yml
conda activate scaledit
npm --prefix frontend ci
cp .env.example .env
```

`make install` can be used to refresh Python development and frontend dependencies after the environment exists.

## Configuration

All machine-dependent roots and server settings are defined by environment variables read in [`backend/runtime_config.py`](backend/runtime_config.py). Direct Python commands and `make` load the same root `.env`; existing shell environment variables take precedence.

| variable | default | purpose |
| --- | --- | --- |
| `REAXIS_DATA_ROOT` | `data` | parent for local runtime data |
| `REAXIS_DATASETS_ROOT` | `data/datasets` | prepared UI datasets |
| `REAXIS_SESSIONS_ROOT` | `data/sessions` | session logs and saved axes/views |
| `REAXIS_UPLOADS_ROOT` | `data/uploads` | uploads when no active dataset exists |
| `REAXIS_OUTPUT_ROOT` | `outputs` | generated analyses and exports |
| `REAXIS_RAW_DATASETS_ROOT` | `data/raw` | manually downloaded source datasets |
| `REAXIS_DATASYNTH_ROOT` | `data/raw` | alternate root for legacy local curated presets |
| `REAXIS_DEFAULT_DATASET` | `ISIC2017` | backend dataset when no name is supplied |
| `REAXIS_BACKEND_HOST` | `0.0.0.0` | Flask bind host |
| `REAXIS_BACKEND_PORT` | `5001` | Flask port |
| `REAXIS_FLASK_DEBUG` | `true` | Flask development debugger/reloader |
| `REAXIS_LLM_PROVIDER` | `gemini_api` | `gemini_api` or `huggingface_local` |
| `REAXIS_GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini model identifier |
| `REAXIS_GEMINI_API_KEY_FILE` | `data/secrets/gemini_api_key.txt` | ignored text file containing the API key |
| `REAXIS_HF_MODEL_PATH` | empty | local Hugging Face model directory |

## Dataset Preparation

Obtain a source dataset, prepare a flat UI dataset under `REAXIS_DATASETS_ROOT`, and precompute at least the CLIP embedding and PCA coordinate caches.

Four curated sources can be downloaded directly by the importer:

```bash
python -m backend.import_curated_dataset --list-presets
python -m backend.import_curated_dataset \
  --preset imagenette_160 \
  --name Imagenette1500 \
  --limit 1500 \
  --methods clip \
  --reduction pca
```

The ordinal datasets are manual downloads because several require registration or impose non-redistribution terms:

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

Validate configuration and cache availability without starting the service:

```bash
make doctor
```

## Run the Interface

Start Flask and Vite together:

```bash
conda activate scaledit
make dev
```

## Ordinal Modeling Study

See [`backend/ordinal_study/README.md`](backend/ordinal_study/README.md).

Main entrypoint:

```bash
python -m backend.run_ordinal_modeling_study --help
```
