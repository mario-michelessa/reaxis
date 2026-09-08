# Reaxis Backend

The backend is a Flask development service for prepared image galleries, metadata axes, semantic-axis creation/refinement, session libraries, and local research logging.

Use the repository-level setup and configuration instructions in [`../README.md`](../README.md). Dataset acquisition and cache structure are documented in [`../data/README.md`](../data/README.md).

## Run

From the repository root:

```bash
conda activate scaledit
make backend
```

`backend/runtime_config.py` reads machine-specific paths and process settings from `.env`. The default bind address is `0.0.0.0:5001`; set `REAXIS_FLASK_DEBUG=false` outside local development.

## Data Contract

The backend scans `REAXIS_DATASETS_ROOT`. Clients pass dataset names such as `EmoSet`, not filesystem paths. The browser currently starts with normalized CLIP layout embeddings and creates axes with raw CLIP, so prepare `embeddings_clip.npz`, `coords_pca2d_clip.npz`, and `embeddings_clip_raw.npz`.

```bash
python -m backend.precompute_embeddings /path/to/prepared-dataset --methods clip --reduction pca
python -m backend.precompute_embeddings /path/to/prepared-dataset --methods clip --raw
make doctor
```

## Endpoint Groups

- Gallery: `/health`, `/datasets`, `/gallery.json`, `/images/...`, `/thumb/...`
- Metadata: `/metadata/summary`, `/metadata/upload`, `/analysis/recommend_scatterplots`
- Axis refinement: `/axis/create`, `/axis/move`, `/axis/update_prompts`
- Saved state: `/axis/library...`, `/visualization/library...`, `/subset/library...`
- Sessions: `/sessions`, `/sessions/open`, `/session/log`
- Language/zero-shot: `/llm/extract_attributes`, `/llm/suggest_values`, `/llm/attribute_distribution`
- Local writes: `/upload`, `/artifacts/export`

The service has no authentication and is not intended for exposure to untrusted networks. See [`../docs/REPOSITORY_AUDIT.md`](../docs/REPOSITORY_AUDIT.md) for the endpoint/path and security audit.

## Source Boundaries

- Live runtime modules remain directly under `backend/`.
- Ordinal modeling code is under `backend/ordinal_study/`.
- Offline participant/session analysis was moved to `postprocessing/user_study/` and is not imported by Flask.
- Generated results belong under `REAXIS_OUTPUT_ROOT` and are ignored by Git.
