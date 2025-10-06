# Simple Image Gallery Backend

Minimal Flask backend that:
- Lists images in a dataset and computes lightweight embeddings (mean RGB) or optional CLIP if available.
- Reduces embeddings to 2D (PCA by default; UMAP if installed) and snaps them to a grid.
- Serves a JSON gallery payload and the corresponding image files.
- Accepts image uploads into the dataset.

This backend is designed to power the existing frontend without changing its design.

## Endpoints

- `GET /health` — quick health check
- `GET /gallery.json?dataset=PATH&method=umap|pca&embed=avg|clip|dino|dift_sd&n_layer=64&n_tile=8` — returns items with `{ id, url, className, x, y, gx, gy }`
- `GET /images/<path>` — serves images relative to the selected dataset
- `POST /upload` — multipart form upload with fields: `file` (required), `class` (optional subfolder), `dataset` (optional root). Returns `{ ok, path }`.

Notes:
- If `dataset` is not provided in `GET /gallery.json`, the server uses `backend/uploads`.
- The server expects precomputed embeddings and will not compute them on demand. If missing, it responds with a warning and no items. See Precompute section below.

## Run Locally

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py  # listens on http://127.0.0.1:5001
```

Configure dataset root:
- Edit `backend/server.py` and set the `DATASET_PATH` constant to your dataset folder. The server always uses this path and ignores any `dataset` query parameter.

Environment variables:
- `HOST` (default `127.0.0.1`)
- `PORT` (default `5001`)

## Dataset Layout

Any folder tree containing images with extensions: `.jpg, .jpeg, .png, .bmp, .tiff, .webp`.
The immediate parent folder name is used as the `className`.

Included sample images live in `backend/uploads`.

## Precompute Embeddings (required)

Before starting the server, precompute embeddings for your dataset. This writes caches the server will load directly:

```bash
python backend/precompute_embeddings.py /path/to/dataset --methods avg,clip,dino,dift_sd
```

Caches are stored under `/path/to/dataset/.cache/embeddings_<method>.npz`.

## Dev Notes

- Dependencies are minimal: `numpy`, `pillow`, `Flask`, `flask-cors`.
- Optional extras (not required): `umap-learn`, `torch`, `transformers`.
- Optional extras (not required): `umap-learn`, `torch`, `transformers`, `timm`, `torchvision`, `diffusers`.
 - If `clip.py`, `dino.py`, or `dift_sd.py` are present in `backend/`, the server will attempt to use their featurizers first (requires GPU and locally available model weights). If they fail to initialize, it falls back gracefully to the lightweight avg embedding.

### Enable advanced embeddings

Install any of the optional packages to unlock models:

- CLIP (backend/clip.py): `pip install torch transformers open-clip-torch` if your local featurizer requires it; otherwise ensure backend/clip.py dependencies are satisfied locally.
- DINO (backend/dino.py): `pip install torch timm torchvision`
- DiFT/SD (backend/dift_sd.py): `pip install torch diffusers`
