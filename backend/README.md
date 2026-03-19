# Simple Image Gallery Backend

Minimal Flask backend that:
- Lists images in a dataset and serves precomputed feature spaces, with `siglip2` as the default semantic embedding.
- Reduces embeddings to 2D (PCA by default; UMAP if installed) and snaps them to a grid.
- Serves a JSON gallery payload and the corresponding image files.
- Accepts image uploads into the dataset.

This backend is designed to power the existing frontend without changing its design.

## Endpoints

- `GET /health` — quick health check
- `GET /gallery.json?dataset=PATH&method=umap|pca&embed=color_rgb|siglip2|clip|dino|dift_sd&n_layer=64&n_tile=8` — returns items with `{ id, url, className, x, y, gx, gy }`
- `GET /images/<path>` — serves images relative to the selected dataset
- `POST /upload` — multipart form upload with fields: `file` (required), `class` (optional subfolder), `dataset` (optional root). Returns `{ ok, path }`.
- `POST /llm/extract_attributes` — extract key measurable attributes from a prompt
- `POST /llm/suggest_values` — suggest ordered values (low→high) for one attribute
- `POST /llm/attribute_distribution` — zero-shot score all images in `[0,1]` for an open-ended attribute and return a distribution + axis

LLM endpoints body examples:

```json
{ "prompt": "compare age, texture, and symmetry", "dataset": "ISIC2017" }
```

```json
{ "attribute": "age", "context": "skin lesion images", "n_values": 5 }
```

```json
{
  "attribute": "age",
  "context": "dermoscopy images",
  "dataset": "ISIC2017",
  "n_values": 5
}
```

Notes:
- If `dataset` is not provided in `GET /gallery.json`, the server uses `backend/uploads`.
- The server expects precomputed embeddings and coordinates and will not compute them on demand.
- `siglip2` is the default semantic embedding. If a dataset has not been backfilled with `siglip2` caches yet, the server falls back to `clip` and returns a warning in the gallery payload.

## Run Locally

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py  # listens on http://127.0.0.1:5002
```

Configure dataset root:
- Edit `backend/server.py` and set the `DATASET_PATH` constant to your dataset folder. The server always uses this path and ignores any `dataset` query parameter.

Runtime config:
- `HOST` and `PORT` can still be set via environment variables for Flask process binding.
- LLM and zero-shot configuration is hardcoded in `backend/constants.py`.

To use a local Hugging Face LLM:
1. Set `HF_LOCAL_MODEL_PATH` directly in `backend/constants.py`.
2. Keep `LLM_PROVIDER = 'huggingface_local'` in `backend/constants.py`.

## Dataset Layout

Any folder tree containing images with extensions: `.jpg, .jpeg, .png, .bmp, .tiff, .webp`.
The immediate parent folder name is used as the `className`.

Included sample images live in `backend/uploads`.

## Precompute Embeddings (required)

Before starting the server, precompute embeddings for your dataset. This writes caches the server will load directly:

```bash
python backend/precompute_embeddings.py
python backend/precompute_embeddings.py /path/to/dataset --methods color_rgb,siglip2,clip,dino,dift_sd
```

Caches are stored under `/path/to/dataset/.cache/embeddings_<method>.npz`.
With no positional dataset arguments, the script scans every directory under `data/datasets/`, defaults to `siglip2`, and skips datasets whose requested caches already exist unless `--include-complete` is passed.

## Import Curated Datasets

To prepare datasets for this UI, use the curated importer. It:

- downloads a source dataset or archive
- flattens images into `data/datasets/<name>/`
- writes `metadata.csv`
- precomputes embedding caches and PCA coordinates

Examples:

```bash
python backend/import_curated_dataset.py --preset stars_hubble --name HubbleStars --limit 1200
python backend/import_curated_dataset.py --preset paintings_wikiart --name WikiArt1500 --limit 1500
python backend/import_curated_dataset.py --preset imagenette_160 --name Imagenette1500 --limit 1500
python backend/import_curated_dataset.py --preset tiny_imagenet_200 --name TinyImageNet1500 --limit 1500
python backend/import_curated_dataset.py --preset inat2021birds_local --name inat2021birds --limit 1500
python backend/import_curated_dataset.py --preset mapillary_vistas_local --name MapillaryVistas --limit 1500 --max-source-pixels 20000000
python backend/import_curated_dataset.py --preset imagenet_n02958343_local --name ImageNet_n02958343 --limit 1500
python backend/import_local_curated_datasets.py --only broden1_224 CUB HAM10000 ImageNet_n02958343 ImageNet_R Imagenette1500 inat2021birds MapillaryVistas
```

Useful flags:

- `--methods color_rgb,siglip2,clip` to prepare the current UI feature spaces
- `--max-edge 512` to resize images for a lighter dataset
- `--max-source-pixels 80000000` to downscale very large source images before the normal import resize step
- `--overwrite` to replace an existing prepared dataset
- `--skip-precompute` if you only want the images and metadata first

Available presets:

- `stars_hubble`: ESA/Hubble imagery filtered toward star and stellar scenes
- `paintings_wikiart`: WikiArt paintings with artist/genre/style metadata
- `imagenette_160`: fast.ai Imagenette subset of ImageNet
- `tiny_imagenet_200`: Stanford Tiny-ImageNet-200

Selected local presets:

- `broden1_224`: Broden images plus annotation-availability flags from `index.csv`
- `cub_local`: CUB split/species folders
- `ham10000_local`: HAM10000 dermoscopy images plus lesion metadata
- `imagenet_n02958343_local`: flat local ImageNet subset for synset `n02958343`; keeps all available images when the source has fewer than the requested limit
- `imagenet_r_local`: ImageNet-R split/style/class tree
- `imagenette_local`: Imagenette `train` and `val` splits
- `inat2021birds_local`: bird images joined with scientific/common name, taxonomy, image size, rights-holder, and geolocation metadata
- `mapillary_vistas_local`: `training` and `validation` street scenes joined with `v2.0` semantic-mask summaries (`testing` is skipped because it has no labels)

The batch runner `backend/import_local_curated_datasets.py` contains the current curated local outputs and per-dataset overrides such as source roots, overwrite behavior, limits, and source-pixel ceilings.

## Dev Notes

- Dependencies are minimal: `numpy`, `pillow`, `Flask`, `flask-cors`.
- Optional extras (not required): `umap-learn`, `torch`, `transformers`.
- Optional extras (not required): `umap-learn`, `torch`, `transformers`, `timm`, `torchvision`, `diffusers`.
- If `clip.py`, `dino.py`, or `dift_sd.py` are present in `backend/`, the server will attempt to use their featurizers first where applicable (requires GPU and locally available model weights).

### Enable advanced embeddings

Install any of the optional packages to unlock models:

- SigLIP2 / CLIP text-image embeddings: `pip install torch transformers`
- CLIP (backend/clip.py): `pip install torch transformers open-clip-torch` if your local featurizer requires it; otherwise ensure backend/clip.py dependencies are satisfied locally.
- DINO (backend/dino.py): `pip install torch timm torchvision`
- DiFT/SD (backend/dift_sd.py): `pip install torch diffusers`
- Local HF LLM (`/llm/*` endpoints, 4-bit quantized): `pip install torch transformers accelerate bitsandbytes`
