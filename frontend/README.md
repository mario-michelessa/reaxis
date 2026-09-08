# Reaxis Frontend

The frontend is a Svelte 4 application built with Vite and UnoCSS. Use the root [`README.md`](../README.md) for complete installation, dataset, and launch instructions.

## Development

```bash
npm ci
npm run dev -- --host 0.0.0.0 --port 5174
```

In development, Vite proxies same-origin `/api` requests to `VITE_BACKEND_PROXY_TARGET`, defaulting to `http://127.0.0.1:5001`. Set `VITE_API_BASE` when the browser should call a separately hosted backend directly.

The root `make dev` command configures both services and enables polling to avoid Linux inotify exhaustion.

## Build

```bash
npm run build
```

The production bundle is written to `frontend/dist/`. Building has no deployment side effects; copy or deploy `dist/` explicitly in infrastructure outside this repository. Configure the production web server to proxy `/api` to Flask.

## Standalone Mode

Set `VITE_STANDALONE=1` or open `/?standalone=1`. Place generated galleries under `frontend/public/datasets/<name>/` and add `frontend/public/datasets/index.json`. This directory is ignored because it can contain third-party images and generated payloads.

To export the default static dataset set, edit `DATASETS` if needed and run:

```bash
scripts/export_static_datasets.sh
```

For a single prepared dataset:

```bash
python -m backend.gallery_backend data/datasets/MyDataset \
  --export_all_dir frontend/public/datasets/MyDataset \
  --method pca \
  --default_method clip
python -m backend.export_metadata \
  data/datasets/MyDataset \
  frontend/public/datasets/MyDataset
```

## Main Components

- `src/App.svelte`: application shell, dataset/session state, saved libraries
- `src/components/PromptSidebar.svelte`: question analysis and axis creation
- `src/components/AxisBuilder.svelte`: axis inspection and feedback
- `src/components/AxesMinimap.svelte`: projection, filtering, lasso, and local grids
- `src/lib/apiBase.js`: API base resolution
- `src/lib/sessionApi.js`: session logging calls
