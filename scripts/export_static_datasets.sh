#!/usr/bin/env bash
set -euo pipefail

DATASETS_ROOT="${REAXIS_DATASETS_ROOT:-data/datasets}"
STATIC_ROOT="${REAXIS_STATIC_EXPORT_ROOT:-frontend/public/datasets}"
DATASETS=(ISIC2017 VIS30KGUI EmoSet)

for dataset in "${DATASETS[@]}"; do
  source_dir="${DATASETS_ROOT}/${dataset}"
  target_dir="${STATIC_ROOT}/${dataset}"
  python -m backend.gallery_backend \
    "${source_dir}" \
    --export_all_dir "${target_dir}" \
    --methods "" \
    --method pca \
    --default_method color_rgb
  python -m backend.export_metadata "${source_dir}" "${target_dir}"
done
