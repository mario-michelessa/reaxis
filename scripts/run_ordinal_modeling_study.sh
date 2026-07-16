#!/usr/bin/env bash
set -euo pipefail

DATASETS="UTKFace,AffectNet,KonIQ10k"
METHODS="text_prior,prompt_ladder,ordinal_ridge,rank_svm,knn_ordinal,kernel_ridge,reaxis_random,reaxis_active"
BUDGETS="0,1,3,5,10,20"
REPETITIONS="5"
OUTPUT_DIR="backend/experiments/ordinal_modeling"

python backend/run_ordinal_modeling_study.py \
  --datasets "${DATASETS}" \
  --methods "${METHODS}" \
  --budgets "${BUDGETS}" \
  --repetitions "${REPETITIONS}" \
  --output-dir "${OUTPUT_DIR}"
