#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-data/rulesafe_multitask/rulesafe_20rules_50x.npz}
OUT=${OUT:-data/dp3_rulesafe_smoke/rulesafe_20rules_40eps_256pts.zarr}
POINT_COUNT=${POINT_COUNT:-256}
MAX_EPISODES=${MAX_EPISODES:-40}
DP3_ROOT=${DP3_ROOT:-third_party/3D-Diffusion-Policy/3D-Diffusion-Policy}

python data_generation/export_dp3_zarr.py \
  --npz "$DATASET" \
  --out "$OUT" \
  --point-count "$POINT_COUNT" \
  --max-episodes "$MAX_EPISODES" \
  --overwrite

python data_generation/validate_dp3_zarr.py \
  --zarr "$OUT" \
  --dp3-root "$DP3_ROOT"
