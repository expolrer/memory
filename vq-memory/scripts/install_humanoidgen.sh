#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/VQ-Memory
ENV_PREFIX="$PROJECT/.conda/vqmemory"
CACHE_ROOT=/root/autodl-tmp/.cache/vqmemory
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
mkdir -p "$PIP_CACHE_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

python -m pip install numpy==1.23.5
python -m pip install torchvision==0.17.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "$PROJECT/third_party/HumanoidGen/requirements.txt" \
  -c "$PROJECT/constraints-cu118.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install -e "$PROJECT/third_party/HumanoidGen"
python -m pip install -e "$PROJECT/third_party/HumanoidGen/third_party/pytorch3d_simplified"

python - <<'PY'
import importlib.util
mods = ['mani_skill', 'sapien', 'mplib', 'open3d', 'zarr', 'humanoidgen']
for mod in mods:
    print(mod, bool(importlib.util.find_spec(mod)))
PY
