#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/VQ-Memory
CONDA=/root/miniconda3/bin/conda
ENV_PREFIX="$PROJECT/.conda/vqmemory"
CACHE_ROOT=/root/autodl-tmp/.cache/vqmemory
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda_pkgs"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

if [ ! -x "$CONDA" ]; then
  echo "conda not found at $CONDA" >&2
  exit 1
fi

if [ ! -d "$ENV_PREFIX" ]; then
  "$CONDA" create -y -p "$ENV_PREFIX" python=3.9 pip
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
python -m pip install --upgrade pip
python -m pip install torch==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "$PROJECT/requirements-minimal.txt"

python - <<'PY'
import torch
print('python_ok')
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
PY
