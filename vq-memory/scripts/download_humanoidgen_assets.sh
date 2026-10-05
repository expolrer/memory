#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/VQ-Memory
ENV_PREFIX="$PROJECT/.conda/vqmemory"
DOWNLOAD_DIR="$PROJECT/downloads/humanoidgen_assets"
HG_ROOT="$PROJECT/third_party/HumanoidGen"
mkdir -p "$DOWNLOAD_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = 'TeleEmbodied/humanoidgen_dataset'
files = ['assets/assets.zip', 'assets/table_assets.zip']
download_dir = Path('/root/autodl-tmp/VQ-Memory/downloads/humanoidgen_assets')
download_dir.mkdir(parents=True, exist_ok=True)
for filename in files:
    print('downloading', filename)
    path = hf_hub_download(repo_id=repo, repo_type='dataset', filename=filename, local_dir=download_dir)
    print(path)
PY

mkdir -p "$HG_ROOT" "$HG_ROOT/humanoidgen/scene_builder/table"
if [ -f "$DOWNLOAD_DIR/assets/assets.zip" ]; then
  rm -rf "$HG_ROOT/assets"
  unzip -q "$DOWNLOAD_DIR/assets/assets.zip" -d "$HG_ROOT"
  if [ -d "$HG_ROOT/assets/assets" ]; then mv "$HG_ROOT/assets/assets" "$HG_ROOT/assets_tmp" && rm -rf "$HG_ROOT/assets" && mv "$HG_ROOT/assets_tmp" "$HG_ROOT/assets"; fi
fi
if [ -f "$DOWNLOAD_DIR/assets/table_assets.zip" ]; then
  rm -rf "$HG_ROOT/humanoidgen/scene_builder/table/assets"
  unzip -q "$DOWNLOAD_DIR/assets/table_assets.zip" -d "$HG_ROOT/humanoidgen/scene_builder/table"
  if [ -d "$HG_ROOT/humanoidgen/scene_builder/table/assets/assets" ]; then
    mv "$HG_ROOT/humanoidgen/scene_builder/table/assets/assets" "$HG_ROOT/humanoidgen/scene_builder/table/assets_tmp"
    rm -rf "$HG_ROOT/humanoidgen/scene_builder/table/assets"
    mv "$HG_ROOT/humanoidgen/scene_builder/table/assets_tmp" "$HG_ROOT/humanoidgen/scene_builder/table/assets"
  fi
fi

find "$HG_ROOT/assets" "$HG_ROOT/humanoidgen/scene_builder/table/assets" -maxdepth 2 -type d 2>/dev/null | sed -n '1,40p'
