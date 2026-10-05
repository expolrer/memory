#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/VQ-Memory
mkdir -p "$PROJECT/third_party"
cd "$PROJECT/third_party"

clone_if_missing() {
  local url="$1"
  local dir="$2"
  if [ ! -d "$dir/.git" ]; then
    git clone --depth 1 "$url" "$dir"
  else
    echo "exists: $dir"
  fi
}

clone_if_missing https://github.com/TeleHuman/HumanoidGen.git HumanoidGen
clone_if_missing https://github.com/YanjieZe/3D-Diffusion-Policy.git 3D-Diffusion-Policy
clone_if_missing https://github.com/Physical-Intelligence/openpi.git openpi
clone_if_missing https://github.com/thu-ml/RoboticsDiffusionTransformer.git RoboticsDiffusionTransformer
clone_if_missing https://github.com/microsoft/CogACT.git CogACT
