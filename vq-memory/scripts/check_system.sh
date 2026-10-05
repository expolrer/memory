#!/usr/bin/env bash
set -euo pipefail

echo "== OS =="
lsb_release -ds || cat /etc/os-release

echo "== GPU =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true

echo "== CUDA =="
ls -ld /usr/local/cuda* 2>/dev/null || true
/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -n 1 || true

echo "== Python/Conda =="
/root/miniconda3/bin/conda --version || true
/root/miniconda3/bin/python --version || true

echo "== Disk =="
df -h / /root/autodl-tmp || true
