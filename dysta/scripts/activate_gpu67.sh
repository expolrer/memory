#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT="/ssd/DySta"
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

echo "Physical GPUs 6,7 are exposed as process-local cuda:0,cuda:1"
