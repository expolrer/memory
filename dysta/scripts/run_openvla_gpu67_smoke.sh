#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3

cd /ssd/DySta/dysta_reproduction
/ssd/DySta/.venv-openvla/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  scripts/smoke_openvla_dysta_ddp.py
