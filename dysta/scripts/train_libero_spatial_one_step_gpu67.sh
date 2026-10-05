#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_MODE=offline
export HF_HOME=/ssd/DySta/hf_cache
export PYTHONHASHSEED=42

PYTHON=/ssd/DySta/.venv-openvla/bin/python
OPENVLA_ROOT=/ssd/DySta/third_party/external_sources/openvla-oft

cd "${OPENVLA_ROOT}"
"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  vla-scripts/finetune.py \
  --vla_path /ssd/DySta/models/openvla-7b \
  --data_root_dir /ssd/DySta/datasets/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /ssd/DySta/runs-smoke \
  --shuffle_buffer_size 256 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --use_dysta True \
  --dysta_history_frames 2 \
  --dysta_history_interval 1 \
  --batch_size 2 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 1 \
  --max_steps 1 \
  --seed 42 \
  --save_freq 100000 \
  --image_aug False \
  --use_lora True \
  --lora_rank 32 \
  --lora_dropout 0.0 \
  --merge_lora_during_training False \
  --wandb_log_freq 1 \
  --run_id_note dysta-one-step-gpu67
