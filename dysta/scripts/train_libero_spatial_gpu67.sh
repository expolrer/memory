#!/usr/bin/env bash
set -euo pipefail

PHYSICAL_GPUS="${PHYSICAL_GPUS:-6,7}"
if [[ ! "${PHYSICAL_GPUS}" =~ ^[0-9]+,[0-9]+$ ]]; then
  printf 'PHYSICAL_GPUS must contain exactly two comma-separated GPU IDs: %s\n' "${PHYSICAL_GPUS}" >&2
  exit 2
fi
IFS=',' read -r GPU_A GPU_B <<< "${PHYSICAL_GPUS}"
if [[ "${GPU_A}" == "${GPU_B}" ]]; then
  printf 'PHYSICAL_GPUS must contain two distinct GPU IDs: %s\n' "${PHYSICAL_GPUS}" >&2
  exit 2
fi
GPU_TAG="gpu${PHYSICAL_GPUS//,/}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPUS}"
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME=/ssd/DySta/hf_cache
export PYTHONHASHSEED=42

PYTHON=/ssd/DySta/.venv-openvla/bin/python
OPENVLA_ROOT=/ssd/DySta/third_party/external_sources/openvla-oft
DATA_ROOT=/ssd/DySta/datasets/modified_libero_rlds
DATASET_NAME="${DATASET_NAME:-libero_spatial_no_noops}"
RUN_ID_NOTE="${RUN_ID_NOTE:-dysta-paper-${GPU_TAG}}"

case "${DATASET_NAME}" in
  libero_spatial_no_noops|libero_object_no_noops|libero_goal_no_noops|libero_10_no_noops) ;;
  *)
    printf 'Unsupported LIBERO dataset: %s\n' "${DATASET_NAME}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${DATA_ROOT}/${DATASET_NAME}/1.0.0" ]]; then
  printf 'Missing TFDS dataset directory: %s\n' "${DATA_ROOT}/${DATASET_NAME}/1.0.0" >&2
  exit 2
fi

cd "${OPENVLA_ROOT}"
"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  vla-scripts/finetune.py \
  --vla_path /ssd/DySta/models/openvla-7b \
  --data_root_dir "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir /ssd/DySta/runs \
  --shuffle_buffer_size 4096 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --use_dysta True \
  --dysta_history_frames 2 \
  --dysta_history_interval 1 \
  --batch_size 4 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 15000 \
  --max_steps 15000 \
  --seed 42 \
  --save_freq 1000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --use_lora True \
  --lora_rank 32 \
  --lora_dropout 0.0 \
  --merge_lora_during_training False \
  --wandb_log_freq 10 \
  --run_id_note "${RUN_ID_NOTE}"
