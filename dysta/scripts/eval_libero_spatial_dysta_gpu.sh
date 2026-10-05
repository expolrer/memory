#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-6}"
TASK_ID="${TASK_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-1}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"

case "${TASK_SUITE_NAME}" in
  libero_spatial) DATASET_NAME=libero_spatial_no_noops ;;
  libero_object) DATASET_NAME=libero_object_no_noops ;;
  libero_goal) DATASET_NAME=libero_goal_no_noops ;;
  libero_10) DATASET_NAME=libero_10_no_noops ;;
  *)
    printf 'Unsupported LIBERO task suite: %s\n' "${TASK_SUITE_NAME}" >&2
    exit 2
    ;;
esac

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH=/ssd/DySta/dysta_reproduction/configs/libero
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONHASHSEED=7

PYTHON=/ssd/DySta/.venv-openvla/bin/python
OPENVLA_ROOT=/ssd/DySta/third_party/external_sources/openvla-oft
BASE_MODEL=/ssd/DySta/models/openvla-7b
CHECKPOINT="${CHECKPOINT:-/ssd/DySta/runs/openvla-7b+${DATASET_NAME}+gb64+lr-0.0005+seed-42+lora-r32+dropout-0.0--image_aug--dysta-paper-gpu67--15000_chkpt}"
LOG_DIR="${LOG_DIR:-/ssd/DySta/eval-logs/${TASK_SUITE_NAME//_/-}}"

if [[ ! -d "${CHECKPOINT}" ]]; then
  printf 'Missing final checkpoint: %s\n' "${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
cd "${OPENVLA_ROOT}"
"${PYTHON}" experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --base_model_path "${BASE_MODEL}" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --use_dysta True \
  --dysta_history_frames 2 \
  --dysta_history_interval 1 \
  --center_crop True \
  --num_open_loop_steps 8 \
  --lora_rank 32 \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --task_id "${TASK_ID}" \
  --num_trials_per_task "${NUM_TRIALS}" \
  --num_steps_wait 10 \
  --env_img_res 256 \
  --local_log_dir "${LOG_DIR}" \
  --raise_episode_errors True \
  --use_wandb False \
  --seed 7 \
  --run_id_note "dysta-${TASK_SUITE_NAME}-task${TASK_ID}-gpu${GPU_ID}"
