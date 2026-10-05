#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=6
export LIBERO_CONFIG_PATH=/ssd/DySta/dysta_reproduction/configs/libero
export PYOPENGL_PLATFORM=egl

cd /ssd/DySta/dysta_reproduction
/ssd/DySta/.venv-openvla/bin/python scripts/smoke_libero_env.py
