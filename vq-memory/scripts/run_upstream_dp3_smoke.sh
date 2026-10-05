#!/usr/bin/env bash
set -euo pipefail

DP3_ROOT=${DP3_ROOT:-third_party/3D-Diffusion-Policy/3D-Diffusion-Policy}
cd "$DP3_ROOT"

WANDB_MODE=${WANDB_MODE:-disabled} python train.py \
  --config-name=simple_dp3 \
  task=rulesafe_smoke \
  name=rulesafe_simple_dp3_smoke \
  exp_name=rulesafe_smoke \
  horizon=4 \
  n_obs_steps=2 \
  n_action_steps=2 \
  policy.use_point_crop=false \
  training.device=cuda:0 \
  training.num_epochs=1 \
  training.max_train_steps=2 \
  training.use_ema=false \
  training.resume=false \
  training.rollout_every=999999 \
  training.checkpoint_every=999999 \
  training.val_every=999999 \
  training.sample_every=999999 \
  dataloader.batch_size=2 \
  dataloader.num_workers=0 \
  dataloader.persistent_workers=false \
  val_dataloader.batch_size=2 \
  val_dataloader.num_workers=0 \
  val_dataloader.persistent_workers=false \
  logging.mode=disabled \
  checkpoint.save_ckpt=false
