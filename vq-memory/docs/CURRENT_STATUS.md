# Current Reproduction Status

## Completed

- Created the reproduction workspace under `/root/autodl-tmp/VQ-Memory`.
- Created a Python 3.9 conda environment at `.conda/vqmemory`.
- Installed PyTorch 2.2.2 + CUDA 11.8; PyTorch detects 4 GPUs.
- Installed HumanoidGen dependencies, SAPIEN 3.0.1, ManiSkill 3.0.0b21, mplib, open3d, and pytorch3d_simplified.
- Cloned reference repositories under `third_party/`:
  - HumanoidGen
  - 3D-Diffusion-Policy
  - openpi
  - RoboticsDiffusionTransformer
  - CogACT
- Patched `mplib/planner.py` so `n_init_qpos=50`, matching HumanoidGen README.
- Implemented testable symbolic `rule001` and `rule020` state machines.
- Implemented VQ-Memory VQ-VAE, codebook clustering, and tokenizer scaffold.
- Added a minimal CPU-PhysX SAPIEN environment wrapper for RuleSafe symbolic tasks.

## Verified commands

```bash
cd /root/autodl-tmp/VQ-Memory
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/VQ-Memory/.conda/vqmemory
python -m pytest -q
python -m train.train_vqvae --config configs/vq_memory.yaml --dry-run
```

## Important SAPIEN note

In this container, `sapien.Scene()` without explicit systems can segfault. Use:

```python
import sapien
from sapien import physx
scene = sapien.Scene([physx.PhysxCpuSystem()])
```

This is encoded in `rulesafe/envs/sapien_safe_env.py`.

## Next required work

1. Download HumanoidGen assets from HuggingFace.
2. Run a native HumanoidGen demo task, likely with rendering disabled first.
3. Build real RuleSafe safe assets: knob, handle, door, lock state.
4. Replace symbolic events with SAPIEN articulated-body state transitions.
5. Generate first 100 demonstrations for `rule001` and `rule020`.
6. Connect generated trajectories to `train/train_vqvae.py`.
