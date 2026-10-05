# HumanoidGen Installation Notes

HumanoidGen is the closest open reference for the RuleSafe data generation pipeline. The cloned commit in this workspace is recorded by `git -C third_party/HumanoidGen rev-parse --short HEAD`.

## Why this environment is separate but shared

The paper-style reproduction needs both VQ-Memory and HumanoidGen in one Python environment so generated trajectories can be tokenized without format conversion friction. The environment lives at:

```bash
/root/autodl-tmp/VQ-Memory/.conda/vqmemory
```

## Version pins

- Python 3.9: required by HumanoidGen README.
- PyTorch 2.2.2 + CUDA 11.8: compatible with the server CUDA stack and 4090D driver.
- torchvision 0.17.2 + CUDA 11.8: matched to PyTorch 2.2.2.
- NumPy 1.23.5: compatible with `numba==0.56.4` from HumanoidGen requirements.

## Install command

```bash
bash scripts/install_humanoidgen.sh
```

## Required manual asset step

HumanoidGen requires assets from HuggingFace:

- `TeleEmbodied/humanoidgen_dataset/assets/assets.zip`
- `TeleEmbodied/humanoidgen_dataset/assets/table_assets.zip`

Expected extraction targets according to HumanoidGen README:

- `third_party/HumanoidGen/assets`
- `third_party/HumanoidGen/humanoidgen/scene_builder/table/assets`

## Required mplib patch

After `mplib` is installed, HumanoidGen asks to change `n_init_qpos` in `mplib/planner.py` from 20 to 50. A later script should locate it with:

```bash
python -c "import mplib; print(mplib.planner.__file__)"
```
