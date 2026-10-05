# Reproduction Roadmap

## Phase 0: System audit

Confirm GPU, CUDA, conda, disk space, and project layout. This server has Ubuntu 20.04.6, CUDA 11.8 directories, and 4 RTX 4090D GPUs.

## Phase 1: Minimal benchmark

Implement a RuleSafe-like environment with one safe object containing:

- `knob`: rotatable component.
- `handle`: open/close component.
- `door`: locked or unlocked articulated door.

Start with two rules:

- `rule001`: the door unlocks after both knob and handle are open.
- `rule020`: the door unlocks after password `11` is entered through knob interactions while the handle controls recording/unlocking.

This phase can be unit-tested without SAPIEN by using symbolic phase transitions.

## Phase 2: Demonstration generation

Rebuild the paper's demonstration pipeline:

1. LLM-style rule description and executable checker.
2. Task decomposition into atomic hand operations and arm motions.
3. OMPL motion planning for reaching interaction poses.
4. SAPIEN execution and logging.

Data schema is defined in `data_generation/schema.py`.

## Phase 3: VQ-Memory tokenizer

Train a VQ-VAE on historical joint-state windows:

- Input: joint states with window size 50.
- Encoder: continuous latent representation.
- Vector quantization: nearest entry in a 256-vector codebook.
- Decoder: reconstructs the joint-state window.
- Post-hoc K-means: compress 256 learned codes into 4 semantic memory tokens.

## Phase 4: Policy integration

Recommended order:

1. DP3 adapter: easier first baseline, memory embedded by MLP/CNN.
2. pi0 adapter: memory as special language tokens.
3. RDT/CogACT adapters: same token injection idea as pi0.

## Phase 5: Evaluation

Report:

- Success Rate: percentage of complete task success.
- Process Score: percentage of correct intermediate stages.
- Ablation: cluster count 256/32/4/2 and memory length 20/40/60.
