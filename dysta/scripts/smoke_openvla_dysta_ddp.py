"""Two-GPU integration smoke test for the patched OpenVLA + DySta path.

The test builds a tiny random-weight OpenVLA model, so it validates the real
OpenVLA classes without requiring the 7B checkpoint. The visual token count,
DySta partition, gate architecture, and auxiliary-loss weights retain the
paper settings.
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from peft import LoraConfig, get_peft_model

from dysta.config import GateConfig, LossConfig, TokenLayout
from dysta.multiframe import DyStaMultiFrameAdapter
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction


PHYSICAL_GPUS = "6,7"
TOKENS_PER_FRAME = 256
NUM_FRAMES = 2
EXPECTED_CONTEXT_TOKENS = 133 + 107 + NUM_FRAMES * 16


def build_tiny_openvla() -> OpenVLAForActionPrediction:
    config = OpenVLAConfig(
        vision_backbone_id="dinosiglip-vit-so-224px",
        llm_backbone_id="llama2-7b-pure",
        text_config={
            "vocab_size": 192,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 1024,
        },
        llm_max_length=1024,
        pad_token_id=0,
        pad_to_multiple_of=64,
        _attn_implementation="eager",
    )

    # A 256x256 image with patch size 16 produces the paper's 256 tokens.
    config.timm_model_ids = ["vit_tiny_patch16_224", "vit_tiny_patch16_224"]
    config.timm_override_act_layers = [None, None]
    config.image_sizes = [256, 256]

    model = OpenVLAForActionPrediction(config)
    model.vision_backbone.set_num_images_in_input(NUM_FRAMES)
    model.dysta_adapter = DyStaMultiFrameAdapter(
        TokenLayout(total_tokens=256, static_tokens=(133, 107), dynamic_tokens=16),
        GateConfig(
            embedding_dim=model.llm_dim,
            num_tokens=TOKENS_PER_FRAME,
            num_levels=2,
            thresholds=(0.8, 0.4),
        ),
        LossConfig(
            alpha_levels=(0.2, 0.1),
            beta_gate=0.1,
            infonce_temperature=0.07,
            gate_lambda=0.1,
        ),
        num_frames=NUM_FRAMES,
        frame_interval=20,
    )
    return model


def run_smoke() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != PHYSICAL_GPUS:
        raise RuntimeError(f"set CUDA_VISIBLE_DEVICES={PHYSICAL_GPUS} before launching")
    if not dist.is_initialized():
        raise RuntimeError("the distributed process group is not initialized")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"expected two DDP processes, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(260203983)
    torch.cuda.manual_seed_all(260203983)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = build_tiny_openvla().to(device=device, dtype=torch.bfloat16)
    dysta_adapter = model.dysta_adapter
    model.dysta_adapter = None
    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            init_lora_weights="gaussian",
        ),
    )
    # Match finetune.py: attach DySta after PEFT has frozen the base model.
    model.model.dysta_adapter = dysta_adapter
    model.train()
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )

    # Different samples per rank ensure that matching gradients really result
    # from DDP all-reduce rather than identical local computation.
    generator = torch.Generator(device=device).manual_seed(260203983 + rank)
    batch_size = 2
    input_ids = torch.randint(3, 128, (batch_size, 12), generator=generator, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    pixel_values = torch.randn(
        batch_size,
        NUM_FRAMES * 6,
        256,
        256,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    output = ddp_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    if output.projector_features.shape != (batch_size, EXPECTED_CONTEXT_TOKENS, model.llm_dim):
        raise AssertionError(f"unexpected DySta context shape: {tuple(output.projector_features.shape)}")
    if output.dysta_gate_probabilities.shape != (batch_size, NUM_FRAMES - 1, 2):
        raise AssertionError(f"unexpected gate shape: {tuple(output.dysta_gate_probabilities.shape)}")

    total_loss = output.loss + output.dysta_auxiliary_loss
    if not torch.isfinite(total_loss):
        raise AssertionError(f"non-finite total loss: {total_loss}")
    total_loss.backward()

    gate_weight = ddp_model.module.dysta_adapter.gate.position_mlp[0].weight
    if gate_weight.grad is None or not torch.isfinite(gate_weight.grad).all():
        raise AssertionError("DySta gate did not receive a finite gradient")
    grad_checksum = gate_weight.grad.float().sum()
    gathered = [torch.zeros_like(grad_checksum) for _ in range(world_size)]
    dist.all_gather(gathered, grad_checksum)
    if not torch.allclose(gathered[0], gathered[1], rtol=1e-4, atol=1e-4):
        raise AssertionError(f"DDP gradient mismatch: {[value.item() for value in gathered]}")

    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "physical_gpus": [6, 7],
                    "world_size": world_size,
                    "gpu_name": torch.cuda.get_device_name(local_rank),
                    "dtype": str(next(model.parameters()).dtype),
                    "tokens_per_frame": TOKENS_PER_FRAME,
                    "frames": NUM_FRAMES,
                    "context_tokens": output.projector_features.shape[1],
                    "task_loss": float(output.loss.detach()),
                    "dysta_infonce_loss": float(output.dysta_infonce_loss.detach()),
                    "dysta_gate_loss": float(output.dysta_gate_loss.detach()),
                    "dysta_auxiliary_loss": float(output.dysta_auxiliary_loss.detach()),
                    "ddp_gradient_checksum": float(grad_checksum.detach()),
                },
                sort_keys=True,
            )
        )



def main() -> None:
    # Importing Prismatic under torchrun initializes Accelerate PartialState,
    # which may create the default process group before this script starts.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    try:
        run_smoke()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
