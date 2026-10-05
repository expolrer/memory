from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from dysta import GateConfig, SharedRecacheGate, gate_prior_loss, tokenwise_infonce


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"expected world_size=2, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(2026 + rank)

    config = GateConfig(
        embedding_dim=64,
        num_tokens=32,
        num_levels=2,
        position_hidden_dims=(96, 48),
        position_out_dim=16,
        token_hidden_dims=(32, 16),
        token_out_dim=8,
        thresholds=(0.8, 0.4),
    )
    gate = DistributedDataParallel(SharedRecacheGate(config).to(device), device_ids=[local_rank])
    references = torch.randn(4, 2, 32, 64, device=device)
    current = torch.randn(4, 32, 64, device=device, requires_grad=True)
    output = gate(references, current, inference=False, hard=False)
    deltas = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], device=device)
    contrastive = tokenwise_infonce(current[:, :8], references[:, 0, :8], temperature=0.1)
    loss = contrastive + 0.1 * gate_prior_loss(output.probabilities, deltas, prior_lambda=0.1)
    loss.backward()

    finite = torch.tensor(
        float(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in gate.parameters())),
        device=device,
    )
    loss_sum = loss.detach().clone()
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "ok" if finite.item() == 1.0 else "non_finite_gradient",
                    "world_size": world_size,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "logical_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    "mean_loss": loss_sum.item() / world_size,
                },
                indent=2,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
