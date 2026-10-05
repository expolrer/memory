from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class PolicyBatchBuilder:
    joint_dim: int = 13
    memory_mode: str = "none"
    raw_memory_steps: int = 40
    vq_memory_length: int = 40
    vq_vocab_size: int = 4

    @property
    def input_dim(self) -> int:
        if self.memory_mode == "none":
            return self.joint_dim
        if self.memory_mode == "raw":
            return self.joint_dim + self.raw_memory_steps * self.joint_dim
        if self.memory_mode == "vq":
            return self.joint_dim + self.vq_memory_length
        raise ValueError(f"unknown memory_mode={self.memory_mode!r}")

    def make_features(self, current_state: torch.Tensor, raw_memory: Optional[torch.Tensor] = None, vq_tokens: Optional[torch.Tensor] = None):
        if self.memory_mode == "none":
            return current_state
        if self.memory_mode == "raw":
            if raw_memory is None:
                raise ValueError("raw_memory is required for raw mode")
            return torch.cat([current_state, raw_memory.reshape(raw_memory.shape[0], -1)], dim=-1)
        if self.memory_mode == "vq":
            if vq_tokens is None:
                raise ValueError("vq_tokens is required for vq mode")
            # Normalize token ids into a compact numeric conditioning vector.
            denom = max(1, self.vq_vocab_size - 1)
            return torch.cat([current_state, vq_tokens.float() / denom], dim=-1)
        raise ValueError(f"unknown memory_mode={self.memory_mode!r}")


class StateActionPolicy(nn.Module):
    """Small state-only action expert used to verify memory injection plumbing.

    This is not a replacement for DP3/pi0/RDT. It is a cheap post-training
    adapter smoke test: the same data path later feeds the real policy models.
    """

    def __init__(self, input_dim: int, action_dim: int = 13, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
