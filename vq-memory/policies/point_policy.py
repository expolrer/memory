from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .state_policy import PolicyBatchBuilder


class PointNetEncoder(nn.Module):
    def __init__(self, point_dim: int = 6, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), nn.ReLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: [B, N, C]
        feat = self.point_mlp(points)
        return feat.max(dim=1).values


@dataclass
class PointPolicyFeatureBuilder:
    point_feature_dim: int = 256
    memory_mode: str = "none"
    raw_memory_steps: int = 40
    joint_dim: int = 13
    vq_memory_length: int = 40
    vq_vocab_size: int = 4

    @property
    def memory_dim(self) -> int:
        return PolicyBatchBuilder(
            joint_dim=self.joint_dim,
            memory_mode=self.memory_mode,
            raw_memory_steps=self.raw_memory_steps,
            vq_memory_length=self.vq_memory_length,
            vq_vocab_size=self.vq_vocab_size,
        ).input_dim - self.joint_dim

    @property
    def input_dim(self) -> int:
        return self.point_feature_dim + self.joint_dim + self.memory_dim

    def make_memory_features(self, current_state: torch.Tensor, raw_memory: Optional[torch.Tensor] = None, vq_tokens: Optional[torch.Tensor] = None):
        state_builder = PolicyBatchBuilder(
            joint_dim=self.joint_dim,
            memory_mode=self.memory_mode,
            raw_memory_steps=self.raw_memory_steps,
            vq_memory_length=self.vq_memory_length,
            vq_vocab_size=self.vq_vocab_size,
        )
        full = state_builder.make_features(current_state, raw_memory=raw_memory, vq_tokens=vq_tokens)
        return full[:, self.joint_dim:]


class PointActionPolicy(nn.Module):
    """DP3-like point-cloud action expert for reproduction plumbing.

    This is intentionally lightweight: PointNet-style encoder + memory features
    + MLP action head. It validates the DP3 observation shape and memory path
    before wiring the full upstream DP3 runner.
    """

    def __init__(self, feature_builder: PointPolicyFeatureBuilder, point_dim: int = 6, hidden_dim: int = 256, action_dim: int = 13):
        super().__init__()
        self.encoder = PointNetEncoder(point_dim=point_dim, out_dim=feature_builder.point_feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_builder.input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, points: torch.Tensor, current_state: torch.Tensor, memory_features: torch.Tensor) -> torch.Tensor:
        point_feat = self.encoder(points)
        feat = torch.cat([point_feat, current_state, memory_features], dim=-1)
        return self.head(feat)
