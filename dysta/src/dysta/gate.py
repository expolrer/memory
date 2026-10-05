from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dysta.config import GateConfig


def _make_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    dimensions = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for index, (in_features, out_features) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(in_features, out_features))
        if index < len(dimensions) - 2:
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class GateOutput:
    logits: Tensor
    probabilities: Tensor
    decisions: Tensor


class SharedRecacheGate(nn.Module):
    """Shared two-stage MLP with a separate binary head for each cache level."""

    def __init__(self, config: GateConfig) -> None:
        super().__init__()
        self.config = config
        self.position_mlp = _make_mlp(
            2 * config.embedding_dim,
            config.position_hidden_dims,
            config.position_out_dim,
            config.dropout,
        )
        self.token_mlp = _make_mlp(
            config.num_tokens,
            config.token_hidden_dims,
            config.token_out_dim,
            config.dropout,
        )
        self.level_heads = nn.ModuleList(
            nn.Linear(config.flattened_dim, 1) for _ in range(config.num_levels)
        )
        self.register_buffer("thresholds", torch.tensor(config.thresholds, dtype=torch.float32))

    def _validate(self, references: Tensor, current: Tensor) -> Tensor:
        expected = (self.config.num_tokens, self.config.embedding_dim)
        if current.ndim != 3 or tuple(current.shape[1:]) != expected:
            raise ValueError(f"current must have shape [batch, {expected[0]}, {expected[1]}]")
        if references.ndim == 3:
            if references.shape != current.shape:
                raise ValueError("shared reference and current tensors must have identical shapes")
            references = references.unsqueeze(1).expand(-1, self.config.num_levels, -1, -1)
        expected_references = (
            current.shape[0],
            self.config.num_levels,
            self.config.num_tokens,
            self.config.embedding_dim,
        )
        if tuple(references.shape) != expected_references:
            raise ValueError(f"references must have shape {expected_references}")
        return references

    def _shared_features(self, reference: Tensor, current: Tensor) -> Tensor:
        paired = torch.cat((reference, current), dim=-1)
        position_features = self.position_mlp(paired)
        channel_features = self.token_mlp(position_features.transpose(1, 2))
        return channel_features.flatten(start_dim=1)

    def forward(
        self,
        references: Tensor,
        current: Tensor,
        *,
        inference: bool | None = None,
        hard: bool = True,
        temperature: float | None = None,
    ) -> GateOutput:
        references = self._validate(references, current)
        inference = (not self.training) if inference is None else inference
        temperature = self.config.gumbel_temperature if temperature is None else temperature
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        binary_logits = []
        for level, head in enumerate(self.level_heads):
            features = self._shared_features(references[:, level], current)
            refresh_logit = head(features).squeeze(-1)
            binary_logits.append(torch.stack((torch.zeros_like(refresh_logit), refresh_logit), dim=-1))
        logits = torch.stack(binary_logits, dim=1)
        probabilities = logits.softmax(dim=-1)[..., 1]

        if inference:
            decisions = (probabilities > self.thresholds.to(probabilities)).to(probabilities.dtype)
        else:
            decisions = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)[..., 1]

        # L1 is the most persistent level. Refreshing it invalidates every
        # less-persistent level below it.
        effective = [decisions[:, 0]]
        for level in range(1, self.config.num_levels):
            effective.append(torch.maximum(effective[-1], decisions[:, level]))
        decisions = torch.stack(effective, dim=1)
        return GateOutput(logits=logits, probabilities=probabilities, decisions=decisions)

    @torch.no_grad()
    def set_thresholds(self, thresholds: Iterable[float]) -> None:
        values = torch.as_tensor(tuple(thresholds), dtype=self.thresholds.dtype, device=self.thresholds.device)
        if values.shape != self.thresholds.shape:
            raise ValueError(f"expected {self.config.num_levels} thresholds")
        if torch.any((values < 0) | (values > 1)):
            raise ValueError("thresholds must lie in [0, 1]")
        self.thresholds.copy_(values)
