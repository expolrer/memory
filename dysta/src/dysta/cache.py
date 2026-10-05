from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import torch
from torch import Tensor

from dysta.config import TokenLayout
from dysta.partition import PartitionedTokens, StaticDynamicPartitioner


class CacheAction(str, Enum):
    APPEND_DYNAMIC = "append_dynamic"
    REBUILD_FROM_LEVEL = "rebuild_from_level"
    REBUILD_ALL = "rebuild_all"


@dataclass(frozen=True)
class KVRecomputePlan:
    action: CacheAction
    reusable_prefix_tokens: int
    refresh_from_level: int | None
    recompute_embeddings: Tensor


class HierarchicalTokenCache:
    """Rollout-time token state and exact causal-prefix recomputation plan."""

    def __init__(self, layout: TokenLayout, *, max_dynamic_frames: int | None = None, detach: bool = True) -> None:
        if max_dynamic_frames is not None and max_dynamic_frames <= 0:
            raise ValueError("max_dynamic_frames must be positive")
        self.layout = layout
        self.max_dynamic_frames = max_dynamic_frames
        self.detach = detach
        self.clear()

    def clear(self) -> None:
        self.static_levels: tuple[Tensor, ...] | None = None
        self.dynamic_history: list[Tensor] = []
        self.reference_tokens: tuple[Tensor, ...] | None = None
        self.last_refresh_steps: Tensor | None = None
        self.step = -1

    @property
    def initialized(self) -> bool:
        return self.static_levels is not None

    def _store(self, tensor: Tensor) -> Tensor:
        return tensor.detach() if self.detach else tensor

    def initialize(self, partition: PartitionedTokens, full_tokens: Tensor) -> KVRecomputePlan:
        if len(partition.static_levels) != self.layout.num_static_levels:
            raise ValueError("partition level count does not match token layout")
        self.static_levels = tuple(self._store(tokens) for tokens in partition.static_levels)
        self.dynamic_history = [self._store(partition.dynamic)]
        self.reference_tokens = tuple(self._store(full_tokens) for _ in partition.static_levels)
        self.last_refresh_steps = torch.zeros(
            full_tokens.shape[0], self.layout.num_static_levels, dtype=torch.long, device=full_tokens.device
        )
        self.step = 0
        return KVRecomputePlan(
            action=CacheAction.REBUILD_ALL,
            reusable_prefix_tokens=0,
            refresh_from_level=0,
            recompute_embeddings=self.context(),
        )

    def references(self) -> Tensor:
        if self.reference_tokens is None:
            raise RuntimeError("cache is not initialized")
        return torch.stack(self.reference_tokens, dim=1)

    def deltas_for_next_step(self) -> Tensor:
        if self.last_refresh_steps is None:
            raise RuntimeError("cache is not initialized")
        return (self.step + 1 - self.last_refresh_steps).clamp_min(1)

    def context(self) -> Tensor:
        if self.static_levels is None:
            raise RuntimeError("cache is not initialized")
        return StaticDynamicPartitioner.pack_context(self.static_levels, self.dynamic_history)

    def _uniform_decisions(self, decisions: Tensor) -> Tensor:
        if decisions.ndim != 2 or decisions.shape[1] != self.layout.num_static_levels:
            raise ValueError("refresh decisions must have shape [batch, num_static_levels]")
        selected = decisions >= 0.5
        for level in range(1, self.layout.num_static_levels):
            selected[:, level] |= selected[:, level - 1]
        if not torch.all(selected == selected[:1]):
            raise ValueError("a single rollout cache requires uniform decisions across its batch")
        return selected

    def update(
        self,
        partition: PartitionedTokens,
        full_tokens: Tensor,
        refresh_decisions: Tensor,
    ) -> KVRecomputePlan:
        if self.static_levels is None or self.reference_tokens is None or self.last_refresh_steps is None:
            raise RuntimeError("initialize the cache before calling update")
        selected = self._uniform_decisions(refresh_decisions)
        self.step += 1

        new_static = []
        new_references = []
        for level, (cached, candidate, reference) in enumerate(
            zip(self.static_levels, partition.static_levels, self.reference_tokens)
        ):
            mask = selected[:, level, None, None]
            new_static.append(self._store(torch.where(mask, candidate, cached)))
            new_references.append(self._store(torch.where(mask, full_tokens, reference)))
            self.last_refresh_steps[:, level] = torch.where(
                selected[:, level],
                torch.full_like(self.last_refresh_steps[:, level], self.step),
                self.last_refresh_steps[:, level],
            )
        self.static_levels = tuple(new_static)
        self.reference_tokens = tuple(new_references)
        self.dynamic_history.append(self._store(partition.dynamic))
        if self.max_dynamic_frames is not None:
            self.dynamic_history = self.dynamic_history[-self.max_dynamic_frames :]

        refresh_levels = torch.where(selected[0])[0]
        if refresh_levels.numel() == 0:
            reusable = self.layout.static_total + (len(self.dynamic_history) - 1) * self.layout.dynamic_tokens
            return KVRecomputePlan(
                action=CacheAction.APPEND_DYNAMIC,
                reusable_prefix_tokens=reusable,
                refresh_from_level=None,
                recompute_embeddings=self.dynamic_history[-1],
            )

        first_level = int(refresh_levels[0].item())
        reusable = sum(self.layout.static_tokens[:first_level])
        recompute = torch.cat((*self.static_levels[first_level:], *self.dynamic_history), dim=1)
        action = CacheAction.REBUILD_ALL if first_level == 0 else CacheAction.REBUILD_FROM_LEVEL
        return KVRecomputePlan(
            action=action,
            reusable_prefix_tokens=reusable,
            refresh_from_level=first_level,
            recompute_embeddings=recompute,
        )


def crop_legacy_past_key_values(past_key_values: Sequence[Any], prefix_tokens: int) -> tuple[Any, ...]:
    """Crop a Transformers legacy KV tuple to a causal prefix.

    Keys and values are expected to use the common [..., sequence, head_dim]
    layout. Any extra per-layer entries are preserved unchanged.
    """

    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be non-negative")
    cropped = []
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError("each legacy cache layer must contain key and value tensors")
        key, value, *extra = layer
        if not isinstance(key, Tensor) or not isinstance(value, Tensor):
            raise TypeError("legacy cache keys and values must be tensors")
        cropped.append((key[..., :prefix_tokens, :], value[..., :prefix_tokens, :], *extra))
    return tuple(cropped)
