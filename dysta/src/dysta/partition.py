from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from dysta.config import TokenLayout


@dataclass(frozen=True)
class PartitionedTokens:
    static_levels: tuple[Tensor, ...]
    dynamic: Tensor

    @property
    def batch_size(self) -> int:
        return self.dynamic.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.dynamic.shape[-1]

    def packed(self) -> Tensor:
        return torch.cat((*self.static_levels, self.dynamic), dim=1)


class StaticDynamicPartitioner(nn.Module):
    """Split projected visual tokens into fixed, contiguous semantic slots.

    This is the minimum-assumption interpretation of Equation 2. The paper does
    not disclose a separate token-routing network. Specialization is therefore
    learned through the VLA task loss and level-wise InfoNCE objectives.
    """

    def __init__(self, layout: TokenLayout) -> None:
        super().__init__()
        self.layout = layout

    def forward(self, tokens: Tensor) -> PartitionedTokens:
        if tokens.ndim != 3:
            raise ValueError(f"expected [batch, tokens, dim], got {tuple(tokens.shape)}")
        if tokens.shape[1] != self.layout.total_tokens:
            raise ValueError(
                f"expected {self.layout.total_tokens} tokens, got {tokens.shape[1]}"
            )
        slices = self.layout.slices
        static = tuple(tokens[:, token_slice, :] for token_slice in slices[:-1])
        dynamic = tokens[:, slices[-1], :]
        return PartitionedTokens(static_levels=static, dynamic=dynamic)

    @staticmethod
    def pack_context(static_levels: Iterable[Tensor], dynamic_history: Iterable[Tensor]) -> Tensor:
        static_levels = tuple(static_levels)
        dynamic_history = tuple(dynamic_history)
        if not static_levels:
            raise ValueError("at least one static level is required")
        if not dynamic_history:
            raise ValueError("at least one dynamic frame is required")
        return torch.cat((*static_levels, *dynamic_history), dim=1)
