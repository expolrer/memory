from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


def _as_tuple(values: Sequence[int] | Sequence[float]) -> tuple:
    return tuple(values)


@dataclass(frozen=True)
class TokenLayout:
    """Contiguous token layout stated by the paper.

    Static levels are ordered from most persistent (L1) to least persistent.
    Dynamic tokens always form the final segment so static KV entries remain a
    reusable prefix.
    """

    total_tokens: int = 256
    static_tokens: Tuple[int, ...] = (133, 107)
    dynamic_tokens: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "static_tokens", _as_tuple(self.static_tokens))
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if not self.static_tokens or any(count <= 0 for count in self.static_tokens):
            raise ValueError("static_tokens must contain positive level sizes")
        if self.dynamic_tokens <= 0:
            raise ValueError("dynamic_tokens must be positive")
        if sum(self.static_tokens) + self.dynamic_tokens != self.total_tokens:
            raise ValueError(
                "static and dynamic token counts must sum to total_tokens: "
                f"{sum(self.static_tokens)} + {self.dynamic_tokens} != {self.total_tokens}"
            )

    @property
    def num_static_levels(self) -> int:
        return len(self.static_tokens)

    @property
    def static_total(self) -> int:
        return sum(self.static_tokens)

    @property
    def slices(self) -> tuple[slice, ...]:
        result = []
        start = 0
        for count in (*self.static_tokens, self.dynamic_tokens):
            result.append(slice(start, start + count))
            start += count
        return tuple(result)

    @classmethod
    def libero_memory(cls) -> "TokenLayout":
        return cls(total_tokens=256, static_tokens=(230,), dynamic_tokens=26)


@dataclass(frozen=True)
class GateConfig:
    """Architecture of the shared recache gate.

    The paper appendix is internally inconsistent: it states that the second
    FFN maps 256 tokens to 128, but also states that the final head receives a
    128 x 64 tensor. Defaults preserve the explicitly reported first-FFN output
    (128) and final-head shape (128 x 64); token_out_dim remains configurable.
    """

    embedding_dim: int = 4096
    num_tokens: int = 256
    num_levels: int = 2
    position_hidden_dims: Tuple[int, ...] = (1024, 256)
    position_out_dim: int = 128
    token_hidden_dims: Tuple[int, ...] = (256, 128)
    token_out_dim: int = 64
    thresholds: Tuple[float, ...] = (0.8, 0.4)
    gumbel_temperature: float = 1.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_hidden_dims", _as_tuple(self.position_hidden_dims))
        object.__setattr__(self, "token_hidden_dims", _as_tuple(self.token_hidden_dims))
        object.__setattr__(self, "thresholds", _as_tuple(self.thresholds))
        if self.embedding_dim <= 0 or self.num_tokens <= 0 or self.num_levels <= 0:
            raise ValueError("gate dimensions and level count must be positive")
        if len(self.thresholds) != self.num_levels:
            raise ValueError("one recache threshold is required per static level")
        if any(not 0.0 <= value <= 1.0 for value in self.thresholds):
            raise ValueError("recache thresholds must lie in [0, 1]")
        if self.gumbel_temperature <= 0:
            raise ValueError("gumbel_temperature must be positive")

    @property
    def flattened_dim(self) -> int:
        return self.position_out_dim * self.token_out_dim


@dataclass(frozen=True)
class LossConfig:
    alpha_levels: Tuple[float, ...] = (0.2, 0.1)
    beta_gate: float = 0.1
    infonce_temperature: float = 0.07
    gate_lambda: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha_levels", _as_tuple(self.alpha_levels))
        if any(value < 0 for value in self.alpha_levels) or self.beta_gate < 0:
            raise ValueError("loss weights must be non-negative")
        if self.infonce_temperature <= 0 or self.gate_lambda <= 0:
            raise ValueError("temperature and gate_lambda must be positive")
