from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dysta.config import GateConfig, LossConfig, TokenLayout
from dysta.gate import SharedRecacheGate
from dysta.losses import gate_prior_loss, tokenwise_infonce


@dataclass(frozen=True)
class DyStaMultiFrameOutput:
    context_embeddings: Tensor
    static_by_level: tuple[Tensor, ...]
    dynamic_by_frame: Tensor
    infonce_by_level: tuple[Tensor, ...]
    weighted_infonce_loss: Tensor
    gate_loss: Tensor
    auxiliary_loss: Tensor
    gate_probabilities: Tensor
    gate_decisions: Tensor


class DyStaMultiFrameAdapter(nn.Module):
    """Compress a temporal image-token sequence using the paper's cache logic.

    Input tokens are ordered frame-major: T blocks of N projected visual tokens.
    The output ordering is all final static levels followed by the dynamic block
    from every frame. This preserves static tokens as a causal KV prefix.
    """

    def __init__(
        self,
        layout: TokenLayout,
        gate_config: GateConfig,
        loss_config: LossConfig,
        *,
        num_frames: int,
        frame_interval: int = 1,
        hard_gate: bool = True,
    ) -> None:
        super().__init__()
        if layout.total_tokens != gate_config.num_tokens:
            raise ValueError("layout and gate token counts must match")
        if layout.num_static_levels != gate_config.num_levels:
            raise ValueError("layout and gate level counts must match")
        if len(loss_config.alpha_levels) != layout.num_static_levels:
            raise ValueError("one alpha loss weight is required per static level")
        if num_frames <= 0 or frame_interval <= 0:
            raise ValueError("num_frames and frame_interval must be positive")
        self.layout = layout
        self.loss_config = loss_config
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.hard_gate = hard_gate
        self.gate = SharedRecacheGate(gate_config)

    @property
    def context_tokens(self) -> int:
        return self.layout.static_total + self.num_frames * self.layout.dynamic_tokens

    def _reshape_frames(self, projected_tokens: Tensor) -> Tensor:
        if projected_tokens.ndim == 4:
            expected = (self.num_frames, self.layout.total_tokens)
            if tuple(projected_tokens.shape[1:3]) != expected:
                raise ValueError(f"expected frame and token dimensions {expected}")
            return projected_tokens
        if projected_tokens.ndim != 3:
            raise ValueError("projected_tokens must have shape [B, T*N, D] or [B, T, N, D]")
        expected_tokens = self.num_frames * self.layout.total_tokens
        if projected_tokens.shape[1] != expected_tokens:
            raise ValueError(f"expected {expected_tokens} flattened tokens, got {projected_tokens.shape[1]}")
        return projected_tokens.reshape(
            projected_tokens.shape[0], self.num_frames, self.layout.total_tokens, projected_tokens.shape[-1]
        )

    def _split_frames(self, frames: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        slices = self.layout.slices
        static = tuple(frames[:, :, token_slice, :] for token_slice in slices[:-1])
        dynamic = frames[:, :, slices[-1], :]
        return static, dynamic

    def _contrastive_losses(self, static: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if static[0].shape[0] < 2 or self.num_frames < 2:
            return tuple(level.sum() * 0.0 for level in static)
        losses = []
        for level in static:
            pair_losses = [
                tokenwise_infonce(
                    level[:, frame],
                    level[:, 0],
                    temperature=self.loss_config.infonce_temperature,
                )
                for frame in range(1, self.num_frames)
            ]
            losses.append(torch.stack(pair_losses).mean())
        return tuple(losses)

    def forward(self, projected_tokens: Tensor) -> DyStaMultiFrameOutput:
        frames = self._reshape_frames(projected_tokens)
        static, dynamic = self._split_frames(frames)
        infonce_by_level = self._contrastive_losses(static)
        weighted_infonce = sum(
            weight * loss for weight, loss in zip(self.loss_config.alpha_levels, infonce_by_level)
        )

        cached_static = [level[:, 0] for level in static]
        references = [frames[:, 0] for _ in range(self.layout.num_static_levels)]
        last_refresh = torch.zeros(
            frames.shape[0], self.layout.num_static_levels, dtype=frames.dtype, device=frames.device
        )
        gate_probabilities = []
        gate_decisions = []
        gate_losses = []

        for frame in range(1, self.num_frames):
            current_step = float(frame * self.frame_interval)
            deltas = current_step - last_refresh
            output = self.gate(
                torch.stack(references, dim=1),
                frames[:, frame],
                inference=not self.training,
                hard=self.hard_gate,
            )
            gate_probabilities.append(output.probabilities)
            gate_decisions.append(output.decisions)
            gate_losses.append(
                gate_prior_loss(
                    output.probabilities,
                    deltas,
                    prior_lambda=self.loss_config.gate_lambda,
                )
            )

            updated_last_refresh = []
            for level in range(self.layout.num_static_levels):
                decision = output.decisions[:, level, None, None]
                cached_static[level] = (
                    decision * static[level][:, frame] + (1.0 - decision) * cached_static[level]
                )
                references[level] = decision * frames[:, frame] + (1.0 - decision) * references[level]
                updated_last_refresh.append(
                    output.decisions[:, level] * current_step
                    + (1.0 - output.decisions[:, level]) * last_refresh[:, level]
                )
            last_refresh = torch.stack(updated_last_refresh, dim=1)

        if gate_losses:
            gate_loss = torch.stack(gate_losses).mean()
            probabilities = torch.stack(gate_probabilities, dim=1)
            decisions = torch.stack(gate_decisions, dim=1)
        else:
            gate_loss = frames.sum() * 0.0
            shape = (frames.shape[0], 0, self.layout.num_static_levels)
            probabilities = frames.new_empty(shape)
            decisions = frames.new_empty(shape)

        dynamic_history = dynamic.flatten(start_dim=1, end_dim=2)
        context = torch.cat((*cached_static, dynamic_history), dim=1)
        auxiliary = weighted_infonce + self.loss_config.beta_gate * gate_loss
        return DyStaMultiFrameOutput(
            context_embeddings=context,
            static_by_level=static,
            dynamic_by_frame=dynamic,
            infonce_by_level=infonce_by_level,
            weighted_infonce_loss=weighted_infonce,
            gate_loss=gate_loss,
            auxiliary_loss=auxiliary,
            gate_probabilities=probabilities,
            gate_decisions=decisions,
        )
