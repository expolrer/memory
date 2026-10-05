from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def tokenwise_infonce(
    anchor: Tensor,
    positive: Tensor,
    *,
    temperature: float = 0.07,
    trajectory_ids: Tensor | None = None,
    symmetric: bool = True,
) -> Tensor:
    """InfoNCE over matching token positions in paired trajectory frames.

    Each batch row is one trajectory pair. The token at the same position in
    the paired frame is positive; same-position tokens from other trajectories
    are negatives, matching the paper's description and commented equation.
    """

    if anchor.ndim != 3 or anchor.shape != positive.shape:
        raise ValueError("anchor and positive must share shape [batch, tokens, dim]")
    if anchor.shape[0] < 2:
        raise ValueError("InfoNCE needs at least two trajectories per batch")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)

    def directional(query: Tensor, key: Tensor) -> Tensor:
        logits = torch.einsum("bnd,cnd->bnc", query, key) / temperature
        batch_size, num_tokens, _ = logits.shape
        if trajectory_ids is not None:
            if trajectory_ids.shape != (batch_size,):
                raise ValueError("trajectory_ids must have shape [batch]")
            same_trajectory = trajectory_ids[:, None] == trajectory_ids[None, :]
            off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=logits.device)
            mask = same_trajectory & off_diagonal
            logits = logits.masked_fill(mask[:, None, :], torch.finfo(logits.dtype).min)
        labels = torch.arange(batch_size, device=logits.device)[:, None].expand(-1, num_tokens)
        return F.cross_entropy(logits.reshape(-1, batch_size), labels.reshape(-1))

    loss = directional(anchor, positive)
    if symmetric:
        loss = 0.5 * (loss + directional(positive, anchor))
    return loss


def hierarchical_infonce_loss(
    anchors: Sequence[Tensor],
    positives: Sequence[Tensor],
    weights: Sequence[float],
    *,
    temperature: float = 0.07,
    trajectory_ids: Tensor | None = None,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    if not (len(anchors) == len(positives) == len(weights)):
        raise ValueError("anchors, positives, and weights need one entry per level")
    if not anchors:
        raise ValueError("at least one static level is required")
    level_losses = tuple(
        tokenwise_infonce(
            anchor,
            positive,
            temperature=temperature,
            trajectory_ids=trajectory_ids,
        )
        for anchor, positive in zip(anchors, positives)
    )
    total = sum(weight * loss for weight, loss in zip(weights, level_losses))
    return total, level_losses


def gate_prior_loss(
    refresh_probabilities: Tensor,
    deltas: Tensor,
    *,
    prior_lambda: float,
    eps: float = 1e-6,
) -> Tensor:
    """Binary cross entropy against p_delta = 1 - exp(-lambda * delta)."""

    if refresh_probabilities.shape != deltas.shape:
        raise ValueError("refresh_probabilities and deltas must have identical shapes")
    if prior_lambda <= 0:
        raise ValueError("prior_lambda must be positive")
    if torch.any(deltas < 0):
        raise ValueError("deltas must be non-negative")
    prior = 1.0 - torch.exp(-prior_lambda * deltas.to(refresh_probabilities.dtype))
    probabilities = refresh_probabilities.clamp(eps, 1.0 - eps)
    return -(prior * probabilities.log() + (1.0 - prior) * (1.0 - probabilities).log()).mean()
