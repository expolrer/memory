from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from dysta.cache import HierarchicalTokenCache, KVRecomputePlan
from dysta.config import GateConfig, TokenLayout
from dysta.gate import GateOutput, SharedRecacheGate
from dysta.partition import StaticDynamicPartitioner


@dataclass(frozen=True)
class DyStaStepOutput:
    context_embeddings: Tensor
    kv_plan: KVRecomputePlan
    gate: GateOutput | None
    deltas: Tensor | None


class DyStaController(nn.Module):
    """Stateful paper-method controller around projected VLA visual tokens."""

    def __init__(
        self,
        layout: TokenLayout,
        gate_config: GateConfig,
        *,
        max_dynamic_frames: int | None = None,
        detach_cache: bool = True,
    ) -> None:
        super().__init__()
        if layout.num_static_levels != gate_config.num_levels:
            raise ValueError("layout and gate must use the same number of static levels")
        if layout.total_tokens != gate_config.num_tokens:
            raise ValueError("layout and gate must use the same visual token count")
        self.layout = layout
        self.partitioner = StaticDynamicPartitioner(layout)
        self.gate = SharedRecacheGate(gate_config)
        self.cache = HierarchicalTokenCache(
            layout, max_dynamic_frames=max_dynamic_frames, detach=detach_cache
        )

    def reset(self) -> None:
        self.cache.clear()

    def forward(
        self,
        projected_visual_tokens: Tensor,
        *,
        inference: bool | None = None,
        hard_gate: bool = True,
    ) -> DyStaStepOutput:
        partition = self.partitioner(projected_visual_tokens)
        if not self.cache.initialized:
            plan = self.cache.initialize(partition, projected_visual_tokens)
            return DyStaStepOutput(
                context_embeddings=self.cache.context(),
                kv_plan=plan,
                gate=None,
                deltas=None,
            )

        deltas = self.cache.deltas_for_next_step()
        gate_output = self.gate(
            self.cache.references(),
            projected_visual_tokens,
            inference=inference,
            hard=hard_gate,
        )
        plan = self.cache.update(
            partition,
            projected_visual_tokens,
            gate_output.decisions,
        )
        return DyStaStepOutput(
            context_embeddings=self.cache.context(),
            kv_plan=plan,
            gate=gate_output,
            deltas=deltas,
        )
