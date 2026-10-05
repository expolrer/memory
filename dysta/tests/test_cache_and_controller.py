import torch

from dysta import CacheAction, DyStaController, GateConfig, TokenLayout
from dysta.cache import crop_legacy_past_key_values


def make_controller() -> DyStaController:
    layout = TokenLayout(total_tokens=12, static_tokens=(5, 4), dynamic_tokens=3)
    gate_config = GateConfig(
        embedding_dim=8,
        num_tokens=12,
        num_levels=2,
        position_hidden_dims=(16, 8),
        position_out_dim=6,
        token_hidden_dims=(12, 8),
        token_out_dim=4,
        thresholds=(0.5, 0.5),
    )
    return DyStaController(layout, gate_config, max_dynamic_frames=4)


def set_head_biases(controller: DyStaController, l1: float, l2: float) -> None:
    with torch.no_grad():
        for head in controller.gate.level_heads:
            head.weight.zero_()
        controller.gate.level_heads[0].bias.fill_(l1)
        controller.gate.level_heads[1].bias.fill_(l2)


def test_reuse_appends_only_current_dynamic_tokens() -> None:
    controller = make_controller().eval()
    set_head_biases(controller, -12.0, -12.0)
    first = torch.zeros(1, 12, 8)
    second = torch.ones(1, 12, 8)

    initial = controller(first, inference=True)
    reused = controller(second, inference=True)

    assert initial.kv_plan.action is CacheAction.REBUILD_ALL
    assert reused.kv_plan.action is CacheAction.APPEND_DYNAMIC
    assert reused.kv_plan.reusable_prefix_tokens == 12
    assert reused.kv_plan.recompute_embeddings.shape == (1, 3, 8)
    assert reused.context_embeddings.shape == (1, 15, 8)
    torch.testing.assert_close(reused.context_embeddings[:, :9], torch.zeros(1, 9, 8))


def test_l2_refresh_reuses_l1_prefix_and_rebuilds_suffix() -> None:
    controller = make_controller().eval()
    set_head_biases(controller, -12.0, 12.0)
    controller(torch.zeros(1, 12, 8), inference=True)
    result = controller(torch.ones(1, 12, 8), inference=True)

    assert result.kv_plan.action is CacheAction.REBUILD_FROM_LEVEL
    assert result.kv_plan.refresh_from_level == 1
    assert result.kv_plan.reusable_prefix_tokens == 5
    assert result.kv_plan.recompute_embeddings.shape == (1, 10, 8)
    torch.testing.assert_close(result.context_embeddings[:, :5], torch.zeros(1, 5, 8))
    torch.testing.assert_close(result.context_embeddings[:, 5:9], torch.ones(1, 4, 8))


def test_legacy_kv_crop_uses_sequence_axis() -> None:
    key = torch.arange(2 * 4 * 10 * 8).reshape(2, 4, 10, 8)
    value = key + 1
    cropped = crop_legacy_past_key_values(((key, value),), prefix_tokens=6)
    assert cropped[0][0].shape == (2, 4, 6, 8)
    torch.testing.assert_close(cropped[0][0], key[..., :6, :])
