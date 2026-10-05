import torch

from dysta import DyStaMultiFrameAdapter, GateConfig, LossConfig, TokenLayout


def make_adapter() -> DyStaMultiFrameAdapter:
    layout = TokenLayout(total_tokens=12, static_tokens=(5, 4), dynamic_tokens=3)
    gate = GateConfig(
        embedding_dim=8,
        num_tokens=12,
        num_levels=2,
        position_hidden_dims=(16, 8),
        position_out_dim=6,
        token_hidden_dims=(12, 8),
        token_out_dim=4,
        thresholds=(0.5, 0.5),
    )
    return DyStaMultiFrameAdapter(
        layout,
        gate,
        LossConfig(alpha_levels=(0.2, 0.1), beta_gate=0.1),
        num_frames=3,
        frame_interval=2,
    )


def test_multiframe_context_has_one_static_copy_and_all_dynamic_frames() -> None:
    adapter = make_adapter().train()
    tokens = torch.randn(4, 3 * 12, 8, requires_grad=True)
    output = adapter(tokens)

    assert output.context_embeddings.shape == (4, 18, 8)
    assert output.dynamic_by_frame.shape == (4, 3, 3, 8)
    assert [value.shape for value in output.static_by_level] == [(4, 3, 5, 8), (4, 3, 4, 8)]
    assert output.gate_probabilities.shape == (4, 2, 2)
    assert torch.isfinite(output.auxiliary_loss)


def test_multiframe_auxiliary_loss_backpropagates() -> None:
    adapter = make_adapter().train()
    tokens = torch.randn(4, 3, 12, 8, requires_grad=True)
    output = adapter(tokens)
    (output.context_embeddings.square().mean() + output.auxiliary_loss).backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert any(parameter.grad is not None for parameter in adapter.gate.parameters())
