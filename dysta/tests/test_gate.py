import torch

from dysta import GateConfig, SharedRecacheGate


def small_gate() -> SharedRecacheGate:
    config = GateConfig(
        embedding_dim=32,
        num_tokens=24,
        num_levels=2,
        position_hidden_dims=(48, 24),
        position_out_dim=12,
        token_hidden_dims=(24, 12),
        token_out_dim=8,
        thresholds=(0.5, 0.5),
    )
    return SharedRecacheGate(config)


def test_gate_uses_level_specific_references_and_shared_trunk() -> None:
    gate = small_gate().eval()
    current = torch.randn(3, 24, 32)
    references = torch.randn(3, 2, 24, 32)
    output = gate(references, current, inference=True)

    assert output.logits.shape == (3, 2, 2)
    assert output.probabilities.shape == (3, 2)
    assert output.decisions.shape == (3, 2)
    assert gate.level_heads[0] is not gate.level_heads[1]


def test_l1_refresh_forces_l2_refresh() -> None:
    gate = small_gate().eval()
    with torch.no_grad():
        for head in gate.level_heads:
            head.weight.zero_()
        gate.level_heads[0].bias.fill_(12.0)
        gate.level_heads[1].bias.fill_(-12.0)
    current = torch.randn(1, 24, 32)
    references = torch.randn(1, 2, 24, 32)

    output = gate(references, current, inference=True)
    torch.testing.assert_close(output.decisions, torch.tensor([[1.0, 1.0]]))
    assert output.probabilities[0, 0] > 0.99
    assert output.probabilities[0, 1] < 0.01


def test_training_gate_is_differentiable() -> None:
    gate = small_gate().train()
    current = torch.randn(2, 24, 32, requires_grad=True)
    references = torch.randn(2, 2, 24, 32)
    output = gate(references, current, inference=False, hard=False)
    output.decisions.mean().backward()

    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
