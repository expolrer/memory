import torch

from dysta import gate_prior_loss, hierarchical_infonce_loss, tokenwise_infonce


def test_matching_pairs_have_lower_infonce_than_shuffled_pairs() -> None:
    torch.manual_seed(7)
    anchor = torch.randn(8, 5, 16)
    positive = anchor + 0.01 * torch.randn_like(anchor)
    matching = tokenwise_infonce(anchor, positive, temperature=0.1)
    shuffled = tokenwise_infonce(anchor, positive.roll(1, dims=0), temperature=0.1)
    assert matching < shuffled


def test_hierarchical_loss_applies_reported_weights() -> None:
    torch.manual_seed(11)
    anchors = (torch.randn(4, 3, 8), torch.randn(4, 2, 8))
    positives = tuple(value + 0.02 * torch.randn_like(value) for value in anchors)
    total, levels = hierarchical_infonce_loss(anchors, positives, (0.2, 0.1))
    torch.testing.assert_close(total, 0.2 * levels[0] + 0.1 * levels[1])


def test_gate_prior_prefers_probability_near_exponential_prior() -> None:
    deltas = torch.tensor([[1.0, 4.0], [2.0, 8.0]])
    prior = 1.0 - torch.exp(-0.2 * deltas)
    matching = gate_prior_loss(prior, deltas, prior_lambda=0.2)
    inverted = gate_prior_loss(1.0 - prior, deltas, prior_lambda=0.2)
    assert matching < inverted
