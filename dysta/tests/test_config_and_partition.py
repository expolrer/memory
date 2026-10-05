import pytest
import torch

from dysta import StaticDynamicPartitioner, TokenLayout


def test_paper_layout_splits_256_tokens() -> None:
    layout = TokenLayout()
    tokens = torch.arange(2 * 256 * 8, dtype=torch.float32).reshape(2, 256, 8)
    split = StaticDynamicPartitioner(layout)(tokens)

    assert [part.shape for part in split.static_levels] == [(2, 133, 8), (2, 107, 8)]
    assert split.dynamic.shape == (2, 16, 8)
    torch.testing.assert_close(split.packed(), tokens)


def test_libero_memory_layout_matches_paper_context_length() -> None:
    layout = TokenLayout.libero_memory()
    assert layout.static_tokens == (230,)
    assert layout.dynamic_tokens == 26
    assert layout.static_total + 20 * layout.dynamic_tokens == 750


def test_invalid_layout_is_rejected() -> None:
    with pytest.raises(ValueError, match="must sum"):
        TokenLayout(total_tokens=256, static_tokens=(133, 100), dynamic_tokens=16)
