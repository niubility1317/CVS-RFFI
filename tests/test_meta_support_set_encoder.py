from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _encoder():
    from cvsrffi.meta_support_set_encoder import SupportSetEncoder

    torch.manual_seed(7)
    return SupportSetEncoder(
        feature_dim=3,
        coefficient_dim=2,
        block_count=2,
        hidden_dim=5,
        lr_min=0.001,
        lr_max=0.01,
    )


def test_support_encoder_is_row_permutation_invariant() -> None:
    """A row-order-sensitive pooling implementation would break this contract."""
    encoder = _encoder()
    features = torch.tensor([[1.0, 0.0, 2.0], [3.0, 1.0, 0.0], [2.0, 4.0, 1.0], [0.0, 2.0, 3.0]])
    labels = torch.tensor([9, 9, 4, 4])
    physical_tokens = ("p0", "p1", "p2", "p3")
    order = torch.tensor([2, 0, 3, 1])

    state_a = encoder(features, labels, physical_tokens)
    state_b = encoder(features[order], labels[order], tuple(physical_tokens[i] for i in order))

    assert torch.allclose(state_a.q, state_b.q, atol=1e-6)
    assert torch.allclose(state_a.uncertainty, state_b.uncertainty, atol=1e-6)
    assert torch.allclose(state_a.block_gates, state_b.block_gates, atol=1e-6)
    assert torch.allclose(state_a.block_lrs, state_b.block_lrs, atol=1e-6)


def test_support_encoder_uses_shared_class_formula_under_label_permutation() -> None:
    """A branch keyed on a concrete class ID would break this label relabeling."""
    encoder = _encoder()
    features = torch.tensor([[1.0, 0.0, 2.0], [3.0, 1.0, 0.0], [2.0, 4.0, 1.0], [0.0, 2.0, 3.0]])
    tokens = ("p0", "p1", "p2", "p3")

    original = encoder(features, torch.tensor([9, 9, 4, 4]), tokens)
    relabeled = encoder(features, torch.tensor([101, 101, -3, -3]), tokens)

    assert torch.allclose(original.q, relabeled.q, atol=1e-6)
    assert torch.allclose(original.uncertainty, relabeled.uncertainty, atol=1e-6)
    assert torch.allclose(original.block_gates, relabeled.block_gates, atol=1e-6)
    assert torch.allclose(original.block_lrs, relabeled.block_lrs, atol=1e-6)


def test_support_encoder_rejects_repeated_physical_tokens() -> None:
    encoder = _encoder()

    with pytest.raises(ValueError, match="physical support tokens must be unique"):
        encoder(torch.ones(2, 3), torch.tensor([0, 1]), ("same", "same"))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_support_encoder_rejects_nonfinite_features(bad_value: float) -> None:
    encoder = _encoder()
    features = torch.tensor([[1.0, bad_value, 2.0], [3.0, 4.0, 5.0]])

    with pytest.raises(ValueError, match="non-finite"):
        encoder(features, torch.tensor([0, 1]), ("p0", "p1"))


def test_support_encoder_bounds_gate_and_learning_rate_outputs() -> None:
    encoder = _encoder()

    state = encoder(torch.ones(2, 3), torch.tensor([0, 1]), ("p0", "p1"))

    assert state.q.shape == (2,)
    assert state.uncertainty.shape == ()
    assert state.block_gates.shape == (2,)
    assert torch.all((state.block_gates >= 0.0) & (state.block_gates <= 1.0))
    assert torch.all((state.block_lrs >= 0.001) & (state.block_lrs <= 0.01))


def test_support_encoder_rejects_collapsed_learning_rate_interval() -> None:
    from cvsrffi.meta_support_set_encoder import SupportSetEncoder

    with pytest.raises(ValueError, match="learning-rate bounds"):
        SupportSetEncoder(
            feature_dim=3,
            coefficient_dim=2,
            block_count=2,
            lr_min=0.01,
            lr_max=0.01,
        )
