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

    mask = torch.ones(4)
    state_a = encoder(features, labels, physical_tokens, mask)
    state_b = encoder(
        features[order],
        labels[order],
        tuple(physical_tokens[i] for i in order),
        mask[order],
    )

    assert torch.allclose(state_a.q, state_b.q, atol=1e-6)
    assert torch.allclose(state_a.uncertainty, state_b.uncertainty, atol=1e-6)
    assert torch.allclose(state_a.block_gates, state_b.block_gates, atol=1e-6)
    assert torch.allclose(state_a.block_lrs, state_b.block_lrs, atol=1e-6)


def test_support_encoder_uses_shared_class_formula_under_label_permutation() -> None:
    """A branch keyed on a concrete class ID would break this label relabeling."""
    encoder = _encoder()
    features = torch.tensor([[1.0, 0.0, 2.0], [3.0, 1.0, 0.0], [2.0, 4.0, 1.0], [0.0, 2.0, 3.0]])
    tokens = ("p0", "p1", "p2", "p3")

    mask = torch.ones(4)
    original = encoder(features, torch.tensor([9, 9, 4, 4]), tokens, mask)
    relabeled = encoder(features, torch.tensor([101, 101, -3, -3]), tokens, mask)

    assert torch.allclose(original.q, relabeled.q, atol=1e-6)
    assert torch.allclose(original.uncertainty, relabeled.uncertainty, atol=1e-6)
    assert torch.allclose(original.block_gates, relabeled.block_gates, atol=1e-6)
    assert torch.allclose(original.block_lrs, relabeled.block_lrs, atol=1e-6)


def test_support_encoder_rejects_repeated_physical_tokens() -> None:
    encoder = _encoder()

    with pytest.raises(ValueError, match="physical support tokens must be unique"):
        encoder(
            torch.ones(2, 3), torch.tensor([0, 1]), ("same", "same"), torch.ones(2)
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_support_encoder_rejects_nonfinite_features(bad_value: float) -> None:
    encoder = _encoder()
    features = torch.tensor([[1.0, bad_value, 2.0], [3.0, 4.0, 5.0]])

    with pytest.raises(ValueError, match="non-finite"):
        encoder(features, torch.tensor([0, 1]), ("p0", "p1"), torch.ones(2))


def test_support_encoder_bounds_gate_and_learning_rate_outputs() -> None:
    encoder = _encoder()

    state = encoder(
        torch.ones(2, 3), torch.tensor([0, 1]), ("p0", "p1"), torch.ones(2)
    )

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


@pytest.mark.parametrize("k_shot", (1, 2, 5, 10, 20))
def test_support_encoder_computes_masked_raw_class_statistics_for_all_k(k_shot: int) -> None:
    encoder = _encoder()
    rows = []
    labels = []
    tokens = []
    for class_id in (0, 1):
        for shot in range(k_shot):
            rows.append([float(class_id + 1), float(shot), float(class_id - shot)])
            labels.append(class_id)
            tokens.append(f"opaque-{class_id}-{shot}")
    features = torch.tensor(rows)
    mask = torch.ones(len(rows))

    state = encoder(features, torch.tensor(labels), tuple(tokens), mask)

    assert state.class_means.shape == (2, 3)
    assert state.class_diag_variances.shape == (2, 3)
    assert state.class_norms.shape == (2, 2)
    assert state.class_stat_flags.shape == (2, 4)
    assert torch.equal(state.class_means[0], features[:k_shot].mean(dim=0))
    assert torch.equal(
        state.class_diag_variances[0], features[:k_shot].var(dim=0, unbiased=False)
    )
    expected_flags = torch.tensor(
        [1.0, float(k_shot >= 2), float(k_shot >= 5), float(k_shot >= 10)]
    )
    assert torch.equal(state.class_stat_flags[0], expected_flags)


def test_support_encoder_mask_excludes_padded_rows_from_statistics_and_state() -> None:
    encoder = _encoder()
    features = torch.tensor(
        [[1.0, 2.0, 3.0], [999.0, 999.0, 999.0], [4.0, 5.0, 6.0], [-999.0, -999.0, -999.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    tokens = ("a0", "a-pad", "b0", "b-pad")
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])

    masked = encoder(features, labels, tokens, mask)
    compact = encoder(
        features[[0, 2]], labels[[0, 2]], ("a0", "b0"), torch.ones(2)
    )

    assert torch.equal(masked.class_means, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert torch.equal(masked.class_diag_variances, torch.zeros(2, 3))
    assert torch.allclose(masked.q, compact.q, atol=1e-6)
    assert torch.allclose(masked.block_gates, compact.block_gates, atol=1e-6)


def test_support_encoder_mean_variance_norms_and_availability_change_domain_state() -> None:
    encoder = _encoder()
    tokens = ("a0", "a1", "b0", "b1")
    labels = torch.tensor([0, 0, 1, 1])
    concentrated = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    )
    dispersed = torch.tensor(
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
    )

    state_a = encoder(concentrated, labels, tokens, torch.ones(4))
    state_b = encoder(dispersed, labels, tokens, torch.ones(4))
    k1_state = encoder(
        concentrated[[0, 2]], labels[[0, 2]], ("a0", "b0"), torch.ones(2)
    )

    assert torch.equal(state_a.class_means, state_b.class_means)
    assert not torch.equal(state_a.class_diag_variances, state_b.class_diag_variances)
    assert not torch.allclose(state_a.q, state_b.q)
    assert not torch.equal(state_a.class_stat_flags, k1_state.class_stat_flags)
    assert not torch.allclose(state_a.q, k1_state.q)


@pytest.mark.parametrize(
    "mask",
    (
        torch.tensor([1.0, 0.5]),
        torch.tensor([1.0, float("nan")]),
        torch.tensor([0.0, 1.0]),
    ),
)
def test_support_encoder_rejects_invalid_or_empty_class_mask(mask: torch.Tensor) -> None:
    encoder = _encoder()

    with pytest.raises(ValueError, match="effective mask|effective row"):
        encoder(torch.ones(2, 3), torch.tensor([0, 1]), ("p0", "p1"), mask)
