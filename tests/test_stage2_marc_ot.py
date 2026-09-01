from __future__ import annotations

import inspect
import math

import pytest
import torch

import cvsrffi.stage2_marc_ot as marc_ot
from cvsrffi.stage2_marc_ot import (
    MARCOTDiagnostics,
    blockwise_primary_projection,
    marc_ot_losses,
    support_bank_transport,
)


def test_public_training_apis_have_no_forbidden_evaluation_surface() -> None:
    for function in (
        support_bank_transport,
        blockwise_primary_projection,
        marc_ot_losses,
    ):
        signature = str(inspect.signature(function)).lower()
        assert not any(word in signature for word in ("query", "truth", "role", "quota"))


def test_support_bank_transport_has_uniform_marginals_and_is_deterministic() -> None:
    support = torch.tensor([[0.0], [2.0]], requires_grad=True)
    bank = torch.tensor([[0.0], [2.0]], requires_grad=True)

    first = support_bank_transport(support, bank, epsilon=0.2, iterations=100)
    second = support_bank_transport(support, bank, epsilon=0.2, iterations=100)

    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.allclose(first.sum(dim=1), torch.full((2,), 0.5), atol=1.0e-5)
    assert torch.allclose(first.sum(dim=0), torch.full((2,), 0.5), atol=1.0e-5)
    first.square().sum().backward()
    assert support.grad is not None and torch.isfinite(support.grad).all()
    assert bank.grad is None


@pytest.mark.parametrize(
    ("support", "bank", "epsilon", "iterations", "message"),
    [
        (torch.ones(2, 2, 1), torch.ones(2, 2), 0.1, 10, "two-dimensional"),
        (torch.ones(2, 2), torch.ones(2, 3), 0.1, 10, "feature dimensions"),
        (torch.tensor([[math.nan, 0.0]]), torch.ones(2, 2), 0.1, 10, "finite"),
        (torch.ones(2, 2), torch.ones(2, 2), 0.0, 10, "epsilon"),
        (torch.ones(2, 2), torch.ones(2, 2), 0.1, 0, "iterations"),
    ],
)
def test_support_bank_transport_rejects_invalid_inputs(
    support: torch.Tensor,
    bank: torch.Tensor,
    epsilon: float,
    iterations: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        support_bank_transport(support, bank, epsilon=epsilon, iterations=iterations)


def test_support_bank_transport_rejects_unconverged_marginals() -> None:
    support = torch.tensor([[0.0], [1.0], [10.0]])
    bank = torch.tensor([[0.0], [10.0]])
    with pytest.raises(ValueError, match="marginals"):
        support_bank_transport(support, bank, epsilon=0.01, iterations=1)


def test_support_bank_transport_converges_for_formal_k10_geometry() -> None:
    support = torch.zeros(60, 685)
    bank = torch.zeros(11, 685)
    support[:, 0] = torch.linspace(-8.0, 8.0, 60)
    bank[:, 0] = torch.linspace(-8.0, 8.0, 11)
    support.requires_grad_()

    plan = support_bank_transport(support, bank, epsilon=0.1, iterations=80)

    assert torch.isfinite(plan).all()
    assert torch.allclose(
        plan.sum(dim=1), torch.full((60,), 1.0 / 60.0), atol=1.0e-4, rtol=0.0
    )
    assert torch.allclose(
        plan.sum(dim=0), torch.full((11,), 1.0 / 11.0), atol=1.0e-4, rtol=0.0
    )
    plan.square().sum().backward()
    assert support.grad is not None and torch.isfinite(support.grad).all()
    assert bank.grad is None


def test_transport_cost_is_invariant_to_repeating_every_feature_coordinate() -> None:
    support = torch.tensor([[0.0], [0.5], [1.5], [2.0]], requires_grad=True)
    bank = torch.tensor([[0.0], [2.0]])
    base = support_bank_transport(support, bank, epsilon=0.2, iterations=100)
    repeated = support_bank_transport(
        support.repeat(1, 685),
        bank.repeat(1, 685),
        epsilon=0.2,
        iterations=100,
    )

    assert torch.allclose(repeated, base, atol=1.0e-6, rtol=0.0)

    labels = torch.tensor([0, 0, 1, 1])
    tokens = ("c0-a", "c0-b", "c1-a", "c1-b")
    logits = torch.tensor([[2.0, -2.0], [1.0, -1.0], [-1.0, 1.0], [-2.0, 2.0]])
    base_loss = marc_ot_losses(
        support,
        labels,
        tokens,
        logits,
        bank,
        ot_epsilon=0.2,
        ot_iterations=100,
    ).transport_loss
    repeated_loss = marc_ot_losses(
        support.repeat(1, 685),
        labels,
        tokens,
        logits,
        bank.repeat(1, 685),
        ot_epsilon=0.2,
        ot_iterations=100,
    ).transport_loss
    assert torch.allclose(repeated_loss, base_loss, atol=1.0e-6, rtol=0.0)


def test_blockwise_projection_only_changes_conflicting_block() -> None:
    primary = {
        "time": [torch.tensor([1.0, 0.0])],
        "freq": [torch.tensor([1.0, 0.0])],
    }
    auxiliary = {
        "time": [torch.tensor([-1.0, 1.0])],
        "freq": [torch.tensor([1.0, 1.0])],
    }

    projected = marc_ot._blockwise_project_grouped(primary, auxiliary, ratio_cap=0.5)

    assert torch.dot(projected["time"][0], primary["time"][0]) >= 0
    assert torch.allclose(projected["time"][0], torch.tensor([0.0, 0.5]), atol=1.0e-6)
    assert torch.allclose(
        projected["freq"][0], torch.tensor([0.3535534, 0.3535534]), atol=1.0e-6
    )
    assert projected.diagnostics["time"]["raw_cosine"] == pytest.approx(-math.sqrt(0.5))
    assert projected.diagnostics["time"]["projected_cosine"] == pytest.approx(0.0, abs=1.0e-7)
    assert projected.diagnostics["time"]["projected_norm"] == pytest.approx(0.5)


def test_blockwise_projection_reuses_canonical_parameter_routing() -> None:
    primary = {
        "id_backbone.t1.weight": [torch.tensor([1.0])],
        "id_backbone.t1.bias": [torch.tensor([1.0])],
    }
    auxiliary = {
        "id_backbone.t1.weight": [torch.tensor([-2.0])],
        "id_backbone.t1.bias": [torch.tensor([1.0])],
    }

    projected = blockwise_primary_projection(primary, auxiliary, ratio_cap=1.0)

    assert tuple(projected) == ("t1",)
    assert len(projected["t1"]) == 2
    assert torch.allclose(projected["t1"][0], torch.tensor([-1.0]))
    assert torch.allclose(projected["t1"][1], torch.tensor([1.0]))
    with pytest.raises(ValueError, match="canonical"):
        blockwise_primary_projection(
            {"id_backbone.cls_head.head.weight": [torch.ones(1)]},
            {"id_backbone.cls_head.head.weight": [torch.ones(1)]},
            ratio_cap=1.0,
        )
    with pytest.raises(ValueError, match="canonical"):
        blockwise_primary_projection(
            {"time": [torch.ones(1)]},
            {"time": [torch.ones(1)]},
            ratio_cap=1.0,
        )


def test_blockwise_projection_handles_none_zero_primary_and_extreme_scales() -> None:
    primary = {
        "missing": [None, torch.tensor([0.0, 0.0])],
        "extreme": [torch.tensor([1.0e20, 0.0], dtype=torch.float64)],
    }
    auxiliary = {
        "missing": [torch.tensor([2.0]), torch.tensor([3.0, 4.0])],
        "extreme": [torch.tensor([-1.0e20, 1.0e20], dtype=torch.float64)],
    }

    projected = marc_ot._blockwise_project_grouped(primary, auxiliary, ratio_cap=0.25)

    assert torch.equal(projected["missing"][0], torch.zeros(1))
    assert torch.equal(projected["missing"][1], torch.zeros(2))
    assert torch.allclose(
        projected["extreme"][0],
        torch.tensor([0.0, 2.5e19], dtype=torch.float64),
        rtol=1.0e-12,
    )
    assert torch.dot(projected["extreme"][0], primary["extreme"][0]) >= 0.0
    assert math.isfinite(projected.diagnostics["extreme"]["projected_norm"])


def test_blockwise_projection_removes_conflict_even_when_eps_is_one() -> None:
    primary = {"time": [torch.tensor([1.0, 0.0])]}
    projected = marc_ot._blockwise_project_grouped(
        primary,
        {"time": [torch.tensor([-1.0, 1.0])]},
        ratio_cap=10.0,
        eps=1.0,
    )
    assert torch.allclose(projected["time"][0], torch.tensor([0.0, 1.0]))
    assert torch.dot(projected["time"][0], primary["time"][0]) >= 0.0


def test_blockwise_projection_keeps_nonnegative_dot_for_small_primary() -> None:
    primary = {"tiny": [torch.tensor([1.0e-30, 0.0], dtype=torch.float64)]}
    projected = marc_ot._blockwise_project_grouped(
        primary,
        {"tiny": [torch.tensor([-1.0, 1.0], dtype=torch.float64)]},
        ratio_cap=1.0,
        eps=1.0,
    )
    assert torch.dot(projected["tiny"][0], primary["tiny"][0]) >= 0.0
    assert float(torch.linalg.vector_norm(projected["tiny"][0])) == pytest.approx(
        1.0e-30, rel=1.0e-12
    )


@pytest.mark.parametrize("side", ["primary", "auxiliary"])
def test_blockwise_projection_rejects_nonfinite_gradients(side: str) -> None:
    name = "id_backbone.t1.weight"
    primary = {name: [torch.ones(1)]}
    auxiliary = {name: [torch.ones(1)]}
    (primary if side == "primary" else auxiliary)[name][0] = torch.tensor([math.inf])
    with pytest.raises(ValueError, match="nonfinite"):
        blockwise_primary_projection(primary, auxiliary, ratio_cap=1.0)


def _loss_fixture(k_shot: int) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], torch.Tensor, torch.Tensor]:
    labels = torch.arange(2).repeat_interleave(k_shot)
    row = torch.arange(k_shot, dtype=torch.float32)[:, None] * 0.01
    class_zero = torch.cat((torch.ones(k_shot, 1), row), dim=1)
    class_one = torch.cat((-torch.ones(k_shot, 1), row), dim=1)
    features = torch.cat((class_zero, class_one), dim=0).requires_grad_()
    tokens = tuple(f"s-{index}" for index in range(len(labels)))
    logits = torch.stack((features[:, 0], -features[:, 0]), dim=1)
    bank = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
    return features, labels, tokens, logits, bank


@pytest.mark.parametrize(
    ("k_shot", "expected_mode"),
    [
        (1, "mean_scale"),
        (2, "diagonal"),
        (5, "low_rank_1"),
        (10, "low_rank_2"),
        (20, "low_rank_2"),
    ],
)
def test_marc_ot_losses_are_support_only_differentiable_and_k_conditioned(
    k_shot: int,
    expected_mode: str,
) -> None:
    features, labels, tokens, logits, bank = _loss_fixture(k_shot)

    result = marc_ot_losses(
        features,
        labels,
        tokens,
        logits,
        bank,
        fold_count=2,
        fold_seed=13,
        ot_epsilon=0.2,
        ot_iterations=100,
        statistic_rank=2,
    )

    assert isinstance(result, MARCOTDiagnostics)
    assert result.k_shot == k_shot
    assert result.statistics_mode == expected_mode
    assert result.class_risk.shape == (2,)
    assert torch.isfinite(result.total)
    assert not hasattr(result, "temporary_prototypes")
    assert not hasattr(result, "persistent_state")
    result.total.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert bank.grad is None


def test_marc_ot_k1_zero_statistics_keep_gradients_finite() -> None:
    features = torch.zeros(2, 2, requires_grad=True)
    labels = torch.tensor([0, 1])
    logits = features @ torch.zeros(2, 2)
    result = marc_ot_losses(
        features,
        labels,
        ("a", "b"),
        logits,
        torch.zeros(2, 2),
        ot_epsilon=0.2,
        ot_iterations=20,
    )
    result.total.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_marc_ot_losses_reuse_existing_old_only_d92_cross_fit() -> None:
    generator = torch.Generator().manual_seed(17)
    labels = torch.arange(6).repeat_interleave(2)
    identity = torch.randn(12, 160, generator=generator, requires_grad=True)
    fft = torch.randn(12, 96, generator=generator)
    head_weight = torch.randn(160, 6, generator=generator)
    logits = identity @ head_weight
    bank = torch.randn(3, 160, generator=generator)

    result = marc_ot_losses(
        identity,
        labels,
        tuple(f"d92-{index}" for index in range(12)),
        logits,
        bank,
        support_fft_features=fft,
        fold_count=2,
        fold_seed=19,
        ot_epsilon=2.0,
        ot_iterations=100,
    )

    assert result.cross_fit_mode == "d92_old_only"
    assert result.class_risk.shape == (6,)
    assert torch.isfinite(result.cross_fit_ce)
    cross_fit_gradient = torch.autograd.grad(result.cross_fit_ce, identity)[0]
    assert torch.isfinite(cross_fit_gradient).all()
    assert torch.linalg.vector_norm(cross_fit_gradient) > 0.0


def test_marc_ot_k1_fft_validation_precedes_explicit_d92_rejection() -> None:
    labels = torch.arange(6)
    tokens = tuple(f"k1-{index}" for index in range(6))
    identity = torch.zeros(6, 160)
    logits = torch.zeros(6, 6)
    bank = torch.zeros(3, 160)

    with pytest.raises(ValueError, match=r"\[N,160\].*\[N,96\]"):
        marc_ot_losses(
            identity[:, :159], labels, tokens, logits, bank[:, :159],
            support_fft_features=torch.zeros(6, 96),
        )
    nonfinite_fft = torch.zeros(6, 96)
    nonfinite_fft[0, 0] = math.inf
    with pytest.raises(ValueError, match="FFT features must be finite"):
        marc_ot_losses(
            identity, labels, tokens, logits, bank,
            support_fft_features=nonfinite_fft,
        )
    with pytest.raises(ValueError, match=r"D92 cross-fit requires K>=2"):
        marc_ot_losses(
            identity, labels, tokens, logits, bank,
            support_fft_features=torch.zeros(6, 96),
        )


def test_marc_ot_losses_reject_unequal_k_and_nonfinite_frozen_head_logits() -> None:
    features, labels, tokens, logits, bank = _loss_fixture(2)
    with pytest.raises(ValueError, match="equal K"):
        marc_ot_losses(
            features[:-1], labels[:-1], tokens[:-1], logits[:-1], bank,
            ot_epsilon=0.2, ot_iterations=50,
        )
    broken = logits.detach().clone()
    broken[0, 0] = math.nan
    with pytest.raises(ValueError, match="frozen head logits"):
        marc_ot_losses(
            features, labels, tokens, broken, bank,
            ot_epsilon=0.2, ot_iterations=50,
        )


def test_marc_ot_losses_reject_nonintegral_labels_and_empty_tokens() -> None:
    features, labels, tokens, logits, bank = _loss_fixture(2)
    bad_labels = labels.float()
    bad_labels[0] = 0.5
    with pytest.raises(ValueError, match="integer"):
        marc_ot_losses(
            features, bad_labels, tokens, logits, bank,
            ot_epsilon=0.2, ot_iterations=50,
        )
    with pytest.raises(ValueError, match="nonempty"):
        marc_ot_losses(
            features, labels, ("", *tokens[1:]), logits, bank,
            ot_epsilon=0.2, ot_iterations=50,
        )
