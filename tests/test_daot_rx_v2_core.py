from __future__ import annotations

import pytest
import torch

from cvsrffi.deployment_orbit import (
    TangentDirectionRegistry,
    sample_single_tx_intervention,
)
from cvsrffi.selective_tangent import chordal_sensitivity, directional_routing_loss
from cvsrffi.tangent_calibration import finite_difference_linearity_error, select_largest_stable_delta


def test_direction_registry_separates_null_mixed_tx_and_secant_directions() -> None:
    registry = TangentDirectionRegistry.default()

    assert registry["rx_filter"].kind == "pure_nuisance"
    assert registry["total_cfo"].kind == "mixed"
    assert registry["pa"].kind == "tx_fingerprint"
    assert registry["clipping"].kind == "secant_only"
    assert registry["quantization"].supports_tangent is False
    assert registry["total_cfo"].budget > 0.0
    assert registry["pa"].null_identity is False


def test_chordal_sensitivity_is_finite_at_identical_features_and_uses_delta_squared() -> None:
    base = torch.tensor([[1.0, 0.0]])
    same = base.clone()
    rotated = torch.tensor([[0.0, 1.0]])

    assert float(chordal_sensitivity(base, same, delta=0.1)) == pytest.approx(0.0)
    assert float(chordal_sensitivity(base, rotated, delta=0.1)) == pytest.approx(200.0)
    assert float(chordal_sensitivity(base, rotated, delta=0.2)) == pytest.approx(50.0)


def test_delta_calibration_selects_largest_locally_linear_step() -> None:
    def evaluate(delta: float) -> torch.Tensor:
        return torch.tensor([delta + 2.0 * delta * delta])

    errors = finite_difference_linearity_error(evaluate, deltas=[0.2, 0.1, 0.05])
    selected = select_largest_stable_delta(errors, max_error=0.15)

    assert errors[0.2] > 0.15
    assert selected == pytest.approx(0.1)


def test_random_tx_intervention_uses_one_replayable_direction_per_sample() -> None:
    x = torch.randn(6, 2, 64)

    first, first_ids, first_signs = sample_single_tx_intervention(
        x,
        seed=77,
        strength=0.03,
        sample_rate_hz=25e6,
    )
    second, second_ids, second_signs = sample_single_tx_intervention(
        x,
        seed=77,
        strength=0.03,
        sample_rate_hz=25e6,
    )

    assert torch.equal(first_ids, second_ids)
    assert torch.equal(first_signs, second_signs)
    assert torch.allclose(first, second)
    assert first_ids.shape == (6,)
    assert set(first_signs.tolist()) <= {-1.0, 1.0}
    assert bool((first - x).abs().sum(dim=(1, 2)).gt(0.0).all())


def test_directional_routing_prefers_nuisance_in_domain_and_tx_in_identity() -> None:
    base_id = torch.tensor([[1.0, 0.0]])
    base_dom = torch.tensor([[1.0, 0.0]])
    good = directional_routing_loss(
        base_id=base_id,
        nuisance_id=torch.tensor([[0.99, 0.01]]),
        fingerprint_id=torch.tensor([[0.0, 1.0]]),
        base_dom=base_dom,
        nuisance_dom=torch.tensor([[0.0, 1.0]]),
        fingerprint_dom=torch.tensor([[0.99, 0.01]]),
        nuisance_margin=0.05,
        fingerprint_margin=0.05,
    )
    bad = directional_routing_loss(
        base_id=base_id,
        nuisance_id=torch.tensor([[0.0, 1.0]]),
        fingerprint_id=torch.tensor([[0.99, 0.01]]),
        base_dom=base_dom,
        nuisance_dom=torch.tensor([[0.99, 0.01]]),
        fingerprint_dom=torch.tensor([[0.0, 1.0]]),
        nuisance_margin=0.05,
        fingerprint_margin=0.05,
    )

    assert float(good["loss"]) == pytest.approx(0.0)
    assert float(bad["loss"]) > 1.0
