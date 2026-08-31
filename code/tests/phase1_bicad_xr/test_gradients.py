from __future__ import annotations

import pytest
import torch

from cvsrffi.phase1_bicad_xr.gradients import (
    GradientRatioAudit,
    GradientRatioController,
    measure_bounded_gradient_ratio,
    project_local_conflicting_gradients,
    project_conflicting_gradient,
    safe_svd,
    scale_explicit_gradients,
)


def test_safe_svd_rejects_nonfinite_input_and_keeps_zero_matrix_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        safe_svd(torch.tensor([[1.0, float("inf")]]))

    u, singular_values, vh = safe_svd(torch.zeros(2, 3), rank=2)

    assert u.shape == (2, 2)
    assert singular_values.shape == (2,)
    assert vh.shape == (2, 3)
    assert torch.isfinite(u).all()
    assert torch.isfinite(singular_values).all()
    assert torch.isfinite(vh).all()


def test_project_conflicting_gradient_removes_only_the_conflicting_component() -> None:
    gradient = torch.tensor([1.0, 1.0])
    opposing = torch.tensor([-1.0, 0.0])

    projected = project_conflicting_gradient(gradient, opposing)

    assert torch.allclose(projected, torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.dot(projected, opposing).item() == pytest.approx(0.0)


def test_gradient_ratio_controller_uses_detached_ema_ratio_and_bounded_scale() -> None:
    controller = GradientRatioController(
        target_ratio=1.0,
        ema_decay=0.0,
        min_scale=0.25,
        max_scale=4.0,
    )

    scale = controller.update(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0]))

    assert scale == pytest.approx(2.0)
    assert controller.ratio == pytest.approx(2.0)
    assert controller.ema_ratio is not None and not controller.ema_ratio.requires_grad


def test_measure_bounded_gradient_ratio_caps_effective_pair_dose_and_reports_audit() -> None:
    audit = measure_bounded_gradient_ratio(
        reference_gradient=torch.tensor([3.0, 4.0]),
        controlled_gradient=torch.tensor([30.0, 40.0]),
        initial_weight=0.02,
        max_ratio=0.05,
    )

    assert isinstance(audit, GradientRatioAudit)
    assert audit.raw_ratio == pytest.approx(0.20)
    assert audit.effective_ratio == pytest.approx(0.05)
    assert audit.scale == pytest.approx(0.25)
    assert audit.effective_weight == pytest.approx(0.005)
    assert audit.effective_ratio <= 0.05 + 1e-8


def test_only_the_explicit_parameter_list_is_scaled() -> None:
    selected = torch.nn.Parameter(torch.tensor([2.0]))
    untouched = torch.nn.Parameter(torch.tensor([3.0]))
    selected.grad = torch.tensor([4.0])
    untouched.grad = torch.tensor([5.0])

    scale_explicit_gradients([selected], 0.25)

    assert selected.grad is not None and selected.grad.item() == pytest.approx(1.0)
    assert untouched.grad is not None and untouched.grad.item() == pytest.approx(5.0)


def test_explicit_gradient_scaling_fails_closed_and_rolls_back_post_mul_overflow() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    original = torch.tensor([torch.finfo(torch.float32).max])
    parameter.grad = original.clone()

    with pytest.raises(ValueError, match="finite"):
        scale_explicit_gradients([parameter], 2.0)

    assert parameter.grad is not None
    assert torch.equal(parameter.grad, original)
    assert torch.isfinite(parameter.grad).all()


def test_gradient_ratio_controller_rejects_nonfinite_raw_ratio_without_state_mutation() -> None:
    controller = GradientRatioController(
        target_ratio=1e308,
        ema_decay=0.5,
        min_scale=0.0,
        max_scale=1e308,
    )
    first = controller.update(
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
    )
    before = controller.ema_ratio

    with pytest.raises(ValueError, match="finite"):
        controller.update(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
        )

    assert first == pytest.approx(1e308)
    assert before is not None
    assert controller.ema_ratio is not None
    assert torch.equal(controller.ema_ratio, before)
    assert controller.last_scale == pytest.approx(1e308)


def test_local_projection_allowlist_changes_only_identity_fusion_and_projection() -> None:
    gradients = {
        "identity_last_block.weight": torch.tensor([1.0, 1.0]),
        "fusion.bias": torch.tensor([1.0, 1.0]),
        "projection.weight": torch.tensor([1.0, 1.0]),
        "shared_stem.weight": torch.tensor([1.0, 1.0]),
    }
    references = {
        name: torch.tensor([-1.0, 0.0]) for name in gradients
    }

    projected = project_local_conflicting_gradients(gradients, references)

    for name in ("identity_last_block.weight", "fusion.bias", "projection.weight"):
        assert torch.allclose(projected[name], torch.tensor([0.0, 1.0]))
    assert torch.equal(projected["shared_stem.weight"], gradients["shared_stem.weight"])
    assert torch.equal(gradients["identity_last_block.weight"], torch.tensor([1.0, 1.0]))
