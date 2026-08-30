from __future__ import annotations

import pytest
import torch

from cvsrffi.phase1_bicad_xr.gradients import (
    GradientRatioController,
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


def test_only_the_explicit_parameter_list_is_scaled() -> None:
    selected = torch.nn.Parameter(torch.tensor([2.0]))
    untouched = torch.nn.Parameter(torch.tensor([3.0]))
    selected.grad = torch.tensor([4.0])
    untouched.grad = torch.tensor([5.0])

    scale_explicit_gradients([selected], 0.25)

    assert selected.grad is not None and selected.grad.item() == pytest.approx(1.0)
    assert untouched.grad is not None and untouched.grad.item() == pytest.approx(5.0)
