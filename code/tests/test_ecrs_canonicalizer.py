from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import (  # noqa: E402
    AnalyticCanonicalizer,
    ContentEstimator,
    NuisanceEstimator,
)


def _as_iq(z: torch.Tensor) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=1)


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0], x[:, 1])


def _nmse(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    return (reference - estimate).abs().square().mean() / reference.abs().square().mean()


def test_analytic_canonicalizer_inverts_only_cfo_phase_and_scalar_gain() -> None:
    torch.manual_seed(3)
    clean = torch.randn(3, 64, dtype=torch.complex64)
    nuisance = torch.tensor(
        [[0.012, 0.7, math.log(1.8)], [-0.018, -1.1, math.log(0.6)], [0.0, 0.2, 0.0]],
        dtype=torch.float32,
    )
    n = torch.arange(clean.size(1), dtype=torch.float32)
    phase = 2.0 * math.pi * nuisance[:, :1] * n + nuisance[:, 1:2]
    perturbed = clean * nuisance[:, 2:3].exp() * torch.exp(1j * phase)

    canonicalizer = AnalyticCanonicalizer()
    restored = _as_complex(canonicalizer(_as_iq(perturbed), nuisance))

    assert _nmse(clean, restored) < _nmse(clean, perturbed)
    torch.testing.assert_close(restored, clean, rtol=2e-5, atol=2e-5)
    assert not any("fir" in name.lower() or "conjugate" in name.lower() for name, _ in canonicalizer.named_parameters())


def test_nuisance_and_content_estimators_are_bounded_finite_single_view_modules() -> None:
    torch.manual_seed(4)
    x = torch.randn(2, 2, 64, requires_grad=True)
    nuisance = NuisanceEstimator()(x)
    assert nuisance.shape == (2, 3)
    assert torch.isfinite(nuisance).all()
    assert torch.all(nuisance[:, 0].abs() <= 0.05)
    assert torch.all(nuisance[:, 1].abs() <= math.pi)
    assert torch.all(nuisance[:, 2].abs() <= 2.0)

    canonical = AnalyticCanonicalizer()(x, nuisance)
    s_hat, confidence = ContentEstimator()(canonical)
    assert s_hat.shape == (2, 64)
    assert s_hat.is_complex()
    assert confidence.shape == (2, 64)
    assert torch.isfinite(s_hat.real).all() and torch.isfinite(s_hat.imag).all()
    assert torch.all((confidence >= 0.0) & (confidence <= 1.0))


def test_identity_loss_can_detach_content_estimator() -> None:
    estimator = ContentEstimator()
    x = torch.randn(2, 2, 32, requires_grad=True)
    s_hat, _ = estimator(x)
    identity_projection = torch.nn.Linear(64, 3)
    logits = identity_projection(torch.view_as_real(s_hat.detach()).flatten(1))
    logits.square().mean().backward()
    assert all(parameter.grad is None for parameter in estimator.parameters())
