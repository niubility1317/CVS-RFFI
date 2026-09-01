from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import WeightedRidgeLayer  # noqa: E402


def test_weighted_ridge_recovers_known_complex_coefficients() -> None:
    torch.manual_seed(11)
    batch, samples, features = 2, 96, 28
    phi = torch.randn(batch, samples, features, dtype=torch.complex64)
    truth = torch.randn(batch, features, dtype=torch.complex64)
    target = torch.einsum("bnk,bk->bn", phi, truth)
    confidence = torch.rand(batch, samples)

    result = WeightedRidgeLayer(alpha_lambda=1e-7)(phi, target, confidence)

    torch.testing.assert_close(result["resp_coef"], truth, rtol=2e-3, atol=2e-3)
    assert result["resp_cov_diag"].shape == (batch, features)
    assert torch.all(result["weights"] >= 0.05)
    torch.testing.assert_close(result["weights"].mean(dim=1), torch.ones(batch), atol=1e-5, rtol=1e-5)
    for key in ("gram_eigenvalues", "log_condition", "effective_rank", "effective_sample_size", "coverage", "nmse"):
        assert key in result["resp_quality"]


def test_weighted_ridge_uses_second_cholesky_then_augmented_lstsq_fallback() -> None:
    torch.manual_seed(12)
    phi = torch.randn(1, 40, 28, dtype=torch.complex64)
    target = torch.randn(1, 40, dtype=torch.complex64)
    layer = WeightedRidgeLayer(alpha_lambda=1e-4)
    original = layer._cholesky_solve
    calls = 0

    def fail_once(matrix, rhs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return original(matrix, rhs)

    layer._cholesky_solve = fail_once
    second = layer(phi, target, torch.ones(1, 40))
    assert second["ridge_info"].item() == 1

    zero_phi = torch.zeros(1, 12, 28, dtype=torch.complex64)
    qr = WeightedRidgeLayer(alpha_lambda=0.0)(
        zero_phi, torch.zeros(1, 12, dtype=torch.complex64), torch.ones(1, 12)
    )
    assert qr["ridge_info"].item() == 2
    assert torch.isfinite(qr["resp_coef"].real).all()
    assert "torch.inverse" not in inspect.getsource(WeightedRidgeLayer)
