from __future__ import annotations

"""RED contract tests for the frozen Phase1 CLIC token operators."""

import sys
from pathlib import Path

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase1_clic import (  # noqa: E402
    CLIC_EMBED_DIM,
    CLIC_INIT_SEED,
    CLIC_INPUT_LENGTH,
    CLIC_LAGS,
    CLICConfig,
    CLICConfigError,
    CLICRuntimeError,
    CLICTokenBatch,
    totalized_clic_tokens,
)


def _nonzero_iq(*, batch: int = 2, length: int = CLIC_INPUT_LENGTH) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32)
    phase = 0.17 * t.square() / length + 0.031 * t
    phase = phase.unsqueeze(0) + torch.arange(batch, dtype=torch.float32).unsqueeze(1) * 0.23
    return torch.stack((phase.cos(), phase.sin()), dim=1)


def _apply_complex_gain_phase_cfo(
    x: torch.Tensor,
    *,
    magnitude: float,
    phase: float,
    omega: float,
) -> torch.Tensor:
    t = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device)
    z = torch.complex(x[:, 0], x[:, 1])
    transform = magnitude * torch.exp(1j * (phase + omega * t))
    transformed = z * transform
    return torch.stack((transformed.real, transformed.imag), dim=1)


def test_clic_public_api_imports():
    assert CLIC_LAGS == (1, 2, 4, 8)
    assert CLIC_INPUT_LENGTH == 256
    assert CLIC_EMBED_DIM == 160
    assert CLIC_INIT_SEED == 7281164
    assert CLICConfig is not None
    assert CLICTokenBatch is not None
    assert callable(totalized_clic_tokens)


def test_clic_token_shape_and_fixed_lags():
    x = torch.randn(3, 2, 256, dtype=torch.float32)
    c = totalized_clic_tokens(x, operator_mode="raw_phase_control")
    g = totalized_clic_tokens(x, operator_mode="complex_local_invariant_curvature")

    assert CLIC_LAGS == (1, 2, 4, 8)
    assert c.tokens.shape == g.tokens.shape == (3, 16, 256)
    assert c.valid_mask.shape == g.valid_mask.shape == (3, 4, 256)
    assert c.reliability.shape == g.reliability.shape == (3, 4, 256)
    assert c.valid_mask.dtype is torch.bool
    assert g.valid_mask.dtype is torch.bool
    assert torch.all((c.reliability >= 0) & (c.reliability <= 1))
    assert torch.all((g.reliability >= 0) & (g.reliability <= 1))


def test_clic_zero_domain_is_totalized_and_nonfinite_fails_closed():
    zero = torch.zeros(2, 2, 256)
    out = totalized_clic_tokens(zero, operator_mode="complex_local_invariant_curvature")

    assert torch.count_nonzero(out.tokens) == 0
    assert torch.count_nonzero(out.valid_mask) == 0
    assert torch.count_nonzero(out.reliability) == 0
    assert torch.count_nonzero(out.valid_fraction) == 0
    assert torch.count_nonzero(out.reliability_mean) == 0

    bad = zero.clone()
    bad[0, 0, 9] = float("nan")
    with pytest.raises(CLICRuntimeError, match="non-finite"):
        totalized_clic_tokens(bad, operator_mode="complex_local_invariant_curvature")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_clic_any_nonfinite_received_iq_fails_closed(bad_value: float):
    bad = torch.ones(1, 2, 256, dtype=torch.float32)
    bad[0, 1, 9] = bad_value
    with pytest.raises(CLICRuntimeError, match="non-finite"):
        totalized_clic_tokens(bad, operator_mode="raw_phase_control")


def test_g_operator_is_invariant_to_complex_gain_phase_and_linear_cfo():
    x = _nonzero_iq()
    transformed = _apply_complex_gain_phase_cfo(
        x,
        magnitude=1.7,
        phase=0.4,
        omega=0.03,
    )

    before = totalized_clic_tokens(x, operator_mode="complex_local_invariant_curvature")
    after = totalized_clic_tokens(
        transformed,
        operator_mode="complex_local_invariant_curvature",
    )

    torch.testing.assert_close(before.tokens, after.tokens, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(before.valid_mask, after.valid_mask)
    torch.testing.assert_close(before.reliability, after.reliability, rtol=2e-5, atol=2e-6)


def test_short_input_and_unknown_operator_fail_closed():
    with pytest.raises(CLICConfigError):
        totalized_clic_tokens(torch.zeros(1, 2, 16), operator_mode="raw_phase_control")
    with pytest.raises(CLICConfigError):
        totalized_clic_tokens(torch.zeros(1, 2, 256), operator_mode="other")
