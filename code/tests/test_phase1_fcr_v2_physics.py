from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase1_fcr_types import FCRConfig  # noqa: E402
from cvsrffi.phase1_fcr_v2_factors import FCRV2FactorEncoder  # noqa: E402
from cvsrffi.phase1_fcr_v2_physics import (  # noqa: E402
    IdentityInitializedPhysicsDecoder,
    apply_multipath,
    apply_sfo,
    apply_sto,
    complex_gram,
)


def _real_iq(batch_size: int = 2, length: int = 256) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(batch_size, 2, length)


def _complex_tone(length: int = 256, cycles: float = 5.0) -> torch.Tensor:
    index = torch.arange(length, dtype=torch.float32)
    return torch.exp(1j * (2.0 * math.pi * cycles * index / length))


def test_content_code_is_capacity_limited() -> None:
    config = FCRConfig()
    encoder = FCRV2FactorEncoder(config)
    canonical = _real_iq()
    residual = 0.25 * _real_iq()
    z_adv = torch.randn(canonical.size(0), 160)

    out = encoder(canonical, residual, z_adv)

    assert out.z_s.shape == (canonical.size(0), config.input_len // config.content_stride, 16)
    assert out.z_s.shape[-1] <= 16
    assert out.z_f_id.shape == (canonical.size(0), 160)
    assert out.z_f_dev.shape == (canonical.size(0), 160)
    assert out.z_f_dev.requires_grad
    assert out.canonical_residual is not None
    assert out.canonical_residual.shape == (canonical.size(0), config.input_len)
    torch.testing.assert_close(out.z_f_id.norm(dim=1), torch.ones(canonical.size(0)), atol=1e-5, rtol=1e-5)


def test_identity_decoder_starts_as_identity_channel() -> None:
    decoder = IdentityInitializedPhysicsDecoder(FCRConfig())
    iq = _complex_tone().unsqueeze(0).repeat(2, 1)

    decoded = decoder.identity_forward(iq)
    rebuilt = torch.complex(decoded.mu_iq[:, 0], decoded.mu_iq[:, 1])

    torch.testing.assert_close(rebuilt, iq, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(decoded.delta_f, torch.zeros_like(iq))


def test_sto_shifts_and_sfo_creates_phase_slope() -> None:
    impulse = torch.zeros(1, 32, dtype=torch.complex64)
    impulse[:, 5] = 1.0 + 0.0j
    shifted = apply_sto(impulse, torch.tensor([3.0]))

    assert shifted.abs().argmax(dim=1).item() == 8

    tone = torch.ones(1, 64, dtype=torch.complex64)
    warped = apply_sfo(tone, torch.tensor([0.01]))
    phase_step = torch.angle(warped[:, 1:] * warped[:, :-1].conj()).abs().mean()

    assert phase_step.item() > 0.0


def test_multipath_and_complex_gram_have_expected_structure() -> None:
    impulse = torch.zeros(1, 16, dtype=torch.complex64)
    impulse[:, 0] = 1.0 + 0.0j
    taps = torch.tensor([[1.0 + 0.0j, 0.5 + 0.25j, 0.0 + 0.0j]], dtype=torch.complex64)

    echoed = apply_multipath(impulse, taps)
    basis = torch.stack((echoed, echoed.conj()), dim=-1)
    gram = complex_gram(basis)

    assert echoed.abs()[0, 1].item() > 0.0
    assert gram.shape == (1, 2, 2)
    assert torch.isfinite(gram.real).all()
    assert torch.isfinite(gram.imag).all()
    torch.testing.assert_close(gram, gram.conj().transpose(-1, -2), atol=1e-5, rtol=1e-5)
