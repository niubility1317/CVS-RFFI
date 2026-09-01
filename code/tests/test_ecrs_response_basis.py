from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import ResponseBasis  # noqa: E402


def test_fixed_response_basis_has_28_physical_columns_and_no_parameters() -> None:
    basis = ResponseBasis()
    phi = basis(torch.randn(3, 64, dtype=torch.complex64))
    assert phi.shape == (3, 64, 28)
    assert phi.dtype == torch.complex64
    assert list(basis.parameters()) == []
    assert basis.block_slices == {
        "pa": slice(0, 8),
        "iq": slice(8, 16),
        "cross": slice(16, 20),
        "slew": slice(20, 28),
    }


def test_response_basis_obeys_physical_common_phase_laws() -> None:
    torch.manual_seed(9)
    basis = ResponseBasis()
    s_hat = torch.randn(2, 48, dtype=torch.complex64)
    psi = torch.tensor(0.73)
    rotation = torch.exp(1j * psi)
    phi = basis(s_hat)
    rotated = basis(s_hat * rotation)

    positive = torch.cat([phi[..., :8], phi[..., 16:]], dim=-1)
    positive_rotated = torch.cat([rotated[..., :8], rotated[..., 16:]], dim=-1)
    torch.testing.assert_close(positive_rotated, positive * rotation, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(rotated[..., 8:16], phi[..., 8:16] * rotation.conj(), rtol=2e-4, atol=2e-5)
