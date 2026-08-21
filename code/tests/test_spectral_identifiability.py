import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.spectral_identifiability import (  # noqa: E402
    build_center_mask,
    extract_sid_fft96,
    validate_sid_mask,
)


def test_sid_fft96_has_fixed_groups_and_finite_unit_norm():
    torch.manual_seed(7)
    iq = torch.randn(4, 2, 256)
    mask = build_center_mask(256, half_width=48, dc_notch=2)

    features, diagnostics = extract_sid_fft96(iq, mask, mode="sid")

    assert features.shape == (4, 96)
    assert torch.isfinite(features).all()
    assert torch.allclose(features.norm(dim=1), torch.ones(4), atol=1e-5)
    assert tuple(diagnostics["group_norms"].shape) == (4, 5)


def test_phase_mode_zeroes_non_phase_groups():
    iq = torch.randn(2, 2, 128)
    mask = torch.ones(128, dtype=torch.bool)

    features, _ = extract_sid_fft96(iq, mask, mode="phase")

    assert torch.count_nonzero(features[:, :24]) == 0
    assert torch.count_nonzero(features[:, 64:]) == 0


def test_empty_or_wrong_shape_mask_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        validate_sid_mask(torch.zeros(128), 128)
    with pytest.raises(ValueError, match="fft_bins"):
        validate_sid_mask(torch.ones(64), 128)
