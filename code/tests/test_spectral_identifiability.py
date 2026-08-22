import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.spectral_identifiability import (  # noqa: E402
    SIDFFT96Residual,
    build_center_mask,
    extract_sid_fft96,
    load_sid_mask,
    select_sid_mask,
    SpectralIdentifiabilityAccumulator,
    validate_sid_mask,
)
from scripts import audit_phase1_spectral_identifiability as spectral_audit  # noqa: E402


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


def test_identifiability_prefers_tx_separating_low_domain_band():
    accumulator = SpectralIdentifiabilityAccumulator(num_bands=2, feature_dim=1)
    for tx in (0, 1):
        for rx in (0, 1):
            accumulator.update(
                np.array([[float(tx) * 4.0], [float(rx) * 4.0]]),
                tx=tx,
                rx=rx,
                day=0,
                view=0,
            )

    stats = accumulator.finalize()

    assert stats["j_score"][0] > stats["j_score"][1]
    assert stats["tx_scatter"][0] > stats["tx_scatter"][1]
    assert stats["domain_scatter"][1] > stats["domain_scatter"][0]


def test_domain_scatter_is_conditioned_on_tx_under_imbalanced_domain_composition():
    accumulator = SpectralIdentifiabilityAccumulator(num_bands=1, feature_dim=1)
    domain_tx_counts = {
        0: {0: 3, 1: 1},
        1: {0: 1, 1: 3},
    }
    for rx, tx_counts in domain_tx_counts.items():
        for tx, count in tx_counts.items():
            for _ in range(count):
                accumulator.update(
                    np.array([[float(tx) * 10.0]]),
                    tx=tx,
                    rx=rx,
                    day=0,
                    view=0,
                )

    stats = accumulator.finalize()

    assert stats["tx_scatter"][0] > 0.0
    assert stats["domain_scatter"][0] == pytest.approx(0.0)


def test_mask_selection_is_stable_on_equal_scores():
    mask = select_sid_mask({"j_score": np.ones(8)}, keep_fraction=0.25, dc_notch=0)

    assert np.flatnonzero(mask).tolist() == [0, 1]


def test_torch_mask_boundary_returns_current_numpy_uint8_array():
    assert hasattr(spectral_audit, "_torch_mask_to_numpy")
    mask = torch.tensor([True, False, True])

    converted = spectral_audit._torch_mask_to_numpy(mask)

    assert converted.__class__ is np.ndarray
    assert converted.dtype == np.uint8
    assert converted.tolist() == [1, 0, 1]


def test_load_sid_mask_crosses_numpy_torch_boundary_via_python_values(tmp_path, monkeypatch):
    artifact = tmp_path / "sid_mask.npz"
    np.savez_compressed(artifact, mask=np.asarray([1, 0, 1, 0], dtype=np.uint8))
    original_as_tensor = torch.as_tensor

    def guarded_as_tensor(value, *args, **kwargs):
        assert not isinstance(value, np.ndarray)
        return original_as_tensor(value, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", guarded_as_tensor)

    loaded = load_sid_mask(artifact, fft_bins=4)

    assert loaded.dtype == torch.bool
    assert loaded.tolist() == [True, False, True, False]


def test_sid_residual_energy_is_bounded_relative_to_raw_embedding():
    module = SIDFFT96Residual(
        embedding_dim=8,
        mode="sid",
        mask=torch.ones(64, dtype=torch.bool),
        residual_scale=1.0,
        max_residual_ratio=0.10,
    )
    with torch.no_grad():
        for parameter in module.projector.parameters():
            parameter.fill_(100.0)
    iq = torch.randn(4, 2, 64)
    z_raw = torch.randn(4, 8)

    output = module(iq, z_raw)

    allowed = 0.10 * z_raw.norm(dim=1)
    assert torch.all(output["sid_delta"].norm(dim=1) <= allowed + 1e-6)
    assert torch.allclose(output["z_sid"], z_raw + output["sid_delta"])
    assert torch.all(output["sid_delta_raw"].norm(dim=1) >= output["sid_delta"].norm(dim=1))
