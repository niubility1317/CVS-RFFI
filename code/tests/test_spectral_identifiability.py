import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import spectral_identifiability as spectral  # noqa: E402
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


def test_clustered_bootstrap_reports_nontrivial_stable_selection_probability():
    rng = np.random.default_rng(19)
    accumulator = SpectralIdentifiabilityAccumulator(num_bands=2, feature_dim=1)
    for cluster in range(12):
        tx = cluster % 2
        rx = (cluster // 2) % 3
        for view in (0, 1):
            descriptor = np.asarray(
                [[5.0 * tx + 0.01 * cluster], [rng.normal() + 0.3 * view]],
                dtype=np.float64,
            )
            accumulator.update(
                descriptor,
                tx=tx,
                rx=rx,
                day=cluster % 2,
                view=view,
                cluster=cluster,
            )

    stats = accumulator.finalize(
        bootstrap_repeats=64,
        bootstrap_keep_fraction=0.5,
        bootstrap_seed=71,
    )

    probability = stats["bootstrap_selection_probability"]
    assert probability[0] >= 0.8
    assert probability[1] <= 0.2
    assert not np.all(probability == 1.0)
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


def _iq_from_shifted_spectrum(spectrum: torch.Tensor) -> torch.Tensor:
    signal = torch.fft.ifft(torch.fft.ifftshift(spectrum, dim=-1), dim=-1)
    return torch.stack((signal.real, signal.imag), dim=1)


def test_phase_features_do_not_create_edges_across_discontinuous_mask_segments():
    fft_bins = 64
    mask = torch.zeros(fft_bins, dtype=torch.bool)
    mask[[10, 11, 40, 41]] = True
    base = torch.zeros(1, fft_bins, dtype=torch.complex64)
    base[0, 10] = torch.polar(torch.tensor(1.0), torch.tensor(0.0))
    base[0, 11] = torch.polar(torch.tensor(1.0), torch.tensor(0.2))
    base[0, 40] = torch.polar(torch.tensor(1.0), torch.tensor(0.4))
    base[0, 41] = torch.polar(torch.tensor(1.0), torch.tensor(0.6))
    shifted = base.clone()
    shifted[0, 40:42] *= torch.polar(torch.tensor(1.0), torch.tensor(1.1))

    first, _ = extract_sid_fft96(_iq_from_shifted_spectrum(base), mask, mode="phase")
    second, _ = extract_sid_fft96(_iq_from_shifted_spectrum(shifted), mask, mode="phase")

    assert torch.allclose(first, second, atol=2e-5, rtol=0)


def test_exact_fftshift_mirror_indices_include_dc_and_nyquist_fixed_points():
    mirror_indices = getattr(spectral, "exact_fftshift_mirror_indices", None)
    assert mirror_indices is not None

    assert mirror_indices(8).tolist() == [0, 7, 6, 5, 4, 3, 2, 1]


def test_quadratic_log_amplitude_trend_is_removed_before_identity_pooling():
    fft_bins = 64
    frequency = torch.linspace(-1.0, 1.0, fft_bins)
    log_amplitude = 0.3 + 0.4 * frequency + 0.25 * frequency.square()
    trend_residual = getattr(spectral, "quadratic_log_amplitude_residual", None)
    assert trend_residual is not None

    residual, _ = trend_residual(
        log_amplitude.unsqueeze(0),
        torch.ones(fft_bins, dtype=torch.bool),
    )

    assert residual.abs().max().item() < 2e-4


def test_quality_vector_tracks_low_energy_bins_and_has_finite_gradients():
    fft_bins = 64
    strong_spectrum = torch.zeros(1, fft_bins, dtype=torch.complex64)
    strong_spectrum[0, 20:44] = 1.0 + 0.2j
    strong_iq = _iq_from_shifted_spectrum(strong_spectrum).requires_grad_(True)
    faded_iq = torch.zeros_like(strong_iq, requires_grad=True)
    mask = torch.ones(fft_bins, dtype=torch.bool)

    strong_feature, strong_diag = extract_sid_fft96(strong_iq, mask, mode="sid")
    faded_feature, faded_diag = extract_sid_fft96(faded_iq, mask, mode="sid")
    (strong_feature.square().sum() + faded_feature.square().sum()).backward()

    assert strong_diag["quality"].shape == (1, 7)
    assert faded_diag["quality"].shape == (1, 7)
    assert strong_diag["valid_bin_ratio"].item() > faded_diag["valid_bin_ratio"].item()
    assert torch.isfinite(strong_iq.grad).all()
    assert torch.isfinite(faded_iq.grad).all()


def test_hierarchical_identifiability_separates_receiver_main_effect_from_tx_interaction():
    common = SpectralIdentifiabilityAccumulator(num_bands=1, feature_dim=1)
    interaction = SpectralIdentifiabilityAccumulator(num_bands=1, feature_dim=1)
    for tx in (0, 1):
        for rx in (0, 1):
            common.update(
                np.asarray([[10.0 * tx + 3.0 * rx]]),
                tx=tx,
                rx=rx,
                day=0,
                view=0,
            )
            interaction.update(
                np.asarray([[10.0 * tx + 3.0 * rx + 4.0 * tx * rx]]),
                tx=tx,
                rx=rx,
                day=0,
                view=0,
            )

    common_stats = common.finalize()
    interaction_stats = interaction.finalize()

    assert common_stats["rx_main_scatter"][0] > 0.0
    assert common_stats["tx_rx_interaction_scatter"][0] == pytest.approx(0.0, abs=1e-12)
    assert interaction_stats["tx_rx_interaction_scatter"][0] > 0.0
    assert "tx_day_interaction_scatter" in common_stats
    assert "tx_view_interaction_scatter" in common_stats


def test_hsid_role_masks_are_deterministic_and_disjoint():
    selector = getattr(spectral, "select_hsid_role_masks", None)
    assert selector is not None
    stats = {
        "j_score": np.asarray([9.0, 8.0, 1.0, 0.5, 0.25, 0.1]),
        "nonlinear_score": np.asarray([0.5, 0.4, 9.0, 8.0, 0.2, 0.1]),
        "domain_score": np.asarray([0.1, 0.2, 0.3, 0.4, 9.0, 8.0]),
        "bootstrap_selection_probability": np.ones(6),
    }

    masks = selector(stats, common_fraction=1 / 3, nonlinear_fraction=1 / 3, domain_fraction=1 / 3)

    assert np.flatnonzero(masks["common_mask"]).tolist() == [0, 1]
    assert np.flatnonzero(masks["nonlinear_mask"]).tolist() == [2, 3]
    assert np.flatnonzero(masks["domain_mask"]).tolist() == [4, 5]


def test_hsid_role_masks_apply_the_declared_dc_notch_to_every_role():
    scores = np.arange(9, dtype=np.float64)
    stats = {
        "j_score": scores,
        "nonlinear_score": scores[::-1].copy(),
        "domain_score": np.roll(scores, 2),
        "bootstrap_selection_probability": np.ones(9),
    }

    masks = spectral.select_hsid_role_masks(
        stats,
        common_fraction=0.3,
        nonlinear_fraction=0.3,
        domain_fraction=0.3,
        dc_notch=1,
    )

    for mask in masks.values():
        assert not mask[3:6].any()


def test_independent_hsid_prototype_evidence_is_raw_preserving_and_bounded():
    evidence_cls = getattr(spectral, "HSIDPrototypeEvidence", None)
    assert evidence_cls is not None
    module = evidence_cls(num_classes=3, spectral_dim=16, alpha_max=0.2)
    features = torch.randn(4, 96)
    raw_logits = torch.randn(4, 3)
    quality = torch.randn(4, 7)

    initial = module(features, raw_logits, quality)
    assert torch.equal(initial["fused_logits"], raw_logits)
    assert initial["spectral_embedding"].shape == (4, 16)
    assert initial["spectral_logits"].shape == (4, 3)
    assert torch.all(initial["fusion_gate"] == 0.0)

    with torch.no_grad():
        module.fusion_alpha.fill_(1.0)
    active = module(features, raw_logits, quality)
    assert torch.all(active["fusion_gate"] >= 0.0)
    assert torch.all(active["fusion_gate"] <= 0.2 + 1e-7)


def test_hsid_fusion_alpha_can_recover_from_negative_optimizer_state():
    torch.manual_seed(23)
    module = spectral.HSIDPrototypeEvidence(num_classes=3, spectral_dim=16, alpha_max=0.2)
    with torch.no_grad():
        module.fusion_alpha.fill_(-1e-3)
    output = module(torch.randn(4, 96), torch.randn(4, 3), torch.randn(4, 7))

    assert torch.count_nonzero(output["fusion_gate"]) == 0
    output["fused_logits"][:, 0].sum().backward()
    assert module.fusion_alpha.grad is not None
    assert module.fusion_alpha.grad.abs().item() > 0.0
