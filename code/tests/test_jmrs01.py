from __future__ import annotations

import math

import pytest
import torch

from cvsrffi.jmrs01 import (
    ALLOWED_S0_ROWS,
    JMRS01Config,
    build_mechanism,
    mechanism_loss,
    validate_s0_rows,
)


def _tone_batch(batch: int = 4, length: int = 256) -> torch.Tensor:
    n = torch.arange(length, dtype=torch.float32)
    phase = 2.0 * math.pi * (0.071 * n + 0.00002 * n.square())
    signal = torch.complex(torch.cos(phase), torch.sin(phase))
    return torch.stack((signal.real, signal.imag), dim=0).repeat(batch, 1, 1)


def test_s0_registry_rejects_removed_symbol_dependent_d2() -> None:
    assert ALLOWED_S0_ROWS == ("M0", "R1", "R2", "D1", "P1", "P2", "S1")
    with pytest.raises(ValueError, match="known transmitted symbols"):
        validate_s0_rows(["M0", "D2"])


@pytest.mark.parametrize("row", ["R1", "R2", "D1", "P1", "P2", "S1"])
def test_mechanism_contract_is_finite_and_within_budget(row: str) -> None:
    torch.manual_seed(7)
    cfg = JMRS01Config(z_dim=64, num_classes=6, embedding_dim=32)
    model = build_mechanism(row, cfg)
    iq = _tone_batch()
    z_id = torch.randn(4, 64)

    output = model(iq=iq, z_id=z_id)

    assert output.embedding.shape == (4, 32)
    assert output.logits.shape == (4, 6)
    assert output.reliability.shape == (4,)
    assert torch.isfinite(output.embedding).all()
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.reliability).all()
    assert ((0.0 <= output.reliability) & (output.reliability <= 1.0)).all()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) <= 50_000


def test_rc_feature_correction_is_norm_bounded() -> None:
    cfg = JMRS01Config(z_dim=64, num_classes=6, correction_radius=0.25)
    model = build_mechanism("R1", cfg)
    output = model(iq=_tone_batch(), z_id=torch.randn(4, 64) * 100.0)

    norms = output.diagnostics["correction_norm"]
    assert torch.all(norms <= 0.250001)


def test_dsq_masks_spectral_nulls_without_nonfinite_ratios() -> None:
    cfg = JMRS01Config(z_dim=64, num_classes=6, spectral_mask_ratio=0.1)
    model = build_mechanism("D1", cfg)
    iq = torch.zeros(3, 2, 256)
    output = model(iq=iq, z_id=torch.randn(3, 64))

    assert torch.isfinite(output.embedding).all()
    assert torch.equal(output.reliability, torch.zeros(3))
    assert torch.equal(output.diagnostics["valid_bin_fraction"], torch.zeros(3))


@pytest.mark.parametrize("row", ["P1", "P2"])
def test_phase_innovation_disables_unobservable_low_amplitude_samples(row: str) -> None:
    cfg = JMRS01Config(z_dim=64, num_classes=6, amplitude_mask_ratio=0.2)
    model = build_mechanism(row, cfg)
    output = model(iq=torch.zeros(3, 2, 256), z_id=torch.randn(3, 64))

    assert torch.equal(output.reliability, torch.zeros(3))
    assert torch.equal(output.diagnostics["valid_sample_fraction"], torch.zeros(3))


def test_sham_features_are_reproducible_for_same_seed() -> None:
    cfg = JMRS01Config(z_dim=64, num_classes=6, seed=91)
    iq = _tone_batch()
    z_id = torch.randn(4, 64)
    a = build_mechanism("S1", cfg)(iq=iq, z_id=z_id).embedding
    b = build_mechanism("S1", cfg)(iq=iq, z_id=z_id).embedding

    torch.testing.assert_close(a, b)


def test_unified_loss_is_finite_and_backpropagates() -> None:
    cfg = JMRS01Config(z_dim=64, num_classes=6)
    model = build_mechanism("D1", cfg)
    clean = model(iq=_tone_batch(), z_id=torch.randn(4, 64))
    satellite = model(iq=_tone_batch() * 0.8, z_id=torch.randn(4, 64))
    labels = torch.tensor([0, 0, 1, 1])
    receivers = torch.tensor([0, 1, 0, 1])

    losses = mechanism_loss(clean, satellite, labels, receivers)

    assert set(losses) >= {"total", "ce", "clean_sat", "class_cond_rx", "tx_margin"}
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
