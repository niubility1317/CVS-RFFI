import torch

from cvsrffi.phase1_fcr_physics import (
    FisherIdentifiabilityGate,
    FrozenFingerprintFeatureBank,
)


def _iq(batch_size: int = 2, length: int = 256) -> torch.Tensor:
    torch.manual_seed(31)
    return torch.randn(batch_size, 2, length)


def test_frozen_bank_has_no_parameters_and_returns_deterministic_named_finite_blocks() -> None:
    bank = FrozenFingerprintFeatureBank()
    signal = _iq()

    first = bank(signal)
    second = bank(signal)

    expected = {
        "iq_non_circularity",
        "am_am",
        "am_pm",
        "memory_residual",
        "spectral_shoulder",
        "phase_noise_psd",
        "amplitude_conditioned_residual",
        "cyclostationary",
    }
    assert sum(parameter.numel() for parameter in bank.parameters()) == 0
    assert set(first.blocks) == expected
    for name in expected:
        assert torch.isfinite(first.blocks[name]).all()
        torch.testing.assert_close(first.blocks[name], second.blocks[name])


def test_fisher_gate_detaches_quality_and_downweights_pa_for_low_papr() -> None:
    gate = FisherIdentifiabilityGate()
    gram = torch.eye(4, requires_grad=True)
    snr_db = torch.tensor([20.0], requires_grad=True)
    low_papr = torch.ones(1, 2, 256, requires_grad=True)
    index = torch.arange(256, dtype=torch.float32)
    high_complex = (1.0 + 4.0 * (index == 40).float()) * torch.exp(1j * index * 0.09)
    high_papr = torch.stack((high_complex.real, high_complex.imag)).unsqueeze(0).requires_grad_()

    low = gate(low_papr, gram, snr_db)
    high = gate(high_papr, gram, snr_db)

    assert low.block_weights["pa"].item() < high.block_weights["pa"].item()
    for output in (low, high):
        assert all(not value.requires_grad for value in output.quality.values())
        assert all(torch.isfinite(value).all() for value in output.block_weights.values())
        assert all(((value >= 0) & (value <= 1)).all() for value in output.block_weights.values())


def test_fisher_gate_degrades_degenerate_quality_without_nan_or_amplification() -> None:
    gate = FisherIdentifiabilityGate()
    bad = gate(torch.zeros(2, 2, 256), torch.zeros(4, 4), torch.tensor(float("nan")))

    assert bad.quality["gram_effective_rank"].max().item() == 0.0
    assert bad.quality["snr_quality"].max().item() == 0.0
    assert max(value.max().item() for value in bad.block_weights.values()) <= 1.0
    assert all(torch.isfinite(value).all() for value in bad.block_weights.values())
