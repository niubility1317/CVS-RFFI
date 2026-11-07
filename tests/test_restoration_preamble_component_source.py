import torch

from experiments.pamf_phase3_restoration.preamble_component_source import (
    EDGE_HYBRIDS,
    EDGE_REUSED,
    components,
    counterfactuals,
)


def test_exact_component_support_and_orthogonality():
    gen = torch.Generator().manual_seed(7)
    clean, base = torch.randn(2, 3, 2, 256, generator=gen, dtype=torch.float64)
    repeat, contrast, outside = components(base, clean)
    torch.testing.assert_close(repeat + contrast + outside, clean - base)
    assert not (repeat[:, :, :80].any() or repeat[:, :, 160:].any())
    assert not (contrast[:, :, :80].any() or contrast[:, :, 160:].any())
    assert not outside[:, :, 80:160].any()
    torch.testing.assert_close((repeat * contrast).sum((1, 2)), torch.zeros(3, dtype=base.dtype))


def test_known_symbol_phase_and_zero_energy_packet():
    gen = torch.Generator().manual_seed(13)
    pattern = torch.randn(2, 20, generator=gen, dtype=torch.complex128)
    phase = torch.exp(0.83j * torch.arange(4, dtype=torch.float64))
    rotated = (phase[None, :, None] * pattern[:, None, :]).reshape(2, 80)
    base = torch.zeros(2, 2, 256, dtype=torch.float64)
    base[0, :, 80:160] = torch.stack((rotated[0].real, rotated[0].imag))
    clean = base.clone()
    clean[0, :, 80:160] += torch.stack((rotated[1].real, rotated[1].imag))
    clean[1, 0, 80:160] = 1
    repeat, contrast, outside = components(base, clean)
    torch.testing.assert_close(repeat, clean - base)
    torch.testing.assert_close(contrast, torch.zeros_like(base))
    assert not outside.any()


def test_each_subset_has_expected_residual_energy():
    gen = torch.Generator().manual_seed(17)
    base, clean = torch.randn(2, 6, 2, 256, generator=gen, dtype=torch.float64)
    outputs, energies = counterfactuals(base, clean)
    assert set(outputs) == set(range(1, 7))
    for mask, output in outputs.items():
        omitted = [bit for bit in range(3) if not mask & (1 << bit)]
        torch.testing.assert_close((clean - output).square().sum((1, 2)), energies[:, omitted].sum(1))


def test_front_tail_split_preserves_original_energy_and_mask_meaning():
    gen = torch.Generator().manual_seed(19)
    base, clean = torch.randn(2, 3, 2, 256, generator=gen, dtype=torch.float64)
    _, old_energy = counterfactuals(base, clean)
    hybrid, energy = counterfactuals(base, clean, edge_split=True)
    assert set(hybrid) == set(EDGE_HYBRIDS)
    assert sorted(EDGE_HYBRIDS + EDGE_REUSED) == list(range(16))
    torch.testing.assert_close(energy[:, :2], old_energy[:, :2])
    torch.testing.assert_close(energy[:, 2:].sum(1), old_energy[:, 2])
    torch.testing.assert_close(hybrid[4][:, :, :80], clean[:, :, :80])
    torch.testing.assert_close(hybrid[4][:, :, 80:], base[:, :, 80:])
    torch.testing.assert_close(hybrid[8][:, :, :160], base[:, :, :160])
    torch.testing.assert_close(hybrid[8][:, :, 160:], clean[:, :, 160:])
