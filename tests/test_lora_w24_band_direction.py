import torch

from experiments.pamf_lora_main.w24_band_direction import BANDS, matched_spectra


def test_one_band_replaces_direction_at_fixed_edit_energy():
    learned = torch.zeros((1, 8, 384), dtype=torch.complex64)
    noise = torch.zeros_like(learned)
    for _, lo, hi in BANDS:
        learned[..., lo] = 2j
        noise[..., -hi] = -7
    learned[..., 0] = 0.3
    noise[..., 0] = -40
    patched, masks, measurements = matched_spectra(learned, noise)

    assert measurements.shape == (1, len(BANDS), 5)
    for i, (_, lo, hi) in enumerate(BANDS):
        mask = masks[i]
        edit = torch.where(mask, patched, learned)
        assert edit[0, 0, lo] == 0
        torch.testing.assert_close(edit[0, 0, -hi], learned.new_tensor(2))
        assert edit[0, 0, 0] == learned[0, 0, 0]
        torch.testing.assert_close(edit[..., ~mask], learned[..., ~mask], rtol=0, atol=0)
        torch.testing.assert_close(edit.abs().square().sum(), learned.abs().square().sum())
    torch.testing.assert_close(patched.abs().square().sum(), learned.abs().square().sum())


def test_empty_noise_or_learned_band_stays_at_the_observed_edit():
    learned = torch.zeros((2, 8, 384), dtype=torch.complex64)
    noise = torch.zeros_like(learned)
    learned[0, :, 8] = 1j
    noise[1, :, 9] = 1
    patched, _, info = matched_spectra(learned, noise)

    torch.testing.assert_close(patched, learned, rtol=0, atol=0)
    assert torch.isfinite(info).all()
    assert not info[..., -1].any()
