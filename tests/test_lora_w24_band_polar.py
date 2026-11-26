import torch

from experiments.pamf_lora_main.w24_band_polar import band_interventions


def test_polar_controls_change_only_the_requested_factor():
    learn = torch.zeros(1, 8, 384, dtype=torch.complex64)
    noise = torch.zeros_like(learn)
    learn[..., 10], learn[..., -30], learn[..., 35] = 2j, 1, 4j
    noise[..., 10], noise[..., -30], noise[..., 35] = -3, -4j, 3j
    phase, amplitude, same, exact = band_interventions(learn, noise)
    torch.testing.assert_close(phase[..., 10], torch.full_like(phase[..., 10], 2))
    torch.testing.assert_close(phase[..., -30], torch.full_like(phase[..., -30], 1j))
    torch.testing.assert_close(amplitude[..., 10] / amplitude[..., -30], torch.full_like(amplitude[..., 10], .75j))
    for result in (phase, amplitude, same):
        torch.testing.assert_close(result.abs().square().sum(), learn.abs().square().sum())
    for result in (phase, amplitude, same, exact):
        assert torch.equal(result[..., 35], learn[..., 35])
    torch.testing.assert_close(exact[..., 10], -noise[..., 10])
    torch.testing.assert_close(exact[..., -30], -noise[..., -30])


def test_absent_exact_noise_leaves_all_branches_at_frozen_half():
    learn = torch.ones(2, 8, 384, dtype=torch.complex64)
    for result in band_interventions(learn, torch.zeros_like(learn)):
        assert torch.equal(result, learn)
