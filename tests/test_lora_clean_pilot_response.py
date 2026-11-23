import torch

from experiments.pamf_lora_main.clean_pilot_response import (
    CleanPilotResponse, preserve_corroborated_edit, relative_response,
)


def test_clean_response_removes_repeatable_hardware_and_leaves_supported_direction():
    frequency = torch.fft.fftfreq(256)
    # Odd real and even zero-mean imaginary responses are orthogonal.
    hardware = .1j * (frequency.square() - frequency[frequency.abs() <= .25].square().mean())
    channel = .05 * frequency
    observed_clean = (1 + hardware + torch.linspace(-1, 1, 20)[:, None] * hardware).unsqueeze(1)
    calibration = CleanPilotResponse()
    calibration.fit(observed_clean)
    support = torch.ones(2)
    response = calibration(torch.stack((1 + hardware, 1 + hardware + channel))[:, None], support)
    assert (response[0] - 1).abs().max() < 1e-6
    assert (response[1] - (1 + channel)).abs().max() < 1e-6
    no_evidence = calibration(torch.ones_like(response), torch.zeros_like(support))
    torch.testing.assert_close(no_evidence, torch.ones_like(response), rtol=0, atol=0)
    scaled = (1 + hardware + channel)[None, None, :] * (2 - 3j)
    torch.testing.assert_close(relative_response(scaled), relative_response(scaled / (2 - 3j)))


def test_corroboration_protects_aligned_component_and_keeps_unrelated_detail():
    direction = torch.tensor([[[1, -1, 0, 0]]], dtype=torch.complex64)
    unrelated = torch.tensor([[[0, 0, 1j, -1j]]], dtype=torch.complex64)
    original = torch.zeros_like(direction)
    coarse = .6 * direction + unrelated
    fine = .2 * direction + 2 * unrelated
    actual = preserve_corroborated_edit(original, coarse, fine, direction)
    torch.testing.assert_close(actual, .6 * direction + 2 * unrelated)
    torch.testing.assert_close(preserve_corroborated_edit(original, -coarse, fine, direction), fine)
    torch.testing.assert_close(preserve_corroborated_edit(original, coarse, fine, original), fine)
    torch.testing.assert_close(preserve_corroborated_edit(original, coarse, 2 * coarse, direction), 2 * coarse)
    opposing_fine = -.2 * direction + 2 * unrelated
    torch.testing.assert_close(preserve_corroborated_edit(original, 10 * direction, opposing_fine, direction),
                               .8 * direction + 2 * unrelated)
