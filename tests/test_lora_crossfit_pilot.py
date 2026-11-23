import torch

from experiments.pamf_lora_main.crossfit_pilot import (
    LAGS, apply_transfer, estimate_shared, regularized_inverse,
)
from experiments.pamf_lora_main.physical import ChirpResponse


def make_observation(split_conflict=False):
    reference = ChirpResponse().chirp.expand(1, 6, -1)
    taps = torch.zeros(1, len(LAGS), dtype=torch.complex64)
    taps[:, LAGS.index(2)] = .2 - .08j
    waveform = apply_transfer(reference, taps)
    if split_conflict:
        waveform[:, 1::2] = apply_transfer(reference[:, 1::2], -taps)
    return waveform * reference.conj(), reference


def test_crossfit_shares_only_predictive_response_and_keeps_carrier():
    observed, reference = make_observation()
    variance = torch.zeros(1, 6, 1)
    quality = torch.ones_like(variance)
    transfer, support = estimate_shared(observed, reference, quality, variance)
    assert transfer.shape == (1, 1, 256) and support.shape == (1,)
    assert support.item() > .9
    assert ((transfer - 1).abs() <= .451).all()
    restored = regularized_inverse(observed, observed, reference, variance, transfer)
    torch.testing.assert_close(restored.mean(-1), observed.mean(-1))
    old = (observed - observed.mean(-1, keepdim=True)).abs().square().mean()
    new = (restored - restored.mean(-1, keepdim=True)).abs().square().mean()
    assert new < .05 * old


def test_conflicting_or_empty_groups_fall_back_to_identity():
    observed, reference = make_observation(split_conflict=True)
    variance = torch.zeros(1, 6, 1)
    for wave, quality in [(observed, torch.ones_like(variance)),
                          (observed, torch.zeros_like(variance)),
                          (torch.zeros_like(observed), torch.ones_like(variance))]:
        transfer, support = estimate_shared(wave, reference, quality, variance)
        torch.testing.assert_close(support, torch.zeros_like(support), atol=0, rtol=0)
        torch.testing.assert_close(transfer, torch.ones_like(transfer), atol=0, rtol=0)
