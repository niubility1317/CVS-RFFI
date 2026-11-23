import numpy as np
import pytest
import torch

from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import replay
from experiments.pamf_lora_main.fixed_action_noise import paired
from experiments.pamf_unrolled_restore.signal_retention import retention_loss


@pytest.mark.parametrize('route', ['full', 'residual'])
def test_re_noise_keeps_registered_iq_and_the_noiseless_waveform(route):
    clock = np.arange(8192) % 1024
    raw = np.exp(2j*np.pi*clock*(clock/1024-.5)*.055).astype(np.complex64)
    key = (raw, VIEWS[-1], SOURCE[0], 'fixed-action-source-test')
    original, new = paired(key, route, 'source-V')
    ordinary, noise = replay(route, VIEWS[-1], (raw, SOURCE[0], key[-1]))
    assert np.array_equal(original, ordinary)
    np.testing.assert_allclose(new[1], ordinary-noise, atol=7e-7, rtol=3e-6)
    assert not np.array_equal(new[0], original)
    assert all(np.array_equal(a, b) for a, b in zip((original, new), paired(key, route, 'source-V')))


def test_retention_penalizes_weak_band_shrinkage_and_backpropagates():
    time = torch.arange(256)
    signal = .06*torch.exp(2j*torch.pi*5*time/256) + .025j*torch.exp(-2j*torch.pi*18*time/256)
    signal = signal[None, None].expand(4, 6, -1)
    exact = torch.stack((signal, signal), dim=1)
    assert retention_loss(exact, signal) == 0
    scaled = (exact*.65).requires_grad_()
    penalty = retention_loss(scaled, signal)
    penalty.backward()
    assert penalty > .1 and scaled.grad.norm() > 0
