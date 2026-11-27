import torch

from experiments.pamf_lora_main.w24_online_additive_source import sample_additive


def test_full_block_agc_scales_retained_observation_and_source_together():
    delay = 31
    signal = torch.full((2, 8192 + delay), 4., dtype=torch.complex64)
    signal[0, :delay] = 20.
    main = torch.zeros_like(signal)
    received, source = sample_additive(signal, main, main, delay, torch.tensor([20., 3.]))
    expected_gain = torch.tensor([signal[0].abs().square().mean().rsqrt(), 10**(-3/20)])
    assert received.shape == (2, 8192)
    torch.testing.assert_close(received, signal[:, delay:] * expected_gain[:, None])
    torch.testing.assert_close(received, source, atol=0, rtol=0)


def test_noise_redraw_preserves_proper_input_and_image_covariance():
    signal = torch.zeros((12, 8192), dtype=torch.complex64)
    alpha, beta = 0.75 + .10j, .15 - .12j
    main, image = torch.full_like(signal, alpha), torch.full_like(signal, beta)
    def draw(seed):
        y, s = sample_additive(signal, main, image, 0, torch.zeros(12),
                               generator=torch.Generator().manual_seed(seed))
        assert torch.equal(s, signal)
        return y

    first, second = draw(7), draw(8)
    assert abs(first.abs().square().mean() - (abs(alpha)**2 + abs(beta)**2)) < .01
    assert abs(first.square().mean() - 2*alpha*beta) < .01
    assert abs((first * second.conj()).mean()) < .01
