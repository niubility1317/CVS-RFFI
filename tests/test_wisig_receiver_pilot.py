import torch

from experiments.pamf_repl_view_shuffle.receiver_pilot import (
    PERIOD, READOUTS, SIGNS, TONES, channel_estimate, estimate, restore,
)


def preamble(delay=0):
    z = torch.zeros(PERIOD, dtype=torch.complex64)
    z[list(TONES)] = torch.tensor(SIGNS, dtype=torch.float32).to(torch.complex64) * (1 + 1j)
    return torch.fft.ifft(z, norm='ortho').roll(delay).repeat(13)[:256]


def test_observed_pilot_separates_cfo_and_dominant_delay_from_relative_response():
    time = torch.arange(256)
    alpha = .014
    clean = (1 - 2j) * preamble(4)
    iq = clean[None] * torch.exp(1j * alpha * (time - 120))
    obs = estimate(iq)
    taps, _, confidence, delay, mismatch = channel_estimate(obs)
    assert int(delay.item()) == 4
    torch.testing.assert_close(obs.slope, torch.tensor([alpha]), atol=2e-7, rtol=0)
    torch.testing.assert_close(obs.aligned, clean[None], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(taps, torch.tensor([[0, 0, 1, 0, 0]], dtype=taps.dtype), atol=2e-5, rtol=2e-5)
    assert confidence.item() > .99 and mismatch.item() < 1e-10
    output, info = restore(iq, clean_floor=.01)
    assert tuple(output) == READOUTS and info['cfo_hz'].item() > 0
    torch.testing.assert_close(output['eq_keep_cfo'], iq, atol=2e-5, rtol=2e-5)
    assert torch.equal(output['repeat_wiener'], iq)


def test_repetition_control_preserves_the_unobserved_preamble_tail():
    ideal = preamble()[None].expand(16, -1)
    rng = torch.Generator().manual_seed(8)
    noise = .25 * torch.complex(
        torch.randn(ideal.shape, generator=rng), torch.randn(ideal.shape, generator=rng))
    observed = ideal + noise
    clean = estimate(ideal)
    obs = estimate(observed)
    assert clean.noise_power.max() == 0
    assert obs.noise_power.median() > .05
    output, _ = restore(observed, clean_floor=.01)
    filtered = output['repeat_wiener']
    assert torch.equal(filtered[:, :40], observed[:, :40])
    assert torch.equal(filtered[:, 200:], observed[:, 200:])
    assert (filtered-ideal).abs().square().mean() < (observed-ideal).abs().square().mean()
    for x in output.values():
        assert x.shape == observed.shape and torch.isfinite(x).all()
