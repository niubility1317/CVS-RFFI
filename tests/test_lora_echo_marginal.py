import torch

from experiments.pamf_lora_main.crossfit_pilot import paired_action_inputs
from experiments.pamf_lora_main.echo_marginal import (
    CommonSpectrum, echo_log_evidence, marginal_system, posterior_update,
)


def test_nuisance_marginal_matches_dense_posterior_and_evidence():
    generator = torch.Generator().manual_seed(720)
    noise = torch.rand(3, 1, generator=generator) + .2
    prior = torch.rand(6, generator=generator) + .2
    columns = .5 * torch.randn(3, 6, 4, generator=generator)
    residual = torch.randn(3, 6, generator=generator)
    system = marginal_system(prior, noise, columns)
    point, echo = posterior_update(residual, prior, system)
    h0 = torch.diag_embed(prior + noise)
    h1 = h0 + columns @ columns.mT
    result = torch.linalg.solve(h1, residual[..., None]).squeeze(-1)
    torch.testing.assert_close(point - echo, prior * result)
    score = .5 * ((residual * (residual / (prior + noise) - result)).sum(-1)
                  + torch.linalg.slogdet(h0).logabsdet - torch.linalg.slogdet(h1).logabsdet)
    torch.testing.assert_close(echo_log_evidence(residual, system), score)
    no_echo = marginal_system(prior, noise, torch.zeros_like(columns))
    _, delta = posterior_update(residual, prior, no_echo)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0, rtol=0)
    torch.testing.assert_close(echo_log_evidence(residual, no_echo), torch.zeros(3), atol=0, rtol=0)


def test_injects_only_common_band_and_keeps_packet_differences():
    generator = torch.Generator().manual_seed(721)
    spectrum = CommonSpectrum()
    wave = torch.randn(2, 6, 256, generator=generator, dtype=torch.complex64)
    weights = wave.real.new_full((len(wave), 6, 1), 1/6)
    current = spectrum.vector(wave, weights)
    desired = current * .82
    out = spectrum.install(wave, current, desired)
    torch.testing.assert_close(spectrum.vector(out, weights), desired)
    torch.testing.assert_close(out[:, 1:] - out[:, :1], wave[:, 1:] - wave[:, :1])
    fft = torch.fft.fft(out - wave, norm='ortho')
    protected = torch.ones(256, dtype=torch.bool)
    protected[spectrum.bins] = False
    torch.testing.assert_close(fft[..., protected], torch.zeros_like(fft[..., protected]), atol=3e-7, rtol=0)


def test_noise_only_pairs_keep_the_controlled_waveform_difference():
    source = torch.ones((4, 6, 256), dtype=torch.complex64)
    change = torch.full_like(source, .05j)
    batch = paired_action_inputs(source, lambda x, _: x + .1, torch.Generator().manual_seed(725),
                                 change, with_identity=True, echo_probability=0.)
    torch.testing.assert_close(batch[:, 2] - batch[:, 0], change)
    torch.testing.assert_close(batch[-2:, 0], source[-2:], atol=0, rtol=0)
    torch.testing.assert_close(batch.imag[:, :2], torch.zeros_like(batch.imag[:, :2]), atol=0, rtol=0)
