"""Wideband observation-consistent unfolding for aligned complex windows."""

import torch
from torch import nn

from experiments.pamf_reception_restore.models import Core, mirror_spectrum


EDIT_RMS = .18


def action(value, main, image):
    return main * value + image * mirror_spectrum(value)


def adjoint(value, main, image):
    return main.conj() * value + mirror_spectrum(image.conj() * value)


def bounded_edit(delta):
    power = delta.abs().square().mean(dim=-1, keepdim=True)
    return delta * (1. + power / EDIT_RMS**2).rsqrt()


class JointUnroll(nn.Module):
    """Reestimate complex action and noise after each accepted waveform update."""

    def __init__(self):
        super().__init__()
        self.steps = nn.ModuleList(Core(incoming=16, outgoing=7) for _ in range(3))

    def forward(self, received, coherence, *, iterative=True):
        observed = torch.fft.fft(received, norm='ortho')
        packet_weight = coherence.square().clamp_min(.001)
        consensus = (observed * packet_weight).sum(1, keepdim=True) / packet_weight.sum(1, keepdim=True)
        consensus = consensus.expand_as(observed)
        x = observed
        main, image = torch.ones_like(x), torch.zeros_like(x)
        logvar = (1. - coherence).clamp(.002, .7).log().expand_as(x.real)
        prior = torch.zeros_like(x)
        stages = []
        for step in self.steps:
            innovation = observed - action(x, main, image)
            values = torch.stack((observed, x, consensus, innovation, main - 1., image, prior), dim=2)
            features = torch.cat((values.real, values.imag, logvar.unsqueeze(2) / 6.,
                                  coherence.expand_as(logvar).unsqueeze(2)), dim=2)
            raw = step(features.flatten(0, 1)).unflatten(0, observed.shape[:2])
            a, b, c, d, variance, p, q = raw.unbind(2)
            main = 1. + torch.complex((main.real - 1. + .35 * a).clamp(-.75, 1.),
                                      (main.imag + .35 * b).clamp(-.85, .85))
            image = torch.complex((image.real + .18 * c).clamp(-.35, .35),
                                  (image.imag + .18 * d).clamp(-.35, .35))
            logvar = (logvar + variance).clamp(-9., 1.)
            prior = prior + .10 * torch.complex(p, q)
            prediction = x + prior
            error = observed - action(prediction, main, image)
            # The learned variance relaxes observation consistency in noisy bins.
            inverse = adjoint(error * (.06 / (.06 + logvar.exp())), main, image)
            estimate = prediction + inverse / (1. + main.abs().square() + image.abs().square())
            detail = estimate - observed
            detail = torch.cat((torch.zeros_like(detail[..., :1]), detail[..., 1:]), dim=-1)
            estimate = observed + bounded_edit(detail)
            stages.append((estimate, main, image, logvar.exp()))
            if iterative:
                x = estimate
                prior = torch.zeros_like(prior)
        return stages


def joint_loss(stages, clean, noise, received):
    """Keep the inferred action honest on real source replay components."""
    wanted, epsilon, observed = (torch.fft.fft(x, norm='ortho') for x in (clean, noise, received))
    noiseless = observed - epsilon
    bins = torch.fft.fftfreq(clean.shape[-1], device=clean.device).abs()
    bands = ((bins > 0.) & (bins <= 1/32), (bins > 1/32) & (bins <= 1/8), bins > 1/8)
    detail_risk = []
    physical_risk = []
    for estimate, main, image, variance in stages:
        energy = wanted.abs().square()
        error = (estimate - wanted).abs().square()
        detail_risk.append(torch.stack([error[..., band].mean(-1) /
                                      (.015 + energy[..., band].mean(-1)) for band in bands]).mean())
        clean_replay = (action(wanted, main, image) - noiseless).abs().square().mean()
        solved_replay = (action(estimate, main, image) - noiseless).abs().square().mean()
        # Estimate noise power per band in expectation, not the replay's noise phasor.
        estimated = torch.stack([variance[..., b].mean(-1) for b in bands]).clamp_min(1e-5)
        actual = torch.stack([epsilon[..., b].abs().square().mean(-1) for b in bands])
        noise_risk = (estimated.log() + actual / estimated).mean()
        physical_risk.append(.15 * clean_replay + .08 * solved_replay + .015 * noise_risk)
    # Independent source draws are in the same demodulated coordinates.
    final_shape = stages[-1][0][..., 1:]
    repeat = (final_shape[::2] - final_shape[1::2]).abs().square().mean()
    primary = .65 * detail_risk[-1] + .35 * torch.stack(detail_risk).mean()
    return primary + torch.stack(physical_risk).mean() + .15 * repeat, detail_risk[-1]
