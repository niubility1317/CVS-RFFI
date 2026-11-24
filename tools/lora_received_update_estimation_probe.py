"""Check receiver-only LoRa gain estimates and variance proxies under isolated perturbations."""

import json
from math import log

import torch

from experiments.pamf_lora_main.received_update import ReceivedUpdate


def summary(value):
    return value.quantile(torch.tensor([.1, .5, .9])).tolist()


def run(count=512):
    rng = torch.Generator().manual_seed(2026112427)
    t = torch.arange(256, dtype=torch.float32) / 256
    phase = torch.rand((count, 6, 1), generator=rng) * 2 * torch.pi
    amp = .05 * torch.rand((count, 6, 1), generator=rng)
    source = (1 + amp * torch.cos(2 * torch.pi * 12 * t + phase)) * torch.exp(
        .08j * torch.sin(2 * torch.pi * 19 * t - phase))
    perturbation = (.14 + .07j) * torch.cos(2 * torch.pi * (.4 * t + .15))
    perturbation = perturbation + (.10 - .15j) * torch.sin(2 * torch.pi * .3 * t)
    direction = torch.tensor([-.8, -.8, -.2, -.2, 1., 1.]).reshape(1, 6, 1)
    action = (1 + direction * perturbation).expand_as(source)
    altered = source * action
    white = torch.randn(source.shape, dtype=source.dtype, generator=rng)
    shared = torch.randn((count, 1, 256), dtype=source.dtype, generator=rng)
    frequency = torch.fft.fftfreq(256).abs()
    colored = torch.fft.ifft(torch.fft.fft(shared, norm='ortho') * (frequency <= .16), norm='ortho')
    noises = {'independent_white': white, 'shared_colored': .92 * colored + .08 * white}
    results = {}
    for name, raw_noise in noises.items():
        noise = raw_noise / raw_noise.abs().square().mean((1, 2), keepdim=True).sqrt()
        results[name] = {}
        for snr in (0, 6, 18):
            epsilon = noise * altered.abs().square().mean((1, 2), keepdim=True).sqrt() * 10 ** (-snr / 20)
            received = altered + epsilon
            spectrum = torch.fft.fft(received, norm='ortho')
            outer = spectrum[..., frequency > .25].abs().square().median(-1, keepdim=True).values / log(2.)
            estimate, _, held = ReceivedUpdate.estimate(received, outer)
            # Each true channel group has the same mean, fixing the relative
            # gauge. The correct source and noise are used only below.
            error = (estimate - action).abs().square().mean((1, 2)).sqrt()
            identity = (1 - action).abs().square().mean((1, 2)).sqrt()
            exact_noise = epsilon.abs().square().mean((1, 2))
            predictor = (estimate * source - received).abs().square().mean((1, 2))
            passthrough = (source - received).abs().square().mean((1, 2))
            results[name][snr] = {
                'relative_action_rmse_quantiles': summary(error),
                'identity_rmse': float(identity[0]),
                'improved_action_fraction': float((error < identity).float().mean()),
                'outer_to_real_noise_quantiles': summary(outer.mean((1, 2)) / exact_noise),
                'hold_to_real_noise_quantiles': summary(held.mean((1, 2)) / exact_noise),
                'forward_residual_ratio': summary(predictor / passthrough),
            }
    return dict(n=count, normalization='dechirped 256 kS/s, same true action gauge in both folds',
                diagnostics_only=True, routes=results)


if __name__ == '__main__':
    print(json.dumps(run(), allow_nan=False))
