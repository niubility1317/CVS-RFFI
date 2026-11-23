"""Estimate a bounded forward response on disjoint received preamble groups."""

import torch

from .channel_blind import ChannelBlind, zero_carrier


LAGS = (-4, -3, -2, -1, 1, 2, 3, 4)
EDGE = max(abs(k) for k in LAGS)


def delayed(value):
    return torch.stack([value.roll(lag, dims=-1) for lag in LAGS], dim=-1)


def apply_transfer(signal, taps):
    return signal + (delayed(signal) * taps[:, None, None, :]).sum(-1)


def generic_taps(signal, rng):
    shape = (signal.shape[0], len(LAGS))
    kw = dict(device=signal.device, generator=rng)
    tap = torch.polar(torch.rand(shape, **kw), 2 * torch.pi * torch.rand(shape, **kw))
    taper = signal.real.new_tensor([1. / (1 + abs(lag)) for lag in LAGS])
    tap = tap * taper
    radius = .48 * torch.rand((len(signal), 1), **kw).square()
    return radius * tap / tap.abs().sum(-1, keepdim=True).clamp_min(1e-8)


def paired_action_inputs(clean, perturb, rng, change, *, with_identity):
    def draw():
        taps = generic_taps(clean, rng)
        filtered = apply_transfer(clean, taps)
        scale = .15 + .85 * torch.rand((len(clean), 1, 1), device=clean.device, generator=rng).square()
        measurement = filtered + scale * (perturb(filtered, rng) - filtered)
        return measurement, taps

    first, taps = draw()
    second, _ = draw()
    changed = first + apply_transfer(change, taps)
    if with_identity:
        first[-2:] = second[-2:] = clean[-2:]
        changed[-2:] = clean[-2:] + change[-2:]
    return torch.stack((first, second, changed), dim=1)


def centered(value, dim):
    return value - value.mean(dim=dim, keepdim=True)


@torch.no_grad()
def estimate_shared(observed, template, quality, noise_variance):
    """Fit two mutually held-out relative responses in the original IQ coordinates."""
    phase = template[..., EDGE:-EDGE]
    # Background direct-carrier amplitudes need not agree across the chirps.
    gain = observed.mean(-1, keepdim=True).real.clamp_min(.15)
    response = observed[..., EDGE:-EDGE] / gain
    residual = centered(response, -1) * phase
    columns = delayed(template)[..., EDGE:-EDGE, :] * phase.conj()[..., None]
    design = centered(columns, -2) * phase[..., None]
    weight = quality.squeeze(-1).square()
    ridge = .02 + .20 * noise_variance.mean(dim=(1, 2))
    eye = torch.eye(len(LAGS), dtype=design.dtype, device=design.device)

    def group(part):
        x = design[:, part].flatten(1, 2)
        y = residual[:, part].flatten(1, 2)
        weights = weight[:, part, None].expand(-1, -1, design.shape[-2]).flatten(1, 2)
        total = weights.sum(1).clamp_min(1e-8)
        xw = x * weights.sqrt()[..., None]
        yw = y * weights.sqrt()
        gram = (xw.mH @ xw) / total[:, None, None]
        cross = (xw.mH @ yw[..., None]) / total[:, None, None]
        beta = torch.linalg.solve_ex(gram + ridge[:, None, None] * eye, cross).result
        return beta, xw, yw

    def held_gain(group_data, other_beta):
        _, matrix, signal = group_data
        original = signal.abs().square().sum(-1)
        error = (signal - (matrix @ other_beta).squeeze(-1)).abs().square().sum(-1)
        gain = (1 - error / original.clamp_min(1e-8)).clamp(0, 1)
        return torch.where(original > 1e-8, gain, 0.)

    odd, even = group(slice(None, None, 2)), group(slice(1, None, 2))
    support = torch.minimum(held_gain(odd, even[0]), held_gain(even, odd[0]))
    # A group with no usable evidence cannot corroborate the other group.
    usable = (weight[:, ::2].sum(-1) > .01) & (weight[:, 1::2].sum(-1) > .01)
    support = torch.where(usable, support, 0.)
    taps = .5 * (odd[0] + even[0]).squeeze(-1) * support.sqrt()[:, None]
    frequency = torch.fft.fftfreq(observed.shape[-1], device=observed.device)
    lags = frequency.new_tensor(LAGS)
    angle = -2 * torch.pi * lags[:, None] * frequency
    kernel = taps @ torch.polar(torch.ones_like(angle), angle)
    limited = kernel * (.45 / kernel.abs().clamp_min(1e-8)).clamp_max(1.)
    return 1 + limited[:, None, :], support


def regularized_inverse(observed, proposal, template, variance, response):
    y = torch.fft.fft(observed * template, norm='ortho')
    prior = torch.fft.fft(proposal * template, norm='ortho')
    weight = observed.shape[1] * variance
    spectrum = (response.conj() * y + weight * prior) / (response.abs().square() + weight)
    refined = torch.fft.ifft(spectrum, norm='ortho') * template.conj()
    return observed + zero_carrier(refined - observed)


class CrossfitPilot(ChannelBlind):
    def __init__(self, widths):
        super().__init__(widths=widths, companion='mirror')

    def forward(self, observed, quality, template, *, diagnostics=False):
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        coarse, prior, variance = self.propose(observed, quality, mixed_conv=mixed)
        transfer, support = estimate_shared(observed, template, quality, variance)
        fine = regularized_inverse(observed, prior, template, variance, transfer)
        if diagnostics:
            engineering = regularized_inverse(observed, observed, template, variance, transfer)
            return coarse, fine, engineering, support
        return coarse, fine
