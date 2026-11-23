"""LoRa leave-one-chirp-out priors in a locally weighted, lapped transform."""

from math import log

import torch
from torch import nn
from torch.nn import functional as F

from .channel_blind_source import waveform_risk
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, transfer_risk, weak_difference


def analysis(value, window):
    size = window.numel()
    repeated = torch.cat((value[..., -(size//2):], value), dim=-1)
    frames = repeated.unfold(-1, size, size//2)
    return torch.fft.fftshift(torch.fft.fft(frames * window, norm='ortho'), dim=-1)


def synthesis(value, window):
    frames = torch.fft.ifft(torch.fft.ifftshift(value, dim=-1), norm='ortho') * window
    start, end = frames.chunk(2, dim=-1)
    return (end + start.roll(-1, dims=-2)).flatten(-2)


def other_mean(value):
    return (value.sum(1, keepdim=True) - value) / (value.shape[1] - 1)


def controller(inputs, outputs):
    layers = nn.Sequential(nn.Conv2d(inputs, 16, (1, 5), padding=(0, 2)), nn.SiLU(),
                           nn.Conv2d(16, 16, (3, 1), padding=(1, 0)), nn.SiLU(),
                           nn.Conv2d(16, 16, (1, 5), padding=(0, 4), dilation=(1, 2)), nn.SiLU(),
                           nn.Conv2d(16, outputs, 1))
    nn.init.normal_(layers[-1].weight, std=.01)
    nn.init.zeros_(layers[-1].bias)
    return layers


class LappedPosterior(nn.Module):
    def __init__(self, local):
        super().__init__()
        self.local = local
        self.prior = controller(7, 3)
        self.refine = controller(12, 2)
        freq = torch.fft.fftshift(torch.fft.fftfreq(64))
        self.register_buffer('window', (torch.pi*(torch.arange(64)+.5)/64).sin(),
                             persistent=False)
        self.register_buffer('frequency', freq, persistent=False)
        self.register_buffer('outer', freq.abs() > .25, persistent=False)
        self.register_buffer('noncarrier', freq != 0, persistent=False)

    def estimate_power(self, coefficients):
        # The sine taper has mean square 1/2. The windowed coefficient variance
        # therefore uses half the time-domain sample variance.
        values = coefficients[..., self.outer].abs().square()
        return values.median(dim=-1, keepdim=True).values / log(2.)

    def transformed(self, observed):
        return analysis(observed, self.window)

    def precision(self, value, reference, spread, noise, support_noise, variance, *, scalar_readout):
        return .85 * noise / (.02 + noise + variance), None

    @staticmethod
    def fields(net, source, mixed):
        values = torch.stack(source, dim=2).flatten(0, 1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            out = net(values)
        return out.float().unflatten(0, source[0].shape[:2]).unbind(2)

    def restore(self, observed, quality, *, stationary_readout=False, scalar_readout=False,
                reference_action=None, noise_override=None, observation_edit=None, project_edit=None):
        value = self.transformed(observed)
        reference = other_mean(value)
        spread = (other_mean(value.abs().square()) - reference.abs().square()).clamp_min(0.)
        if reference_action is not None:
            reference = reference * reference_action.unsqueeze(-1)
        measured = self.estimate_power(value)
        pooled = measured.mean(dim=-2, keepdim=True).expand_as(measured)
        noise = .75 * measured + .25 * pooled if self.local and not stationary_readout else pooled
        if noise_override is not None:
            noise = noise_override.expand_as(noise)
        support_noise = other_mean(noise).clamp_min(0.)
        f = self.frequency.expand_as(spread)
        t = torch.linspace(-1., 1., value.shape[-2], device=value.device)[None, None, :, None].expand_as(spread)
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        a, b, scale = self.fields(self.prior, (reference.real, reference.imag, spread.log1p(),
                                support_noise.log1p().expand_as(spread), f, t, reference.abs().log1p()), mixed)
        power = spread + .002
        prior = reference + .50 * power.sqrt() * torch.complex(a.tanh(), b.tanh())
        variance = .002 + (.006 + power) * F.softplus(scale)
        weight, companion = self.precision(value, reference, spread, noise, support_noise, variance,
                                           scalar_readout=scalar_readout)
        innovation = prior - value
        first = weight * innovation
        if companion is not None:
            first = first + companion * innovation.conj()
        if observation_edit is not None:
            first = first + self.transformed(observation_edit)
        current = value + first
        projected_first = None
        if project_edit is not None:
            projected_first = project_edit(synthesis(first, self.window))
            current = value + self.transformed(projected_first)
        q = quality[..., None].expand_as(spread)
        c, d = self.fields(self.refine, (value.real, value.imag, current.real, current.imag,
                           innovation.real, innovation.imag, scale.clamp(-4., 4.),
                           spread.log1p(), noise.log1p().expand_as(spread), weight, f, q), mixed)
        amplitude = (noise+.002).sqrt()
        correction = torch.complex(c.tanh(), d.tanh())
        second = first + .50 * weight * amplitude * correction
        if companion is not None:
            second = second + .50 * companion * amplitude * correction.conj()
        outputs = []
        for index, edit in enumerate((first, second)):
            # The overlap-add frame retains the original time grid and mean.
            if index == 0 and projected_first is not None:
                timed = projected_first
            else:
                timed = synthesis(edit, self.window)
                if project_edit is not None:
                    timed = project_edit(timed)
            timed = timed - timed.mean(-1, keepdim=True)
            outputs.append(observed + timed)
        return *outputs, prior, variance

    def forward(self, observed, quality):
        return self.restore(observed, quality)[:2]


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    source = complex_iq(iq)
    clean, phase, _, level = coordinates(tracker, iq)
    clean_frame = phase * level
    weak = weak_difference(clean)
    delta = weak * clean_frame
    inputs = paired_inputs(source, recipe.perturb, rng, delta, with_identity=with_identity)
    observed, phase, quality, level = coordinates(tracker, real_iq(inputs.flatten(0, 1)))
    frame = phase * level
    target = torch.stack((source, source, source + delta), dim=1).flatten(0, 1) / frame
    coarse, fine, prior, variance = model.restore(observed, quality)
    error = (model.transformed(target) - prior).abs().square()
    risk = (error / variance + variance.log())[..., model.noncarrier].mean()
    physical = (fine * frame).unflatten(0, inputs.shape[:2])
    consistency = ((physical[:, 0] - physical[:, 1]) / clean_frame).abs().square().mean()
    passage = transfer_risk((physical[:, 2] - physical[:, 0]) / clean_frame, weak)
    values = torch.stack((waveform_risk(coarse, target), waveform_risk(fine, target),
                          consistency, passage, risk))
    first, second, paired = recipe.loss_weights(step)
    loss = first*values[0] + second*values[1] + paired*values[2] + .35*values[3] + .08*values[4]
    return loss, values
