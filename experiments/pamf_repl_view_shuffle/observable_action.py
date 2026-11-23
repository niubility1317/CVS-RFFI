"""WiSig restoration with a short receive action fitted to the current IQ."""

import torch
from torch import nn

from experiments.pamf_reception_restore.models import Core
from .receiver_pilot import estimate, phase, power


LAGS = (-2, -1, 0, 1, 2)
DELAY_MAX = 40


def shift_index(index, delay, lag=0):
    position = index - delay[:, None] - lag
    return position.clamp(0, index.numel() - 1), (position >= 0) & (position < index.numel())


def columns(x, delay, index, ramp):
    taps = []
    for lag in LAGS:
        coordinates, valid = shift_index(index, delay, lag)
        taps.append(x.gather(1, coordinates) * valid)
    middle = taps[LAGS.index(0)]
    return torch.stack((*taps, middle * ramp, middle.conj()), dim=-1)


def forward_action(x, h, delay, index, ramp):
    return (columns(x, delay, index, ramp) * h[:, None, :]).sum(dim=-1)


def adjoint(value, h, delay, index, ramp):
    # Transpose the finite-window gather in the real complex inner product.
    result = torch.zeros_like(value)
    for k, lag in enumerate(LAGS):
        coordinates, valid = shift_index(index, delay, lag)
        term = h[:, k:k+1].conj() * value
        if lag == 0:
            term = term + h[:, -2:-1].conj() * ramp * value + h[:, -1:] * value.conj()
        result = result.scatter_add(1, coordinates, term * valid)
    return result


def estimate_delay(x, observed, delays):
    # Pad the cross-correlation to exclude circular matches. The mild lag
    # penalty resolves near ties from the repeating part of the preamble.
    length = 2 * x.shape[-1]
    cross = torch.fft.fft(observed, n=length) * torch.fft.fft(x, n=length).conj()
    correlation = torch.fft.ifft(cross, n=length)[:, delays].abs()
    correlation = correlation * (1 - .08 * delays.abs() / DELAY_MAX)
    return delays[correlation.argmax(dim=-1)]


def fit_action(coarse, observed, delay, index, ramp, regularizer, center):
    design = columns(coarse, delay, index, ramp)
    fitting = design[:, ::2, :]
    gram = fitting.mH @ fitting / fitting.shape[1]
    rhs = fitting.mH @ observed[:, ::2, None] / fitting.shape[1]
    h = torch.linalg.solve_ex(gram + regularizer, rhs + center, check_errors=False).result.squeeze(-1)
    residual = observed[:, 1::2] - (design[:, 1::2] * h[:, None, :]).sum(dim=-1)
    variance = power(residual).mean(dim=-1, keepdim=True)
    return h, variance


def iq_channels(x):
    return torch.stack((x.real, x.imag), dim=1)


def amplitude_evidence(received, noise):
    spectrum = torch.fft.fft(received, norm='ortho')
    energy = power(spectrum)
    neighborhood = sum(energy.roll(offset, dims=-1) for offset in (-2, -1, 0, 1, 2)) / 5
    signal = (neighborhood - noise).clamp_min(0)
    confidence = .75 * signal / (signal + noise + 1e-8)
    return spectrum, confidence


def preserve_spectral_amplitude(proposal, observed_spectrum, confidence):
    hypothesis = torch.fft.fft(proposal, norm='ortho')
    magnitude = hypothesis.abs()
    stable = magnitude > 1e-8
    unit = torch.where(stable, hypothesis / magnitude.clamp_min(1e-8),
                       observed_spectrum / observed_spectrum.abs().clamp_min(1e-8))
    envelope = torch.lerp(magnitude, observed_spectrum.abs(), confidence)
    return torch.fft.ifft(envelope * unit, norm='ortho')


class ObservableAction(nn.Module):
    def __init__(self, fitted, *, protect_amplitude=False):
        super().__init__()
        self.fitted = fitted
        self.protect_amplitude = protect_amplitude
        self.coarse = Core(incoming=5, widths=(24, 36, 52, 64))
        self.fine = Core(incoming=9, widths=(24, 36, 52, 64))
        self.register_buffer('index', torch.arange(256), persistent=False)
        self.register_buffer('ramp', torch.linspace(-1, 1, 256), persistent=False)
        self.register_buffer('delays', torch.arange(-DELAY_MAX, DELAY_MAX + 1), persistent=False)
        penalties = torch.tensor([.18, .18, .035, .18, .18, .25, .22], dtype=torch.complex64)
        self.register_buffer('regularizer', torch.diag(penalties), persistent=False)
        center = torch.zeros(len(penalties), 1, dtype=penalties.dtype)
        center[LAGS.index(0)] = penalties[LAGS.index(0)]
        self.register_buffer('center', center, persistent=False)

    def forward(self, y, *, return_unprotected=False):
        obs = estimate(y)
        rms = power(y).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        x = obs.aligned / rms
        context = torch.stack((obs.coherence, obs.disturbance.clamp(0, 2)), dim=1)
        side = torch.cat((context[:, :, None].expand(-1, -1, x.shape[-1]),
                          self.ramp.expand(len(x), 1, -1)), dim=1)
        inputs = torch.cat((iq_channels(x), side), dim=1)
        mixed = self.training and x.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            change = self.coarse(inputs)
        coarse = x + torch.complex(change[:, 0].float(), change[:, 1].float())
        if self.protect_amplitude:
            observed_spectrum, confidence = amplitude_evidence(y / rms, obs.noise_power[:, None] / rms.square())
            coarse = preserve_spectral_amplitude(coarse * obs.rotation, observed_spectrum,
                                                 confidence) * obs.rotation.conj()
        delay = estimate_delay(coarse.detach(), x, self.delays)
        h, variance = fit_action(coarse, x, delay, self.index, self.ramp, self.regularizer, self.center)
        if self.fitted:
            innovation = x - forward_action(coarse, h, delay, self.index, self.ramp)
            correction = adjoint(innovation, h, delay, self.index, self.ramp)
        else:
            correction = x - coarse
        fine_inputs = torch.cat((inputs, iq_channels(coarse), iq_channels(correction)), dim=1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            detail = self.fine(fine_inputs)
        prior = coarse + torch.complex(detail[:, 0].float(), detail[:, 1].float())
        trust = .35 / (.35 + variance)
        if self.fitted:
            residual = x - forward_action(prior, h, delay, self.index, self.ramp)
            norm_bound = h.abs().sum(dim=-1, keepdim=True).square()
            update = adjoint(residual, h, delay, self.index, self.ramp) / (1 + norm_bound)
        else:
            update = .5 * (x - prior)
        fine = prior + .8 * trust * update
        unrestricted = fine
        if self.protect_amplitude:
            fine = preserve_spectral_amplitude(fine * obs.rotation, observed_spectrum,
                                               confidence) * obs.rotation.conj()
        basis = rms * obs.rotation
        info = dict(action_delay=delay, held_residual=variance[:, 0], coherence=obs.coherence,
                    action_gain=h[:, LAGS.index(0)].abs(), image_gain=h[:, -1].abs())
        if return_unprotected:
            return coarse * basis, fine * basis, info, unrestricted * basis
        return coarse * basis, fine * basis, info


def generic_observation(signal, generator):
    """Independent continuous augmentation, returning IQ without latent parameters."""
    batches, _, length = signal.shape
    device = signal.device

    def rand(*size):
        return torch.rand(size, device=device, generator=generator)

    def noise(*size):
        args = dict(device=device, generator=generator)
        return torch.complex(torch.randn(size, **args), torch.randn(size, **args)) * (.5 ** .5)

    lag = torch.randint(-DELAY_MAX, DELAY_MAX + 1, (batches, 1, 1), device=device, generator=generator)
    lag = lag * (rand(batches, 1, 1) < .65)
    t = torch.arange(length, device=device)
    where = t - lag
    where = torch.where(where < 0, -where, where.clamp_max(2 * (length - 1) - where))
    delayed = signal.gather(-1, where.expand_as(signal))
    echoes = delayed + .20 * noise(batches, 1, 1) * delayed.roll(1, -1)
    echoes = echoes + .15 * noise(batches, 1, 1) * delayed.roll(-4, -1)
    echoes = echoes + .12 * noise(batches, 1, 1) * delayed.conj()
    rotation = torch.pi * (2 * rand(batches, 1, 1) - 1)
    rotation = rotation + .65 * torch.pi * (2 * rand(batches, 1, 1) - 1) * (2 * t / (length - 1) - 1)
    signal_gain = (1.0 * (2 * rand(batches, 1, 1) - 1)).exp()
    disturbance = noise(batches, 1, length)
    disturbance = disturbance + (.6 * (2 * rand(batches, 1, 1) - 1)) * disturbance.roll(1, -1)
    rms = power(signal[:, :1]).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    dose = rand(batches, 1, 1).square()
    return phase(rotation) * signal_gain * echoes + rms * dose * disturbance


def weak_probe(clean):
    spectrum = torch.fft.fft(clean.roll(1, 0) - clean, norm='ortho')
    weak = spectrum.abs().quantile(.75, dim=-1, keepdim=True)
    return .25 * torch.fft.ifft(spectrum * (spectrum.abs() <= weak), norm='ortho')


def waveform_risk(output, clean):
    error = power(output - clean).mean(dim=-1)
    return (error / power(clean).mean(dim=-1).clamp_min(.01)).log1p().mean()


def objective(model, clean, generator):
    difference = weak_probe(clean)
    target = torch.stack((clean, clean, clean + difference), dim=1)
    pair = generic_observation(target[:, (0, 2)], generator)
    other = generic_observation(target[:, 1:2], generator)
    received = torch.stack((pair[:, 0], other[:, 0], pair[:, 1]), dim=1)
    received[-4:] = target[-4:]
    coarse, refined, _ = model(received.flatten(0, 1))
    output = refined.unflatten(0, target.shape[:2])
    delta = output[:, 2] - output[:, 0]
    transport = (power(delta - difference).mean(-1) / (.002 + power(difference).mean(-1))).log1p().mean()
    risks = torch.stack((waveform_risk(coarse, target.flatten(0, 1)), waveform_risk(refined, target.flatten(0, 1)),
                         waveform_risk(output[:, 0], output[:, 1]), transport))
    return (risks * risks.new_tensor((.35, 1., .08, .35))).sum(), risks
