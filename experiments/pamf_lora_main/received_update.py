"""Calibrate amplitude and phase updates against LoRa's received preamble."""

from math import log

import torch
from torch import nn
from torch.nn import functional as F

from .channel_blind_source import waveform_risk
from .common_anchor import constrained_step, head, iq_field
from .dura_repair import complex_iq, real_iq
from .joint_spectrum import limit
from .joint_unroll_source import coordinates


def masked_center(value, valid):
    value = torch.where(valid, value, 0.)
    mean = value.sum(-2, keepdim=True) / valid.sum(-2, keepdim=True).clamp_min(1)
    return torch.where(valid, value - mean, 0.)


def solve_gains(projection, precision, active, *, scalar):
    """Maximize the regularized quadratic on the active part of a unit box."""
    f, g = projection.masked_fill(~active, 0.).unbind(-1)
    a, c = precision.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).unbind(-1)
    cross = precision[..., 0, 1] * active.all(-1)
    if scalar:
        aa, cc = active.unbind(-1)
        denominator = a * aa + c * cc + 2 * cross
        gain = ((f + g) / denominator.clamp_min(1e-8)).clamp(0, 1)
        return gain.unsqueeze(-1) * active
    z, o = torch.zeros_like(f), torch.ones_like(f)
    determinant = (a * c - cross.square()).clamp_min(1e-16)
    x, y = (c * f - cross * g) / determinant, (a * g - cross * f) / determinant
    gx = torch.stack((z, z, o, f / a, (f - cross) / a, x), -1).clamp(0, 1)
    gy = torch.stack((z, g / c, (g - cross) / c, z, o, y), -1).clamp(0, 1)
    gx, gy = gx * active[..., :1], gy * active[..., 1:]
    gain = 2 * (f[..., None] * gx + g[..., None] * gy - cross[..., None] * gx * gy)
    gain = gain - a[..., None] * gx.square() - c[..., None] * gy.square()
    inside = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    gain = torch.cat((gain[..., :-1], gain[..., -1:].masked_fill(~inside[..., None], -torch.inf)), -1)
    chosen = gain.argmax(-1, keepdim=True)
    return torch.cat((gx.gather(-1, chosen), gy.gather(-1, chosen)), -1)


def update_coordinates(tracker, iq):
    y, rotation, quality, rms = coordinates(tracker, iq)
    shared_rms = rms.square().mean(1, keepdim=True).sqrt()
    return y * (rms / shared_rms), rotation, quality, shared_rms


class ReceivedUpdate(nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar
        self.coarse = head(12, 2)
        self.refine = head(16, 2)
        self.risk = nn.Linear(12, 4)
        nn.init.normal_(self.risk.weight, std=.01)
        nn.init.zeros_(self.risk.bias)
        f = torch.fft.fftfreq(256).abs() * 256
        bands = torch.stack(((f > 0) & (f <= 8), (f > 8) & (f <= 32), (f > 32) & (f <= 64)))
        self.register_buffer('bands', bands.float(), persistent=False)
        self.register_buffer('outer', f > 64, persistent=False)
        self.register_buffer('kappa', torch.zeros(2))
        self.register_buffer('axis_floor', torch.full((2, 3), .005))

    @staticmethod
    def estimate(y, sigma):
        # Opposite chirp group supplies the reference; blocks of eight samples
        # alternate in the fit and validation fold at each of four time knots.
        even = y[:, ::2].mean(1, keepdim=True)
        odd = y[:, 1::2].mean(1, keepdim=True)
        reference = torch.cat((odd, even), 1).repeat(1, y.shape[1] // 2, 1)
        v = reference.unflatten(-1, (4, -1, 8))
        z = y.unflatten(-1, (4, -1, 8))
        ridge = .05 + .25 * sigma
        powers = v.abs().square().mean(-1)
        cross = ((z - v) * v.conj()).mean(-1)
        power = torch.stack((powers[..., ::2].mean(-1), powers[..., 1::2].mean(-1)), -1)
        moment = torch.stack((cross[..., ::2].mean(-1), cross[..., 1::2].mean(-1)), -1)
        change = moment / (power + ridge.unsqueeze(-1))
        change = change * (.55 / change.abs().clamp_min(1e-7)).clamp_max(1.)
        # Score each fold with the other fold's fit. Include a unit-gain
        # option and interpolate four conservative estimates for recovery.
        spread = (z - v).abs().square().mean(-1)
        noise = torch.stack((spread[..., ::2].mean(-1), spread[..., 1::2].mean(-1)), -1)
        predicted = change.flip(-1)
        residual = noise - 2 * (predicted.conj() * moment).real + predicted.abs().square() * power
        residual = residual.clamp_min(0.)
        trust = ((noise - residual).mean(-1) / (.02 + noise.mean(-1))).clamp(0, 1)
        relative = (change.mean(-1) * trust).resolve_conj()
        sampled = torch.view_as_real(relative).permute(0, 1, 3, 2).flatten(0, 1)
        action = F.interpolate(sampled, size=y.shape[-1], mode='linear', align_corners=False)
        action = 1 + torch.complex(*action.unflatten(0, y.shape[:2]).unbind(2))
        return action, reference, residual.mean((2, 3), keepdim=False).unsqueeze(-1)

    def proposal(self, network, fields):
        values = torch.stack(fields, dim=2).flatten(0, 1)
        bf16 = self.training and values.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            output = network(values)
        return torch.complex(*output.float().unflatten(0, fields[0].shape[:2]).tanh().unbind(2))

    @staticmethod
    def components(values):
        common = values.mean(1, keepdim=True)
        return torch.stack((common.expand_as(values), values - common), 1)

    def directions(self, original, update):
        signal = self.components(original)
        correction = self.components(torch.fft.fft(update, norm='ortho'))
        floor = self.axis_floor @ self.bands
        valid = (signal.abs() > floor[None, :, None, :]) & self.bands.any(0)
        unit = signal / signal.abs().clamp_min(1e-7)
        amplitude = (correction * unit.conj()).real * unit
        phase = correction - amplitude
        d = torch.stack((amplitude, phase), 2)
        common = torch.where(valid[:, :1, None], d[:, :1], 0.)
        private = masked_center(d[:, 1:], valid[:, 1:, None])
        return torch.cat((common, private), 1)

    def moments(self, directions, target):
        a, p = directions.unbind(2)
        square = torch.stack((a.abs().square(), p.abs().square(), (a.conj() * p).real), dim=-2)
        ha, hp, hap = (square.sum(-3) @ self.bands.T).unbind(-2)
        precision = torch.stack((ha, hap, hap, hp), -1).unflatten(-1, (2, 2))
        projection = None
        if target is not None:
            delta = self.components(torch.fft.fft(target, norm='ortho'))
            projections = ((directions.detach().conj() * delta.unsqueeze(2)).real.sum(-2) @ self.bands.T)
            projection = projections.transpose(-1, -2)
        return precision, projection

    def correction(self, y, original, edit, quality, sigma, mismatch, rms, *, scalar, target, stage,
                   phase_margin=True):
        d = self.directions(original, edit)
        gram, projection = self.moments(d, target)
        diagonal = gram.diagonal(dim1=-2, dim2=-1).clamp_min(0.)
        energy = (self.components(original).abs().square().sum(-2) @ self.bands.T).unsqueeze(-1)
        scale = (diagonal + 1e-8).sqrt() * (1 + energy).sqrt()
        active = diagonal > 1e-8
        ab = gram[..., 0, 1:2] / (diagonal.prod(-1, keepdim=True) + 1e-12).sqrt()
        rank = diagonal.shape[:-1] + (1,)
        quality, sigma, mismatch = [v.mean((1, 2)).reshape(-1, 1, 1, 1).expand(rank)
                                    for v in (quality, sigma, mismatch)]
        part = torch.arange(2, device=y.device).reshape(1, 2, 1, 1).expand(rank)
        band = F.one_hot(torch.arange(3, device=y.device), 3).reshape(1, 1, 3, 3)
        features = torch.cat((diagonal.log1p(), energy.log1p(), ab.clamp(-1, 1), quality,
                              sigma.log1p(), mismatch.log1p(), part, band.expand(*rank[:-1], 3),
                              active.any(-1, keepdim=True)), -1)
        mean, raw_uncertainty = self.risk(features).chunk(2, -1)
        estimate = (.60 * diagonal + torch.tanh(mean) * scale) * active
        sigma_b = (.01 + F.softplus(raw_uncertainty)) * scale
        ridge = .01 * diagonal.mean(-1, keepdim=True).clamp_min(1e-8)
        precision = gram + torch.diag_embed(ridge.expand_as(diagonal))
        penalty = self.kappa[stage] * sigma_b
        if not phase_margin:
            penalty = penalty * penalty.new_tensor([1., 0.])
        gain = solve_gains(estimate - penalty, precision, active, scalar=scalar)
        multiplier = gain.transpose(-1, -2) @ self.bands
        combined = (d * multiplier.unsqueeze(-2)).sum((1, 2))
        restored = y + limit(torch.fft.ifft(combined, norm='ortho'), y, rms, y.real.new_tensor(.10))
        return restored, (estimate, sigma_b, projection, active, gain)

    def states(self, y, quality, rms, *, scalar=None, reference=None, identity=False, fixed=False,
               anchor_common=False, learning_only=False, phase_margin=True):
        original = torch.fft.fft(y, norm='ortho')
        sigma = original[..., self.outer].abs().square().median(-1, keepdim=True).values / log(2.)
        action, other, mismatch = self.estimate(y, sigma)
        if identity or learning_only:
            action = torch.ones_like(action)
        variance = (.015 + sigma + .25 * mismatch).clamp(.015, 3.)
        if fixed or learning_only:
            variance = torch.full_like(variance, .10)
        center = y.mean(1, keepdim=True).expand_as(y)
        span = torch.linspace(-1., 1., 256, device=y.device).expand_as(y.real)
        fields = (*iq_field(y, center, other, y - center, action - 1.),
                  quality.expand_as(span), variance.log1p().expand_as(span))
        target = None if reference is None else reference - y
        scalar = self.scalar if scalar is None else scalar
        proposal = y + .45 * self.proposal(self.coarse, fields)
        edit = proposal - y if learning_only else constrained_step(
            y, proposal, action, variance, 4., anchor=anchor_common)
        if anchor_common:
            edit = edit - edit.mean(1, keepdim=True)
        args = (quality, sigma, mismatch, rms)
        first, first_stats = self.correction(y, original, edit, *args, scalar=scalar, target=target,
                                             stage=0, phase_margin=phase_margin)
        innovation = action.conj() * (y - action * first)
        proposal = first + .30 * self.proposal(self.refine, (*fields, *iq_field(first, innovation)))
        edit = proposal - y if learning_only else constrained_step(
            y, proposal, action, variance, 3., anchor=anchor_common)
        if anchor_common:
            edit = edit - edit.mean(1, keepdim=True)
        final, second_stats = self.correction(y, original, edit, *args, scalar=scalar, target=target,
                                              stage=1, phase_margin=phase_margin)
        return first, final, (first_stats, second_stats), (action, sigma, mismatch)


def pairing(tracker, iq, recipe, rng, bands, *, with_identity):
    source = complex_iq(iq)
    clean, rotation, _, level = update_coordinates(tracker, iq)
    difference = .5 * (clean.roll(1, dims=0) - clean)
    detail = torch.fft.ifft(torch.fft.fft(difference, norm='ortho') * bands.any(0), norm='ortho')
    detail = detail * rotation * level
    extra = source + detail
    draws = torch.stack((recipe.perturb(source, rng), recipe.perturb(source, rng)), 1)
    replay = torch.Generator(device=iq.device)
    # Use the same generic random draws for two different legal source IQ views.
    replay.set_state(rng.get_state())
    left = recipe.perturb(source, rng)
    right = recipe.perturb(extra, replay)
    draws = torch.cat((draws, left[:, None], right[:, None]), 1)
    reference = torch.stack((source, source, source, extra), 1)
    if with_identity:
        draws[-2:] = reference[-2:]
    y, phase, quality, rms = update_coordinates(tracker, real_iq(draws.flatten(0, 1)))
    frame = phase * rms
    reference = reference.flatten(0, 1) / frame
    return y, quality, rms, reference, frame, detail, rotation * level


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    y, quality, rms, reference, frame, detail, clean_frame = pairing(
        tracker, iq, recipe, rng, model.bands, with_identity=with_identity)
    first, final, states, _ = model.states(y, quality, rms, reference=reference)
    risk = y.real.new_zeros(())
    for estimate, error_scale, projection, active, _ in states:
        standard = (estimate - projection) / error_scale
        value = F.smooth_l1_loss(standard, torch.zeros_like(standard), reduction='none')
        risk = risk + ((value + error_scale.log()) * active).sum() / active.sum().clamp_min(1)
    waves = (final * frame).reshape(len(iq), 4, *y.shape[1:])
    agreement = ((waves[:, 0] - waves[:, 1]) / clean_frame).abs().square().mean()
    change = model.components(torch.fft.fft((waves[:, 3] - waves[:, 2]) / clean_frame, norm='ortho'))
    expected = model.components(torch.fft.fft(detail / clean_frame, norm='ortho'))
    band_error = (change - expected).abs().square() @ model.bands.T
    band_energy = expected.abs().square() @ model.bands.T
    weak = (band_error / (.02 + band_energy)).log1p().mean()
    metrics = torch.stack((waveform_risk(first, reference), waveform_risk(final, reference),
                           agreement, weak, risk))
    low, high, pair = recipe.loss_weights(step)
    growth = min(1., step / (recipe.steps / 5))
    return low * metrics[0] + high * metrics[1] + pair * agreement + .22 * weak + (.01 + .03 * growth) * risk, metrics
