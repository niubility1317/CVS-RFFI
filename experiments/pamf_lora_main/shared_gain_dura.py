"""One native-IQ recovery graph for short WiSig windows and full LoRa captures."""

import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.preamble_noise import WhitenedAction, preamble_weights
from experiments.pamf_lora_main.received_quality import clean_geometry_weight
from experiments.pamf_lora_main.w24_native_dura import capture_scale
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_uncertainty_restore.models import weighted_map


def residual_network(inputs, outputs, *, circular=False, zero_output=True):
    padding_mode = 'circular' if circular else 'replicate'
    layers = nn.Sequential(
        nn.Conv1d(inputs, 24, 7, padding=3, padding_mode=padding_mode), nn.SiLU(),
        nn.Conv1d(24, 24, 5, padding=8, dilation=4, padding_mode=padding_mode), nn.SiLU(),
        nn.Conv1d(24, 24, 5, padding=16, dilation=8, padding_mode=padding_mode), nn.SiLU(),
        nn.Conv1d(24, outputs, 3, padding=1, padding_mode=padding_mode),
    )
    if zero_output:
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    return layers


def retain_observed_component(observed, estimate):
    """Bound the removed received component and the orthogonal correction."""
    strength = observed.abs().square().mean(-1, keepdim=True).clamp_min(1e-12)
    radial = (estimate * observed.conj()).real.mean(-1, keepdim=True) / strength
    correction = estimate - radial * observed
    relative = (correction.abs().square().mean(-1, keepdim=True) / strength).clamp_min(1e-12).sqrt()
    return radial.clamp(.80, 1.20) * observed + (.75 / relative).clamp(max=1.) * correction


class SharedGainDura(nn.Module):
    """Estimate degradation, then incorporate a real gain in each DURA prior."""

    def __init__(self, *, packet_gain=False, retain_received=False, clean_noise_floor=None):
        super().__init__()
        self.packet_gain = packet_gain
        self.retain_received = retain_received
        self.register_buffer('clean_noise_floor', clean_noise_floor, persistent=False)
        self.coarse = residual_network(2, 2)
        self.context_features = residual_network(6, 16, zero_output=False)
        self.channel = nn.Linear(16, 64 + 4)
        self.local_phase = nn.Conv1d(16, 1, 1)
        nn.init.zeros_(self.channel.weight)
        nn.init.zeros_(self.channel.bias)
        nn.init.zeros_(self.local_phase.weight)
        nn.init.zeros_(self.local_phase.bias)
        self.gain = residual_network(6, 1, circular=True)
        self.priors = nn.ModuleList(residual_network(9, 2) for _ in range(2))
        delays = torch.zeros(32)
        delays[0] = 1.
        self.register_buffer('identity_kernel', delays)

    def estimate(self, observed, coarse):
        power = torch.stack((observed.abs().square(), (observed - coarse).abs().square()), dim=1)
        context = torch.cat((pair(observed), pair(coarse), power), dim=1)
        decimated = F.avg_pool1d(context, 4)
        features = self.context_features(decimated)
        global_features = self.channel(features.mean(dim=-1))
        kernel = .25 * torch.complex(global_features[:, :32].tanh(), global_features[:, 32:64].tanh())
        kernel = kernel + self.identity_kernel
        kernel = kernel / kernel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        phase_knots = self.local_phase(F.adaptive_avg_pool1d(features, 8)).tanh() * torch.pi
        phase = F.interpolate(phase_knots, size=observed.shape[-1], mode='linear', align_corners=False)[:, 0]
        main = torch.polar(torch.ones_like(phase), phase)
        image = .18 * torch.complex(global_features[:, 64].tanh(), global_features[:, 65].tanh())
        log_variance = -5. + 5. * global_features[:, 66].tanh()
        quality_logit = global_features[:, 67]
        return Action(kernel, main, image[:, None].expand_as(main)), log_variance.exp(), quality_logit

    def spectral_update(self, observed, state, innovation, variance):
        y, x, residual = torch.fft.fft(torch.stack((observed, state, innovation)), norm='ortho').unbind()
        noise = variance[:, None]
        powers = [value.abs().square() / noise for value in (y, x, residual)]
        relative = residual * x.conj() / (noise * ((1. + powers[1]) * (1. + powers[2])).sqrt())
        features = torch.stack((*[p.log1p() for p in powers], relative.real, relative.imag,
                                noise.log().expand_as(y.real) / 8), dim=1)
        coefficient = .5 * self.gain(features)[:, 0].tanh()
        if self.packet_gain:
            coefficient = coefficient.mean(dim=-1, keepdim=True)
        return torch.fft.ifft(coefficient * residual, norm='ortho')

    def restoration(self, raw):
        if raw.ndim != 2 or not raw.is_complex() or raw.shape[-1] not in (256, 8192):
            raise ValueError('Expected batched native complex IQ with 256 or 8192 samples')
        scale = capture_scale(raw)
        observed = raw / scale
        coarse = observed + .35 * complex_iq(self.coarse(pair(observed)))
        action, variance, quality = self.estimate(observed, coarse)
        likelihood, measurement = action, observed
        if self.clean_noise_floor is not None:
            likelihood = WhitenedAction(action, preamble_weights(observed, self.clean_noise_floor))
            measurement = likelihood.whiten(observed)
        log_variance = (variance.log() / 8)[:, None, None].expand(-1, 1, observed.shape[-1])
        current = coarse
        stages = []
        for network in self.priors:
            innovation = likelihood.adjoint(measurement - likelihood(current))
            fitted = current + self.spectral_update(observed, current, innovation, variance)
            features = torch.cat((pair(observed), pair(coarse), pair(fitted), pair(innovation), log_variance), dim=1)
            prior = fitted + .35 * complex_iq(network(features))
            current = weighted_map(likelihood, measurement, prior, variance, prior_variance=.20,
                                   steps=2, initial=current)
            if self.retain_received:
                current = retain_observed_component(observed, current)
            stages.append(current * scale)
        return stages, action, variance, quality

    def forward(self, raw, *, clean_cut):
        stages, _, _, quality = self.restoration(raw)
        if isinstance(clean_cut, dict):
            smooth = clean_geometry_weight(raw, clean_cut)
            return raw + smooth[:, None] * (stages[-1] - raw)
        take = ((quality - clean_cut) / 2.).clamp(0., 1.)
        smooth = take.square() * (3. - 2. * take)
        return raw + smooth[:, None] * (stages[-1] - raw)
