"""LoRa complex shrinkage with separate repeated and chirp-specific evidence."""

from math import log

import torch
from torch import nn

from .channel_blind_source import waveform_risk
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, transfer_risk, weak_difference
from .spectral_block_recovery import head


def spectrum(wave):
    return torch.fft.fftshift(torch.fft.fft(wave, norm='ortho'), dim=-1)


def wave(spectrum):
    return torch.fft.ifft(torch.fft.ifftshift(spectrum, dim=-1), norm='ortho')


def shrink(value, common, unique, attenuation, *, separate):
    """Match edit energy per packet and bin while changing the edit direction."""
    repeated = attenuation[:, :1] * common
    individual = attenuation[:, 1:] * unique
    edit = -(repeated + individual)
    if not separate:
        mass = torch.linalg.vector_norm(value, dim=1, keepdim=True)
        amount = torch.linalg.vector_norm(edit, dim=1, keepdim=True)
        edit = -value * (amount / torch.where(mass > 0, mass, 1.))
    return edit


class SplitShrink(nn.Module):
    def __init__(self, separate):
        super().__init__()
        self.separate = separate
        self.coarse = head(9, 2)
        self.refine = head(13, 2)
        for net in (self.coarse, self.refine):
            nn.init.normal_(net[-1].weight, std=.01)
            nn.init.constant_(net[-1].bias, -1.5)
        freq = torch.fft.fftshift(torch.fft.fftfreq(256))
        self.register_buffer('frequency', freq, persistent=False)
        self.register_buffer('outer', freq.abs() > .25, persistent=False)
        self.register_buffer('weak', (freq.abs() <= .125) & (freq != 0), persistent=False)

    def forward(self, observed, quality, *, scalar_readout=False):
        value = spectrum(observed)
        common = value.mean(dim=1, keepdim=True)
        unique = value - common
        common_power = common.abs().square()
        spread = unique.abs().square().mean(dim=1, keepdim=True)
        first = value[:, ::2].mean(dim=1, keepdim=True)
        second = value[:, 1::2].mean(dim=1, keepdim=True)
        agreement = (first.conj() * second).real
        mismatch = .5 * (first - second).abs().square()
        noise = value[..., self.outer].abs().square().flatten(1).median(dim=1).values / log(2.)
        noise = noise[:, None, None]
        trust = agreement.relu() / (.015 + agreement.relu() + mismatch)
        allowed = torch.cat((1. - .8 * trust * self.weak, torch.ones_like(trust)), dim=1)
        allowed = .70 * (noise / (noise + .02)) * allowed * (self.frequency != 0)
        features = torch.cat((common_power.log1p(), spread.log1p(),
                              agreement.sign() * agreement.abs().log1p(), mismatch.log1p(), trust,
                              noise.log1p().expand_as(trust), quality.mean(dim=1, keepdim=True).expand_as(trust),
                              self.frequency.expand_as(trust), self.weak.expand_as(trust)), dim=1)
        use_amp = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = self.coarse(features).float()
        gains = allowed * logits.sigmoid()
        initial_edit = shrink(value, common, unique, gains, separate=self.separate)
        coarse = value + initial_edit
        coarse_power = coarse.abs().square().mean(dim=1, keepdim=True)
        observed_residual = initial_edit.abs().square().mean(dim=1, keepdim=True)
        features = torch.cat((features, gains, coarse_power.log1p(), observed_residual.log1p()), dim=1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = logits + .6 * self.refine(features).float()
        final_edit = shrink(value, common, unique, allowed * logits.sigmoid(), separate=self.separate)
        if scalar_readout:
            initial_edit = shrink(value, common, unique, gains, separate=False)
            final_edit = shrink(value, common, unique, allowed * logits.sigmoid(), separate=False)
            coarse = value + initial_edit
        return wave(coarse), wave(value + final_edit)


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    source = complex_iq(iq)
    clean, phase, _, level = coordinates(tracker, iq)
    clean_frame = phase * level
    detail = weak_difference(clean)
    delta = detail * clean_frame
    inputs = paired_inputs(source, recipe.perturb, rng, delta, with_identity=with_identity)
    observed, phase, quality, level = coordinates(tracker, real_iq(inputs.flatten(0, 1)))
    frame = phase * level
    target = torch.stack((source, source, source + delta), dim=1).flatten(0, 1) / frame
    coarse, fine = model(observed, quality)
    physical = (fine * frame).unflatten(0, inputs.shape[:2])
    consistency = ((physical[:, 0] - physical[:, 1]) / clean_frame).abs().square().mean()
    passage = transfer_risk((physical[:, 2] - physical[:, 0]) / clean_frame, detail)
    costs = torch.stack((waveform_risk(coarse, target), waveform_risk(fine, target), consistency, passage))
    first, second, paired = recipe.loss_weights(step)
    loss = first * costs[0] + second * costs[1] + paired * costs[2] + .35 * costs[3]
    return loss, costs
