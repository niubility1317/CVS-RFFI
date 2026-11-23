"""Source-calibrated disturbance and signed recovery utility for full practical."""

import torch
from torch import nn

from .dura_repair import complex_iq, no_tone, real_iq
from .full_dura_detail import extra_correction, half_baseline, scale_detail
from .residual_dura_bandfit import correction
from .residual_dura_gate import DisturbanceFeatures, FEATURES


MIN_GAIN = .02
MAX_DOSE = .0375
RISK_FEATURES = FEATURES + 4


def actions(eligible, disturbed, gains):
    value, position = gains.masked_fill(~eligible, -torch.inf).max(1)
    return torch.where(disturbed & (value > MIN_GAIN), position + 1, 0)


def signed_gain(base, proposals):
    return ((base[:, None] - proposals) / (base[:, None] + .002)).mean(-1)


class Utility(nn.Module):
    def __init__(self):
        super().__init__()
        self.detect = nn.Linear(FEATURES, 1)
        self.regress = nn.Sequential(nn.Linear(RISK_FEATURES, 48), nn.SiLU(), nn.Linear(48, 2))
        self.register_buffer('center', torch.zeros(RISK_FEATURES))
        self.register_buffer('scale', torch.ones(RISK_FEATURES))
        self.register_buffer('boundary', torch.tensor(float('inf')))

    def detection(self, features):
        normalized = (features[:, :FEATURES] - self.center[:FEATURES]) / self.scale[:FEATURES]
        return self.detect(normalized.clamp(-10., 10.))[:, 0]

    def utility(self, features):
        return self.regress(((features - self.center) / self.scale).clamp(-10., 10.))


class Proposals:
    def __init__(self, base, prior, response, r17, projector):
        self.base, self.prior, self.response = base, prior, response
        self.r17, self.projector = r17, projector
        self.features = DisturbanceFeatures().to(next(response.parameters()).device).eval()

    @torch.no_grad()
    def candidates(self, x):
        half, _ = half_baseline(self.base, self.prior, x)
        accepted = torch.zeros(len(half), dtype=torch.bool, device=half.device)
        extra, adjusted = extra_correction(self.response, half, accepted)
        accepted[extra] = True
        reference = half.clone()
        reference[extra] = scale_detail(half[extra], adjusted, .5)
        coherent = self.response.eligible(half)
        expanded = coherent & ~accepted
        response_wave = reference.clone()
        if expanded.any():
            raw = half[expanded]
            response_wave[expanded] = scale_detail(raw, raw + self.response.proposal(raw), .5)
        y = reference.flatten(0, 1)
        coords = self.projector.coordinates(y)
        c = correction(self.projector, self.r17(y), coords, no_tone(coords[0]))
        d = (c * coords[1] * coords[2]).reshape_as(complex_iq(reference))
        energy = complex_iq(reference).abs().square().mean((1, 2)).sqrt()
        ratio = d.abs().square().mean((1, 2)).sqrt() / energy.clamp_min(1e-8)
        bounded = d * (MAX_DOSE / ratio.clamp_min(1e-8)).clamp_max(1.)[:, None, None]
        other = reference + real_iq(bounded)
        can_expand = ~accepted
        eligible = torch.stack((expanded, can_expand), dim=1)
        t = complex_iq(response_wave - reference)
        p = torch.stack((t, bounded), dim=1)
        intensity = p.abs().square().mean((2, 3)).sqrt() / energy.clamp_min(1e-8)[:, None]
        irregularity = p.abs().square().mean(-1).sqrt().std(-1) / energy.clamp_min(1e-8)[:, None]
        descriptor, _ = self.features(x)
        features = torch.cat((descriptor, intensity, irregularity), dim=1)
        return torch.stack((reference, response_wave, other)), eligible, features, accepted

    @torch.no_grad()
    def gains(self, candidates, clean):
        tracker = self.projector.tracker
        observed, rot, *_ = tracker.track(candidates[0])
        expected, *_ = tracker.track(clean)
        rms = observed.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        target_rms = expected.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        target = no_tone(expected / target_rms)
        error = no_tone(complex_iq(candidates.flatten(0, 1)).reshape(3, *observed.shape)
                        * rot.conj() / rms) - target
        residual = torch.fft.fft(error, norm='ortho').abs().square().mean(2)
        offset = torch.fft.fftfreq(256, device=observed.device).abs() * 256
        bands = ((offset > 0) & (offset <= 8), (offset >= 9) & (offset <= 32), offset > 32)
        costs = torch.stack([residual[..., b].mean(-1) for b in bands], dim=-1).transpose(0, 1)
        return signed_gain(costs[:, 0], costs[:, 1:])
