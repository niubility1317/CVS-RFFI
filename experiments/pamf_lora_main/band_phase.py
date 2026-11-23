"""Calibrate companion-chirp phase separately in two near-carrier bands."""

import torch
from torch import nn

from experiments.pamf_restoration.models import zero_output

from .complex_vector import fft
from .denoise_components import bands
from .packet_near import PacketNear, phase_transfer


class BandPhase(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimators = nn.ModuleList(
            nn.Sequential(nn.Linear(11, 24), nn.SiLU(), zero_output(nn.Linear(24, 3)))
            for _ in range(2)
        )

    def forward(self, observed, current, quality, selectors):
        wide, wide_quality = phase_transfer(current[..., selectors[0] | selectors[1]])
        count = current.shape[1]
        eye = torch.eye(count, device=current.device, dtype=torch.bool)
        chirp = torch.arange(count, device=current.device, dtype=current.real.dtype)
        distance = (chirp[:, None]-chirp[None]).abs().expand_as(wide.real) / max(count-1, 1)
        q = quality[..., 0]
        q_i, q_j = q[:, :, None].expand_as(wide.real), q[:, None, :].expand_as(wide.real)
        outputs = []
        for index, (band, estimate) in enumerate(zip(selectors, self.estimators, strict=True)):
            raw, raw_quality = phase_transfer(observed[..., band])
            phase, coherence = phase_transfer(current[..., band])
            evidence = torch.stack((raw.real, raw.imag, raw_quality, phase.real, phase.imag,
                                    coherence, wide.real, wide.imag, wide_quality, q_i*q_j,
                                    distance), dim=-1)
            mix, rotation, trust = estimate(evidence).unbind(-1)
            mix = .5*(mix+mix.transpose(-1, -2))
            rotation = .5*(rotation-rotation.transpose(-1, -2))
            trust = .5*(trust+trust.transpose(-1, -2))
            # Start with wider interpolation in 1–8 kHz and a stronger
            # carrier-frame anchor in 9–32 kHz.
            source = .75*phase+.25*wide
            weight = (mix+(0.7, -1.1)[index]).sigmoid()
            vec = 1.+weight*(source-1.)
            unit = torch.where(vec.abs()>1e-5, vec/vec.abs().clamp_min(1e-5), 1.)
            unit = unit*torch.polar(torch.ones_like(rotation), .8*rotation.tanh())
            confidence = (((coherence+.1)/1.2).logit()+trust).sigmoid()
            outputs.append((unit.masked_fill(eye, 1.), confidence.masked_fill(eye, 1.)))
        return outputs


class CalibratedPacketNear(PacketNear):
    def __init__(self):
        super().__init__(packet=True)
        self.calibrator = BandPhase()

    def forward(self, observed, baseline, quality, frozen_update, *, phase_readout=False):
        current = baseline+torch.fft.ifft(frozen_update, norm='ortho')
        select = bands(current.shape[-1], current.device)
        estimates = self.calibrator(fft(observed), fft(current), quality, select)
        near = select[0] | select[1]
        is_outer_near = select[1][near][None, :, None, None]
        transfer = tuple(torch.where(is_outer_near, high[:, None], low[:, None])
                         for low, high in zip(*estimates, strict=True))
        common = self.context(observed, current, quality, packet=True, transport=transfer)
        local = self.proposal(observed, current, quality)
        result = self.reorient(frozen_update, .3*local+.3*common)
        return (result, estimates) if phase_readout else result


def phase_risk(estimates, clean_reception):
    """Supervise phase and confidence on two chirps from the same noiseless replay."""
    truth = fft(clean_reception)
    select = bands(truth.shape[-1], truth.device)
    pairs = torch.triu(torch.ones((truth.shape[1], truth.shape[1]),
                                 device=truth.device, dtype=torch.bool), diagonal=1)
    losses = []
    for (unit, confidence), band in zip(estimates, select, strict=True):
        exact, reliability = phase_transfer(truth[..., band])
        rho = reliability[:, pairs]
        angular = (unit[:, pairs]-exact[:, pairs]).abs().square()
        quality = (confidence[:, pairs]-rho).square()
        losses.append((rho*angular+.25*quality).mean())
    return torch.stack(losses)
