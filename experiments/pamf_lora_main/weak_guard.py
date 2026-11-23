"""Recognize weak diffuse disturbances in the real LoRa guard bands."""

import math

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

from .dura_repair import complex_iq


GUARD_FEATURES = 8
FIT_SEED = 2026113107
HELD_SEED = 2026113137
HELD_DOSES = (.0075, .015, .03, .06, .12)


class GuardFloor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        freq = (256 * torch.fft.fftfreq(256)).round().abs()
        window = torch.hann_window(256)
        self.register_buffer('window', window / window.square().mean().sqrt())
        self.register_buffer('guards', torch.stack(((freq >= 70) & (freq <= 91),
                                                   (freq >= 92) & (freq <= 119))))

    def forward(self, iq):
        raw = complex_iq(iq)
        rms = raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        power = torch.fft.fft((raw / rms) * self.window, norm='ortho').abs().square()
        summaries = []
        for mask in self.guards:
            band = power[..., mask]
            floor = band.median(-1).values.clamp_min(1e-12)
            average = band.mean(-1).clamp_min(1e-12)
            for statistic in (floor.log(), (floor / average).log()):
                ordered = statistic.sort(dim=1).values
                summaries.extend((ordered[:, 2], ordered[:, 4]))
        return torch.stack(summaries, dim=1)


class WeakGuard(torch.nn.Module):
    def __init__(self, record):
        super().__init__()
        self.features = GuardFloor()
        for field in ('midpoint', 'dispersion', 'weight', 'offset', 'threshold'):
            self.register_buffer(field, torch.as_tensor(record[field], dtype=torch.float32))

    def forward(self, iq):
        score = apply_scores(self.features(iq), self.midpoint, self.dispersion, self.weight, self.offset)
        return score > self.threshold


def proposed_doses(rms, old_threshold, old, recognized):
    extension = (~old & recognized).to(rms.dtype)
    base = .5 * old.to(rms.dtype)
    gentle = .5 * (rms / old_threshold).square().clamp(max=1.)
    return torch.stack((base, base + .25 * extension, base + .5 * extension, base + gentle * extension))


def monotone_scores(positive_inputs, negative_inputs):
    """Use paired generic corruptions for clean-versus-impaired logistic fitting."""
    midpoint = negative_inputs.median(dim=0).values
    p25,p75 = torch.quantile(negative_inputs, negative_inputs.new_tensor([.25,.75]),dim=0)
    dispersion = (p75-p25).clamp_min(.10)
    a = ((negative_inputs-midpoint)/dispersion).clamp(-10,10).double().numpy()
    b = ((positive_inputs-midpoint)/dispersion).clamp(-10,10).double().numpy()
    values = np.concatenate((a,b))
    target = np.concatenate((np.zeros(len(a)),np.ones(len(b))))
    weights = np.concatenate((np.full(len(a),.5/len(a)),np.full(len(b),.5/len(b))))
    def loss(state):
        w = state[:-1]
        z = values@w + state[-1]
        objective = np.sum(weights*(np.logaddexp(0,z)-target*z)) + .025 * np.dot(w,w)
        delta = weights*(expit(z)-target)
        gradient = np.concatenate((values.T@delta+.05*w,[delta.sum()]))
        return objective,gradient
    result=minimize(loss,np.zeros(values.shape[1]+1),method='L-BFGS-B',jac=True,
                    bounds=[(0,None)]*values.shape[1]+[(None,None)],options=dict(maxiter=150))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Generic weak-disturbance fit failed: {result.message}')
    return midpoint,dispersion,negative_inputs.new_tensor(result.x[:-1]),float(result.x[-1])


def apply_scores(row, midpoint, dispersion, weight, bias):
    return ((row-midpoint)/dispersion).clamp(-10,10) @ weight + bias


def generic_noise(iq, rng, amplitude=None):
    """Generic independent complex noise, either flat or smoothly colored."""
    size=iq.shape
    opts=dict(device=iq.device,generator=rng)
    draw=torch.complex(torch.randn(size,**opts),torch.randn(size,**opts))
    frequency=torch.fft.fftfreq(size[-1],device=iq.device)
    angle=2*torch.pi*(frequency-torch.rand((len(iq),1,1),**opts))
    depth=.95*torch.rand((len(iq),1,1),**opts)
    flat=torch.rand((len(iq),1,1),**opts)<.40
    profile=(1+torch.where(flat,0.,depth)*angle.cos()).sqrt()
    draw=torch.fft.ifft(torch.fft.fft(draw) * profile)
    if amplitude is None:
        amplitude=.006*torch.exp(math.log(.16/.006)*torch.rand((len(iq),1,1),**opts))
    return iq + amplitude * draw/2**.5
