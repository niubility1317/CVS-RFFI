"""Received-IQ proxy conditions. These are NOT estimates of transmitter PA drive.

Observation error is calibrated from source-training controlled degradations, not
from class CE, predicted class confidence, query statistics, or receiver labels.
"""
import torch
from torch import nn


@torch.no_grad()
def received_conditions(x):
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[-1] < 8:
        raise ValueError("received IQ must be [B,2,L>=8]")
    if not torch.isfinite(x).all():
        raise ValueError("received IQ contains nonfinite values")
    x = x.float()
    power = x.square().sum(1)
    mean_power = power.mean(-1).clamp_min(1e-10)
    amp = (power + 1e-10).sqrt()
    cv = amp.std(-1, unbiased=False) / amp.mean(-1).clamp_min(1e-5)
    # Gain invariant receive-shape/transition proxies, not a physical PA claim.
    transition = (x[..., 1:] - x[..., :-1]).square().sum(1).mean(-1) / mean_power
    state = torch.stack((cv, transition / 4.), -1)
    roughness = (x[..., 2:] - 2*x[..., 1:-1] + x[..., :-2]).square().sum(1).mean(-1)
    roughness = (roughness / (6*mean_power)).clamp(0, 8)
    re = (x[:,0,1:]*x[:,0,:-1] + x[:,1,1:]*x[:,1,:-1])
    im = (x[:,1,1:]*x[:,0,:-1] - x[:,0,1:]*x[:,1,:-1])
    denom = (amp[:,1:]*amp[:,:-1]).clamp_min(1e-10)
    coherence = torch.stack((re/denom, im/denom), -1).mean(1).square().sum(-1).clamp(0,1).sqrt()
    crest = (power.amax(-1) / mean_power).clamp(1, 100)
    quality = torch.stack((roughness, 1-coherence, (crest-1)/10), -1)
    # Repeated plateaus are an explicit clipping proxy. Do not encode the pattern as identity.
    peak = x.abs().amax((1,2), keepdim=True)
    plateau = (x.abs() >= peak * .999).float().mean((1,2))
    valid = (mean_power > 1e-9) & (plateau < .20)
    return {"state": state, "quality": quality, "valid": valid, "plateau": plateau}


class ObservationErrorModel(nn.Module):
    def __init__(self, feature_dim, floor=1e-5, ceiling=1.0):
        super().__init__()
        if not 0 < floor < ceiling:
            raise ValueError("invalid observation variance bounds")
        self.floor, self.ceiling = float(floor), float(ceiling)
        self.register_buffer("coefficients", torch.zeros(4, feature_dim))
        self.register_buffer("calibrated", torch.tensor(False))
        self.register_buffer("calibration_bias", torch.zeros(feature_dim))

    @torch.no_grad()
    def fit(self, quality, reference, degraded, *, role):
        if role != "L_s":
            raise ValueError("quality calibration may fit only source training L_s")
        if quality.ndim != 2 or quality.shape[-1] != 3 or reference.shape != degraded.shape:
            raise ValueError("invalid calibration pair shapes")
        if reference.shape != (quality.shape[0], self.coefficients.shape[1]) or len(quality)<4:
            raise ValueError("need at least four source calibration pairs")
        design = torch.cat((torch.ones_like(quality[:,:1]), quality), 1).double()
        target = (degraded-reference).double().square()
        coef = torch.linalg.solve(design.T@design + .01*torch.eye(4,device=design.device,dtype=design.dtype), design.T@target)
        # Projected gradient NNLS: nonnegative quality-to-variance map.
        coef = coef.clamp_min(0)
        step = 1/(torch.linalg.matrix_norm(design,ord=2).square()+.01)
        for _ in range(100):
            coef = (coef-step*(design.T@(design@coef-target)+.01*coef)).clamp_min(0)
        self.coefficients.copy_(coef.to(self.coefficients))
        self.calibration_bias.copy_((degraded-reference).mean(0))
        self.calibrated.fill_(True)

    def forward(self, quality):
        if not bool(self.calibrated):
            raise RuntimeError("observation error is uncalibrated; fit source-training degradation pairs first")
        design = torch.cat((torch.ones_like(quality[:,:1]), quality), 1)
        return (design @ self.coefficients).clamp(self.floor,self.ceiling)
