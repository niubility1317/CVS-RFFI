"""Match LoRa carrier gain to the scale of recovered complex detail."""
from math import log

import torch
from torch import nn

from .antithetic_detail import antithetic_inputs, objective
from .channel_blind import zero_carrier
from .joint_unroll_source import coordinates
from .receiver_physics import apply_generic_physics
from .refinement_bound import bound_refinement
from .spectral_covariance import SharedSpectrumRecovery
from .weak_signal_probe import observed_noise_rms


def packet_frame(local_frame):
    """Keep physical chirp-to-chirp gain in a single packet normalization."""
    local, rotation, quality, rms = local_frame
    packet_rms = rms.square().mean(1, keepdim=True).sqrt()
    return local*(rms/packet_rms), rotation, quality, packet_rms


def observed_frame(tracker, iq):
    return packet_frame(coordinates(tracker, iq))


def synthesize(observed, shape, gain, *, coupled):
    carrier = observed.mean(-1, keepdim=True)
    detail = zero_carrier(shape)
    corrected_detail = detail*gain if coupled else detail
    return carrier*gain + corrected_detail


def carrier_error(proposal, reference):
    observed = proposal.mean(-1).abs()
    target = reference.mean(-1).abs()
    return ((observed-target).square()/(.04+target.square())).log1p().mean()


def amplitude_risk(coarse, fine, target, half, half_target, *, source_frame, received_half):
    values = torch.stack((carrier_error(coarse,target),carrier_error(fine,target),carrier_error(half,half_target)))
    return .25*values.mean(), values


def amplitude_perturb(base, changed, *, generator):
    """Draw general smooth and independent gain variation with no scenario inputs."""
    shape = (*base.shape[:2],1)
    opts = dict(device=base.device,dtype=base.real.dtype,generator=generator)
    iid = torch.randn(shape,**opts)
    ramp = torch.linspace(-1,1,shape[1],device=base.device,dtype=base.real.dtype)[None,:,None]
    slope = torch.randn((shape[0],1,1),**opts)
    fraction = torch.rand((shape[0],1,1),**opts)
    strength = torch.rand((shape[0],1,1),**opts)
    variation = strength*((1-fraction)*iid + fraction*slope*ramp)
    log_gain = (variation-variation.mean(1,keepdim=True)).clamp(-1,1)
    gain = log_gain.exp()
    return base*gain, changed*gain


def pair_builder(signal, change, perturb, rng, *, with_identity):
    def transform(base, moved):
        base, moved = apply_generic_physics(base,moved,generator=rng)
        return amplitude_perturb(base,moved,generator=rng)

    return antithetic_inputs(signal,change,perturb,rng,with_identity=with_identity,transform=transform)


def training_objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    return objective(model,tracker,iq,rng,with_identity=with_identity,recipe=recipe,step=step,
                     split_risk=False,build_inputs=pair_builder,private_risk_weight=.20,lag_weight=.25,
                     observation_frame=observed_frame,carrier_risk=amplitude_risk)


class CarrierCoupledRecovery(SharedSpectrumRecovery):
    def __init__(self, widths, *, coupled):
        super().__init__(widths, correlated=True)
        self.scale_detail = coupled
        self.envelope = nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,2))
        nn.init.zeros_(self.envelope[-1].weight)
        nn.init.zeros_(self.envelope[-1].bias)

    def log_gains(self, observed, quality):
        amp = observed.mean(-1,keepdim=True).abs().clamp_min(1e-4).log()
        rad = observed.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).log()*.5
        local_noise = observed_noise_rms(observed)
        log_noise = local_noise.clamp_min(1e-4).log()
        level = amp.median(1,keepdim=True).values.expand_as(amp)
        spread = (amp-level).abs().mean(1,keepdim=True).expand_as(amp)
        shared_noise = log_noise.median(1,keepdim=True).values.expand_as(amp)
        features = torch.cat((amp,amp-level,rad,quality,log_noise,log_noise-shared_noise,level,spread),dim=-1)
        return log(4)*self.envelope(features).tanh()

    def gains(self, observed, quality):
        return self.log_gains(observed,quality).exp().unbind(-1)

    def forward(self, observed, quality, rotation):
        coarse, fine = super().forward(observed,quality,rotation)
        # Limit the detail edit before composing with either carrier gain.
        coarse, fine = bound_refinement(observed,coarse,fine,torch.ones_like(quality))
        gains = self.gains(observed,quality)
        return tuple(synthesize(observed,wave,gain.unsqueeze(-1),coupled=self.scale_detail)
                     for wave,gain in zip((coarse,fine),gains,strict=True))
