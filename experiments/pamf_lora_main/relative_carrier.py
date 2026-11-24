"""Train relative carrier recovery against an independently tracked source reference."""
import torch

from .antithetic_detail import objective
from .carrier_coupled import CarrierCoupledRecovery, observed_frame, pair_builder


def match_packet_power(value, reference):
    actual = value.abs().square().mean((-2, -1), keepdim=True)
    wanted = reference.abs().square().mean((-2, -1), keepdim=True)
    nonzero = (actual > 1e-12) & (wanted > 1e-12)
    scale = (wanted.clamp_min(1e-12) / actual.clamp_min(1e-12)).sqrt()
    return torch.where(nonzero, value*scale, reference)


def mix(observed, proposed, dose):
    return match_packet_power(torch.lerp(observed,proposed,dose),observed)


def relative_carrier(wave):
    level = wave.mean(-1).abs().clamp_min(1e-4).log()
    return level - level.mean(-1,keepdim=True)


def amplitude_risk(coarse, fine, _target, _half, _half_target, *, source_frame, received_half):
    # The reference does not inherit the noisy observation's phase slope.
    reference = relative_carrier(source_frame)[:,None,:]

    def error(wave):
        difference = relative_carrier(wave) - reference
        return difference.square().log1p().mean()

    shape = received_half.shape[:2]
    scores = torch.stack((error(coarse.unflatten(0,shape)),error(fine.unflatten(0,shape)),error(received_half)))
    return .25*scores.mean(),scores


def normalized_pairs(clean, change, perturb, rng, *, with_identity):
    candidates = pair_builder(clean,change,perturb,rng,with_identity=with_identity)
    return match_packet_power(candidates,clean[:,None])


def training_objective(model, tracker, iq, rng, *, with_identity, recipe, step, source_shape=False,
                       first_map_weight=0.):
    return objective(model,tracker,iq,rng,with_identity=with_identity,recipe=recipe,step=step,
                     split_risk=False,build_inputs=normalized_pairs,private_risk_weight=.20,lag_weight=.25,
                     observation_frame=observed_frame,carrier_risk=amplitude_risk,mix=mix,
                     source_shape=source_shape, first_map_weight=first_map_weight)


class RelativeCarrierRecovery(CarrierCoupledRecovery):
    def gains(self, observed, quality):
        log_gain = .5*self.log_gains(observed,quality)
        log_gain = log_gain-log_gain.mean(1,keepdim=True)
        return log_gain.exp().unbind(-1)

    def forward(self, observed, quality, rotation):
        stages = super().forward(observed,quality,rotation)
        return tuple(match_packet_power(proposal,observed) for proposal in stages)
