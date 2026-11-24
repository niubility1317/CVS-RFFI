"""A width-controlled DURA backbone with receive-estimated LoRa action and noise."""

from dataclasses import dataclass

import torch
from torch import nn

from experiments.pamf_reception_restore.models import Core
from .band_allocation import fft, ifft
from .channel_blind import zero_carrier
from .channel_blind_source import waveform_risk
from .common_anchor import iq_field
from .dura_repair import complex_iq, real_iq
from .joint_spectrum import limit, split
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, transfer_risk, weak_difference
from .received_update import ReceivedUpdate
from .relative_echo import shift


WIDTHS = {'base': (16, 24, 36, 48), 'expanded': (32, 48, 72, 96)}
PRIOR_VARIANCE = .085
MAX_EDIT = .55


@dataclass
class ObservedAction:
    main: torch.Tensor
    image: torch.Tensor
    echo: torch.Tensor

    def __call__(self, x):
        return self.main*x + self.image*x.conj() + self.echo*shift(x, 1)

    def adjoint(self, y):
        return self.main.conj()*y + self.image*y.conj() + shift(self.echo.conj()*y, -1)


def dot(a, b):
    return (a.conj()*b).real.sum(-1, keepdim=True)


def robust_observation_variance(residual, variance):
    """Freeze a bounded complex Student-t weight over each four-sample block."""
    normalized_power = residual.abs().square() / variance
    local = normalized_power.unflatten(-1, (-1, 4))
    inflation = ((4. + local.mean(-1, keepdim=True)) / 5.).clamp(1., 8.)
    return variance * inflation.expand_as(local).flatten(-2)


def map_two(action, observed, current, prior, variance, residual=None):
    """Two warm-start CG updates with the DURA innovation reused."""
    weight = variance.reciprocal()
    if residual is None:
        residual = action.adjoint(weight*(observed-action(current)))
    residual = residual - (current-prior)/PRIOR_VARIANCE
    direction = residual
    squared = dot(residual, residual)
    for step in range(2):
        normal = action.adjoint(weight*action(direction)) + direction/PRIOR_VARIANCE
        alpha = squared / dot(direction, normal).clamp_min(1e-18)
        current = current + alpha*direction
        if step == 0:
            residual = residual-alpha*normal
            following = dot(residual, residual)
            direction = residual+(following/squared.clamp_min(1e-18))*direction
            squared = following
    return current


def polyphase_core(incoming, outgoing, widths):
    return Core(incoming=2*incoming, outgoing=2*outgoing, widths=widths)


def read_core(module, fields):
    real = fields[0]
    values = torch.stack([v.expand_as(real) for v in fields], -2).flatten(0, 1)
    values = values.unflatten(-1, (128, 2)).transpose(-1, -2).flatten(1, 2)
    mixed = module.training and values.is_cuda and torch.cuda.is_bf16_supported()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
        output = module(values)
    full = output.float().unflatten(1, (-1, 2)).transpose(-1, -2).flatten(-2)
    return full.unflatten(0, real.shape[:2]).unbind(2)


class DuraCapacity(nn.Module):
    """Time-frequency coarse proposal and two native multiscale prior/CG stages."""

    def __init__(self, size):
        super().__init__()
        widths = WIDTHS[size]
        light = tuple(max(8, 4*((n+7)//8)) for n in widths)
        self.frequency = polyphase_core(5, 2, light)
        self.coarse = polyphase_core(9, 2, light)
        self.risk = polyphase_core(9, 7, (8, 12, 16, 24))
        self.priors = nn.ModuleList(polyphase_core(13, 2, widths) for _ in range(2))
        band = 256*torch.fft.fftfreq(256).abs()
        self.register_buffer('noise_band', band > 64, persistent=False)

    @staticmethod
    def update(r, i, y, rms):
        proposed = zero_carrier(torch.complex(r.tanh(), i.tanh()))
        return limit(proposed, y, rms, y.real.new_tensor(MAX_EDIT))

    def stages(self, y, quality, rms, *, use_action=True, use_noise=True, solve=True, robust=False,
               receiver_variance=None):
        spec = fft(y)
        if receiver_variance is None:
            noise = (spec[..., self.noise_band].abs().square().median(-1, keepdim=True).values
                     / .69314718056).clamp(1e-5, 4.)
        else:
            noise = receiver_variance.clamp(1e-5, 4.)
        log_noise = noise.log().clamp(-11., 2.)/5.
        average = .5*(y[:, ::2].mean(1, keepdim=True) + y[:, 1::2].mean(1, keepdim=True))
        opposite = torch.cat((y[:, 1::2].mean(1, keepdim=True), y[:, ::2].mean(1, keepdim=True)),1).repeat(1, 3, 1)
        flip = (-torch.arange(256, device=y.device)) % 256
        f_real, f_imag = read_core(self.frequency, (*iq_field(spec, spec[..., flip].conj()), log_noise))
        spec_edit = torch.complex(f_real.tanh(), f_imag.tanh())*noise.sqrt()
        estimate = zero_carrier(ifft(spec_edit))
        pack = (*iq_field(y, opposite, estimate, y-average), quality)
        c_real, c_imag = read_core(self.coarse, pack)
        c = y + limit(zero_carrier(estimate+torch.complex(c_real.tanh(), c_imag.tanh())),
                      y, rms, y.real.new_tensor(MAX_EDIT))
        nominal, _, mismatch = ReceivedUpdate.estimate(y, noise.mean(1, keepdim=True))
        inputs = (*iq_field(y, c, opposite, nominal-1.), log_noise)
        gr, gi, ir, ii, er, ei, log_excess = (x.tanh() for x in read_core(self.risk, inputs))
        # Take four-sample means before interpolation to keep rapidly varying action
        # coefficients from absorbing peripheral source details.
        gr, gi, ir, ii, er, ei, log_excess = (
            x.unflatten(-1, (64,4)).mean(-1).repeat_interleave(4, dim=-1)
            for x in (gr, gi, ir, ii, er, ei, log_excess)
        )
        estimator = ObservedAction(nominal+.24*torch.complex(gr, gi),
                                   .12*torch.complex(ir, ii), .15*torch.complex(er, ei))
        variance = .001 + noise*torch.exp(1.2*log_excess)
        used = variance if use_noise else torch.full_like(variance, .04)
        action = estimator if use_action else ObservedAction(torch.ones_like(y),
                                                             torch.zeros_like(y), torch.zeros_like(y))
        current = c
        stages = []
        for network in self.priors:
            remaining = y-action(current)
            innovation = action.adjoint(remaining)
            normalized = innovation / (.001 + innovation.abs().square().mean(-1, keepdim=True)).sqrt()
            fields = (*iq_field(y, c, current, normalized, opposite, action.main-1.),
                      used.log().clamp(-11., 2.)/5.)
            real, imag = read_core(network, fields)
            prior = c + self.update(real, imag, y, rms)
            solve_variance = robust_observation_variance(remaining, used) if robust else used
            current = (map_two(action, y, current, prior, solve_variance, action.adjoint(remaining/solve_variance))
                       if solve else prior)
            current = y + limit(zero_carrier(current-y), y, rms, y.real.new_tensor(MAX_EDIT))
            stages.append(current)
        return (c, *stages), estimator, variance, noise, mismatch


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    raw = complex_iq(iq)
    natural, phase, _, level = coordinates(tracker, iq)
    carrier = phase*level
    weak = weak_difference(natural)
    change = weak*carrier
    views = paired_inputs(raw, recipe.perturb, rng, change, with_identity=with_identity)
    y, phase, quality, rms = coordinates(tracker, real_iq(views.flatten(0, 1)))
    frame = phase*rms
    ground = torch.stack((raw, raw, raw+change),1).flatten(0, 1)/frame
    outputs, action, variance, _, _ = model.stages(y, quality, rms)
    coarse, first, final = outputs
    final_half = y + .5*(final-y)
    corruption = (y - action(ground)).abs().square()
    action_risk = (corruption/(.01+variance.detach())).log1p().mean()
    uncertainty = (corruption.detach()/variance + variance.log()).mean()
    output = (final_half*frame).unflatten(0, views.shape[:2])
    transported = split((output[:, 2]-output[:, 0])/carrier)
    source_change = split(weak)
    retention = transfer_risk(transported[:, :1], source_change[:, :1]) + transfer_risk(
        transported[:, 1:], source_change[:, 1:])
    consistency = (((output[:,0]-output[:,1])/carrier).abs().square()).mean()
    identity = ((output-views)[-2:, :2].abs().square().mean() if with_identity else y.real.new_zeros(()))
    values = torch.stack((waveform_risk(coarse, ground), waveform_risk(first, ground),
                          waveform_risk(final, ground), waveform_risk(final_half, ground),
                          retention, consistency, action_risk, uncertainty, identity))
    a, b, c = recipe.loss_weights(step)
    return values @ values.new_tensor((.6*a, .4*a, .35*b, .65*b, .30, c, .07, .015, .35)), values
