import math
import torch
from torch import nn
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import real_dot
from experiments.pamf_reception_restore.models import Core
from experiments.pamf_restoration.models import complex_iq, pair
from .design import NAMES


def weighted_map(action, observed, prior, variance, prior_variance=SCALE**2, steps=6, *, initial=None):
    if variance.shape == (len(observed),):
        variance = variance[:, None].expand_as(observed.real)
    elif variance.shape == (len(observed), 1):
        variance = variance.expand_as(observed.real)
    if (variance.shape != observed.shape or variance.is_complex() or not torch.isfinite(variance).all() or (variance <= 0).any()
            or not math.isfinite(prior_variance) or prior_variance <= 0 or steps < 1):
        raise ValueError('Positive current-window variance and prior required')
    weight = variance.reciprocal()
    rhs = action.adjoint(weight*(observed-action(prior)))
    if initial is None:
        delta = torch.zeros_like(prior)
        residual = rhs
    else:
        if initial.shape != prior.shape or not initial.is_complex() or not torch.isfinite(initial).all():
            raise ValueError('Finite matched current-window warm start required')
        delta = initial-prior
        residual = rhs-action.adjoint(weight*action(delta))-delta/prior_variance
    direction = residual
    squared = real_dot(residual, residual)
    for _ in range(steps):
        normal = action.adjoint(weight*action(direction))+direction/prior_variance
        alpha = squared/real_dot(direction, normal).clamp_min(1e-20)
        delta = delta+alpha[:, None]*direction
        residual = residual-alpha[:, None]*normal
        following = real_dot(residual, residual)
        direction = residual+(following/squared.clamp_min(1e-20))[:, None]*direction
        squared = following
    return prior+delta


def variance_nll(action, received, clean, variance):
    # The clean waveform is a training/scoring target, never a forward input.
    with torch.no_grad():
        error = complex_iq(received)-action(complex_iq(clean))
        real_squared_error = error.abs().square()/2
    return (real_squared_error/variance+variance.log()).mean(-1)


class Confidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = Core(incoming=6, outgoing=1, widths=(16, 24, 32, 48))

    def forward(self, received, coarse, predicted_received):
        value = self.network(torch.cat((received, coarse, predicted_received), 1))[:, 0]
        return torch.exp(4*value.tanh())


class UncertaintyRestorer(nn.Module):
    def __init__(self, kind, frozen, excess=None):
        super().__init__()
        if kind not in (*NAMES, 'global_floor'):
            raise ValueError('Registered restoration kind required')
        if kind in ('uncertain_map_prior', 'global_floor') and (excess is None or not math.isfinite(excess) or excess < 1e-6):
            raise ValueError('Positive fixed source-only excess calibration required')
        self.kind = kind
        self.frozen = frozen.requires_grad_(False).eval()
        if kind != 'global_floor':
            self.prior = Core(incoming=4)
        if kind == 'uncertain_map_prior':
            self.confidence = Confidence()
        self.register_buffer('excess_variance', torch.tensor(0. if excess is None else excess, dtype=torch.float32))

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def training_outputs(self, received):
        if received.ndim != 3 or received.shape[1:] != (2, 256):
            raise ValueError('Only current received IQ2x256 is accepted')
        if self.kind == 'direct_control':
            with torch.no_grad():
                base = self.frozen(received)
            prior = base+SCALE*self.prior(torch.cat((received, base), 1))
            return prior, None, None, base
        with torch.no_grad():
            base, action, native = self.frozen.estimate(received)
        variance = native[:, None].expand(-1, received.shape[-1])
        if self.kind == 'global_floor':
            prior = base
            variance = variance+self.excess_variance
        else:
            prior = base+SCALE*self.prior(torch.cat((received, base), 1))
            if self.kind == 'uncertain_map_prior':
                with torch.no_grad():
                    predicted_received = pair(action(complex_iq(base)))
                variance = variance+self.excess_variance*self.confidence(received, base, predicted_received)
        # Variance learns a proper source residual score, not posterior reward.
        output = pair(weighted_map(action, complex_iq(received), complex_iq(prior), variance.detach()))
        return output, variance, action, base

    def forward(self, received):
        return self.training_outputs(received)[0]
