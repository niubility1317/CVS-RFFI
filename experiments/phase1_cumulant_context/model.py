"""DURA refinement conditioned on finite-window complex fourth cumulants.

These noisy finite-window statistics need not identify a transmitter and carry
no promise of invariance to receiver or propagation. The network receives no
transmitter label, clean reference or recognizer output at inference.
"""
import torch
from torch import nn
from experiments.phase1_dura_orthogonal.large_context_stable import StableLargeContextDura
from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate

LAGS = (0, 1, 2, 4, 8, 16, 32)
FEATURES = 3 * len(LAGS)
CUMULANT_WEIGHT = .1


def cumulant_features(iq):
    """Bounded globally centered fourth-order moment differences.

    The second statistic has three factors at the first time and one at
    the lagged time; it is not the conventional 3-unconjugated/1-conjugated
    complex cumulant. Lagged slices are not separately recentered, so their
    finite-window values are approximations rather than exact sample cumulants.
    The existing variable name does not change its formula.
    """
    if iq.ndim != 3 or iq.shape[1:] != (2, 256):
        raise ValueError('IQ must have shape [batch, 2, 256]')
    z = torch.complex(iq[:, 0], iq[:, 1])
    z = z - z.mean(-1, keepdim=True)
    z = z / z.abs().square().mean(-1, keepdim=True).clamp_min(1e-10).sqrt()
    results = []
    for lag in LAGS:
        a, b = (z, z) if lag == 0 else (z[:, :-lag], z[:, lag:])
        power_a, power_b = a.abs().square(), b.abs().square()
        cross = (a*b.conj()).mean(-1)
        pseudo = (a*b).mean(-1)
        c22 = ((power_a*power_b).mean(-1) - power_a.mean(-1)*power_b.mean(-1)
               - cross.abs().square() - pseudo.abs().square())
        c31 = ((power_a*a*b.conj()).mean(-1) - 2*power_a.mean(-1)*cross
               - a.square().mean(-1)*(a.conj()*b.conj()).mean(-1))
        results.extend((c22, c31.real, c31.imag))
    value = torch.stack(results, 1)
    return value / (1 + value.abs())


def cumulant_loss(output, clean):
    if output.shape != clean.shape or len(output) == 0:
        raise ValueError('Matching nonempty source IQ required')
    return (cumulant_features(output)-cumulant_features(clean).detach()).square().mean()


class CumulantContextRecovery(StableLargeContextDura):
    def __init__(self, width=384, layers=8):
        super().__init__(width=width, layers=layers)
        self.statistics = nn.Sequential(nn.Linear(3*FEATURES, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, received, anchor, innovation, variance, action=None):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape
                or innovation.shape != received.shape or variance.shape != (len(received),)):
            raise ValueError('Matching current IQ and DURA context required')
        power = anchor.square().mean((1, 2), keepdim=True).clamp_min(1e-12)
        scale = power.sqrt()
        noise = torch.log1p(variance[:, None, None].clamp_min(0.)/power)
        features = torch.cat((received/scale, anchor/scale, innovation/scale,
                              noise.expand(-1, 1, received.shape[-1])), 1)
        local = self.local(features)
        stats = torch.cat([cumulant_features(x) for x in (received, anchor, innovation)], 1)
        tokens = self.down(local).transpose(1, 2) + self.position + self.statistics(stats)[:, None, :]
        context = self.global_context(tokens).transpose(1, 2).repeat_interleave(4, dim=-1)
        delta = self.out(self.decode(torch.cat((local, context), 1)))
        delta = delta*torch.rsqrt(1+delta.square().sum(1, keepdim=True))*scale*self.residual_scale
        return anchor + noise_gate(anchor, variance)*delta
