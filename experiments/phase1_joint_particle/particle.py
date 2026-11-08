"""Source-trainable joint IQ proposals and per-proposal complex gain fitting.

This deterministic finite ensemble is not a calibrated posterior sampler.  The
physical action is the frozen DURA estimate of the *current* received window;
no simulator action, classifier, label, second observation or batch statistic is
an inference input.  Its complex gain is re-estimated separately for every
proposal.  Only the denoising network is trainable.
"""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.polyphase_conv import polyphase_conv1d


PARTICLES = 4
STEPS = 3
LATENT_SEED = 2026110811
GAIN_RIDGE = 0.1
PROPOSAL_SCALE = 0.25
PHYSICS_SCALE = 0.1


def _pair(value):
    return torch.stack((value.real, value.imag), dim=1)


def _complex(value):
    return torch.complex(value[:, 0], value[:, 1])


def _rms(value):
    return value.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()


def _repeat_action(action, particles):
    """Build matching independent action rows; never mix physical records."""
    if particles == 1:
        return action
    return type(action)(action.kernel.repeat_interleave(particles, dim=0),
                        action.main.repeat_interleave(particles, dim=0),
                        action.image.repeat_interleave(particles, dim=0))


def fitted_gain(action, candidate, observed, anchor_energy):
    """Ridge complex gain toward one from observation, separately per row."""
    return _gain_from_prediction(action(_complex(candidate)), _complex(observed), anchor_energy)


def _gain_from_prediction(prediction, target, anchor_energy):
    if prediction.shape != target.shape or anchor_energy.shape != (len(target),):
        raise ValueError('Matched observation, action and reference energy required')
    energy = prediction.abs().square().sum(-1)
    ridge = GAIN_RIDGE * anchor_energy.clamp_min(1e-8)
    return ((prediction.conj() * target).sum(-1) + ridge) / (energy + ridge)


def physical_direction(action, candidate, observed, anchor_energy):
    """Real-linear adjoint of the gain-adjusted observation residual."""
    prediction, target = action(_complex(candidate)), _complex(observed)
    gain = _gain_from_prediction(prediction, target, anchor_energy)
    residual = target - gain[:, None] * prediction
    return _pair(action.adjoint(gain.conj()[:, None] * residual))


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.first = nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation)
        self.dilation = dilation
        self.norm1 = nn.GroupNorm(8, width)
        self.second = nn.Conv1d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, width)

    def forward(self, value):
        hidden = F.gelu(self.norm1(polyphase_conv1d(value, self.first.weight, self.first.bias, self.dilation)))
        return F.gelu(value + self.norm2(self.second(hidden)))


class JointParticleRecovery(nn.Module):
    """Three source-trainable proposal updates, producing a single IQ window."""
    def __init__(self, width=192, steps=STEPS):
        super().__init__()
        if width < 8 or width % 8 or steps < 1:
            raise ValueError('Width must be a positive multiple of eight; steps positive')
        self.steps = int(steps)
        self.entry = nn.Sequential(nn.Conv1d(10, width, 7, padding=3),
                                   nn.GroupNorm(8, width), nn.GELU())
        self.blocks = nn.Sequential(*(ResidualBlock(width, d) for d in (1, 2, 4, 8, 16, 32)))
        self.output = nn.Conv1d(width, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        generator = torch.Generator(device='cpu').manual_seed(LATENT_SEED)
        half = torch.randn((PARTICLES // 2, 2, 256), generator=generator)
        self.register_buffer('fixed_latent', torch.cat((half, -half), dim=0))

    def forward(self, received, anchor, innovation, variance, action, *, latent=None, validate=True):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or
                anchor.shape != received.shape or innovation.shape != received.shape or
                variance.shape != (len(received),)):
            raise ValueError('Matching current IQ windows and per-record variance required')
        if validate and (not all(torch.isfinite(x).all() for x in (received, anchor, innovation, variance))
                         or (variance <= 0).any()):
            raise ValueError('Finite inputs and positive variance required')
        if latent is None:
            latent = self.fixed_latent[None].expand(len(received), -1, -1, -1)
        if latent.ndim != 4 or latent.shape[0] != len(received) or latent.shape[2:] != (2, 256) or latent.shape[1] < 1:
            raise ValueError('Latent must have shape [batch, particles, 2, 256]')
        if validate and not torch.isfinite(latent).all():
            raise ValueError('Finite latent proposals required')
        particles = latent.shape[1]
        current_action = _repeat_action(action, particles)
        repeat = lambda x: x.repeat_interleave(particles, dim=0)
        y, base, initial_innovation, var = map(repeat, (received, anchor, innovation, variance))
        noise = latent.reshape(-1, 2, 256).to(device=y.device, dtype=y.dtype)
        scale = _rms(base)
        sigma = torch.minimum(var.sqrt()[:, None, None], scale * 0.25)
        state = base + sigma * noise
        baseline_signal = current_action(_complex(base))
        anchor_energy = baseline_signal.abs().square().sum(-1)
        baseline_direction = physical_direction(current_action, base, y, anchor_energy)
        log_noise = torch.log1p(var[:, None, None] / scale.square()).expand(-1, 1, 256)
        for index in range(self.steps):
            fraction = torch.full_like(log_noise, (index + 1) / self.steps)
            features = torch.cat((y/scale, base/scale, state/scale, initial_innovation/scale,
                                  log_noise, fraction), dim=1)
            delta = PROPOSAL_SCALE * scale * self.output(self.blocks(self.entry(features))).tanh()
            proposed = base + delta
            direction = physical_direction(current_action, proposed, y, anchor_energy) - baseline_direction
            magnitude = _rms(direction)
            bounded = scale * torch.tanh(magnitude / scale) * direction / magnitude
            candidate = proposed + PHYSICS_SCALE * bounded
            remaining = (self.steps - index - 1) / self.steps
            state = candidate + remaining * sigma * noise
        return candidate.reshape(len(received), particles, 2, 256).mean(dim=1)
