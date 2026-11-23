"""Joint restoration with a regularized receiver-side preamble estimate."""

import torch

from .channel_blind import ChannelBlind, generic_observation, zero_carrier


LAGS = (-1,0,1)


def echo_taps(clean, rng):
    """Draw two local complex echoes independently of the test channel."""
    shape = (len(clean), 1, 1, 2)
    angle = (2. * torch.pi) * torch.rand(shape, device=clean.device, generator=rng)
    strength = .25 * torch.rand(shape, device=clean.device, generator=rng).square()
    return torch.polar(strength, angle).unbind(-1)


def apply_echo(clean, taps):
    first, second = taps
    return clean + first * clean.roll(1, -1) + second * clean.roll(-2, -1)


def generic_distortion(clean, rng):
    """Generate a broad two-echo stress family independent of test channels."""
    return generic_observation(apply_echo(clean, echo_taps(clean, rng)), rng)


def estimate_pilot(observed, template, variance):
    matrix = torch.stack([template.roll(k,-1) for k in LAGS],dim=-1)
    hermitian = matrix.conj().transpose(-2,-1)/matrix.shape[-2]
    ridge = (variance+.02)[...,None]
    eye = torch.eye(len(LAGS),device=observed.device,dtype=observed.dtype)
    identity = observed.new_tensor((0.,1.,0.))
    metric = hermitian@matrix+ridge*eye
    samples = (observed*template)[...,None]
    rhs = hermitian@samples+ridge*identity[:,None]
    taps = torch.linalg.solve_ex(metric,rhs,check_errors=False).result.squeeze(-1)
    frequency = torch.fft.fftfreq(observed.shape[-1],device=observed.device)
    lag = frequency.new_tensor(LAGS)
    angles = -2.*torch.pi*lag[:,None]*frequency
    kernel = torch.polar(torch.ones_like(angles),angles)
    return taps@kernel


def pilot_inverse(observed, proposed, template, variance):
    response = estimate_pilot(observed,template,variance)
    received = torch.fft.fft(observed*template,norm='ortho')
    prior = torch.fft.fft(proposed*template,norm='ortho')
    strength = variance*observed.shape[1]
    gain = response.abs().square()
    refined = (response.conj()*received+strength*prior)/(gain+strength).clamp_min(1e-4)
    refined = torch.fft.ifft(refined,norm='ortho')*template.conj()
    return observed+zero_carrier(refined-observed)


class PilotJoint(ChannelBlind):
    def __init__(self, *, equalize):
        super().__init__()
        self.equalize = equalize

    def forward(self, observed, quality, template):
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        coarse,proposed,variance = self.propose(observed,quality,mixed_conv=mixed)
        strength = variance*observed.shape[1]
        if self.equalize:
            refined = pilot_inverse(observed,proposed,template,variance)
        else:
            refined = (observed+strength*proposed)/(1.+strength)
        return coarse,refined
