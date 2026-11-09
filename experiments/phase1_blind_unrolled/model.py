"""End-to-end blind widely-linear channel estimation and IQ restoration.

The finite 33-tap circular channel is a learned approximation, not a claim
that the real channel is stationary, circular or uniquely identifiable.
The recovery model consumes received IQ only; no DURA or recognizer is used.
"""
import torch
from torch import nn
from torch.nn import functional as F

TAPS = 33
WIDTH = 192
STEPS = 4


def response(coefficients):
    if coefficients.ndim != 2 or coefficients.shape[1] != TAPS or not coefficients.is_complex():
        raise ValueError('Complex 33-tap coefficients required')
    h = torch.cat((coefficients[:, TAPS//2:], coefficients.new_zeros(len(coefficients), 256-TAPS),
                   coefficients[:, :TAPS//2]), 1)
    return torch.fft.fft(h, dim=-1)


def forward_action(signal, linear, image):
    if signal.ndim != 2 or signal.shape[1] != 256 or not signal.is_complex():
        raise ValueError('Complex 256-sample signal required')
    return torch.fft.ifft(response(linear)*torch.fft.fft(signal, dim=-1) +
                          response(image)*torch.fft.fft(signal.conj(), dim=-1), dim=-1)


def adjoint_action(residual, linear, image):
    return torch.fft.ifft(response(linear).conj()*torch.fft.fft(residual, dim=-1) +
        response(image.conj()).conj()*torch.fft.fft(residual.conj(), dim=-1), dim=-1)


def pair(z):
    return torch.stack((z.real, z.imag), 1)


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation),
            nn.GroupNorm(8,width),nn.GELU(),nn.Conv1d(width,width,3,padding=1),nn.GroupNorm(8,width))
    def forward(self,x):
        return F.gelu(x + .2*self.block(x))


class BlindUnrolledRecovery(nn.Module):
    def __init__(self, width=WIDTH, steps=STEPS):
        super().__init__()
        if width < 8 or width%8 or steps < 1: raise ValueError('Valid width and steps required')
        self.steps = steps
        self.channel_encoder = nn.Sequential(nn.Conv1d(2,width,7,padding=3),nn.GELU(),
            *(ResidualBlock(width,k) for k in (1,2,4,8)),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.channel_head = nn.Linear(width,4*TAPS)
        nn.init.zeros_(self.channel_head.weight);nn.init.zeros_(self.channel_head.bias)
        self.priors = nn.ModuleList()
        for _ in range(steps):
            prior = nn.Sequential(nn.Conv1d(8,width,7,padding=3),nn.GroupNorm(8,width),nn.GELU(),
                *(ResidualBlock(width,k) for k in (1,2,4,8,16,32)),nn.Conv1d(width,2,1))
            nn.init.zeros_(prior[-1].weight);nn.init.zeros_(prior[-1].bias)
            self.priors.append(prior)

    def channel(self, received):
        coefficients = .15*self.channel_head(self.channel_encoder(received)).tanh().reshape(-1,4,TAPS)
        linear = torch.complex(coefficients[:,0],coefficients[:,1])
        image = torch.complex(coefficients[:,2],coefficients[:,3])
        impulse = torch.zeros_like(linear);impulse[:,TAPS//2] = 1.
        return linear+impulse, image

    def forward(self, received, anchor=None, innovation=None, variance=None, action=None, *, return_physics=False):
        if received.ndim != 3 or received.shape[1:] != (2,256):
            raise ValueError('One received IQ window is required')
        scale = received.square().mean((1,2),keepdim=True).clamp_min(1e-12).sqrt()
        y = torch.complex((received/scale)[:,0],(received/scale)[:,1])
        linear,image = self.channel(pair(y))
        estimate = y
        for prior in self.priors:
            residual = y-forward_action(estimate,linear,image)
            direction = adjoint_action(residual,linear,image)
            features = torch.cat((pair(y),pair(estimate),pair(residual),pair(direction)),1)
            delta = prior(features)
            delta = .25*delta*torch.rsqrt(1+delta.square().sum(1,keepdim=True))
            estimate = estimate+torch.complex(delta[:,0],delta[:,1])
        restored = pair(estimate)*scale
        if return_physics:
            projected = pair(forward_action(estimate,linear,image))*scale
            return restored,projected
        return restored
