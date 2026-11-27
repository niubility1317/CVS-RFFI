"""Undistilled R02/R07/R17 stages on continuous, 1 MS/s LoRa preambles."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import Action, soft_map
from experiments.pamf_reception_restore.models import ComplexTFUNet, Core
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_uncertainty_restore.models import weighted_map


NATIVE_LENGTH = 8192
STAGES = ('raw_r02', 'raw_r07', 'raw_r17')


def capture_scale(iq):
    return iq.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()


class NativeActionEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = Core(incoming=4,outgoing=4)
        self.global_head = nn.Linear(128,129)
        nn.init.zeros_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)
        impulse = torch.zeros(64)
        # The source pairs share the capture time origin.
        impulse[0] = 1.
        self.register_buffer('initial_kernel',impulse)

    def forward(self, received, coarse):
        if received.ndim != 3 or received.shape[1:] != (2, NATIVE_LENGTH) or received.shape != coarse.shape:
            raise ValueError('Expected sole received/coarse 2 x 8192 continuous IQ')
        features = self.core.encoder(torch.cat((received,coarse),1))
        value = features[-1]
        global_value = self.global_head(value.mean(-1))
        kernel = torch.complex(global_value[:, :64] + self.initial_kernel,global_value[:, 64:128])
        kernel = kernel / kernel.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        for module,skip in zip(self.core.decode,features[-2::-1],strict=True):
            value = module(torch.cat((F.interpolate(value,size=skip.shape[-1],mode='linear',
                                                    align_corners=False),skip),1))
        point = self.core.out(value)
        main = 1 + torch.complex(point[:, 0],point[:, 1])
        image = torch.complex(point[:, 2],point[:, 3])
        log_variance = math.log(1e-8) * (1 - torch.sigmoid(global_value[:, -1] + .351))
        return Action(kernel,main,image), log_variance.exp()


class NativeDura(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse = ComplexTFUNet()
        self.estimator = NativeActionEstimator()
        prior = Core(incoming=9)
        self.priors = nn.ModuleList([prior, copy.deepcopy(prior), copy.deepcopy(prior)])

    def estimate(self, received):
        coarse = self.coarse(received)
        action, variance = self.estimator(received, coarse)
        return coarse, action, variance

    def unroll(self, received, coarse, action, variance):
        observed = complex_iq(received)
        current = coarse
        logvar = (variance.log().clamp(-16, 4) / 8)[:, None, None].expand(-1, 1, NATIVE_LENGTH)
        variance = variance[:, None].expand_as(observed.real)
        for core in self.priors:
            innovation = pair(action.adjoint(observed - action(complex_iq(current))))
            innovation = innovation / (innovation.square().mean((1, 2), keepdim=True) + 1e-6).sqrt()
            proposal = coarse + SCALE * core(torch.cat((received, coarse, current, innovation, logvar), 1))
            current = pair(weighted_map(action, observed, complex_iq(proposal), variance,
                                        steps=2, initial=complex_iq(current)))
        return current

    def forward(self, raw, stage='raw_r17'):
        if raw.ndim != 2 or raw.shape[-1] != NATIVE_LENGTH or not raw.is_complex() or stage not in STAGES:
            raise ValueError('Expected registered native LoRa samples and stage')
        scale = capture_scale(raw)
        incoming = pair(raw / scale)
        if stage == 'raw_r02':
            output = self.coarse(incoming)
        else:
            coarse, action, variance = self.estimate(incoming)
            if stage == 'raw_r07':
                output = pair(soft_map(action, complex_iq(incoming), complex_iq(coarse), variance, SCALE**2))
            else:
                output = self.unroll(incoming, coarse, action, variance)
        return complex_iq(output) * scale


def source_pair(raw, rng):
    """Generic lawful capture-rate simulation and its training-only action."""
    clean = raw / capture_scale(raw)
    batch, n = clean.shape
    options = dict(device=clean.device, dtype=clean.real.dtype, generator=rng)
    draw = torch.rand((batch, 6), **options)
    identity = draw[:, :1] < .15
    channel = (draw[:, :1] >= .15) & (draw[:, :1] < .55)
    kernel = clean.new_zeros((batch, 64))
    delays = (1, 3, 7, 13, 24, 48)
    taps = torch.randn((batch, len(delays)), device=raw.device, dtype=raw.dtype, generator=rng)
    kernel[:, 0] = 1.
    kernel[:, delays] = .075 * channel * taps
    kernel = kernel / kernel.norm(dim=-1, keepdim=True)
    time = torch.linspace(-.5, .5, n, device=clean.device, dtype=clean.real.dtype)
    phase = channel * (.8 * (draw[:, 1:2] - .5) + 8 * math.pi * (draw[:, 2:3] - .5) * time)
    main = torch.polar(torch.ones_like(phase), phase)
    image = (.16 * channel * torch.complex(draw[:, 3:4] - .5, draw[:, 4:5] - .5)).expand_as(main)
    action = Action(kernel, main, image)
    snr_db = -8. + 32. * draw[:, 5:6]
    half_variance = torch.where(identity, 0., .5 * 10 ** (-snr_db / 10.))
    perturbation = torch.randn(clean.shape, device=raw.device, dtype=raw.dtype, generator=rng)
    measured = action(clean) + (2 * half_variance).sqrt() * perturbation
    measured = torch.where(identity, clean, measured)
    scale = capture_scale(measured)
    return clean / scale, measured / scale, action, (half_variance / scale.square()).clamp_min(1e-8).flatten()


def iq_error(output, clean):
    return ((output - clean).abs().square().mean(-1) /
            clean.abs().square().mean(-1).clamp_min(1e-12)).mean()


def coarse_loss(output, clean):
    values = [torch.fft.fft(x, norm='ortho').abs().square().clamp_min(1e-6).log() for x in (output, clean)]
    return iq_error(output, clean) + .1 * (values[0] - values[1]).abs().mean()


def action_loss(estimated, variance, exact, expected_variance, observed, rng):
    probes = [torch.randn(observed.shape, device=observed.device, dtype=observed.dtype, generator=rng)
              for _ in range(2)]
    risk = torch.stack([iq_error(estimated(x), exact(x)) for x in probes]).mean()
    return risk + .1 * (variance.log() - expected_variance.log()).square().mean()
