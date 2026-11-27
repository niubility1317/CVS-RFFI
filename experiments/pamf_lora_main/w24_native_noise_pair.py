"""Generic independent noisy pairs and a residual second stage for native LoRa."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.w24_native_dura import capture_scale


PAIR_WEIGHT = 1.


class PairedNoiseViews:
    """Expose two simulated waveforms, with independent noise and no truth traces."""

    def __init__(self, device, *, shared_action):
        self.shared_action = shared_action
        self.selected = torch.zeros((), dtype=torch.int64, device=device)
        self.acted = torch.zeros_like(self.selected)

    def __call__(self, clean, rng):
        batch, length = clean.shape
        options = dict(device=clean.device, dtype=clean.real.dtype, generator=rng)
        draws = torch.rand((batch, 6), **options)
        strong = draws[:, :1] < 1/3
        identity = draws[:, :1] >= .9
        acted = ~(strong | identity)
        self.selected += strong.sum()
        self.acted += acted.sum()

        delays = (1, 4, 13, 32)
        kernel = clean.new_zeros((batch, max(delays)+1))
        kernel[:, 0] = 1.
        taps = torch.randn((batch, len(delays)), device=clean.device, dtype=clean.dtype, generator=rng)
        kernel[:, delays] = .085 * acted * taps
        kernel = kernel / kernel.norm(dim=-1, keepdim=True)
        t = torch.linspace(-.5, .5, length, device=clean.device, dtype=clean.real.dtype)
        phase = acted * (.8*(draws[:, 1:2]-.5) + 6*math.pi*(draws[:, 2:3]-.5)*t)
        amplitude = 1. + acted * .45 * (2*draws[:, 3:4]-1)
        main = torch.polar(amplitude.expand_as(phase), phase)
        image = (.10*acted*torch.complex(draws[:, 4:5]-.5, draws[:, 5:6]-.5)).expand_as(main)
        warped = torch.where(acted, Action(kernel, main, image)(clean), clean)

        draw_snr = torch.rand((batch, 1), **options)
        snr = torch.where(strong, -8. + 12.*draw_snr, -4. + 28.*draw_snr)
        scale = (~identity) * capture_scale(warped) * 10 ** (-snr/20.)
        first, second = (torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=rng) * scale
                         for _ in range(2))
        witness = warped if self.shared_action else clean
        return warped+first, witness+second


def intermediate_loss(frontend, coarse, witness, coarse_clean, clean):
    error = coarse - witness
    center = frontend.aperture(error)[..., :384]
    return .75*center.abs().square().mean() + .25*error.abs().square().mean() + (
        coarse_clean - clean).abs().square().mean()


def feature_stack(coarse, removed):
    return torch.stack((coarse.real, coarse.imag, removed.real, removed.imag), dim=-2)


class RefinementBranch(nn.Module):
    def __init__(self, inputs, dilations):
        super().__init__()
        width = 32
        self.stem = nn.Conv1d(inputs, width, 7, padding=3)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv1d(width,width,5,padding=2*d,dilation=d,groups=width),
                          nn.Conv1d(width,width,1),nn.SiLU()) for d in dilations
        )
        self.edit = nn.Conv1d(width,2,5,padding=2)
        nn.init.zeros_(self.edit.weight)
        nn.init.zeros_(self.edit.bias)

    def forward(self, features):
        x = F.silu(self.stem(features.flatten(0,1)))
        for block in self.blocks:
            x = x + block(x)
        return torch.complex(*self.edit(x).float().unflatten(0, features.shape[:2]).unbind(dim=-2))


class NoisePairRefiner(nn.Module):
    """Reconcile the raw and intermediate IQ without another synchronization."""

    def __init__(self):
        super().__init__()
        self.time = RefinementBranch(4, (1,4,16,64))
        self.frequency = RefinementBranch(6, (1,4,16))

    def forward(self, observed, coarse):
        removed = observed - coarse
        reconstructed, residual = (torch.fft.fft(v,norm='ortho') for v in (coarse,removed))
        spectral = torch.cat((feature_stack(reconstructed,residual),
                              reconstructed.abs().log1p().unsqueeze(-2),
                              residual.abs().log1p().unsqueeze(-2)),dim=-2)
        edit = self.time(feature_stack(coarse,removed)) + torch.fft.ifft(self.frequency(spectral),norm='ortho')
        return .35 * edit / (1 + edit.abs().square()).sqrt()
