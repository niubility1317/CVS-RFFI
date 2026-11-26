"""Task-trained W24 amplitude calibration over a fixed source recovery."""

import torch
from torch import nn


class TaskGain(nn.Module):
    def __init__(self, parent, *, adaptive, seed):
        super().__init__()
        self.parent = parent.eval().requires_grad_(False)
        self.offset = nn.Parameter(torch.zeros(()))
        self.local = None
        if adaptive:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.local = nn.Sequential(nn.Linear(8,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
            nn.init.zeros_(self.local[-1].weight)
            nn.init.zeros_(self.local[-1].bias)

    def gain(self, y, quality, rms, edges):
        logits = self.offset
        if self.local is not None:
            spectrum = torch.fft.fft(y,norm='ortho')
            outer = spectrum[...,self.parent.noise_band].abs().square().median(-1,keepdim=True).values
            noise = (outer/.69314718056).clamp(1e-6,4.).log()/4
            edge = (edges.abs().square().mean(-1,keepdim=True)/rms.square()).clamp(1e-6,100.).log()/4
            relative = (rms/rms.mean(1,keepdim=True)).clamp(.01,100.).log()/4
            local = torch.cat((noise,quality,edge,relative),dim=-1)
            context = local.mean(1,keepdim=True).expand_as(local)
            logits = logits + self.local(torch.cat((local,context),dim=-1))
        return .5 + .45 * logits.tanh()

    def propose(self, y, quality, rms, edges):
        with torch.no_grad():
            (first, final), = self.parent.propose(y,quality,rms,edges)
        gain = self.gain(y,quality,rms,edges)
        # The caller applies a half edit. Preserve the parent's initial endpoint exactly.
        return (first,final + (2*gain-1) * (final-y)),
