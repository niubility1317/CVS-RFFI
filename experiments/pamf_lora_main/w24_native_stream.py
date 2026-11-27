"""Refine physical IQ with a matched test of context across native chirp boundaries."""

import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_lora_main.w24_native_dura import capture_scale


class NativeStreamRefiner(nn.Module):
    """Predict a bounded complex correction conditioned on the original and coarse IQ."""

    def __init__(self, *, continuous):
        super().__init__()
        self.continuous = continuous
        width = 48
        self.input = nn.Conv1d(4, width, 7, padding=3)
        self.context = nn.ModuleList(
            nn.Sequential(nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation, groups=width),
                          nn.Conv1d(width, width, 1), nn.SiLU())
            for dilation in (1, 4, 16, 64, 256)
        )
        self.edit = nn.Conv1d(width, 2, 5, padding=2)
        nn.init.zeros_(self.edit.weight)
        nn.init.zeros_(self.edit.bias)

    def forward(self, raw, coarse):
        scale = capture_scale(raw)
        samples = raw / scale
        previous_edit = (coarse - raw) / scale
        inputs = torch.stack((samples.real, samples.imag, previous_edit.real, previous_edit.imag), dim=1)
        if not self.continuous:
            inputs = inputs.unflatten(-1, (8, 1024)).transpose(1, 2).flatten(0, 1)
        feature = F.silu(self.input(inputs))
        for block in self.context:
            feature = feature + block(feature)
        edit = self.edit(feature)
        if not self.continuous:
            edit = edit.unflatten(0, (len(raw), 8)).transpose(1, 2).flatten(-2)
        delta = torch.complex(*edit.float().unbind(dim=1))
        return coarse + .25 * scale * delta / (1 + delta.abs().square()).sqrt()
