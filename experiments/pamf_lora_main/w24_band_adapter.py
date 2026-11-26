"""Equal-capacity time and frequency adapters for the final W24 MAP prior."""

import importlib

import torch
from torch import nn


class BandAdapter(nn.Module):
    def __init__(self, *, frequency):
        super().__init__()
        self.frequency = frequency
        bins = (torch.fft.fftfreq(384) * 384).round().abs()
        self.register_buffer('band', (bins >= 9) & (bins <= 32), persistent=False)
        self.filter = nn.Sequential(
            nn.Conv1d(9, 24, 5, padding=2),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv1d(24, 24, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(24, 2, 3, padding=1),
        )
        nn.init.zeros_(self.filter[-1].weight)
        nn.init.zeros_(self.filter[-1].bias)

    def forward(self, observed, current, innovation, prior_edit, noise):
        inputs = torch.stack((observed, current, innovation, prior_edit), dim=-2)
        spectrum = torch.fft.fft(inputs, norm='ortho') * self.band
        if self.frequency:
            fields = torch.fft.fftshift(spectrum, dim=-1)
        else:
            fields = torch.fft.ifft(spectrum, norm='ortho')
        noise_field = noise.log().clamp(-11., 2.) / 5.
        channels = torch.cat((fields.real, fields.imag, noise_field.unsqueeze(-2).expand_as(fields.real[..., :1, :])),
                             dim=-2)
        real, imag = self.filter(channels.flatten(0, 1)).unflatten(0, observed.shape[:2]).unbind(-2)
        edit = torch.complex(real, imag)
        if self.frequency:
            edit = torch.fft.ifftshift(edit, dim=-1)
        else:
            edit = torch.fft.fft(edit, norm='ortho')
        return torch.fft.ifft(edit * self.band, norm='ortho')


def restorer(training, *, frequency):
    from experiments.pamf_lora_main.w24_final_radius import projected_stages

    package = training.DuraW24.__module__.rpartition('.')[0]
    archive = importlib.import_module('.dura_capacity', package)

    class SecondPriorAdapter(training.DuraW24):
        def __init__(self):
            super().__init__()
            self.requires_grad_(False)
            self.band_adapter = BandAdapter(frequency=frequency)
            self.train()

        def train(self, mode=True):
            # Fixed source modules use their inference precision in both runs.
            super().train(False)
            self.band_adapter.train(mode)
            return self

        def stages(self, y, quality, rms, *, condition=()):
            return projected_stages(self, archive, y, quality, rms, condition=condition,
                                    expand_radius=False, adapter=self.band_adapter)

    return SecondPriorAdapter()


def load_parent(model, weights, *, new_module='band_adapter'):
    state = model.state_dict()
    base = set(state) - {f'{new_module}.{key}' for key in getattr(model, new_module).state_dict()}
    if set(weights) != base:
        raise ValueError('The fixed base and source parent weights differ')
    state.update(weights)
    model.load_state_dict(state, strict=True)
