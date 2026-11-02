"""Independent complex patch-dictionary prior for an initial IQ estimate."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class EquivariantPatchDictionary(nn.Module):
    """Synthesize a bounded residual using phase-invariant patch assignments.

    The model consumes only current received IQ, an independently constructed
    initial waveform and a current-window estimated log noise variance.  It has
    no DURA dependency and starts as the identity on the initial waveform.
    """

    def __init__(self, atoms=128, width=48, patch=32, stride=8, temperature=0.15):
        super().__init__()
        if (atoms < 2 or width < 1 or patch < 1 or stride < 1 or patch > 256
                or (256-patch) % stride or stride > patch or temperature <= 0):
            raise ValueError('A covering patch grid and positive dictionary dimensions are required')
        self.atoms = atoms
        self.patch = patch
        self.stride = stride
        self.temperature = float(temperature)
        self.atom_real = nn.Parameter(torch.randn(atoms, patch)/math.sqrt(patch))
        self.atom_imag = nn.Parameter(torch.randn(atoms, patch)/math.sqrt(patch))
        self.entry = nn.Conv1d(7, width, 3, padding=1)
        self.layers = nn.ModuleList(nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in (1, 2, 4))
        self.gate_output = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def forward(self, received, initial, logvar):
        if (received.ndim != 3 or received.shape[1:] != (2, 256)
                or initial.shape != received.shape or logvar.shape != (len(received),)):
            raise ValueError('Matching IQ windows and one estimated log variance are required')
        if not all(torch.isfinite(x).all() for x in (received, initial, logvar)):
            raise FloatingPointError('Finite current-window inputs are required')
        raw_parts = received.unfold(-1, self.patch, self.stride)
        base_parts = initial.unfold(-1, self.patch, self.stride)
        raw = torch.complex(raw_parts[:, 0], raw_parts[:, 1])
        base = torch.complex(base_parts[:, 0], base_parts[:, 1])
        dictionary = torch.complex(self.atom_real, self.atom_imag)
        atom_energy = dictionary.abs().square().sum(-1).clamp_min(1e-8)
        dot = torch.einsum('btp,kp->btk', base, dictionary.conj())
        coeff = dot/atom_energy
        patch_energy = base.abs().square().sum(-1).clamp_min(1e-8)
        residual = (patch_energy[..., None]-dot.abs().square()/atom_energy).clamp_min(0)
        relative_error = residual/patch_energy[..., None]
        weights = torch.softmax(-relative_error/self.temperature, dim=-1)
        decoded = torch.einsum('btk,btk,kp->btp', weights.to(coeff.dtype), coeff, dictionary)

        raw_power = raw.abs().square().mean(-1)
        base_power = base.abs().square().mean(-1)
        power = (2*received.square().mean((1, 2))).clamp_min(1e-8)[:, None]
        coherence = (raw.conj()*base).mean(-1)/(raw_power*base_power).clamp_min(1e-12).sqrt()
        entropy = -(weights*weights.clamp_min(1e-12).log()).sum(-1)/math.log(self.atoms)
        noise = (logvar[:, None]-power.log()).clamp(-16, 8)/8
        features = torch.stack(((raw_power+1e-8).log()-power.log(),
                                (base_power+1e-8).log()-power.log(), coherence.real, coherence.imag,
                                (weights*relative_error).sum(-1), entropy, noise.expand_as(raw_power)), dim=1)
        hidden = F.silu(self.entry(features))
        for layer in self.layers:
            hidden = hidden+0.2*F.silu(layer(hidden))
        strength = 0.5*self.gate_output(hidden).tanh()[:, 0, :, None]
        delta = strength*(decoded-base)
        pair = torch.stack((delta.real, delta.imag), dim=1)
        folded = F.fold(pair.permute(0, 1, 3, 2).reshape(len(initial), 2*self.patch, -1),
                        output_size=(1, 256), kernel_size=(1, self.patch), stride=(1, self.stride))
        count = F.fold(initial.new_ones((1, self.patch, pair.shape[2])), output_size=(1, 256),
                       kernel_size=(1, self.patch), stride=(1, self.stride))
        return initial+folded[:, :, 0]/count[:, :, 0]
