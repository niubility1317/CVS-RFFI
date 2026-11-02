"""Phase-equivariant complex edge-protected fusion around frozen DURA."""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.phase1_dura_blendgate.model import align_to_anchor, make_model as make_reference_model


class ComplexAlignedBlend(nn.Module):
    def __init__(self, template):
        super().__init__()
        if not template.preserve_edges:
            raise ValueError('Edge-protected fusion template required')
        self.expert = template.expert.eval().requires_grad_(False)
        self.entry = template.entry
        self.layers = template.layers
        self.gate_output = nn.Conv1d(24, 2, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def train(self, mode=True):
        super().train(mode)
        self.expert.eval()
        return self

    def forward(self, received, initial, log_variance):
        if (received.ndim != 3 or received.shape[1:] != (2, 256)
                or initial.shape != received.shape or log_variance.shape != (len(received),)):
            raise ValueError('Matching IQ windows and one blind log variance required')
        if not all(torch.isfinite(v).all() for v in (received, initial, log_variance)):
            raise FloatingPointError('Finite waveform evidence required')
        with torch.no_grad():
            candidate = self.expert(received, received, log_variance)
            aligned = align_to_anchor(candidate, initial, preserve_edges=True)
        if not torch.isfinite(aligned).all():
            raise FloatingPointError('Finite aligned expert output required')
        power = received.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        x = torch.complex(received[:, 0], received[:, 1]) / scale[:, 0]
        anchor = torch.complex(initial[:, 0], initial[:, 1]) / scale[:, 0]
        expert = torch.complex(aligned[:, 0], aligned[:, 1]) / scale[:, 0]
        cross = anchor.conj() * expert
        noise = ((log_variance - power[:, 0, 0].log()).clamp(-16, 8) / 8)[:, None, None]
        evidence = torch.stack((torch.log1p(x.abs().square()), torch.log1p(anchor.abs().square()),
                                torch.log1p(expert.abs().square()), cross.real.clamp(-16, 16),
                                cross.imag.clamp(-16, 16)), dim=1)
        h = F.silu(self.entry(torch.cat((evidence, noise.expand(-1, 1, 256)), dim=1)))
        for layer in self.layers:
            h = h + 0.2 * F.silu(layer(h))
        coefficient = 0.5 * self.gate_output(h).tanh()
        difference = torch.complex(aligned[:, 0] - initial[:, 0], aligned[:, 1] - initial[:, 1])
        correction = torch.complex(coefficient[:, 0], coefficient[:, 1]) * difference
        return initial + torch.stack((correction.real, correction.imag), dim=1)


def make_model():
    template = make_reference_model()
    template.preserve_edges = True
    return ComplexAlignedBlend(template)
