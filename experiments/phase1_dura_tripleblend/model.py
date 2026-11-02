"""Continuous raw/DURA/spectral waveform fusion with frozen reader context."""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.phase1_dura_blendgate.model import align_to_anchor, make_model as make_reference_model


class TripleBlend(nn.Module):
    def __init__(self, expert):
        super().__init__()
        self.expert = expert.eval().requires_grad_(False)
        self.entry = nn.Conv1d(9, 32, 7, padding=3)
        self.condition = nn.Sequential(nn.Linear(166, 64), nn.SiLU(), nn.Linear(64, 32))
        self.layers = nn.ModuleList(nn.Conv1d(32, 32, 5, dilation=d, padding=2*d) for d in (1, 2, 4, 8))
        self.gate_output = nn.Conv1d(32, 4, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def train(self, mode=True):
        super().train(mode); self.expert.eval(); return self

    def forward(self, received, packed, log_variance):
        n = len(received)
        if received.shape != (n, 2, 256) or packed.shape != (n, 3, 256) or log_variance.shape != (n,):
            raise ValueError('One IQ window and matching frozen-reader context required')
        if not all(torch.isfinite(v).all() for v in (received, packed, log_variance)):
            raise FloatingPointError('Finite single-window evidence required')
        anchor = packed[:, :2]
        features = packed[:, 2, :160].detach()
        logits = packed[:, 2, 160:166].detach()
        condition = self.condition(torch.cat((F.normalize(features, dim=1), F.softmax(logits, dim=1)), dim=1))
        with torch.no_grad():
            raw = align_to_anchor(received, anchor, preserve_edges=True)
            candidate = self.expert(received, received, log_variance)
            spectral = align_to_anchor(candidate, anchor, preserve_edges=True)
        if not torch.isfinite(raw).all() or not torch.isfinite(spectral).all():
            raise FloatingPointError('Finite aligned expert inputs required')
        power = received.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        x = torch.complex(received[:, 0], received[:, 1]) / scale[:, 0]
        a = torch.complex(anchor[:, 0], anchor[:, 1]) / scale[:, 0]
        b = torch.complex(raw[:, 0], raw[:, 1]) / scale[:, 0]
        c = torch.complex(spectral[:, 0], spectral[:, 1]) / scale[:, 0]
        ab, ac = a.conj()*b, a.conj()*c
        noise = ((log_variance - power[:, 0, 0].log()).clamp(-16, 8)/8)[:, None, None]
        evidence = torch.stack((torch.log1p(x.abs().square()), torch.log1p(a.abs().square()),
            torch.log1p(b.abs().square()), torch.log1p(c.abs().square()),
            ab.real.clamp(-16,16), ab.imag.clamp(-16,16), ac.real.clamp(-16,16), ac.imag.clamp(-16,16)),dim=1)
        h = F.silu(self.entry(torch.cat((evidence, noise.expand(-1,1,256)),dim=1)) + condition[:, :, None])
        for layer in self.layers: h = h + .2 * F.silu(layer(h))
        w = .35 * self.gate_output(h).tanh()
        raw_delta = torch.complex(raw[:,0]-anchor[:,0], raw[:,1]-anchor[:,1])
        expert_delta = torch.complex(spectral[:,0]-anchor[:,0], spectral[:,1]-anchor[:,1])
        delta = torch.complex(w[:,0],w[:,1])*raw_delta + torch.complex(w[:,2],w[:,3])*expert_delta
        return anchor + torch.stack((delta.real,delta.imag),dim=1)


def make_model():
    template = make_reference_model()
    return TripleBlend(template.expert)
