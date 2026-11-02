"""Phase-equivariant multiscale contextual correction around frozen DURA."""
import torch
from torch import nn
from torch.nn import functional as F


class ContextRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv1d(5, 48, 7, padding=3)
        self.down1 = nn.Conv1d(48, 64, 5, stride=2, padding=2)
        self.down2 = nn.Conv1d(64, 96, 5, stride=2, padding=2)
        self.context_norm = nn.LayerNorm(96)
        self.context = nn.MultiheadAttention(96, 4, dropout=0.0, batch_first=True)
        self.up1 = nn.Conv1d(160, 64, 5, padding=2)
        self.up0 = nn.Conv1d(112, 48, 5, padding=2)
        self.gate_output = nn.Conv1d(48, 2, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def forward(self, received, initial, log_variance):
        shape = received.shape
        if (received.ndim != 3 or shape[1:] != (2, 256) or initial.shape != shape
                or log_variance.shape != (shape[0],)):
            raise ValueError('Matching IQ windows and one blind log variance are required')
        if not all(torch.isfinite(v).all() for v in (received, initial, log_variance)):
            raise FloatingPointError('Finite received evidence is required')
        power = received.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        z = torch.complex(received[:, 0], received[:, 1]) / scale[:, 0]
        a = torch.complex(initial[:, 0], initial[:, 1]) / scale[:, 0]
        index = z.abs().argmax(dim=1, keepdim=True)
        pivot = z.gather(1, index)
        unit = pivot / pivot.abs().clamp_min(1e-8)
        zr, ar = z * unit.conj(), a * unit.conj()
        noise = ((log_variance - power[:, 0, 0].log()).clamp(-16, 8) / 8)[:, None, None]
        features = torch.stack((zr.real, zr.imag, ar.real, ar.imag), dim=1)
        x0 = F.silu(self.entry(torch.cat((features, noise.expand(-1, 1, 256)), dim=1)))
        x1 = F.silu(self.down1(x0))
        x2 = F.silu(self.down2(x1))
        q = self.context_norm(x2.transpose(1, 2))
        context, _ = self.context(q, q, q, need_weights=False)
        x2 = x2 + 0.2 * context.transpose(1, 2)
        h1 = F.silu(self.up1(torch.cat((F.interpolate(x2, size=128, mode='linear', align_corners=False), x1), dim=1)))
        h0 = F.silu(self.up0(torch.cat((F.interpolate(h1, size=256, mode='linear', align_corners=False), x0), dim=1)))
        delta = self.gate_output(h0).tanh()
        complex_delta = torch.complex(delta[:, 0], delta[:, 1]) * unit
        correction = torch.stack((complex_delta.real, complex_delta.imag), dim=1) * scale
        return initial + 0.25 * correction
