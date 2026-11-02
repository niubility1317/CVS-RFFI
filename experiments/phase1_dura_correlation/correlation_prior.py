"""Phase-equivariant global temporal correlation for a DURA prior stage."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CorrelationPriorCorrection(nn.Module):
    """Attend within one IQ window using phase-invariant temporal evidence."""

    def __init__(self, width=32, heads=4, key_size=8):
        super().__init__()
        self.heads, self.key_size = heads, key_size
        self.entry = nn.Conv1d(11, width, 7, padding=3)
        self.local = nn.Conv1d(width, width, 5, padding=2)
        self.query = nn.Conv1d(width, heads*key_size, 1)
        self.key = nn.Conv1d(width, heads*key_size, 1)
        self.phase_coeff = nn.Parameter(torch.zeros(heads, 2))
        self.coefficients = nn.Conv1d(width, heads*3*2, 1)
        nn.init.zeros_(self.coefficients.weight)
        nn.init.zeros_(self.coefficients.bias)

    def forward(self, received, current, prior, posterior, log_variance):
        shape = received.shape
        if (received.ndim != 3 or shape[1:] != (2, 256)
                or any(x.shape != shape for x in (current, prior, posterior))
                or log_variance.shape != (shape[0],)):
            raise ValueError('Four aligned IQ windows and one log variance per window are required')
        if not all(torch.isfinite(x).all() for x in (received, current, prior, posterior, log_variance)):
            raise FloatingPointError('Finite current-window evidence is required')
        batch, _, length = shape
        streams = torch.stack((received-prior, current-prior, posterior-prior), dim=1)
        scale = prior.square().mean((1, 2), keepdim=True).clamp_min(1e-8).sqrt()
        reference, normalized = prior/scale, streams/scale[:, None]
        rr, ri = reference[:, 0:1], reference[:, 1:2]
        sr, si = normalized[:, :, 0], normalized[:, :, 1]
        magnitude = torch.cat(((reference.square().sum(1, keepdim=True)+1e-12).sqrt(),
                               (sr.square()+si.square()+1e-12).sqrt()), dim=1)
        dot, cross = sr*rr+si*ri, si*rr-sr*ri
        noise = ((log_variance-2*scale[:, 0, 0].log()).clamp(-16, 8)/8)[:, None, None]
        h = F.silu(self.entry(torch.cat((magnitude, dot, cross, noise.expand(-1, 1, length)), dim=1)))
        h = h+0.2*F.silu(self.local(h))
        q = self.query(h).reshape(batch, self.heads, self.key_size, length).transpose(-1, -2)
        k = self.key(h).reshape(batch, self.heads, self.key_size, length)
        logits = torch.matmul(q, k)/math.sqrt(self.key_size)
        unit = torch.complex(rr[:, 0], ri[:, 0])
        unit = unit/(unit.abs()+1e-6)
        coherence = unit[:, :, None]*unit[:, None, :].conj()
        logits = logits + self.phase_coeff[None, :, 0, None, None]*coherence.real[:, None]
        logits = logits + self.phase_coeff[None, :, 1, None, None]*coherence.imag[:, None]
        attention = logits.softmax(dim=-1)
        vectors = torch.complex(streams[:, :, 0], streams[:, :, 1])
        values = torch.einsum('bhts,bcs->bhct', attention.to(vectors.dtype), vectors)
        weights = self.coefficients(h).tanh().reshape(batch, 2, self.heads, 3, length)
        delta = (torch.complex(weights[:, 0], weights[:, 1])*values).sum((1, 2))
        return prior+(0.25/math.sqrt(self.heads*3))*torch.stack((delta.real, delta.imag), dim=1)
