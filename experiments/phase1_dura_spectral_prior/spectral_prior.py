"""Frequency-coupled posterior correction for a DURA prior stage."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SpectralPosteriorCorrection(nn.Module):
    """Predict complex frequency responses using phase-invariant evidence.

    Four current-window IQ streams and a blind noise estimate are the only
    inputs. The zero output head leaves the existing DURA prior unchanged.
    """

    def __init__(self, width=32):
        super().__init__()
        self.entry = nn.Conv1d(11, width, 7, padding=3, padding_mode='circular')
        self.layers = nn.ModuleList(
            nn.Conv1d(width, width, 5, padding=2*d, dilation=d, padding_mode='circular')
            for d in (1, 2, 4, 8))
        self.coefficients = nn.Conv1d(width, 6, 1)
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
        residuals = torch.stack((received-prior, current-prior, posterior-prior), dim=1)
        p = torch.complex(prior[:, 0], prior[:, 1])
        r = torch.complex(residuals[:, :, 0], residuals[:, :, 1])
        pf = torch.fft.fft(p, dim=-1, norm='ortho')
        rf = torch.fft.fft(r, dim=-1, norm='ortho')
        scale = prior.square().mean((1, 2)).clamp_min(1e-8).sqrt()[:, None]
        pabs = (pf.real.square()+pf.imag.square()+1e-12).sqrt()
        rabs = (rf.real.square()+rf.imag.square()+1e-12).sqrt()
        amplitude = torch.cat((torch.log1p(pabs[:, None]/scale[:, None]),
                               torch.log1p(rabs/scale[:, None])), dim=1)
        coherence = rf*pf[:, None].conj()/(rabs*pabs[:, None]).clamp_min(1e-8)
        noise = ((log_variance-2*scale[:, 0].log()).clamp(-16, 8)/8)[:, None, None]
        features = torch.cat((amplitude, coherence.real, coherence.imag,
                              noise.expand(-1, 1, shape[-1])), dim=1)
        h = F.silu(self.entry(features))
        for layer in self.layers:
            h = h+0.2*F.silu(layer(h))
        weights = self.coefficients(h).tanh()
        delta_f = (torch.complex(weights[:, :3], weights[:, 3:])*rf).sum(1)
        delta = torch.fft.ifft(delta_f, dim=-1, norm='ortho')
        return prior+(0.20/math.sqrt(3))*torch.stack((delta.real, delta.imag), dim=1)
