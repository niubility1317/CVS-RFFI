"""DURA refinement with a bounded fourth-order frequency-coupling branch."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate

OFFSETS = (1, 2, 4, 8)
QUARTIC_WEIGHT = .1
BRANCH_SCALE = .1


def closure(spectrum):
    """Bounded per-frequency products; these are not cumulant estimates."""
    if spectrum.ndim != 2 or spectrum.shape[1] != 256 or not spectrum.is_complex():
        raise ValueError('Complex spectra with 256 bins required')
    value = torch.stack([spectrum.roll(d, 1)*spectrum.roll(-d, 1)*spectrum.conj().square()
                         for d in OFFSETS], dim=1)
    return value/(1+value.abs())


def normalized_spectrum(iq):
    if iq.ndim != 3 or iq.shape[1:] != (2, 256):
        raise ValueError('Matching [batch,2,256] IQ windows required')
    z = torch.complex(iq[:, 0], iq[:, 1])
    scale = z.abs().square().mean(1, keepdim=True).clamp_min(1e-12).sqrt()
    return torch.fft.fft(z/scale, dim=1, norm='ortho'), scale


def quartic_loss(output, clean):
    if output.shape != clean.shape or len(output) == 0:
        raise ValueError('Matching nonempty source IQ pairs required')
    a = closure(normalized_spectrum(output)[0]).flatten(1)
    b = closure(normalized_spectrum(clean)[0]).flatten(1).detach()
    return torch.cat(((a-b).real, (a-b).imag), dim=1).square().mean()


class DuraQuarticRecovery(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.base = DuraMultiscaleSpectral(width=width, taps=9)
        self.spectral_entry = nn.Conv1d(4+2*len(OFFSETS), width, 3, padding=1)
        self.spectral_hidden = nn.Conv1d(width, width, 3, padding=1)
        self.spectral_head = nn.Conv1d(width, 2, 1)
        nn.init.zeros_(self.spectral_head.weight)
        nn.init.zeros_(self.spectral_head.bias)

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        base = self.base(received, anchor, innovation, variance, validate=validate)
        spectrum, scale = normalized_spectrum(base)
        received_complex = torch.complex(received[:, 0], received[:, 1])
        received_spectrum = torch.fft.fft(received_complex/scale, dim=1, norm='ortho')
        product = closure(spectrum)
        features = torch.cat((spectrum.real[:, None], spectrum.imag[:, None],
                              received_spectrum.real[:, None], received_spectrum.imag[:, None],
                              product.real, product.imag), dim=1)
        hidden = F.silu(self.spectral_entry(features))
        output = self.spectral_head(F.silu(self.spectral_hidden(hidden))).tanh()
        delta = torch.fft.ifft(torch.complex(output[:, 0], output[:, 1]), dim=1, norm='ortho')
        correction = torch.stack((delta.real, delta.imag), 1)*scale[:, :, None]
        return base + BRANCH_SCALE*noise_gate(anchor, variance)*correction
