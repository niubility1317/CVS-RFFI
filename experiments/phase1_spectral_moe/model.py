"""Source-trained frequency mixture of three frozen IQ restorers."""
import torch
from torch import nn
from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate

class SpectralMixture(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(9, width, 7, padding=3, padding_mode='circular'), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2, padding_mode='circular'), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2, padding_mode='circular'), nn.GELU(),
            nn.Conv1d(width, 3, 1))
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            self.network[-1].bias.copy_(torch.tensor([3., 0., 0.]))

    def forward(self, received, anchor, joint, diffusion, variance):
        if received.ndim != 3 or received.shape[1:] != (2, 256) or any(
                t.shape != received.shape for t in (anchor, joint, diffusion)) or variance.shape != (len(received),):
            raise ValueError('Matching single-window IQ and per-record variance required')
        signals = (received, anchor, joint, diffusion)
        spectra = [torch.fft.fft(torch.complex(x[:, 0], x[:, 1]), norm='ortho') for x in signals]
        scale = anchor.square().mean((1, 2)).clamp_min(1e-12).sqrt()
        features = torch.stack([part / scale[:, None] for spectrum in spectra
                                for part in (spectrum.real, spectrum.imag)], 1)
        noise = torch.log1p(variance / scale.square())[:, None, None].expand(-1, 1, 256)
        weights = self.network(torch.cat((features, noise), 1)).softmax(1)
        combined = sum(weights[:, i] * spectra[i+1] for i in range(3))
        signal = torch.fft.ifft(combined, norm='ortho')
        iq = torch.stack((signal.real, signal.imag), 1)
        return anchor + noise_gate(anchor, variance) * (iq-anchor)

class CombinedRecovery(nn.Module):
    def __init__(self, gate, joint, diffusion):
        super().__init__()
        self.gate, self.joint, self.diffusion = gate, joint, diffusion

    def forward(self, received, anchor, innovation, variance, action):
        return self.gate(received, anchor, self.joint(received, anchor, innovation, variance, action),
                         self.diffusion(received, anchor, innovation, variance), variance)
