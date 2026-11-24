"""Shen et al. receiver-adversarial CNN with a WiSig preamble front end."""
import torch
from torch import nn
from torch.nn import functional as F


def spectrogram(iq):
    """Map a WiSig 256-point, 25 MS/s preamble to a 26 x 14 CIS."""
    if iq.ndim != 3 or iq.shape[1:] != (2, 256):
        raise ValueError('Expected a batch of two-channel WiSig preambles with 256 points')
    signal = torch.complex(iq[:, 0], iq[:, 1])
    signal = signal / signal.abs().square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    # Boxcar STFT with a 32-point window and 16-point hop, no padding.
    frames = signal.unfold(-1, 32, 16)
    spectrum = torch.fft.fftshift(torch.fft.fft(frames, dim=-1), dim=-1)
    # Keep the nominal 20 MHz Wi-Fi band from the 25 MS/s representation.
    power = spectrum[..., 3:29].abs().square().clamp_min(1e-12)
    # Log-domain form of log10(abs(STFT[t+1] / STFT[t])**2).
    log_power = power.log10()
    return log_power.diff(dim=1).transpose(1, 2).unsqueeze(1).contiguous()


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features):
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


class ResidualBlock(nn.Module):
    def __init__(self, channels, filters):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(channels, filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1),
        )
        self.skip = nn.Identity() if channels == filters else nn.Conv2d(channels, filters, 1)

    def forward(self, x):
        return F.relu(self.path(x) + self.skip(x))


class ReceiverAdversarialCNN(nn.Module):
    def __init__(self, transmitters, receivers):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.ZeroPad2d((2, 3, 2, 3)),
            nn.Conv2d(1, 32, 7, stride=2),
            nn.ReLU(),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 6 * 3, 512),
        )
        self.tx = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, transmitters))
        self.rx = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, receivers))
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def features(self, iq):
        return self.encode(spectrogram(iq))

    def encode(self, images):
        return F.normalize(self.backbone(images), dim=1)

    def classify_features(self, features):
        return self.tx(features)

    def receiver_logits(self, features):
        return self.rx(ReverseGradient.apply(features))
