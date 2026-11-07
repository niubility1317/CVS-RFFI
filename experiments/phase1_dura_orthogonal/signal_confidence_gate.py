"""Source-IQ-trained per-record confidence for a frozen DURA correction."""
import torch
from torch import nn


class SignalConfidenceGate(nn.Module):
    """Choose a bounded correction using only received signal and DURA context."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(9, 96, 7, stride=2, padding=3), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv1d(96, 160, 7, stride=2, padding=3), nn.GroupNorm(8, 160), nn.GELU(),
            nn.Conv1d(160, 192, 7, stride=2, padding=3), nn.GroupNorm(8, 192), nn.GELU())
        self.head = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, received, anchor, innovation, proposal, variance):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape
                or innovation.shape != received.shape or proposal.shape != received.shape
                or variance.shape != (len(received),)):
            raise ValueError('Matching 256-sample IQ and per-record variance required')
        power = anchor.square().mean((1, 2), keepdim=True).clamp_min(1e-12)
        scale = power.sqrt()
        delta = proposal - anchor
        noise = torch.log1p(variance[:, None, None].clamp_min(0) / power).expand(-1, 1, 256)
        features = torch.cat((received/scale, anchor/scale, innovation/scale, delta/scale, noise), dim=1)
        alpha = torch.sigmoid(self.head(self.encoder(features).mean(-1))).unsqueeze(-1)
        return anchor + alpha*delta


def source_gate_loss(output, anchor, clean):
    if output.shape != anchor.shape or output.shape != clean.shape or output.ndim != 3:
        raise ValueError('Paired source IQ required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    error = (output-clean).square().mean((1, 2))/energy
    change = (output-anchor).square().mean((1, 2))/energy
    return (error + .02*change).mean()
