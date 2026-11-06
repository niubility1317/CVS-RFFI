"""Unlabeled source-IQ cross-view auxiliary representation, not a classifier."""
import torch
from torch import nn
from torch.nn import functional as F

SSL_FEATURE_WEIGHT = .05
SSL_OFFDIAGONAL_WEIGHT = .01
SSL_FEATURE_DIM = 16


class SourceIQEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
        self.projection = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, SSL_FEATURE_DIM))

    def forward(self, iq):
        if iq.ndim != 3 or iq.shape[1:] != (2, 256):
            raise ValueError('Source IQ [batch, 2, 256] required')
        scale = iq.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        hidden = self.layers(iq/scale)
        mean = hidden.mean(-1)
        std = hidden.var(-1, unbiased=False).clamp_min(1e-12).sqrt()
        return F.normalize(self.projection(torch.cat((mean, std), 1)), dim=1, eps=1e-12)


def cross_view_ssl_loss(first, second):
    if first.ndim != 2 or first.shape != second.shape or first.shape[1] != SSL_FEATURE_DIM or len(first) < 2:
        raise ValueError('Matching source feature batches of at least two records required')
    def standardize(value):
        center = value-value.mean(0, keepdim=True)
        return center/(center.square().mean(0, keepdim=True)+1e-4).sqrt()
    correlation = standardize(first).T @ standardize(second)/len(first)
    diagonal = correlation.diagonal()
    offdiag = correlation-torch.diag_embed(diagonal)
    return (diagonal-1).square().mean() + SSL_OFFDIAGONAL_WEIGHT*offdiag.square().sum()/SSL_FEATURE_DIM


def frozen_feature_loss(encoder, restored, clean):
    if restored.shape != clean.shape or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('Matching source IQ and frozen auxiliary encoder required')
    with torch.no_grad():
        target = encoder(clean)
    return (encoder(restored)-target).square().sum(1).mean()
