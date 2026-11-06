"""Source IQ cross-half representation; no device labels or inference encoder."""
import torch
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.joint_ssl_iq import SourceIQEncoder, cross_view_ssl_loss


class PatchIQEncoder(SourceIQEncoder):
    def forward(self, iq):
        if iq.ndim != 3 or iq.shape[1:] != (2, 128):
            raise ValueError('Source IQ half-window [batch, 2, 128] required')
        scale = iq.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        hidden = self.layers(iq/scale)
        mean = hidden.mean(-1)
        std = hidden.var(-1, unbiased=False).clamp_min(1e-12).sqrt()
        return F.normalize(self.projection(torch.cat((mean, std), 1)), dim=1, eps=1e-12)


def patch_views(iq):
    if iq.ndim != 3 or iq.shape[1:] != (2, 256):
        raise ValueError('Source IQ [batch, 2, 256] required')
    return torch.cat((iq[..., :128], iq[..., 128:]), dim=0)


def patch_cross_view_loss(encoder, received, clean):
    if received.shape != clean.shape:
        raise ValueError('Matching paired source IQ required')
    received_first, received_second = encoder(patch_views(received)).chunk(2)
    clean_first, clean_second = encoder(patch_views(clean)).chunk(2)
    return .5*(cross_view_ssl_loss(received_first, clean_second) +
               cross_view_ssl_loss(received_second, clean_first))


def frozen_patch_feature_loss(encoder, restored, clean):
    if restored.shape != clean.shape or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('Matching source IQ and frozen auxiliary encoder required')
    with torch.no_grad():
        target = encoder(patch_views(clean))
    return (encoder(patch_views(restored))-target).square().sum(1).mean()
