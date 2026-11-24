"""Preserve complex spectral relations to other real source packets during fitting."""

import torch

from .natural_detail import detail_bands


def anchor_risk(prediction, reference):
    """Return off-diagonal complex similarity risk for normalized clean anchors."""
    predicted = prediction - prediction.mean(dim=0, keepdim=True)
    clean = reference - reference.mean(dim=0, keepdim=True)
    # Set both scales with the clean row, so shrinking a recovered row cannot
    # cancel its scale error. The floor is in normalized source-IQ power units.
    scale = (clean.abs().square().mean(-1, keepdim=True) + 2e-4).sqrt()
    anchor = clean / scale
    error = (predicted - clean[:, None]) / scale[:, None]
    relations = (error.transpose(0, 1) @ anchor.mH) / anchor.shape[-1]
    off_diagonal = ~torch.eye(len(clean), device=clean.device, dtype=torch.bool)
    masked = relations.abs().square().log1p() * off_diagonal
    return masked.sum() / (relations.shape[0] * off_diagonal.sum().clamp_min(1))


def packet_relation_risk(prediction, reference):
    """Measure common and private spectral geometry in both near-carrier bands."""
    recovered = torch.fft.fft(prediction, norm='ortho')
    clean = torch.fft.fft(reference, norm='ortho')
    recovered_mean = recovered.mean(-2, keepdim=True)
    clean_mean = clean.mean(-2, keepdim=True)
    components = ((recovered_mean, clean_mean), (recovered - recovered_mean, clean - clean_mean))
    return torch.stack([anchor_risk(est[..., b].flatten(-2), target[..., b].flatten(-2))
                        for est, target in components for b in detail_bands(clean)])
