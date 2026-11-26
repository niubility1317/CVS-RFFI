"""Penalize repeatable private-spectrum errors of the deployed W24 readout."""

import torch


BIAS_WEIGHT = .10


def private_spectrum(iq):
    spectrum = torch.fft.fft(iq, norm='ortho')
    detail = torch.cat((spectrum[..., 1:65], spectrum[..., -64:]), dim=-1)
    return detail - detail.mean(dim=-2, keepdim=True)


def coherent_error(prediction, center, clean_frame, *, with_identity):
    """Estimate mean squared conditional bias from two independent corruptions."""
    paired = prediction[:, :2]
    if with_identity:
        paired, center, clean_frame = paired[:-2], center[:-2], clean_frame[:-2]
    error = private_spectrum((paired - center[:, None]) / clean_frame[:, None])
    source = private_spectrum(center / clean_frame)
    scale = source.abs().square() + .01
    # Conditional independence makes the signed cross term unbiased. Taking an
    # absolute value or clamping it would also penalize independent noise.
    return ((error[:, 0] * error[:, 1].conj()).real / scale).mean()
