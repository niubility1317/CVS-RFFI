"""Control the complex response of a denoiser to the actual received signal."""

import torch


def retention_loss(restored, signal):
    """Two independent noise exposures share one observed-domain signal."""
    desired = torch.fft.fft(signal, norm='ortho')
    error = torch.fft.fft(restored-signal[:, None], norm='ortho')
    bins = torch.fft.fftfreq(signal.shape[-1], device=signal.device).abs()
    edges = ((bins > 0) & (bins <= 1/32), (bins > 1/32) & (bins <= 1/8))
    output = []
    for band in edges:
        reference = desired[..., band]
        energy = reference.abs().square().sum(-1)
        reliable = energy / (energy + .008)
        gain_error = ((error[..., band]*reference[:, None].conj()).sum(-1) /
                      energy[:, None].clamp_min(.008))
        bias = gain_error.mean(1).abs().square()
        low = (-.10-gain_error.real).clamp_min(0.).square().mean(1)
        output.append(((bias + .5*low)*reliable).mean())
    return .6*output[0] + .4*output[1]
