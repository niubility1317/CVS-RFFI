"""Source paired-IQ auxiliary loss matching the spectral quotient diagnostic."""
import torch

REGULARIZATION = .01
WEIGHT = .1


def _adjacent_quotient(iq):
    spectrum = torch.fft.fft(torch.complex(iq[:, 0], iq[:, 1]), dim=-1, norm='ortho')
    power = spectrum.abs().square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
    left, right = spectrum[:, :-1], spectrum[:, 1:]
    return right * left.conj() / (left.abs().square() + REGULARIZATION * power)


def spectral_quotient_loss(output, source_clean):
    """Mean phase-invariant quotient cosine loss for matching source IQ pairs.

    The clean input is a paired source training target, never target-side truth.
    The base IQ objective remains necessary, including to avoid zero-output collapse.
    """
    if (output.ndim != 3 or output.shape[1:] != (2, 256)
            or source_clean.shape != output.shape or len(output) == 0
            or output.device != source_clean.device or output.dtype != source_clean.dtype
            or output.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Matching floating-point (N, 2, 256) source IQ required')
    observed = _adjacent_quotient(output)
    reference = _adjacent_quotient(source_clean)
    inner = (observed * reference.conj()).sum(dim=-1).abs().square()
    energy = observed.abs().square().sum(dim=-1) * reference.abs().square().sum(dim=-1)
    return (1 - inner / energy.clamp_min(1e-12)).clamp(0, 1).mean()
