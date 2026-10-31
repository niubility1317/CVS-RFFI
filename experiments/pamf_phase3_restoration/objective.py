"""Paired source objectives; labels and deployment state are not inputs."""
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_WEIGHT = .1
INPUT_GRADIENT_CAP = .5


class PairedObjective(nn.Module):
    def __init__(self):
        super().__init__()
        for n in (32, 64, 128):
            self.register_buffer('window' + str(n), torch.hann_window(n), persistent=False)

    def forward(self, restored, clean):
        if restored.shape != clean.shape or restored.ndim != 3 or restored.shape[1:] != (2, 256):
            raise ValueError('Matched source IQ pairs [N,2,256] required')
        target = clean.detach()
        energy = target.square().mean((1, 2)).clamp_min(1e-4)
        iq = (restored-target).square().mean((1, 2))/energy
        z, reference = torch.complex(restored[:, 0], restored[:, 1]), torch.complex(target[:, 0], target[:, 1])
        terms = []
        for n in (32, 64, 128):
            kwargs = dict(n_fft=n, hop_length=n//4, window=getattr(self, 'window'+str(n)),
                          center=False, onesided=False, return_complex=True)
            actual, expected = torch.stft(z, **kwargs).abs(), torch.stft(reference, **kwargs).abs()
            terms.append((actual-expected).abs().mean((1, 2))/expected.mean((1, 2)).clamp_min(1e-4))
        spectral = torch.stack(terms).mean(0)
        return iq + .05*spectral, dict(iq_nmse=iq, spectral_error=spectral)


def feature_guide(reader, restored, clean, target_features):
    """Frozen full F0 feature cosine, with detached per-record IQ-gradient cap."""
    if target_features.shape != (len(restored), 160):
        raise ValueError('Aligned source F0 clean features [N,160] required')
    features = reader.features(restored)
    if features.shape != target_features.shape:
        raise ValueError('Actual full original F0 fused160 output required')
    error = 1-F.cosine_similarity(features, target_features.detach(), dim=1, eps=1e-8)
    raw = FEATURE_WEIGHT*error
    gradient = torch.autograd.grad(raw.sum(), restored, retain_graph=True)[0]
    actual = gradient.detach().flatten(1).norm(dim=1)
    energy = clean.detach().square().mean((1, 2)).clamp_min(1e-4)
    physical = 2*(restored.detach()-clean.detach()).flatten(1).norm(dim=1)/(512*energy)
    weight = (INPUT_GRADIENT_CAP*physical/actual.clamp_min(1e-12)).clamp(max=1).detach()
    return weight*raw, dict(feature_error=error, weight=weight,
                           input_gradient_norm=actual, IQ_gradient_norm=physical)
