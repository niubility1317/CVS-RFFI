"""Preserve source clean-IQ neighborhood distributions without identities or F."""
import torch
from torch.nn import functional as F

GEOMETRY_WEIGHT = .2
GEOMETRY_TEMPERATURE = .5


def iq_geometry_loss(restored, clean):
    if restored.shape != clean.shape or restored.ndim != 3 or restored.shape[1:] != (2, 256) or len(restored) < 2:
        raise ValueError('At least two matching source IQ pairs are required')
    clean_vectors = F.normalize(clean.flatten(1), dim=1, eps=1e-12).detach()
    restored_vectors = F.normalize(restored.flatten(1), dim=1, eps=1e-12)
    clean_logprob = F.log_softmax(clean_vectors @ clean_vectors.T / GEOMETRY_TEMPERATURE, dim=1)
    restored_logprob = F.log_softmax(restored_vectors @ clean_vectors.T / GEOMETRY_TEMPERATURE, dim=1)
    return (clean_logprob.exp()*(clean_logprob-restored_logprob)).sum(1).mean()
