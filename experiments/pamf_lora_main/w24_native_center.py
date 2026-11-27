"""Source-loss projection for the fixed native LoRa center sampling geometry."""

from functools import lru_cache

import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn

from .data import WINDOW_START


NATIVE_CHIRP, N_CHIRPS, CENTER = 1024, 8, 384
NATIVE_LENGTH = NATIVE_CHIRP * N_CHIRPS
CONTEXT, CROP = 125, 1250
UP = CENTER // 8


@lru_cache(maxsize=1)
def geometry():
    start = np.arange(N_CHIRPS, dtype=np.int64) * NATIVE_CHIRP + WINDOW_START - CONTEXT
    crop = np.abs(start[:, None] + np.arange(CROP, dtype=np.int64))
    crop = np.where(crop >= NATIVE_LENGTH, 2 * NATIVE_LENGTH - 2 - crop, crop)
    coefficients = resample_poly(np.eye(CROP, dtype=np.float32), UP, CONTEXT, axis=-1)
    return crop, np.ascontiguousarray(coefficients[:, UP:UP+CENTER])


class NativeCenterProjection(nn.Module):
    """Differentiable linear crop/resample, before packet normalization."""
    def __init__(self):
        super().__init__()
        crop, weights = geometry()
        self.register_buffer('crop', torch.frombuffer(crop.copy(), dtype=torch.int64).reshape(crop.shape))
        self.register_buffer('filter', torch.frombuffer(weights.copy(), dtype=torch.float32).reshape(weights.shape))

    def forward(self, raw):
        cropped = torch.view_as_real(raw[:, self.crop]).transpose(-1, -2)
        sampled = (cropped @ self.filter).transpose(-1, -2).contiguous()
        return torch.view_as_complex(sampled)


def direction_terms(project, error, oracle_edit):
    center_error, oracle = project(torch.cat((error, oracle_edit))).chunk(2)
    edit = oracle + center_error
    want, got = (z.abs().square().mean((-2, -1)) for z in (oracle, edit))
    correlation = (edit * oracle.conj()).real.mean((-2, -1))
    cosine = (correlation / ((want + 1e-8) * (got + 1e-8)).sqrt()).clamp(-1., 1.)
    reliability = want / (want + .002)
    angle = (reliability * (1. - cosine)).mean()
    with torch.no_grad():
        normalized_error = (center_error.abs().square().mean((-2, -1)) / (want + .002)).mean()
        weighted_cosine = (reliability * cosine).sum() / reliability.sum().clamp_min(1e-9)
        relative_strength = (reliability * (got / want.clamp_min(1e-8)).sqrt()).sum()
        relative_strength = relative_strength / reliability.sum().clamp_min(1e-9)
        diagnostics = torch.stack((normalized_error, weighted_cosine, relative_strength))
    return angle, diagnostics
