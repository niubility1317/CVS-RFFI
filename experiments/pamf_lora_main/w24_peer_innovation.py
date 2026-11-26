"""Condition W24 priors on received innovation and local reference dispersion."""

import importlib

import torch
from torch.nn.functional import avg_pool1d


def peer_moments(y):
    """Estimate corresponding-sample moments from the other chirps of this packet."""
    count = y.shape[1] - 1
    mean = (y.sum(dim=1, keepdim=True) - y) / count
    power = y.abs().square()
    second = (power.sum(dim=1, keepdim=True) - power) / count
    return mean, (second - mean.abs().square()).clamp_min(0.)


def temporal_average(value):
    return avg_pool1d(value, 9, stride=1, padding=4, count_include_pad=False)


def innovation_fields(y, *, use_peers):
    if use_peers:
        reference, spread = peer_moments(y)
        spread = temporal_average(spread)
    else:
        reference = torch.complex(temporal_average(y.real), temporal_average(y.imag))
        spread = (temporal_average(y.abs().square()) - reference.abs().square()).clamp_min(0.)
    error = (y - reference) / (.05 + spread).sqrt()
    return error.real.clamp(-6., 6.), error.imag.clamp(-6., 6.), (spread / .05).log1p().clamp_max(6.)


def restorer(training, *, use_peers):
    archived = importlib.import_module(training.DuraW24.__module__)

    class InnovationW24(training.DuraW24):
        def __init__(self):
            archived.DuraCapacity.__init__(self, 'expanded', samples=archived.WIDE, condition_fields=9)

        def stages(self, y, quality, rms, *, condition=(), **options):
            fields = innovation_fields(y, use_peers=use_peers)
            return super().stages(y, quality, rms, condition=(*condition, *fields), **options)

    return InnovationW24()
