"""Partition the learned broadband LoRa correction without rotating other bins."""

import torch

from .broadband_vector import CAP, partitions
from .complex_budget import chirp_rms


SWAPS = ('1_8', '9_32', '33_64', '65_128', 'near32', 'far32', 'all')
LEARNED = ('narrow', 'broad', 'broad_half', 'broad_near32', 'broad_far32', 'broad_far64',
           'split32', 'split64')


def masks(spectrum):
    a, b, c, d = partitions(spectrum.shape[-1], spectrum.device, True)
    return dict(zip(SWAPS, (a, b, c, d, a | b, c | d, a | b | c | d), strict=True))


def blend_near(narrow, outer):
    # Keep the outer learned update fixed and use the remaining chirp dose for near.
    remaining = (CAP**2-chirp_rms(outer).square()).clamp_min(0.).sqrt()
    return outer + narrow * (remaining/chirp_rms(narrow).clamp_min(1e-8)).clamp_max(1.)


def deploy_vectors(narrow, broad):
    region = masks(broad)
    near = broad*region['near32']
    far = broad*region['far32']
    far64 = broad*region['65_128']
    return dict(zip(LEARNED, (narrow, broad, .5*broad, near, far, far64,
                             blend_near(narrow, far), blend_near(narrow, far64)), strict=True))


def source_swaps(learned, oracle):
    altered = {}
    gap = 0.
    for key, selected in masks(learned).items():
        b, a = learned*selected, oracle*selected
        size, known_size = chirp_rms(b), chirp_rms(a)
        paired = torch.where(known_size > 1e-8, a*(size/known_size.clamp_min(1e-8)), b)
        kept = learned*~selected
        altered[f'known_{key}'] = kept + a
        altered[f'matched_{key}'] = kept + paired
        gap = max(gap, float((chirp_rms(paired)-size).abs().max()))
    return altered, gap
