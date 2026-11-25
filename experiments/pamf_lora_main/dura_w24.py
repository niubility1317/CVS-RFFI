"""Lift a frozen DURA correction onto an eight-chirp, two-rate LoRa aperture."""

import numpy as np
from scipy.signal import resample_poly
import torch

from .data import WINDOW_START
from .dura_repair import complex_iq, real_iq
from .packet_data import PACKET_LEN


NARROW = 256
WIDE = 384
EDGE = 24
CHIRPS = 8
STARTS = np.arange(CHIRPS) * 1024
CROP = STARTS[:, None] + WINDOW_START - 125 + np.arange(1250)
CROP = np.where(CROP < 0, -CROP, CROP)
CROP = np.where(CROP >= PACKET_LEN, 2 * PACKET_LEN - 2 - CROP, CROP)
EDGES = STARTS[:, None] + np.r_[0:EDGE, 1024-EDGE:1024]


def apertures(packet):
    """Use W24's capture clock and its packet RMS for both representations."""
    crops = packet[..., CROP]
    narrow = resample_poly(crops, 32, 125, axis=-1)[..., 32:288]
    core = resample_poly(crops, 48, 125, axis=-1)[..., 48:432]
    rms = np.sqrt(np.mean(np.abs(core)**2, axis=(-2, -1), keepdims=True))
    if not np.isfinite(rms).all() or (rms <= 1e-12).any():
        raise ValueError('LoRa W24 requires finite, nonempty received IQ')
    wide = np.concatenate((core / rms, packet[..., EDGES] / rms), axis=-1)
    narrow = narrow / rms
    return tuple(np.stack((x.real, x.imag), axis=-2).astype(np.float32, copy=False)
                 for x in (narrow, wide))


def lift(delta):
    """Evaluate the band-limited complex correction on the W24 center grid."""
    spec = torch.fft.fft(complex_iq(delta), norm='forward')
    result = spec.new_zeros((*spec.shape[:-1], WIDE))
    middle = NARROW // 2
    result[..., :middle] = spec[..., :middle]
    result[..., -middle+1:] = spec[..., middle+1:]
    result[..., (middle, -middle)] = .5 * spec[..., middle:middle+1]
    return real_iq(torch.fft.ifft(result, norm='forward'))


def readouts(wide, narrow, restore, active):
    """Keep the observed 1 MHz edge samples and add only the center correction."""
    if not active.any():
        return wide, wide
    selected = narrow[active]
    middle = selected[:, 1:7]
    left, right = selected[:, :6], selected[:, 2:]
    stacked = torch.cat((middle, left, right), dim=0)
    edits = lift(restore(stacked) - stacked)
    center_edit, left_edit, right_edit = edits.chunk(3, dim=0)
    reference, candidate = wide.clone(), wide.clone()
    reference[active, 1:7, :, :WIDE] += center_edit
    wide_edit = torch.cat((left_edit[:, :2], .5 * (left_edit[:, 2:] + right_edit[:, :4]), right_edit[:, 4:]), dim=1)
    candidate[active, :, :, :WIDE] += wide_edit
    return reference, candidate
