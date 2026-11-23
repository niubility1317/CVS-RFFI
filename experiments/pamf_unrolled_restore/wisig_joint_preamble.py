"""Align WiSig by fitting one observable response to both legacy preamble fields."""

from dataclasses import dataclass

import numpy as np

from experiments.cvs_rff_physics.known_excitation import template
from .wisig_short_training import _estimate, _remove_cfo
from .wisig_timing import ANCHOR, NOMINAL, PERIOD, STARTS, long_template, readouts as local_readouts


SIZE = 256
READOUTS = ('direct', 'local_transition', 'joint_scalar', 'joint_channel')
TAP_OFFSETS = {'joint_scalar': (0,), 'joint_channel': (0, -1, 1)}
IN_ANCHOR = np.abs(STARTS - NOMINAL) <= ANCHOR
OBSERVED = np.r_[np.arange(70, 150), np.arange(158, SIZE)]
WEIGHT = np.r_[np.full(80, 1 / np.sqrt(80)), np.full(SIZE - 158, 1 / np.sqrt(SIZE - 158))]
RIDGE = .15


@dataclass(frozen=True)
class PreambleFit:
    delta: np.ndarray
    margin: np.ndarray
    taps: np.ndarray


def waveform(relative):
    return np.where(relative < 0, template(relative), long_template(relative))


def build_bank(taps):
    relative = OBSERVED[None, :, None] - STARTS[:, None, None] - np.array(taps)
    design = (waveform(relative) * WEIGHT[None, :, None]).astype(np.complex64)
    adjoint = design.conj().swapaxes(1, 2)
    gram = adjoint @ design
    penalty = np.diag([0.] + [RIDGE] * (len(taps) - 1)).astype(np.float32)
    return adjoint, np.linalg.inv(gram + penalty)


BANKS = {name: build_bank(taps) for name, taps in TAP_OFFSETS.items()}


def fit(synced, name):
    """Fit a shared complex response on fixed receive coordinates for all shifts."""
    adjoint, inverse = BANKS[name]
    observed = synced[:, OBSERVED] * WEIGHT.astype(np.float32)
    rhs = np.einsum('dmn,bn->bdm', adjoint, observed, optimize=True)
    coeff = np.einsum('dmk,bdk->bdm', inverse, rhs)
    # At the ridge optimum this is the reduction in the penalized residual.
    explained = (coeff.conj() * rhs).sum(axis=2).real
    energy = np.maximum((np.abs(observed)**2).sum(axis=1), 1e-8)
    scores = explained / energy[:, None]
    anchored = np.where(IN_ANCHOR, scores, -np.inf).max(axis=1)
    candidate = np.where(~IN_ANCHOR, scores, -np.inf).argmax(axis=1)
    row = np.arange(len(synced))
    return PreambleFit(STARTS[candidate] - NOMINAL, scores[row, candidate] - anchored, coeff[row, candidate])


def restore(z, estimate, name, omega, threshold):
    """Copy the observed overlap and extrapolate only the unobserved edge."""
    delta = np.where(estimate.margin > threshold, estimate.delta, 0)
    coordinate = np.arange(SIZE)[None, :] + delta[:, None]
    valid = (coordinate >= 0) & (coordinate < SIZE)
    y = np.take_along_axis(z, np.clip(coordinate, 0, SIZE - 1), axis=1)
    row, col = np.nonzero(~valid)
    if len(row):
        phase = np.exp(1j * omega[row] * coordinate[row, col])
        basis = waveform(col[:, None] - NOMINAL - np.array(TAP_OFFSETS[name]))
        y[row, col] = phase * (basis * estimate.taps[row]).sum(axis=1)
    return y, valid


def observable(iq):
    z = iq[:, 0] + 1j * iq[:, 1]
    est, _ = _estimate(z)
    return z, est, _remove_cfo(z, est.coarse_rad_per_sample)


def readouts(iq, thresholds):
    local, est = local_readouts(iq, threshold=thresholds['local_transition'])
    z = iq[:, 0] + 1j * iq[:, 1]
    omega = est.coarse_rad_per_sample
    sync = _remove_cfo(z, omega)
    values = {'direct': iq, 'local_transition': local['timing_protocol']}
    for name in TAP_OFFSETS:
        restored, _ = restore(z, fit(sync, name), name, omega, thresholds[name])
        values[name] = np.stack((restored.real, restored.imag), axis=1).astype(iq.dtype, copy=False)
    return values, est
