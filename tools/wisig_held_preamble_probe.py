"""Compare same-window and held-preamble scoring on fixed unlabelled source WiSig."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pamf_unrolled_restore.wisig_joint_preamble import (
    IN_ANCHOR, OBSERVED, RIDGE, STARTS, TAP_OFFSETS, WEIGHT, observable, waveform,
)
from experiments.pamf_unrolled_restore.wisig_timing import NOMINAL, estimate_timing


COUNT = 512


def design(taps):
    basis = waveform(OBSERVED[None, :, None] - STARTS[:, None, None] - np.asarray(taps))
    return (basis * WEIGHT[None, :, None]).astype(np.complex64)


def ridge_bank(taps):
    basis = design(taps)
    groups = (np.arange(len(OBSERVED)), *[np.flatnonzero((OBSERVED // 8) % 2 == i) for i in (0, 1)])
    penalty = np.diag([0.] + [RIDGE] * (len(taps) - 1)).astype(np.float32)
    banks = []
    for columns in groups:
        sample_basis = basis[:, columns]
        adjoint = sample_basis.conj().transpose(0, 2, 1)
        gram = adjoint @ sample_basis
        projection = adjoint.reshape(-1, len(columns)).T.copy()
        banks.append((columns, projection, gram, np.linalg.inv(gram + penalty)))
    return tuple(banks)


def estimate(synced, bank):
    observed = synced[:, OBSERVED] * WEIGHT.astype(np.float32)
    fits = []
    for columns, projection, gram, inverse in bank:
        part = observed[:, columns]
        rhs = (part @ projection).reshape(len(observed), len(STARTS), gram.shape[-1])
        h = np.einsum('dkj,bdj->bdk', inverse, rhs)
        energy = (part.real**2 + part.imag**2).sum(-1, keepdims=True)
        fits.append((rhs, h, gram, energy))
    rhs, h, _, energy = fits[0]
    apparent = np.sum((h.conj() * rhs).real, axis=-1) / np.maximum(energy, 1e-8)
    explained = []
    for (_, trained, _, _), (rhs, _, gram, _) in ((fits[1], fits[2]), (fits[2], fits[1])):
        projected = np.einsum('dkj,bdj->bdk', gram, trained)
        explained.append(np.sum((trained.conj() * (2*rhs - projected)).real, axis=-1))
    predictive = sum(explained) / np.maximum(fits[1][3] + fits[2][3], 1e-8)
    return apparent, predictive


def choose(scores):
    outer = np.where(~IN_ANCHOR, scores, -np.inf)
    closest = np.where(IN_ANCHOR, scores, -np.inf).max(1)
    return STARTS[outer.argmax(1)] - NOMINAL, outer.max(1) - closest


def formats(iq, banks):
    local, margin, _ = estimate_timing(iq[:, 0] + 1j*iq[:, 1])
    _, _, synced = observable(iq)
    result = {'existing_transition': (local, margin)}
    for name, bank in banks.items():
        for score_name, score in zip(('same_samples', 'held_samples'), estimate(synced, bank), strict=True):
            result[f'{name}_{score_name}'] = choose(score)
    return result


def test_packets(clean, banks, margins):
    rng = np.random.default_rng(2026112471)
    case = {'unmodified': (clean, 0)}
    time = np.arange(clean.shape[-1])
    z = clean[:, 0] + 1j*clean[:, 1]
    rms = np.sqrt(np.mean(np.abs(z)**2, axis=-1, keepdims=True))
    ramp = np.exp(1j*(1.05*np.sin(7*time/255) + .018*time)).astype(np.complex64)
    noise = ((rng.normal(size=z.shape) + 1j*rng.normal(size=z.shape))*rms*0.14).astype(np.complex64)
    noisy = z*ramp + noise
    for input_name, wave in (('shift', z), ('shift_phase_noise', noisy)):
        for amount in (-28, -12, 12, 28):
            coordinates = time + amount
            padded = np.where(coordinates < 0, -coordinates, np.minimum(coordinates, 510-coordinates))
            shifted = wave[:, padded]
            case[f'{input_name}_{amount:+d}'] = (np.stack((shifted.real, shifted.imag), axis=1), -amount)
    out = {}
    for name, (iq, expected) in case.items():
        out[name] = {}
        values = formats(iq, banks)
        for readout, (delta, confidence) in values.items():
            active = confidence > margins[readout]
            guarded = np.where(active, delta, 0)
            close = np.abs(delta - expected) <= 2
            out[name][readout] = dict(active=int(active.sum()), correct_when_active=int((active & close).sum()),
                                      near_two_unconditional=int(close.sum()),
                                      near_two_with_guard=int((np.abs(guarded-expected)<=2).sum()),
                                      median_absolute_error=float(np.median(np.abs(guarded-expected))))
    return out


def probe(source_root):
    from experiments.pamf_unrolled_restore.wisig_uncertain_source import source_iq

    x, available, calibration = source_iq(source_root)
    held = available[:COUNT]
    banks = {name: ridge_bank(taps) for name, taps in TAP_OFFSETS.items()}
    margins = {}
    for part in np.array_split(calibration, 32):
        for name, (_, confidence) in formats(x[part], banks).items():
            margins[name] = max(margins.get(name, -np.inf), float(confidence.max()))
    # Add a positive margin in score units even when the maximum gain is negative.
    margins = {name: value+.02*abs(value)+1e-6 for name, value in margins.items()}
    return dict(calibration=len(calibration), probes=len(held), threshold=margins,
                target_reads=0, V_reads=0, F_reads=0, TX_reads=0, checkpoint_reads=0,
                metrics=test_packets(x[held], banks, margins))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    args = parser.parse_args()
    print(json.dumps(probe(args.source_root), allow_nan=False))
