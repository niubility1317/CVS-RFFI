"""Compare same-window and held-preamble scoring on fixed unlabelled source WiSig."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pamf_unrolled_restore.wisig_joint_preamble import TAP_OFFSETS, observable
from experiments.pamf_unrolled_restore.wisig_smooth_preamble import choose, estimate, ridge_bank
from experiments.pamf_unrolled_restore.wisig_timing import estimate_timing


COUNT = 512


def formats(iq, banks):
    local, margin, _ = estimate_timing(iq[:, 0] + 1j*iq[:, 1])
    _, _, synced = observable(iq)
    result = {'existing_transition': (local, margin)}
    for name, bank in banks.items():
        for score_name, score in zip(('same_samples', 'held_samples'), estimate(synced, bank)[:2], strict=True):
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


def probe(source_root, *, varying=False):
    from experiments.pamf_unrolled_restore.wisig_uncertain_source import source_iq

    x, available, calibration = source_iq(source_root)
    held = available[COUNT:2*COUNT] if varying else available[:COUNT]
    banks = {name: ridge_bank(taps) for name, taps in TAP_OFFSETS.items()}
    if varying:
        banks['piecewise_scalar'] = ridge_bank((0,), varying=True)
    margins = {}
    for part in np.array_split(calibration, 32):
        for name, (_, confidence) in formats(x[part], banks).items():
            margins[name] = max(margins.get(name, -np.inf), float(confidence.max()))
    # Add a positive margin in score units even when the maximum gain is negative.
    margins = {name: value+.02*abs(value)+1e-6 for name, value in margins.items()}
    return dict(calibration=len(calibration), probes=len(held), held_offset=COUNT if varying else 0, threshold=margins,
                target_reads=0, V_reads=0, F_reads=0, TX_reads=0, checkpoint_reads=0,
                metrics=test_packets(x[held], banks, margins))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    parser.add_argument('--varying', action='store_true')
    args = parser.parse_args()
    print(json.dumps(probe(args.source_root, varying=args.varying), allow_nan=False))
