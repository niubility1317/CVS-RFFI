"""Test WiSig timing from short-lag phase differences on TX-hidden source IQ."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.pamf_unrolled_restore.wisig_joint_preamble import OBSERVED, STARTS, WEIGHT, observable, waveform
from experiments.pamf_unrolled_restore.wisig_smooth_preamble import SMOOTH, choose, estimate
from experiments.pamf_unrolled_restore.wisig_timing import PERIOD
from experiments.pamf_unrolled_restore.wisig_uncertain_source import source_iq


LAGS = (1, 2, 4)
OFFSETS = (-28, -12, 12, 28)
ARMS = ('smooth', 'differential', 'joint')
COUNT = 512
WEIGHTS = WEIGHT.astype(np.float32)**2


def product(later, earlier):
    power = np.abs(later)**2 + np.abs(earlier)**2
    return 2. * later * earlier.conj() / np.maximum(power, 1e-10)


def bank():
    relative = OBSERVED[None] - STARTS[:, None]
    base = waveform(relative)
    return tuple(np.asarray(product(base, waveform(relative-lag)).T, dtype=np.complex64)
                 for lag in LAGS)


def differential(y, template):
    # Estimate reliability on the same received sample region for all hypotheses.
    level = np.maximum(np.sqrt(np.mean(np.abs(y)**2, -1, keepdims=True)), 1e-8)
    y = y/level
    later = y[:, OBSERVED]
    scores = []
    for lag, expected in zip(LAGS, template, strict=True):
        earlier = y[:, OBSERVED-lag]
        observed = product(later, earlier)
        reliability = np.minimum(np.abs(later)*np.abs(earlier), 2.) * WEIGHTS
        # Weighted 1 - half squared complex mismatch. The observed-only term
        # does not change the ranking, but fixes the scale for the joint score.
        candidate = (observed * reliability) @ expected.conj()
        mismatch = (np.abs(observed)**2 * reliability).sum(-1, keepdims=True)
        mismatch = mismatch + reliability @ (np.abs(expected)**2)
        norm = np.maximum(reliability.sum(-1, keepdims=True), 1e-8)
        scores.append(1. + (candidate.real - .5*mismatch)/norm)
    return np.mean(scores, axis=0)


def readouts(iq, templates):
    _, _, sync = observable(iq)
    smooth = estimate(sync, SMOOTH)[0]
    phase_diff = differential(sync, templates)
    return {key: choose(value) for key, value in zip(ARMS, (smooth, phase_diff, .5*(smooth+phase_diff)), strict=True)}


def cases(iq):
    raw = iq[:, 0] + 1j*iq[:, 1]
    t = np.arange(raw.shape[-1])
    rng = np.random.default_rng(2026112598)
    power = np.sqrt(np.mean(np.abs(raw)**2, -1, keepdims=True))
    standard = ((rng.standard_normal(raw.shape)+1j*rng.standard_normal(raw.shape))/np.sqrt(2.)).astype(np.complex64)
    phase = 1.05*np.sin(7.*t/255.) + .018*t
    mild = raw*np.exp(1j*phase) + .14*power*standard
    direction = rng.uniform(-1., 1., size=(len(raw), 1))
    phase = 1.85*np.sin(11.*t/255.+2.4*direction) + .020*direction*t
    envelope = np.exp(.40*np.cos(5.*t/255.+2.*direction))
    strong = raw*envelope*np.exp(1j*phase) + .30*power*standard
    yield 'clean', iq, 0
    for name, wave in (('shift', raw), ('mild', mild), ('strong', strong)):
        for amount in OFFSETS:
            p = t+amount
            indices = np.where(p < 0, -p, np.minimum(p, 2*(len(t)-1)-p))
            shifted = wave[:, indices]
            yield f'{name}_{amount:+d}', np.stack((shifted.real, shifted.imag), 1).astype(np.float32), -amount


def probe(source_root):
    begun = perf_counter()
    source, available, calibration = source_iq(source_root)
    held = available[2*COUNT:3*COUNT]
    if len(held) != COUNT or np.intersect1d(held, calibration).size:
        raise ValueError('Need distinct legally sourced calibration and holdout IQ')
    templates = bank()
    peaks = dict.fromkeys(ARMS, -np.inf)
    for chunk in np.array_split(calibration, 32):
        for key, (_, score) in readouts(source[chunk], templates).items():
            peaks[key] = max(peaks[key], float(score.max()))
    thresholds = {key: peak+.02*abs(peak)+1e-6 for key, peak in peaks.items()}
    outcomes = {}
    for case, iq, shift in cases(source[held]):
        outcomes[case] = {}
        for key, (delta, support) in readouts(iq, templates).items():
            if not np.isfinite(support).all():
                raise FloatingPointError('Nonfinite WiSig timing evidence')
            active = support > thresholds[key]
            guarded = np.where(active, delta, 0)
            errors = guarded - shift
            aliases = np.abs((errors + PERIOD//2)%PERIOD - PERIOD//2)
            correct = np.abs(errors) <= 2
            outcomes[case][key] = dict(active=int(active.sum()), correct=int(correct.sum()),
                unguarded=int((np.abs(delta-shift)<=2).sum()), alias_only=int(((aliases<=2)&~correct).sum()),
                median_absolute_error=float(np.median(np.abs(errors))))
    return dict(status='VERIFIED', source_role='U_s', held=len(held), calibration=len(calibration),
                held_offset=2*COUNT, thresholds=thresholds, results=outcomes, seconds=perf_counter()-begun,
                F_reads=0, TX_reads=0, V_reads=0, target_reads=0, checkpoint_reads=0, trace_reads=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    print(json.dumps(probe(parser.parse_args().source_root), allow_nan=False))
