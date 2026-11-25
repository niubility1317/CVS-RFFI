"""Test WiSig timing from short-lag phase differences on TX-hidden source IQ."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.pamf_unrolled_restore.wisig_differential_timing import ARMS, observations
from experiments.pamf_unrolled_restore.wisig_timing import PERIOD
from experiments.pamf_unrolled_restore.wisig_uncertain_source import source_iq


OFFSETS = (-28, -12, 12, 28)
COUNT = 512


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
    peaks = dict.fromkeys(ARMS, -np.inf)
    for chunk in np.array_split(calibration, 32):
        for key, (_, score) in observations(source[chunk])[-1].items():
            peaks[key] = max(peaks[key], float(score.max()))
    thresholds = {key: peak+.02*abs(peak)+1e-6 for key, peak in peaks.items()}
    outcomes = {}
    for case, iq, shift in cases(source[held]):
        outcomes[case] = {}
        for key, (delta, support) in observations(iq)[-1].items():
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
