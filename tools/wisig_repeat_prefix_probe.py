"""Measure received-repeat prefix extrapolation on independent source U IQ."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_unrolled_restore import wisig_short_training_source as registered
from experiments.pamf_unrolled_restore.wisig_differential_timing import observations
from experiments.pamf_unrolled_restore.wisig_differential_timing_source import RUN
from experiments.pamf_unrolled_restore.wisig_repeat_prefix import MODES, readouts
from experiments.pamf_unrolled_restore.wisig_smooth_preamble import TIME
from experiments.pamf_unrolled_restore.wisig_uncertain_source import source_iq


COUNT = 512
SHIFTS = (12, 28, 36)
FIELDS = ('complex_nmse', 'amplitude_nmse', 'phase_cosine')


def cases(clean):
    rng = np.random.default_rng(2026112599)
    std = np.sqrt(np.mean(np.abs(clean)**2, -1, keepdims=True))
    noise = ((rng.standard_normal(clean.shape)+1j*rng.standard_normal(clean.shape))/np.sqrt(2.)).astype(np.complex64)
    seed = rng.uniform(-1., 1., (len(clean), 1))
    phase = 1.4*np.sin(9.*TIME/255. + 1.5*seed)
    gain = np.exp(.38*np.cos(5.*TIME/255. + seed))
    return dict(none=clean, white=clean+.16*std*noise,
                warped=(clean*gain*np.exp(1j*phase)+.30*std*noise).astype(np.complex64))


def fidelity(wave, reference):
    power = np.abs(reference)**2
    desired = np.maximum(power.sum(-1), 1e-8)
    gain = np.maximum(np.abs(wave)*np.abs(reference), 1e-8)
    phase = ((wave*reference.conj()).real*power/gain).sum(-1)/desired
    error = (np.abs(wave-reference)**2).sum(-1)/desired
    amplitude = ((np.abs(wave)-np.abs(reference))**2).sum(-1)/desired
    return np.stack((error, amplitude, phase), -1)


def summarize(waves, reference):
    values = np.stack([fidelity(waves[mode], reference) for mode in MODES])
    nmse = values[:, :, 0]
    gain = 100.*(nmse[0] - nmse)/np.maximum(nmse[0], 1e-8)
    return dict(n=len(reference), median=np.median(values, axis=1).tolist(),
                relative_reduction=np.median(gain, axis=1).tolist(),
                improves=(nmse<nmse[0]).sum(1).tolist())


def source_threshold(rt):
    root = Path(rt['run_root'])
    if rt['run_id'] != RUN or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('The CPU probe requires the registered differential timing rule')
    record = read(root/'threshold.json')
    if any(record[k] for k in ('source_V_reads', 'target_reads', 'TX_reads', 'F_reads', 'optimizer_updates')):
        raise ValueError('The inherited threshold must come from source U only')
    return record['thresholds']['differential']


def probe(rt):
    started = perf_counter()
    threshold = source_threshold(rt)
    wave, available, calibration = source_iq(Path(rt['source_root']))
    held = available[3*COUNT:4*COUNT]
    if len(held)!=COUNT or np.intersect1d(held, calibration).size:
        raise ValueError('Independent U role is required')
    ground = wave[held]
    clean = ground[:, 0] + 1j*ground[:, 1]
    results = {}
    for kind, complete in cases(clean).items():
        for amount in SHIFTS:
            p = TIME+amount
            incoming = complete[:, np.minimum(p, 2*(len(TIME)-1)-p)]
            iq = np.stack((incoming.real, incoming.imag), 1)
            raw, info, response, fits = observations(iq)
            delta, margin = fits['differential']
            selected = np.where(margin > threshold, delta, 0)
            outputs = readouts(raw, selected, info.coarse_rad_per_sample, response)
            inside = (TIME + selected[:, None] >= 0) & (TIME + selected[:, None] < len(TIME))
            original = raw[np.arange(len(raw))[:, None], (TIME+selected[:, None]).clip(0, len(TIME)-1)]
            prefix = TIME+selected[:, None] < 0
            for x in outputs.values():
                if not np.array_equal(x[inside], original[inside]) or not np.array_equal(x[~prefix], outputs['public'][~prefix]):
                    raise ValueError('Prefix extrapolation must preserve received overlap, bypass and suffix')
            exact = selected == -amount
            if not exact.any():
                raise ValueError('The engineering condition has no exactly timed holes to compare')
            used = {k: v[exact, :amount] for k, v in outputs.items()}
            results[f'{kind}_{amount}'] = dict(exactly_timed=int(exact.sum()), moved=int((selected!=0).sum()),
                                               hypothetical_received=summarize(used, complete[exact, :amount]),
                                               source_clean=summarize(used, clean[exact, :amount]))
    return dict(status='VERIFIED', source_role='U_s', held=len(held), held_offset=3*COUNT, threshold=threshold,
                fields=FIELDS, modes=MODES, seconds=perf_counter()-started, source_V_reads=0,
                F_reads=0, TX_reads=0, target_reads=0, checkpoint_reads=0, optimizer_updates=0, results=results)


def source_coverage(rt):
    started = perf_counter()
    threshold = source_threshold(rt)
    root = Path(rt['run_root'])
    if read(root/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Read source V timing coverage only after the two routes are frozen and scored')
    cache, _, _, _, ids, rx, _ = registered.handles(rt)
    receivers = {str(key): rx == key for key in np.unique(rx)}
    selectors = {'all': np.ones(len(ids), dtype=bool), **receivers}
    results = {}
    for route, folder in registered.ROUTES.items():
        results[route] = {}
        for view in registered.VIEWS:
            raw = np.load(cache/folder/f'{view}.npy', allow_pickle=False, mmap_mode='r')
            if raw.shape != (len(ids), 2, 256) or raw.dtype != np.float32:
                raise ValueError('Unexpected registered WiSig source view')
            shifts = []
            for begin in range(0, len(raw), 64):
                delta, confidence = observations(raw[begin:begin+64])[-1]['differential']
                shifts.append(np.where(confidence > threshold, delta, 0))
            selected = np.concatenate(shifts)
            results[route][view] = {}
            for group, mask in selectors.items():
                values = selected[mask]
                prefix, suffix = -values[values<0], values[values>0]
                results[route][view][group] = dict(rows=len(values), unshifted=int((values==0).sum()),
                    prefix_packets=len(prefix), suffix_packets=len(suffix),
                    prefix_samples=int(prefix.sum()), suffix_samples=int(suffix.sum()))
    return dict(status='VERIFIED', role='V', rows=len(ids), results=results, F_reads=0,
                TX_reads=0, target_reads=0, threshold_updates=0, seconds=perf_counter()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--coverage', action='store_true')
    args = parser.parse_args()
    print(json.dumps((source_coverage if args.coverage else probe)(read(args.runtime)), allow_nan=False))
