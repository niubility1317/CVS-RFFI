"""Confirm paired WiSig predictions and measure source-only timing after scoring."""

import json
import os
from pathlib import Path
import sys

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_unrolled_restore import wisig_short_training_source as core
from experiments.pamf_unrolled_restore.wisig_joint_preamble_source import RUN as PRIOR
from experiments.pamf_unrolled_restore.wisig_smooth_preamble import TIME, observations, readouts
from experiments.pamf_unrolled_restore.wisig_smooth_preamble_source import MODES, RUN


def reference_timing(received, reference):
    offsets = np.arange(-64, 65)
    n = len(TIME)
    spectra = [np.fft.fft(signal, n=2*n) for signal in (received, reference)]
    correlation = np.fft.ifft(spectra[0] * spectra[1].conj())[:, offsets % (2*n)]

    def power(signal, shift):
        squared = signal.real.astype(np.float64)**2 + signal.imag.astype(np.float64)**2
        prefix = np.pad(squared.cumsum(1), [(0, 0), (1, 0)])
        return prefix[:, np.minimum(n, n+shift)] - prefix[:, np.maximum(0, shift)]

    energy = power(received, offsets) * power(reference, -offsets)
    score = np.abs(correlation)**2 / np.maximum(energy, 1e-8)
    return offsets[score.argmax(1)]


def diagnostics(raw, clean, thresholds, rx):
    y = raw[:, 0]+1j*raw[:, 1]
    x = clean[:, 0]+1j*clean[:, 1]
    oracle = reference_timing(y, x)
    _, _, fits = observations(raw)
    modes, _ = readouts(raw, thresholds)
    overlap = ((TIME + oracle[:, None] >= 0) & (TIME + oracle[:, None] < len(TIME)))
    rows = {}
    for name, wave in modes.items():
        if name == 'direct':
            shift = np.zeros(len(x), dtype=int)
        else:
            key = name.removesuffix('_reflect')
            delta, margin, _ = fits[key]
            shift = np.where(margin > thresholds[key], delta, 0)
        restored = wave[:, 0]+1j*wave[:, 1]
        err = np.abs(x-restored)**2
        power = np.abs(x)**2
        nmse = (err*overlap).sum(1) / np.maximum((power*overlap).sum(1), 1e-8)
        delay_err = np.abs(oracle-shift)
        phases = (restored*x.conj()).real / np.maximum(np.abs(restored)*np.abs(x), 1e-8)
        cosine = (phases*power*overlap).sum(1) / np.maximum((power*overlap).sum(1), 1e-8)
        groups = {'all': np.ones(len(rx), dtype=bool), **{str(value): rx == value for value in np.unique(rx)}}
        rows[name] = {group: dict(count=int(mask.sum()), active=int((shift[mask]!=0).sum()),
                       shift_near_two=int((delay_err[mask]<=2).sum()),
                       median_overlapping_nmse=float(np.median(nmse[mask])),
                       median_overlapping_phase_cosine=float(np.median(cosine[mask])))
                       for group, mask in groups.items()}
    return rows


def audit(rt):
    root = Path(rt['run_root'])
    if (rt['run_id'] != RUN or read(root/'state.json')['status'] != 'SOURCE_SCORED'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('Independent CPU diagnostic requires both source routes fully scored')
    marker = read(root/'predictions/complete.json')
    score = read(root/'source-scores.json')
    if (marker['status'] != 'FROZEN' or marker['source_truth_reads'] or marker['target_reads']
            or marker['modes'] != list(MODES) or score['modes'] != list(MODES)
            or not score['truth_last'] or score['target_reads']):
        raise ValueError('Source prediction or truth-side identity changed')
    prior = root.parent/PRIOR/'predictions'
    previous = read(prior/'complete.json')
    if previous['count'] != marker['count'] or previous['target_reads'] or previous['source_truth_reads']:
        raise ValueError('The historic same-reader control differs')
    cache, _, _, _, ids, rx, _ = core.handles(rt)
    thresholds = read(root/'threshold.json')['thresholds']
    records = {}
    for route, folder in core.ROUTES.items():
        source = cache/folder
        clean = np.load(source/'clean.npy', mmap_mode='r', allow_pickle=False)
        records[route] = {}
        for view in core.VIEWS:
            name = f'{route}_{view}.npz'
            with np.load(root/'predictions'/name, allow_pickle=False) as new, np.load(prior/name, allow_pickle=False) as old:
                if not (np.array_equal(new['ids'], ids) and np.array_equal(old['ids'], ids)):
                    raise ValueError('Original and new source physical order differs')
                for mode in ('direct', 'r69'):
                    if not np.array_equal(new['decision'][MODES.index(mode)], old['decision'][previous['modes'].index(mode)]):
                        raise ValueError('Frozen reader or Original DURA decisions changed')
            raw = np.load(source/f'{view}.npy', mmap_mode='r', allow_pickle=False)
            records[route][view] = diagnostics(raw, clean, thresholds, rx)
    return dict(status='VERIFIED', same_direct_and_r69=True, paired_clean_diagnostic_only=True,
                source_count=len(ids), target_reads=0, routes=records)


if __name__ == '__main__':
    print(json.dumps(audit(read(Path(sys.argv[1]))), allow_nan=False))
