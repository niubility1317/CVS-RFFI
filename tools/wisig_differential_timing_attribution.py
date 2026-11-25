"""Audit already-frozen WiSig source timing, physical fidelity and recognition."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_unrolled_restore import wisig_short_training_source as core
from experiments.pamf_unrolled_restore.wisig_differential_timing import ARMS, observations, readouts
from experiments.pamf_unrolled_restore.wisig_differential_timing_source import READOUTS, RUN
from experiments.pamf_unrolled_restore.wisig_timing import PERIOD
from tools.wisig_smooth_preamble_audit import reference_timing


MODES = (*READOUTS, 'r69')
TIME = np.arange(256)
REGIONS = ('overlap', 'short_training')


def waveform_metrics(wave, reference, valid):
    power = np.abs(reference)**2
    energy = np.maximum((power*valid).sum(1), 1e-8)
    cross = wave*reference.conj()
    scale = np.maximum(np.abs(wave)*np.abs(reference), 1e-8)
    cosine = ((cross.real/scale)*power*valid).sum(1)/energy
    original = (np.abs(wave-reference)**2*valid).sum(1)/energy
    amplitude = ((np.abs(wave)-np.abs(reference))**2*valid).sum(1)/energy
    phase = np.angle((cross*valid).sum(1))
    rotated = wave*np.exp(-1j*phase[:, None])
    diagnostic = (np.abs(rotated-reference)**2*valid).sum(1)/energy
    return np.stack((original, cosine, diagnostic, amplitude), -1)


def collect_packet_evidence(raw, clean, thresholds, decisions, labels):
    raw_c = raw[:, 0] + 1j*raw[:, 1]
    clean_c = clean[:, 0] + 1j*clean[:, 1]
    reference = reference_timing(raw_c, clean_c)
    received_index = TIME + reference[:, None]
    overlap = (received_index >= 0) & (received_index < len(TIME))
    stf = overlap & (TIME[None] >= 40) & (TIME[None] < 160)
    _, _, _, estimates = observations(raw)
    variants, _ = readouts(raw, thresholds)
    evidence = {}
    for name, output in variants.items():
        moved = np.zeros(len(raw), dtype=int)
        if name != 'direct':
            delta, margin = estimates[name]
            moved = np.where(margin > thresholds[name], delta, 0)
        good = np.abs(moved-reference) <= 2
        alias_error = np.abs((moved-reference + PERIOD//2) % PERIOD - PERIOD//2)
        wave = output[:, 0] + 1j*output[:, 1]
        if (not np.array_equal(output[moved == 0], raw[moved == 0])
                or not np.array_equal(decisions[MODES.index(name), moved == 0], decisions[0, moved == 0])):
            raise ValueError('An unshifted received window must exactly follow the direct readout')
        evidence[name] = dict(moved=moved, good=good, alias=(alias_error<=2) & ~good,
                              correct=decisions[MODES.index(name)]==labels,
                              overlap=waveform_metrics(wave, clean_c, overlap),
                              short_training=waveform_metrics(wave, clean_c, stf))
    return evidence


def counts(direct, arm, selected):
    mask = selected
    active = arm['moved'] != 0
    before, after = direct['good'], arm['good']
    cells = {'all': mask, 'bypass': mask & ~active,
             'moved_into_reference': mask & active & ~before & after,
             'moved_away_from_reference': mask & active & before & ~after,
             'moved_both_near': mask & active & before & after,
             'moved_both_far': mask & active & ~before & ~after}
    rows = {}
    for condition, subset in cells.items():
        base, new = direct['correct'][subset], arm['correct'][subset]
        result = dict(rows=int(subset.sum()), correct=int(new.sum()), correct_before=int(base.sum()),
                      repaired=int((new & ~base).sum()), damaged=int((base & ~new).sum()),
                      near_two=int(after[subset].sum()), initial_near_two=int(before[subset].sum()),
                      alias_only=int(arm['alias'][subset].sum()))
        if subset.any():
            result.update({region: dict(direct=np.median(direct[region][subset], 0).tolist(),
                                         output=np.median(arm[region][subset], 0).tolist())
                           for region in REGIONS})
        rows[condition] = result
    return rows


def audit(rt):
    begun = perf_counter()
    root = Path(rt['run_root'])
    if (rt['run_id'] != RUN or read(root/'state.json')['status'] != 'SOURCE_SCORED'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('Requires the already-scored WiSig source experiment and a CPU process')
    marker, scores = read(root/'predictions/complete.json'), read(root/'source-scores.json')
    if (marker['status'] != 'FROZEN' or marker['modes'] != list(MODES) or marker['source_truth_reads']
            or marker['target_reads'] or scores['target_reads'] or not scores['truth_last']):
        raise ValueError('The two-route prediction and truth-last source evidence must close first')
    thresholds = read(root/'threshold.json')['thresholds']
    cache, _, source, _, ids, rx, position = core.handles(rt)
    matrices = {}
    for route in core.ROUTES:
        for view in core.VIEWS:
            with np.load(root/'predictions'/f'{route}_{view}.npz', allow_pickle=False) as saved:
                if not np.array_equal(saved['ids'], ids) or saved['decision'].shape != (len(MODES), len(ids)):
                    raise ValueError('Expected the full source prediction identity')
                matrices[route, view] = saved['decision']
    labels = np.load(source/'V'/'labels.npy', allow_pickle=False)[position]
    groups = {'all': np.ones(len(ids), dtype=bool), **{str(key): rx == key for key in np.unique(rx)}}
    result = {}
    for route, folder in core.ROUTES.items():
        folder = cache/folder
        clean = np.load(folder/'clean.npy', mmap_mode='r', allow_pickle=False)
        result[route] = {}
        for index, view in enumerate(core.VIEWS):
            raw = np.load(folder/f'{view}.npy', mmap_mode='r', allow_pickle=False)
            view_records = collect_packet_evidence(raw, clean, thresholds, matrices[route, view], labels)
            result[route][view] = {}
            for name in ARMS:
                record = {key: counts(view_records['direct'], view_records[name], subset)
                          for key, subset in groups.items()}
                if any(row['all']['correct'] != scores['scores'][route][key]['correct'][index][MODES.index(name)]
                       for key, row in record.items()):
                    raise ValueError('Packet-level attribution differs from the frozen source score')
                result[route][view][name] = record
    return dict(status='VERIFIED', role='V', rows=len(ids), seconds=perf_counter()-begun,
                waveform_fields=('complex_nmse', 'phase_cosine', 'one_scalar_phase_oracle_nmse', 'amplitude_nmse'),
                paired_clean_diagnostic_only=True, F_reads=0, target_reads=0, optimizer_updates=0,
                threshold_updates=0, routes=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    print(json.dumps(audit(read(parser.parse_args().runtime)), allow_nan=False))
