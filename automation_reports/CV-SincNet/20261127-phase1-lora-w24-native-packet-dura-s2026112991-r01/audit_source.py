"""Read immutable source scores and recount each receiver and transmitter."""

import argparse
import json
from pathlib import Path

import numpy as np


RUN = '20261127-phase1-lora-w24-native-packet-dura-s2026112991-r01'
ARMS = ('native_dura_center', 'native_dura_whole')
TERMS = ('half_native_nmse', 'coarse_half_native_nmse', 'first_half_native_nmse', 'common_detail',
         'private_detail', 'native_edges_nmse', 'identity')
CHECKS = ('F_reads', 'TX_reads', 'V_reads', 'target_reads', 'practical_reads', 'trace_reads')
MILESTONES = (2000, 6000, 12000, 18000)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def held_trajectory(log):
    held = {}
    with log.open(encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('CHANNEL_BLIND_HELD '):
                _, step, *values = line.split()
                held[int(step)] = dict(zip(TERMS, map(float, values), strict=True))
    if tuple(held) != MILESTONES:
        raise ValueError('Expected all registered U_s held milestones')
    return held


def paired_counts(delta, views):
    blocks = zip((*views, 'six'), (*delta, delta[1:]), strict=True)
    return {view: dict(repairs=int((d > 0).sum()), harms=int((d < 0).sum()), net=int(d.sum()))
            for view, d in blocks}


def audit(root, source):
    if root.name != RUN or (root / 'native_residual_w24_v1').exists():
        raise ValueError('Expected the registered Full-only run')
    state = read(root / 'state.json')
    if state['status'] not in ('SOURCE_SCORED', 'FORMAL_PREDICTING', 'PREDICTIONS_FROZEN', 'FAILED'):
        raise ValueError('Expected independently scored source predictions')
    route = root / 'full_native'
    frozen = read(route / 'source-predictions.json')
    score = read(route / 'source-scores.json')
    if (frozen['status'] != 'PREDICTIONS_FROZEN' or frozen['source_truth_reads'] or frozen['target_reads']
            or score['status'] != 'ANALYZED' or not score['source_truth_last'] or score['target_reads']
            or frozen['waveform'] != score['waveform'] or score['n_packets'] != 300
            or score['n_practical'] != 1800):
        raise ValueError('Unexpected source prediction or scoring evidence')
    views, modes, ids = (frozen[k] for k in ('views', 'modes', 'physical_ids'))
    if len(ids) != 300 or len(set(ids)) != 300 or len(views) != 7 or len(modes) != 8:
        raise ValueError('Expected the complete ordered 300-packet matrix')
    result = dict(source_state=state, n_packets=len(ids), views=views, modes=modes, score=score)
    smoke = read(root / 'smoke.json')
    if (smoke['status'] != 'PASS' or smoke['target_reads'] or smoke['source_truth_reads']
            or smoke['candidate_steps'] != 24000):
        raise ValueError('Expected matched frozen W24 U_s smoke')
    result['frozen_w24_smoke'] = smoke
    result['training'] = {}
    for arm in ARMS[1:]:
        fit = read(root / arm / f'w24_{arm}-complete.json')
        if (fit['steps'] != 24000 or fit['initialization'] != 'random'
                or fit['native_capture_samples'] != 8192
                or fit['observation_steps'] != 2
                or [fit[k] for k in ('n_fit', 'n_held', 'n_calibration')] != [20000, 500, 0]
                or any(fit[k] != 0 for k in CHECKS)):
            raise ValueError(f'Unmatched fit record for {arm}')
        result['training'][arm] = dict(fit=fit, held=held_trajectory(root / f'{arm}-fit.log'))
    with np.load(route / 'predictions.npz', allow_pickle=False) as bundle:
        predictions = np.stack([bundle[v] for v in views])
    if predictions.shape != (7, 8, 300):
        raise ValueError('Invalid frozen source shape')
    # This diagnostic joins only after the above immutable prediction and score checks.
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as meta:
        all_ids = meta['ids'].tolist()
        if all_ids != read(source / 'contract.json')['role_ids']['V']:
            raise ValueError('Source V physical ID mismatch')
        position = {pid: i for i, pid in enumerate(all_ids)}
        rows = [position[pid] for pid in ids]
        tx, rx = meta['y'][rows], meta['rx_i'][rows]
    hits = predictions == tx
    report = score['scores']
    for i, mode in enumerate(modes):
        if hits[:, i].sum(-1).tolist() != [report[mode]['by_scene'][view] for view in views]:
            raise ValueError(f'Source score recount did not reconcile for {mode}')
    controls = ('direct', 'all_half', 'outer_parent_half', 'frozen_unet_half')
    comparisons = [(f'{arm}_half', baseline) for arm in ARMS for baseline in controls]
    comparisons.append((f'{ARMS[1]}_half', f'{ARMS[0]}_half'))
    comparisons.append((f'{ARMS[1]}_full', f'{ARMS[0]}_full'))
    rng = np.random.default_rng(2026112560)
    receivers = [np.flatnonzero(rx == group) for group in np.unique(rx)]
    draws = np.concatenate([rng.choice(rows, size=(2000, len(rows))) for rows in receivers], axis=1)
    result['comparisons'] = {}
    for candidate, baseline in comparisons:
        delta = hits[:, modes.index(candidate)].astype(np.int16) - hits[:, modes.index(baseline)]
        groups = {}
        for key, index in (('rx', rx), ('tx', tx)):
            groups[key] = {str(int(g)): paired_counts(delta[:, index == g], views) for g in np.unique(index)}
        intervals = {view: np.quantile(d[draws].sum(1), (.025, .975)).tolist()
                     for view, d in zip((*views, 'six'), (*delta, delta[1:].sum(0)), strict=True)}
        result['comparisons'][f'{candidate}_vs_{baseline}'] = dict(all=paired_counts(delta, views),
                                                                  ci95=intervals, **groups)
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for parameter in ('root', 'source'):
        cli.add_argument('--' + parameter, type=Path, required=True)
    print(json.dumps(audit(**vars(cli.parse_args()))))
