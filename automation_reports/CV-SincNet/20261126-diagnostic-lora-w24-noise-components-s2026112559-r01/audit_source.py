"""Recount frozen source-only diagnostic predictions by physical packet."""

import argparse
import json
from pathlib import Path

import numpy as np


RUN = '20261126-diagnostic-lora-w24-noise-components-s2026112559-r01'
BASE = ('direct', 'outer_half', 'exact_center')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def counts(delta, views):
    return {view: dict(repairs=int((d > 0).sum()), harms=int((d < 0).sum()), net=int(d.sum()))
            for view, d in zip((*views, 'six'), (*delta, delta[1:]), strict=True)}


def interaction_of_common_bands(hit, modes, views, rx):
    quartet = ('exact_center', 'far_with_private_near', 'near_with_private_far', 'exact_private')
    a, b, c, d = (hit[:, modes.index(key)].astype(np.int16) for key in quartet)
    contrast = a - b - c + d
    rng = np.random.default_rng(2026112560)
    groups = [np.flatnonzero(rx == group) for group in np.unique(rx)]
    draws = np.concatenate([rng.choice(g, (2000, len(g))) for g in groups], axis=1)
    effects = (*contrast, contrast[1:].sum(0))
    return {view: dict(net=int(z.sum()), ci95=np.quantile(z[draws].sum(1), [.025, .975]).tolist())
            for view, z in zip((*views, 'six'), effects, strict=True)}


def audit(root, source):
    if root.name != RUN or read(root / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the completed source-only panel')
    folder = root / 'full_native'
    proof = read(folder / 'source-predictions.json')
    score = read(folder / 'source-scores.json')
    if (proof['status'] != 'PREDICTIONS_FROZEN' or proof['target_reads'] or proof['source_truth_reads']
            or score['status'] != 'ANALYZED' or not score['source_truth_last'] or score['target_reads']
            or proof['waveform'] != score['waveform'] or not proof['waveform']['exact_noise_source_diagnostic_only']
            or score['n_packets'] != 300 or score['n_practical'] != 1800):
        raise ValueError('Invalid source scoring boundary')
    modes, views, ids = (proof[x] for x in ('modes', 'views', 'physical_ids'))
    if len(modes) != 11 or len(views) != 7 or len(set(ids)) != 300:
        raise ValueError('Invalid frozen diagnostic panel')
    with np.load(folder / 'predictions.npz', allow_pickle=False) as data:
        pred = np.stack([data[v] for v in views])
    if pred.shape != (7, 11, 300):
        raise ValueError('Incomplete source prediction matrix')
    if not np.all(pred[0, 2:] == pred[0, 0]):
        raise ValueError('Exact noise interventions must preserve clean predictions')
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        expected_ids = metadata['ids'].tolist()
        if expected_ids != read(source / 'contract.json')['role_ids']['V']:
            raise ValueError('Source V physical order did not match contract')
        positions = {pid: i for i, pid in enumerate(expected_ids)}
        index = [positions[pid] for pid in ids]
        tx, rx = metadata['y'][index], metadata['rx_i'][index]
    hit = pred == tx
    for i, mode in enumerate(modes):
        if hit[:, i].sum(-1).tolist() != [score['scores'][mode]['by_scene'][v] for v in views]:
            raise ValueError(f'Independent recount did not reproduce {mode}')
    comparisons = [(mode, baseline) for mode in modes[2:] for baseline in BASE if mode != baseline]
    comparisons.append(('private_matched', 'common_matched'))
    changes = {}
    for candidate, baseline in comparisons:
        delta = hit[:, modes.index(candidate)].astype(np.int16) - hit[:, modes.index(baseline)]
        changes[f'{candidate}_vs_{baseline}'] = {
            'all': counts(delta, views),
            'rx': {str(int(g)): counts(delta[:, rx == g], views) for g in np.unique(rx)},
            'tx': {str(int(g)): counts(delta[:, tx == g], views) for g in np.unique(tx)},
        }
    return dict(n_physical=len(ids), modes=modes, views=views, source_score=score, changes=changes,
                posthoc_common_band_interaction=interaction_of_common_bands(hit, modes, views, rx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'source'):
        parser.add_argument('--' + key, type=Path, required=True)
    print(json.dumps(audit(**vars(parser.parse_args()))))
