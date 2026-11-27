"""Recount matched regional-dose decisions from closed independent source scoring."""

import argparse
import json
from pathlib import Path

import numpy as np


MODES = ('direct', 'frozen_half', 'fixed_dose_direction', 'constant_gain', 'adaptive_gain',
         'optimal_amplitude_on_learned_direction')
PAIRS = [('adaptive_gain', 'fixed_dose_direction'), ('adaptive_gain', 'constant_gain'),
         ('constant_gain', 'fixed_dose_direction'), (MODES[-1], 'fixed_dose_direction')]


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def counts(delta):
    return dict(repairs=int((delta > 0).sum()), harms=int((delta < 0).sum()), net=int(delta.sum()))


def recount(root, reference, source_metadata):
    state = load(root/'state.json')
    matrix, score = [load(root/name) for name in ('source-predictions.json', 'source-scores.json')]
    prior = load(reference/'source-predictions.json')
    ids, views = matrix['physical_ids'], matrix['views']
    if (state['status'] != 'SOURCE_SCORED' or state['target_reads']
            or matrix['status'] != 'PREDICTIONS_FROZEN' or matrix['target_reads'] or matrix['source_truth_reads']
            or score['status'] != 'ANALYZED' or score['target_reads'] or not score['source_truth_last']
            or score['waveform'] != matrix['waveform'] or matrix['waveform']['oracle_only_mode'] != MODES[-1]
            or tuple(matrix['modes']) != MODES or tuple(prior['modes']) != MODES[:3]
            or prior['status'] != 'PREDICTIONS_FROZEN' or prior['target_reads'] or prior['source_truth_reads']
            or prior['physical_ids'] != ids or prior['views'] != views
            or len(ids) != 300 or len(set(ids)) != 300 or len(views) != 7 or views[0] != 'clean'):
        raise ValueError('Expected completed, matched, independently scored source artifacts')
    with (np.load(root/'predictions.npz', allow_pickle=False) as p,
          np.load(reference/'predictions.npz', allow_pickle=False) as r):
        predicted = np.stack([p[key] for key in views])
        if predicted.shape != (7, len(MODES), 300) or not np.array_equal(
                predicted[:, :3], np.stack([r[key] for key in views])):
            raise ValueError('Archived source baselines disagree')
    with np.load(source_metadata, allow_pickle=False) as meta:
        order = {key: i for i, key in enumerate(meta['ids'].tolist())}
        positions = [order[key] for key in ids]
        truth, rx = meta['y'][positions], meta['rx_i'][positions]
    hit = dict(zip(MODES, (predicted == truth).transpose(1, 0, 2).astype(np.int8), strict=True))
    for mode, correct in hit.items():
        if correct.sum(-1).tolist() != [score['scores'][mode]['by_scene'][key] for key in views]:
            raise ValueError(f'Independent score does not reconcile for {mode}')
    result = {}
    for candidate, baseline in PAIRS:
        delta = hit[candidate] - hit[baseline]
        key = f'{candidate}_vs_{baseline}'
        six, low = counts(delta[1:]), counts(delta[-1])
        recorded = score['paired_contrasts'][key]
        if (six['net'], low['net']) != (recorded['net_six'], recorded['net_low']):
            raise ValueError(f'Independent pair score does not reconcile for {key}')
        result[key] = dict(six=six, low_urban=low, by_scene=dict(zip(views, map(counts, delta), strict=True)))
        for label, values in (('rx', rx), ('tx', truth)):
            result[key][label] = {
                int(g): dict(six=counts(delta[1:, values == g]), low_urban=counts(delta[-1, values == g]))
                for g in np.unique(values)
            }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('source_metadata', type=Path)
    args = parser.parse_args()
    print(json.dumps(recount(args.root, args.reference, args.source_metadata), indent=2))
