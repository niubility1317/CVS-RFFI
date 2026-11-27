"""Attribute errors using already scored, frozen LoRa source predictions."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def count_groups(root):
    frozen = read(root / 'source-predictions.json')
    scored = read(root / 'source-scores.json')
    if (frozen['status'] != 'PREDICTIONS_FROZEN' or frozen['source_truth_reads']
            or frozen['target_reads'] or scored['status'] != 'ANALYZED'
            or scored['target_reads'] or not scored['source_truth_last']
            or frozen['waveform'] != scored['waveform'] or not frozen['waveform']['oracle_only']):
        raise ValueError('Expected independently scored source-only direction diagnostic')

    ids, views, modes = (frozen[k] for k in ('physical_ids', 'views', 'modes'))
    expected_modes = ['direct', 'frozen_half', 'true_amplitude', 'true_direction',
                      'true_both', 'projected_amplitude']
    if modes != expected_modes or len(ids) != 300 or len(set(ids)) != 300 or len(views) != 7:
        raise ValueError('Unexpected frozen source matrix')
    with np.load(root / 'predictions.npz', allow_pickle=False) as bundle:
        predicted = np.stack([bundle[view] for view in views])
    if predicted.shape != (len(views), len(modes), len(ids)):
        raise ValueError('Predictions do not cover the source matrix')

    with np.load(root / 'V_metadata.npz', allow_pickle=False) as meta:
        rows = {key: index for index, key in enumerate(meta['ids'].tolist())}
        position = [rows[key] for key in ids]
        truth, rx = meta['y'][position], meta['rx_i'][position]
    hits = dict(zip(modes, (predicted == truth).transpose(1, 0, 2), strict=True))
    for mode, correct in hits.items():
        if correct.sum(-1).tolist() != [scored['scores'][mode]['by_scene'][v] for v in views]:
            raise ValueError(f'Independent score does not reconcile for {mode}')

    direct, half = hits['direct'], hits['frozen_half']
    groups = {'harmed_by_half': direct & ~half, 'missed_by_both': ~direct & ~half,
              'repaired_by_half': ~direct & half, 'correct_in_both': direct & half}
    selectors = {view: slice(i, i + 1) for i, view in enumerate(views[1:], 1)}
    selectors['six'] = slice(1, None)
    result = {}
    for scene, sel in selectors.items():
        result[scene] = {}
        for name, group in groups.items():
            active = group[sel]
            result[scene][name] = {
                'n': int(active.sum()),
                'correct': {mode: int((active & hits[mode][sel]).sum()) for mode in modes[2:]},
                'by_source_receiver': {
                    int(receiver): {
                        'n': int(active[:, rx == receiver].sum()),
                        'direction': int((active & hits['true_direction'][sel])[:, rx == receiver].sum()),
                        'direction_and_amplitude': int((active & hits['true_both'][sel])[:, rx == receiver].sum()),
                    } for receiver in np.unique(rx)
                },
            }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scored_source_root', type=Path)
    args = parser.parse_args()
    print(json.dumps(count_groups(args.scored_source_root), indent=2))
