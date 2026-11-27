"""Compare same-packet coverage of frozen learned and true-noise source readouts."""

import argparse
import json
from pathlib import Path

import numpy as np

from recount_error_groups import load_scored, verified_hits


FACTOR_MODES = ['direct', 'frozen_half', 'true_amplitude', 'true_direction',
                'true_both', 'projected_amplitude']
LEARNED_MODES = ['direct', 'frozen_half', 'fixed_dose_direction', 'within_chirp', 'cross_chirp']


def area(learned, known_direction, known_noise):
    remaining = ~learned
    return {
        'n': int(learned.size),
        'any_learned_correct': int(learned.sum()),
        'all_learned_wrong': int(remaining.sum()),
        'all_learned_wrong_true_direction_correct': int((remaining & known_direction).sum()),
        'all_learned_wrong_exact_denoise_correct': int((remaining & known_noise).sum()),
        'any_learned_correct_exact_denoise_wrong': int((learned & ~known_noise).sum()),
    }


def recount(factor_root, peer_root):
    factor, factor_scores, factor_pred = load_scored(factor_root, FACTOR_MODES, 'oracle_only')
    peer, peer_scores, peer_pred = load_scored(peer_root, LEARNED_MODES, 'received_only')
    ids, views = factor['physical_ids'], factor['views']
    if (peer['physical_ids'] != ids or peer['views'] != views
            or not np.array_equal(peer_pred[:, :2], factor_pred[:, :2])):
        raise ValueError('Source packet order, scenes and shared frozen baselines must match')
    if (factor['waveform']['frozen_run'] != peer['waveform']['base']
            or factor['waveform']['candidate'] != 'practical_exact_additive'
            or factor['waveform']['regional_edits'] is not True
            or peer['waveform']['matched_center_edge_energy'] is not True):
        raise ValueError('Frozen learned readouts must use the same Practical parent')

    with np.load(factor_root/'V_metadata.npz', allow_pickle=False) as bank:
        rows = {key: i for i, key in enumerate(bank['ids'].tolist())}
        positions = [rows[packet] for packet in ids]
        truth, rx = bank['y'][positions], bank['rx_i'][positions]
    oracles = verified_hits(factor_pred, truth, factor, factor_scores)
    actual = verified_hits(peer_pred, truth, peer, peer_scores)
    any_actual = np.logical_or.reduce(list(actual.values()))
    masks = (any_actual, oracles['true_direction'], oracles['true_both'])
    views_report = {view: area(*(mask[i] for mask in masks)) for i, view in enumerate(views)}
    views_report['six'] = area(*(mask[1:] for mask in masks))
    by_receiver = {int(receiver): {'six': area(*(mask[1:, rx == receiver] for mask in masks)),
                                  'low_urban': area(*(mask[-1, rx == receiver] for mask in masks))}
                   for receiver in np.unique(rx)}
    return dict(source_only=True, frozen_shared_controls_equal=True, n_physical=len(ids),
                learned_modes=LEARNED_MODES, by_view=views_report, by_source_receiver=by_receiver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ('factor_root', 'peer_root', 'output'):
        parser.add_argument(path, type=Path)
    args = parser.parse_args()
    result = recount(args.factor_root, args.peer_root)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result['by_view'][k] for k in ('six', 'practical_low_urban')}))
