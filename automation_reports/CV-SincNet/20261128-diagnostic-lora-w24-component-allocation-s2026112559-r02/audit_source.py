"""Independently recount the closed source-only component factorial."""

import argparse
import json
from pathlib import Path

import numpy as np


MODES = ('direct', 'frozen_half', 'component_norms', 'component_directions', 'component_both', 'exact_center')


def read(root, file):
    return json.loads((root/file).read_text(encoding='utf-8-sig'))


def summarize(delta):
    return dict(repairs=int((delta > 0).sum()), harms=int((delta < 0).sum()), net=int(delta.sum()))


def audit(root, old):
    proof = read(root, 'source-predictions.json')
    prior = read(old, 'source-predictions.json')
    score = read(root, 'source-scores.json')
    if (proof['status'] != 'PREDICTIONS_FROZEN' or prior['status'] != 'PREDICTIONS_FROZEN'
            or score['status'] != 'ANALYZED' or not score['source_truth_last']
            or any(doc['target_reads'] for doc in (proof, prior, score))
            or proof['source_truth_reads'] or prior['source_truth_reads']
            or proof['waveform'] != score['waveform'] or not proof['waveform']['fixed_center_edit_energy']
            or not proof['waveform']['frozen_native_edges'] or proof['modes'] != list(MODES)):
        raise ValueError('Expected the registered closed source-only component factorial')
    views, ids = proof['views'], proof['physical_ids']
    if views != prior['views'] or ids != prior['physical_ids'] or len(views) != 7 or len(set(ids)) != 300:
        raise ValueError('Registered source physical rows do not match')
    with (np.load(root/'predictions.npz', allow_pickle=False) as bundle,
          np.load(old/'predictions.npz', allow_pickle=False) as previous):
        prediction = np.stack([bundle[view] for view in views])
        prior_modes = [prior['modes'].index(mode) for mode in MODES[:2]]
        if (prediction.shape != (7, 6, 300) or
                not np.array_equal(prediction[:, :2], np.stack([previous[v][prior_modes] for v in views]))):
            raise ValueError('Unexpected frozen prediction shape or changed controls')
    with np.load(old/'V_metadata.npz', allow_pickle=False) as bank:
        position = {value: i for i, value in enumerate(bank['ids'].tolist())}
        idx = [position[value] for value in ids]
        truth, rx = bank['y'][idx], bank['rx_i'][idx]
    hits = dict(zip(MODES, (prediction == truth).transpose(1, 0, 2).astype(np.int8), strict=True))
    for mode, hit in hits.items():
        if hit.sum(-1).tolist() != [score['scores'][mode]['by_scene'][scene] for scene in views]:
            raise ValueError(f'Source counts changed for {mode}')
    pairs = {}
    for contrast, old_score in score['paired_contrasts'].items():
        candidate, control = contrast.split('_vs_')
        delta = hits[candidate] - hits[control]
        six, low = summarize(delta[1:]), summarize(delta[-1])
        if (six['net'], low['net']) != (old_score['net_six'], old_score['net_low']):
            raise ValueError(f'Source contrast changed for {contrast}')
        pairs[contrast] = dict(six=six, low=low, by_view={v: summarize(d) for v, d in zip(views, delta, strict=True)},
                               by_receiver={int(group): {'six': summarize(delta[1:, rx == group]),
                                                        'low': summarize(delta[-1, rx == group])}
                                            for group in np.unique(rx)})
    return dict(source_only=True, n_physical=len(ids), controls_exact=True, pairs=pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'old', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.old)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2)
    print(json.dumps({name: (row['six'], row['low']) for name, row in report['pairs'].items()}))
