"""Recount source-only aperture counterfactuals after the independent scorer."""

import argparse
import json
from pathlib import Path

import numpy as np


RUN = '20261127-diagnostic-lora-w24-native-aperture-factor-s2026112559-r02'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def audit(root, source):
    if read(root/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected source truth-last completion')
    out = root/'full_native'
    frozen = read(out/'source-predictions.json')
    scored = read(out/'source-scores.json')
    if (frozen['status'] != 'PREDICTIONS_FROZEN' or frozen['source_truth_reads'] or frozen['target_reads']
            or scored['status'] != 'ANALYZED' or not scored['source_truth_last'] or scored['target_reads']
            or frozen['waveform'] != scored['waveform'] or not frozen['waveform']['oracle_only']):
        raise ValueError('Invalid diagnostic freeze or scoring proof')
    ids, views, modes = (frozen[k] for k in ('physical_ids', 'views', 'modes'))
    with (np.load(out/'predictions.npz', allow_pickle=False) as p,
          np.load(out/'aperture-measurements.npz', allow_pickle=False) as m):
        preds = np.stack([p[v] for v in views])
        detail = np.stack([m[v] for v in views])
    if (preds.shape != (7, 11, 300) or detail.shape != (7, 300, 12)
            or len(set(ids)) != 300 or not np.isfinite(detail).all()):
        raise ValueError('Expected complete frozen fixed matrix and physical measurements')
    with np.load(source/'V_metadata.npz', allow_pickle=False) as meta:
        source_ids = meta['ids'].tolist()
        if source_ids != read(source/'contract.json')['role_ids']['V']:
            raise ValueError('Mismatched truth-side physical IDs')
        order = {key: i for i, key in enumerate(source_ids)}
        positions = [order[k] for k in ids]
        rx, tx = meta['rx_i'][positions], meta['y'][positions]
    hits = (preds == tx).astype(np.int16)
    counts = dict(zip(modes, hits.sum(-1).T.tolist(), strict=True))
    if any(counts[name] != [scored['scores'][name]['by_scene'][v] for v in views] for name in modes):
        raise ValueError('Independent score did not reconcile')
    rows = {name: hits[:, j] for j, name in enumerate(modes)}

    def paired(delta):
        all_rows = {view: (int((x > 0).sum()), int((x < 0).sum()), int(x.sum()))
                    for view, x in zip((*views, 'six'), (*delta, delta[1:]), strict=True)}
        groups = {}
        for k, key in (('rx', rx), ('tx', tx)):
            groups[k] = {int(g): (int(delta[1:, key == g].sum()), int(delta[-1, key == g].sum()),
                                    int(delta[0, key == g].sum())) for g in np.unique(key)}
        return dict(all=all_rows, **groups)

    pairs = {}
    for name in scored['paired_contrasts']:
        a, b = name.split('_vs_')
        pairs[name] = paired(rows[a] - rows[b])
    rng = np.random.default_rng(2026112560)
    groups = [np.flatnonzero(rx == g) for g in np.unique(rx)]
    draws = np.concatenate([rng.choice(g, (2000, len(g))) for g in groups], axis=1)
    factors = {}
    for mode, center, edge, pivot in (('learned', 'band_center', 'band_edges', 'band_half'),
                                       ('equal', 'equal_center', 'equal_edges', 'equal_both'),
                                       ('exact', 'exact_center', 'exact_edges', 'exact_both')):
        d = rows[pivot] + rows['control_half'] - rows[center] - rows[edge]
        totals = (d[1:].sum(0), d[-1])
        factors[mode] = dict(six=int(totals[0].sum()), low=int(totals[1].sum()),
                             ci95=[np.quantile(a[draws].sum(1), [.025, .975]).tolist() for a in totals])
    return dict(RUN=RUN, scores=scored, counts=counts, pairs=pairs, interactions=factors,
                aperture_check={v: np.median(d, axis=0).reshape(4, 3).tolist()
                                for v, d in zip(views, detail, strict=True)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source'):
        parser.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(audit(**vars(parser.parse_args()))))
