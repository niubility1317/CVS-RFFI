"""Validate the complete blind W24 matrix before opening official LoRa labels."""
import argparse
from pathlib import Path
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .data import label_ids, packet_indices, recording
from .w24_formal import FULL, LIMIT, ROUTE, ROUTES, SEEDS, TARGET, VIEWS, cases, evaluation_modes, evaluation_seeds, receipt


DRAWS = 1600
BOOTSTRAP_SEED = 2026112587
MEASURES = (*VIEWS, 'six_scene')


def load_frozen(predictions, ungated=False):
    records, activity = {route: {} for route in ROUTES}, {route: {} for route in ROUTES}
    for case, route in cases().items():
        part = predictions / case
        expected = receipt(case, ungated)
        n_seeds, n_modes = len(expected['seeds']), len(expected['modes'])
        if read(part / 'complete.json') != expected:
            raise ValueError(f'Incomplete or mismatched W24 shard {case}')
        by_view = read(part / 'activity.json')
        if set(by_view) != {f'{rx}/{v}' for rx in expected['receivers'] for v in VIEWS}:
            raise ValueError(f'Incomplete W24 activity for {case}')
        for rx in expected['receivers']:
            with np.load(part / f'{rx}.npz', allow_pickle=False) as z:
                if set(z.files) != set(VIEWS):
                    raise ValueError(f'Malformed W24 view keys for {case}/{rx}')
                rows = np.stack([z[v] for v in VIEWS])
            if (rows.shape != (len(VIEWS), n_seeds, n_modes, LIMIT)
                    or rows.dtype != np.int16 or not ((0 <= rows) & (rows < 10)).all()):
                raise ValueError(f'Malformed W24 class decisions for {case}/{rx}')
            for j, view in enumerate(VIEWS):
                count = np.asarray(by_view[f'{rx}/{view}'])
                changed = (rows[j, :, 1:] != rows[j, :, :1]).sum(-1)
                if (count.shape != (n_modes-1,) or count.dtype.kind not in 'iu'
                        or not ((0 <= count) & (count <= LIMIT)).all() or (changed > count).any()):
                    raise ValueError(f'Changed W24 labels exceed actual activity for {case}/{rx}/{view}')
                if ungated and count[-1] != LIMIT:
                    raise ValueError(f'Incomplete ungated recovery coverage for {case}/{rx}/{view}')
            records[route][rx] = rows
        activity[route].update(by_view)
    matrices = {route: np.stack([records[route][r] for r in TARGET]) for route in ROUTES}
    if not np.array_equal(matrices[FULL][:, 0, :, 0], matrices[ROUTE][:, 0, :, 0]):
        raise ValueError('Paired direct clean predictions differ across the two routes')
    return matrices, activity


def bootstrap_counts(delta):
    # delta is [receiver, view, reader seed, physical packet]. Each draw
    # samples packets within a fixed receiver and reuses those indices across
    # all seven views and three seeds.
    nrx, nv, ns, n = delta.shape
    combined = np.concatenate((delta, delta.sum(2, keepdims=True)), axis=2)
    combined = np.concatenate((combined, combined[:, 1:].sum(1, keepdims=True)), axis=1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = np.full(n, 1/n)
    total = np.zeros((DRAWS, (nv + 1)*(ns + 1)), dtype=np.float32)
    per_rx = []
    for arr in combined:
        indices = rng.multinomial(n, weights, size=DRAWS).astype(np.float32)
        draw = indices @ arr.reshape(-1, n).T.astype(np.float32)
        total += draw
        per_rx.append(np.quantile(draw, [.025, .975], axis=0).T.reshape(nv+1, ns+1, 2))
    return (np.quantile(total, [.025, .975], axis=0).T.reshape(nv+1, ns+1, 2),
            np.stack(per_rx))


def readout(correct, seeds=SEEDS):
    # correct is [receiver, view, reader seed, physical packet].
    counts = correct.sum(-1)
    result = {}
    for j, seed in enumerate((*seeds, 'pooled')):
        cells = counts[:, :, j] if j < len(seeds) else counts.sum(2)
        denom = LIMIT * (1 if j < len(seeds) else len(seeds))
        views = cells.sum(0)
        result[str(seed)] = dict(
            correct_by_scene=dict(zip(VIEWS, views.tolist(), strict=True)),
            accuracy_by_scene=dict(zip(VIEWS, (views/(len(TARGET)*denom)).tolist(), strict=True)),
            six_correct=int(views[1:].sum()), six_accuracy=float(views[1:].sum()/(len(TARGET)*denom*6)),
            by_receiver={rx: dict(zip(VIEWS, cells[i].tolist(), strict=True)) for i, rx in enumerate(TARGET)})
    return result


def paired(delta, seeds=SEEDS):
    intervals, rx_intervals = bootstrap_counts(delta)
    result = {}
    for j, seed in enumerate((*seeds, 'pooled')):
        block = delta[:, :, j:j+1] if j < len(seeds) else delta
        summary = {}
        for i, group in enumerate(MEASURES):
            d = block[:, i:i+1] if i < len(VIEWS) else block[:, 1:]
            pairs = d.sum((1, 2, 3))
            summary[group] = dict(net=int(pairs.sum()), repairs=int((d>0).sum()), harms=int((d<0).sum()),
                CI95_net=intervals[i, j].tolist(),
                net_by_receiver={rx: dict(net=int(pairs[k]), CI95_net=rx_intervals[k, i, j].tolist())
                                 for k, rx in enumerate(TARGET)})
        result[str(seed)] = summary
    return result


def acceptance(results, ungated=False):
    seeds = evaluation_seeds(ungated)
    wide = evaluation_modes(FULL, ungated)[-1]
    full = results[FULL][wide]['pooled']
    if ungated:
        checks = dict(full_six_at_least_87=full['six_correct'] >= 156600,
                      full_low_urban_at_least_80=full['correct_by_scene']['practical_low_urban'] >= 24000)
    else:
        checks = dict(full_six_at_least_67=full['six_correct'] >= 361800,
                      full_low_urban_at_least_53=full['correct_by_scene']['practical_low_urban'] >= 47700,
                      full_clean_at_least_64490=full['correct_by_scene']['clean'] >= 64490)
    for route in ROUTES:
        names = evaluation_modes(route, ungated)
        current = results[route]
        for baseline in names[:-1]:
            checks[f'{route}/six_vs_{baseline}'] = current[wide]['pooled']['six_correct'] > current[baseline]['pooled']['six_correct']
            checks[f'{route}/clean_vs_{baseline}'] = all(
                current[wide][str(s)]['correct_by_scene']['clean'] >= current[baseline][str(s)]['correct_by_scene']['clean']
                for s in seeds)
    gap = full['accuracy_by_scene']['clean'] - full['six_accuracy']
    return dict(checks=checks, all_numeric_checks_pass=all(checks.values()),
                full_clean_minus_six=gap, additional_gap_at_most_3pp=gap <= .03)


def score(predictions, dataset, output, ungated=False):
    matrices, activity = load_frozen(predictions, ungated)
    seeds = evaluation_seeds(ungated)
    output.mkdir(parents=True, exist_ok=False)
    truth = []
    with zipfile.ZipFile(dataset / 'multiple_receiver_test.zip') as archive:
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', output) as capture:
                truth.append(label_ids(capture, LIMIT)[packet_indices(LIMIT, i)])
    labels = np.asarray(truth)[:, None, None, None, :]
    result, contrasts = {}, {}
    for route in ROUTES:
        modes = evaluation_modes(route, ungated)
        hit = matrices[route] == labels
        result[route] = {m:readout(hit[:, :, :, i], seeds) for i,m in enumerate(modes)}
        candidate = hit[:, :, :, -1].astype(np.int16)
        contrasts[route] = {m:paired(candidate - hit[:, :, :, i], seeds) for i,m in enumerate(modes[:-1])}
    write(output / 'scores.json', dict(status='ANALYZED', truth_last=True, seeds=list(seeds),
        n_physical=len(TARGET)*LIMIT, n_six=len(TARGET)*LIMIT*len(seeds)*6, bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_draws=DRAWS, direct_clean_parity='exact', readouts=result, paired=contrasts,
        acceptance=acceptance(result, ungated), activity=activity))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('predictions','dataset','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--ungated', action='store_true')
    score(**vars(parser.parse_args()))
