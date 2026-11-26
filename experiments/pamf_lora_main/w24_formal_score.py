"""Validate the complete blind W24 matrix before opening official LoRa labels."""
import argparse
from pathlib import Path
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .data import label_ids, packet_indices, recording
from .w24_formal import (
    FULL, LIMIT, ROUTE, ROUTES, SEEDS, TARGET, VIEWS, cases, evaluation_modes,
    evaluation_seeds, receipt,
)


DRAWS = 1600
BOOTSTRAP_SEED = 2026112587
MEASURES = (*VIEWS, 'six_scene')


def load_frozen(predictions, ungated=False, outer_objective=False, third_prior=False, *, observed_plans=None):
    ungated = ungated or outer_objective or bool(third_prior)
    plans = observed_plans if observed_plans is not None else {
        case: receipt(case, ungated, outer_objective, third_prior) for case in cases(third_prior)
    }
    routes = tuple(dict.fromkeys(p['processing_route'] for p in plans.values()))
    records, activity = {route: {} for route in routes}, {route: {} for route in routes}
    for case, expected in plans.items():
        part = predictions / case
        route = expected['processing_route']
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
                if (third_prior or observed_plans is not None) and not (count[-3:] == LIMIT).all():
                    raise ValueError(f'Incomplete matched two-/three-prior coverage for {case}/{rx}/{view}')
                if outer_objective and count[-2] != LIMIT:
                    raise ValueError(f'Incomplete archived all-packet comparator for {case}/{rx}/{view}')
            records[route][rx] = rows
        activity[route].update(by_view)
    matrices = {route: np.stack([records[route][r] for r in TARGET]) for route in routes}
    if len(routes) == 1:
        return matrices, activity
    if not np.array_equal(matrices[FULL][:, 0, :, 0], matrices[ROUTE][:, 0, :, 0]):
        raise ValueError('Paired direct clean predictions differ across the two routes')
    if third_prior and not np.array_equal(matrices[FULL][:, 0, :, -2:], matrices[ROUTE][:, 0, :, -2:]):
        raise ValueError('Same clean packets produced different matched candidate predictions across routes')
    if outer_objective and not np.array_equal(matrices[FULL][:, 0, :, -1], matrices[ROUTE][:, 0, :, -1]):
        raise ValueError('Same clean packets produced different candidate predictions across routes')
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


def acceptance(results, ungated=False, outer_objective=False, third_prior=False):
    ungated = ungated or outer_objective or bool(third_prior)
    seeds = evaluation_seeds(ungated)
    wide = evaluation_modes(FULL, ungated, outer_objective, third_prior)[-1]
    full = results[FULL][wide]['pooled']
    if third_prior:
        return full_target_acceptance(full)
    if ungated:
        checks = dict(full_six_at_least_87=full['six_correct'] >= 156600,
                      full_low_urban_at_least_80=full['correct_by_scene']['practical_low_urban'] >= 24000)
    else:
        checks = dict(full_six_at_least_67=full['six_correct'] >= 361800,
                      full_low_urban_at_least_53=full['correct_by_scene']['practical_low_urban'] >= 47700,
                      full_clean_at_least_64490=full['correct_by_scene']['clean'] >= 64490)
    for route in ROUTES:
        names = evaluation_modes(route, ungated, outer_objective, third_prior)
        current = results[route]
        for baseline in names[:-1]:
            checks[f'{route}/six_vs_{baseline}'] = current[wide]['pooled']['six_correct'] > current[baseline]['pooled']['six_correct']
            checks[f'{route}/clean_vs_{baseline}'] = all(
                current[wide][str(s)]['correct_by_scene']['clean'] >= current[baseline][str(s)]['correct_by_scene']['clean']
                for s in seeds)
    gap = full['accuracy_by_scene']['clean'] - full['six_accuracy']
    return dict(checks=checks, all_numeric_checks_pass=all(checks.values()),
                full_clean_minus_six=gap, additional_gap_at_most_3pp=gap <= .03)


def full_target_acceptance(full):
    clean = full['correct_by_scene']['clean']
    low = full['correct_by_scene']['practical_low_urban']
    gaps = dict(six=162000-full['six_correct'], low_urban=25500-low, clean=28566-clean)
    checks = {f'full_{name}_goal': gap <= 0 for name, gap in gaps.items()}
    return dict(checks=checks, all_numeric_checks_pass=all(checks.values()), full_correct_count_gaps=gaps)


def score_observed_dose(predictions, dataset, output):
    from .w24_observed_dose import CANDIDATE, FULL_CASES, formal_receipt

    plans = {case: formal_receipt(case) for case in FULL_CASES}
    matrices, activity = load_frozen(predictions, ungated=True, observed_plans=plans)
    modes = {FULL: plans[FULL_CASES[0]]['modes']}
    score_matrix(matrices, activity, modes, dataset, output, evaluation_seeds(True),
                 lambda results: full_target_acceptance(results[FULL][CANDIDATE]['pooled']))


def score(predictions, dataset, output, ungated=False, outer_objective=False, third_prior=False):
    ungated = ungated or outer_objective or bool(third_prior)
    matrices, activity = load_frozen(predictions, ungated, outer_objective, third_prior)
    seeds = evaluation_seeds(ungated)
    modes = {route: evaluation_modes(route, ungated, outer_objective, third_prior) for route in matrices}
    score_matrix(matrices, activity, modes, dataset, output, seeds,
                 lambda results: acceptance(results, ungated, outer_objective, third_prior))


def score_matrix(matrices, activity, modes_by_route, dataset, output, seeds, verdict):
    output.mkdir(parents=True, exist_ok=False)
    truth = []
    with zipfile.ZipFile(dataset / 'multiple_receiver_test.zip') as archive:
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', output) as capture:
                truth.append(label_ids(capture, LIMIT)[packet_indices(LIMIT, i)])
    labels = np.asarray(truth)[:, None, None, None, :]
    result, contrasts = {}, {}
    for route in matrices:
        modes = modes_by_route[route]
        hit = matrices[route] == labels
        result[route] = {m:readout(hit[:, :, :, i], seeds) for i,m in enumerate(modes)}
        candidate = hit[:, :, :, -1].astype(np.int16)
        contrasts[route] = {m:paired(candidate - hit[:, :, :, i], seeds) for i,m in enumerate(modes[:-1])}
    write(output / 'scores.json', dict(status='ANALYZED', truth_last=True, seeds=list(seeds),
        n_physical=len(TARGET)*LIMIT, n_six=len(TARGET)*LIMIT*len(seeds)*6, bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_draws=DRAWS, direct_clean_parity='exact' if len(matrices) == 2 else 'archive-verified',
        readouts=result, paired=contrasts,
        acceptance=verdict(result), activity=activity))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('predictions','dataset','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--ungated', action='store_true')
    parser.add_argument('--outer-objective', action='store_true')
    matched = parser.add_mutually_exclusive_group()
    matched.add_argument('--third-prior', action='store_true')
    matched.add_argument('--direction', dest='third_prior', action='store_const', const='direction')
    matched.add_argument('--readout', dest='third_prior', action='store_const', const='readout')
    matched.add_argument('--robust', dest='third_prior', action='store_const', const='robust')
    matched.add_argument('--peer', dest='third_prior', action='store_const', const='peer')
    matched.add_argument('--band-tune', dest='third_prior', action='store_const', const='band-tune')
    matched.add_argument('--band-adapter', dest='third_prior', action='store_const', const='band-adapter')
    matched.add_argument('--band-refine', dest='third_prior', action='store_const', const='band-refine')
    matched.add_argument('--white-corruption', dest='third_prior', action='store_const', const='white-corruption')
    score(**vars(parser.parse_args()))
