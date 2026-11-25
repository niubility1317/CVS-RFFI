"""Read LoRa truth only after the selected candidate's entire target matrix freezes."""

import argparse
from io import BytesIO
from json import loads
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.data import TARGET, VIEWS, label_ids, packet_indices, recording
from experiments.pamf_lora_main.full_packet_data import LIMIT
from experiments.pamf_lora_main.residual_dura_score import prediction
from experiments.pamf_lora_main.residual_reference_replay import completion as residual_replay
from experiments.pamf_lora_main import dura_capacity_target, weak_guard_target as guard
from experiments.pamf_lora_main.weak_guard_target import ROUTES, SEEDS, matrix
from tools.wisig_paired_main_uncertainty import summarize


REPLICATES = 1600
TESTS = {replay.MODES[1]: replay for replay in (guard, dura_capacity_target)}


def collect(receipt, config, output, *, replay=guard, artifacts=()):
    receipt = read(receipt)
    root = PurePosixPath(receipt['run_root'])
    if root.name != replay.RUN or receipt['archive_transfer'] != 'VERIFIED':
        raise ValueError('Expected the registered immutable LoRa target run')
    conn = ['ssh', '-F', str(config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-T', 'N607']
    state = subprocess.check_output([*conn, 'cat ' + shlex.quote(str(root / 'state.json'))], timeout=20)
    if loads(state)['status'] != 'PREDICTIONS_FROZEN':
        raise ValueError('Both routes must freeze before collection')
    reports = {'smoke.json', 'predictions/complete.json', *artifacts, *(f'{name}-complete.json' for name in matrix())}
    expected = reports | {'predictions/' + name for name in replay.expected_files()}
    command = f'tar -czf - -C {shlex.quote(str(root))} predictions smoke.json ' + ' '.join(sorted(reports - {
        'predictions/complete.json', 'smoke.json'}))
    payload = subprocess.check_output([*conn, command], timeout=180)
    with tarfile.open(fileobj=BytesIO(payload), mode='r:gz') as archive:
        files = {m.name: m for m in archive if m.isfile()}
        if set(files) != expected:
            raise ValueError('Transferred LoRa prediction archive disagrees with the target matrix')
        output.mkdir(parents=True, exist_ok=False)
        for name in files:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(files[name]) as source, path.open('xb') as dest:
                dest.write(source.read())


def predictions(folder, mode, count=LIMIT):
    return np.stack([[prediction(folder / f'{rx}__{view}__{mode}.npy', count) for rx in TARGET] for view in VIEWS])


def frozen_rows(output, full, residual, residual_old, *, replay=guard, previous_guard=None):
    p = output / 'predictions'
    if read(p / 'complete.json') != replay.completion():
        raise ValueError('Frozen two-route submission does not match the selected candidate')
    if {a.relative_to(p).as_posix() for a in p.rglob('*.npy')} != replay.expected_files():
        raise ValueError('Target prediction index is incomplete')
    if previous_guard is not None:
        previous_guard = previous_guard / 'predictions'
        if replay is guard or read(previous_guard / 'complete.json') != guard.completion():
            raise ValueError('The optional guard control requires its own prior frozen target matrix')
    if read(residual / 'predictions' / 'complete.json') != residual_replay():
        raise ValueError('Residual frozen baseline reference differs')
    from experiments.pamf_lora_main import residual_dura as old
    expected_old = dict(status='PREDICTIONS_FROZEN', seed=old.SEED, receivers=list(TARGET), per_receiver=LIMIT,
                        views=list(VIEWS), modes=list(old.MODES), practical=old.ROUTE, target_truth_read=False)
    if read(residual_old / 'predictions' / 'complete.json') != expected_old:
        raise ValueError('Unexpected original residual seed02')

    all_routes = {}
    candidate = replay.MODES[1]
    for route in ROUTES:
        all_seeds = []
        for seed in SEEDS:
            if route == 'full':
                baseline = full / 'predictions' / str(seed)
                meta = read(baseline / 'complete.json')
                if (meta['status'] != 'PREDICTIONS_FROZEN' or meta['seed'] != seed or meta['target_truth_read']
                    or meta['receivers'] != list(TARGET) or meta['views'] != list(VIEWS)
                    or meta['per_receiver'] != LIMIT
                    or not {'direct', 'coherent_half', 'complex_response_half'} <= set(meta['modes'])):
                    raise ValueError('Unexpected full practical frozen baseline')
                mode, best = 'coherent_half', ('complex_response_half',)
            else:
                baseline = (residual_old / 'predictions' if seed == old.SEED
                            else residual / 'predictions' / str(seed))
                mode, best = 'dura_half', ()
            prior_direct = predictions(baseline, 'direct')
            current = p / route / str(seed)
            if not np.array_equal(prior_direct, predictions(current, 'direct')):
                raise ValueError(f'Tri-set direct baseline decisions changed for {route} seed {seed}')
            row = dict(direct=prior_direct, dura_half=predictions(baseline, mode),
                       **{name: predictions(baseline, name) for name in best})
            row[candidate] = predictions(current, candidate)
            if previous_guard is not None:
                previous = previous_guard / route / str(seed)
                if not np.array_equal(prior_direct, predictions(previous, 'direct')):
                    raise ValueError(f'Historical guard direct predictions differ for {route} seed {seed}')
                row[guard.MODES[1]] = predictions(previous, guard.MODES[1])
            all_seeds.append(row)
        all_routes[route] = {mode: np.stack([row[mode] for row in all_seeds]) for mode in all_seeds[0]}
    return all_routes


def receiver_pairs(hits, mode, seeds, count):
    receivers = np.repeat(TARGET, count)
    rng = np.random.default_rng(20261124)
    candidate = hits[mode].reshape(len(seeds), len(VIEWS), -1)
    selections = [('all_seeds', slice(None)),
                  *((str(seed), slice(i, i + 1)) for i, seed in enumerate(seeds))]
    pairs = {}
    for name, reference in hits.items():
        if name != mode:
            baseline = reference.reshape(candidate.shape)
            pairs[name] = {key: summarize(baseline[select], candidate[select], receivers, VIEWS, rng,
                                          labels=('baseline', 'candidate')) for key, select in selections}
    return pairs


def score_arrays(rows, truth, mode='guard_half', *, detailed_pairing=False, seeds=None):
    seeds = SEEDS if seeds is None else seeds
    count = truth.shape[-1]
    expected = (len(seeds), len(VIEWS), len(TARGET), count)
    if truth.shape != expected[2:]:
        raise ValueError('Truth rows do not match the physical receiver matrix')
    records, paired, aggregate = [], {}, {}
    by_route = {}
    for route, choices in rows.items():
        if any(decisions.shape != expected for decisions in choices.values()):
            raise ValueError(f'Prediction rows do not match the registered matrix for {route}')
        hits = {name: decisions == truth for name, decisions in choices.items()}
        if detailed_pairing:
            by_route[route] = receiver_pairs(hits, mode, seeds, count)
        candidate = hits[mode]
        baseline_names = tuple(name for name in hits if name != mode)
        for i, seed in enumerate(seeds):
            for j, view in enumerate(VIEWS):
                for receiver, select in [('ALL', slice(None)), *zip(TARGET, range(len(TARGET)), strict=True)]:
                    subject = candidate[i, j, select]
                    compared = {name: dict(repairs=int((subject & ~hits[name][i, j, select]).sum()),
                                           harms=int((~subject & hits[name][i, j, select]).sum()))
                                for name in baseline_names}
                    records.append(dict(route=route, seed=seed, view=view, receiver=receiver, count=subject.size,
                                        correct={name: int(a[i, j, select].sum()) for name, a in hits.items()},
                                        **{f'{mode}_vs': compared}))
        aggregate[route] = {
            name: dict(clean=int(a[:, 0].sum()), six_scene=int(a[:, 1:].sum()), low_urban=int(a[:, -1].sum()))
            for name, a in hits.items()
        }
        paired[route] = {}
        for name in baseline_names:
            reference = hits[name]
            repaired, harmed = candidate & ~reference, reference & ~candidate
            difference = repaired.sum(axis=0, dtype=np.int16) - harmed.sum(axis=0, dtype=np.int16)
            totals = np.concatenate((difference, difference[1:].sum(axis=0, keepdims=True)))
            rng = np.random.default_rng(20261124)
            pools = np.zeros((len(totals), REPLICATES), dtype=np.int64)
            for i in range(len(TARGET)):
                index = rng.integers(0, count, size=(REPLICATES, count))
                for j in range(len(totals)):
                    pools[j] += totals[j, i][index].sum(axis=1)
            paired[route][name] = {}
            for j, view in enumerate((*VIEWS, 'six_scene')):
                selection = j if j < len(VIEWS) else slice(1, None)
                paired[route][name][view] = dict(
                    net=int(totals[j].sum()), repairs=int(repaired[:, selection].sum()),
                    harms=int(harmed[:, selection].sum()), ci95=np.quantile(pools[j], (.025, .975)).tolist())
    scores = dict(aggregate=aggregate, paired=paired, rows=records)
    if detailed_pairing:
        scores['paired_by_seed_and_receiver'] = by_route
    return scores


def dura_capacity_acceptance(aggregate):
    candidate = dura_capacity_target.MODES[1]
    full = aggregate['full'][candidate]
    minima = {'six_scene': 361800, 'low_urban': 47700, 'clean': 64490}
    checks = {name: dict(correct=full[name], minimum=minimum, passed=full[name] >= minimum)
              for name, minimum in minima.items()}
    by_route = {}
    for route in ROUTES:
        current, prior = (aggregate[route][method] for method in (candidate, 'dura_half'))
        six_gain, clean_gain = (current[key] - prior[key] for key in ('six_scene', 'clean'))
        by_route[route] = dict(six_gain=six_gain, clean_gain=clean_gain, passed=six_gain > 0 and clean_gain >= 0)
    gap = abs(6 * full['clean'] - full['six_scene'])
    return dict(passed=all(item['passed'] for item in (*checks.values(), *by_route.values())),
                full_minima=checks, versus_original_dura=by_route,
                full_clean_gap_pp=100 * gap / 540000, full_clean_gap_within_3pp=gap <= 16200)


def score(output, full, residual, residual_old, dataset, *, replay=guard, previous_guard=None):
    rows = frozen_rows(output, full, residual, residual_old, replay=replay, previous_guard=previous_guard)
    smoke = read(output / 'smoke.json')
    if smoke['status'] != 'PASS' or smoke['target_reads'] != 0 or smoke['seeds'] != list(SEEDS):
        raise ValueError('Missing no-query smoke for the exact test readers')
    truth = []
    with zipfile.ZipFile(dataset / 'multiple_receiver_test.zip') as archive:
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', output) as handle:
                truth.append(label_ids(handle, LIMIT)[packet_indices(LIMIT, i)])
    result = score_arrays(rows, np.stack(truth), replay.MODES[1], detailed_pairing=replay is dura_capacity_target)
    expected_full = dict(clean=64490, six_scene=302413, low_urban=37845)
    expected_residual = dict(clean=64307, six_scene=289162, low_urban=38566)
    expected_original_full = dict(clean=64490, six_scene=301599, low_urban=37596)
    if (result['aggregate']['full']['complex_response_half'] != expected_full
        or result['aggregate']['full']['dura_half'] != expected_original_full
        or result['aggregate']['residual']['dura_half'] != expected_residual):
        raise ValueError('Independent truth scoring did not reproduce the registered matched baselines')
    if replay is dura_capacity_target:
        result['formal_acceptance'] = dura_capacity_acceptance(result['aggregate'])
    write(output / 'scores.json', dict(status='TARGET_SCORED', truth_last=True, bootstrap_replicates=REPLICATES, **result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pull = commands.add_parser('collect')
    evaluate = commands.add_parser('score')
    for cmd, names in (
        (pull, ('receipt', 'config', 'output')),
        (evaluate, ('output', 'full', 'residual', 'residual-old', 'dataset')),
    ):
        for name in names:
            cmd.add_argument('--' + name, required=True, type=Path)
        cmd.add_argument('--candidate', choices=tuple(TESTS), default=guard.MODES[1])
    evaluate.add_argument('--previous-guard', type=Path)
    arguments = vars(parser.parse_args())
    arguments['replay'] = TESTS[arguments.pop('candidate')]
    if arguments.pop('action') == 'collect':
        collect(**arguments)
    else:
        score(**arguments)
