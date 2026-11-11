"""Post-score same-packet comparison on matching raw and DURA inputs."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_branch_training.plan import VIEWS
from experiments.pamf_paper_completion.plan import RUN as PAPER_RUN
from experiments.pamf_repl_completion.plan import RUN as PL_RUN
from experiments.pamf_repl_view_shuffle.plan import RUN as TRI_RUN


TASKS = ('cross_rx', 'cross_both')
READOUTS = ('DURA-LU', 'DURA-L', 'raw')


def frozen_row(entry, seed, runs, candidate_input):
    prediction, scores = (Path(entry[key]) for key in ('predictions', 'scores'))
    marker = read(prediction / 'complete.json')
    if (marker['status'] != 'ALL_PREDICTIONS_COMPLETE' or marker['truth_read'] is not False
            or marker['query_fit'] is not False):
        raise ValueError(f'Not a frozen truth-blind prediction: {prediction}')
    input_config = read(runs / marker['run_id'] / 'configs' / f'predict__{marker["row_id"]}.json')
    if (input_config['restored_caches'] != candidate_input['restored_caches']
            or input_config['classes'] != candidate_input['classes']
            or any(input_config['tests'][task]['views_root'] != candidate_input['tests'][task]['views_root']
                   for task in TASKS)):
        raise ValueError(f'Classifier inputs or DURA caches differ: {prediction}')
    scored = read(scores / 'scores.json')
    if (read(scores / 'complete.json')['status'] != 'SCORED_COMPLETE'
            or scored['status'] != 'SCORED' or scored['truth_last'] is not True
            or scored['model_seed'] != seed or scored['row_id'] != marker['row_id']):
        raise ValueError(f'Independent same-seed scores required: {scores}')
    overall = {(r['test'], r['readout'], r['view']): r for r in scored['results']
               if r['dimension'] == 'overall'}
    return prediction, overall


def tri_entry(runs, seed):
    row = f'main_split-factorial-ssl-full-permuted-s{seed}'
    tri = read(runs / PAPER_RUN / 'sources.json')[row]
    expected_source = runs / TRI_RUN / 'F' / f'main_split-repl-shuffled_time-3view-s{seed}' / 'source'
    if Path(tri['source_output']) != expected_source or tri['reused'] is not True:
        raise ValueError('The paper DURA row must reuse the original Tri-SIG-TIME source model')
    return row, tri


def prepare_row(root, row):
    sources = read(root / 'sources.json')
    cfg = read(root / 'configs' / f'score__{row}.json')
    seed = cfg['model_seed']
    if tuple(cfg['views']) != VIEWS:
        raise ValueError('Exactly clean and the six practical scenes are required')
    if cfg['protocol_id'] == 'F0':
        _, tri = tri_entry(root.parent, seed)
        suffix = f'F0-s{seed}' if f'baseline_pl-F0-s{seed}' in sources else f's{seed}'
    else:
        suffix = f'{cfg["protocol_id"]}-s{seed}'
        tri = sources[f'tri_sigtime_ssl-{suffix}']
    candidate_input = read(root / 'configs' / f'predict__{row}.json')
    entries = {'candidate': sources[row], 'PL': sources[f'baseline_pl-{suffix}'],
               'Tri-SIG-TIME': tri}
    if f'weak_branch_fixed-s{seed}' in sources:
        entries['Weak-branch-fixed'] = sources[f'weak_branch_fixed-s{seed}']
    taper = f'taper-control-{cfg["protocol_id"]}-s{seed}'
    if taper in sources:
        entries['Taper-original-guard'] = sources[taper]
    day_physics = f'day-physics-{cfg["protocol_id"]}-s{seed}'
    if day_physics in sources:
        entries['Day-physics'] = sources[day_physics]
    branch_certified = f'branch-certified-{cfg["protocol_id"]}-s{seed}'
    if branch_certified in sources:
        entries['Branch-certified'] = sources[branch_certified]
    fused_physics = f'fused-physics-{cfg["protocol_id"]}-s{seed}'
    if fused_physics in sources:
        entries['Fused-physics'] = sources[fused_physics]
    for arm in ('contrast', 'joint'):
        packet = f'packet-{arm}-{cfg["protocol_id"]}-s{seed}'
        if packet in sources:
            entries[f'Packet-{arm}'] = sources[packet]
    if f'baseline_sup-s{seed}' in sources:
        entries['Sup'] = sources[f'baseline_sup-s{seed}']
    return cfg, {name: frozen_row(entry, seed, root.parent, candidate_input) for name, entry in entries.items()}


def paired_counts(a, b, mask):
    n = int(mask.sum())
    correct_a, correct_b = int((a & mask).sum()), int((b & mask).sum())
    fixed, broken = int((a & ~b & mask).sum()), int((~a & b & mask).sum())
    return dict(count=n, candidate_accuracy=100 * correct_a / n, baseline_accuracy=100 * correct_b / n,
                fixed=fixed, broken=broken, net=fixed - broken, delta_pp=100 * (fixed - broken) / n)


def compare_row(cfg, entries):
    populations = {}
    for task in TASKS:
        ids, predictions = None, {}
        for name, (path, _) in entries.items():
            with np.load(path / task / 'predictions.npz', allow_pickle=False) as data:
                if ids is None:
                    ids = data['ids'].copy()
                    if len(np.unique(ids)) != len(ids):
                        raise ValueError(f'Duplicate physical ID in {task}')
                elif not np.array_equal(ids, data['ids']):
                    raise ValueError(f'Physical IDs differ in {task} for {name}')
                predictions[name] = {(route, view): data[f'{route}__{view}'].copy()
                                     for route in READOUTS for view in VIEWS}
        populations[task] = ids, predictions
    if np.intersect1d(populations[TASKS[0]][0], populations[TASKS[1]][0]).size:
        raise ValueError('Combined target populations must be disjoint')

    correct, groups = {}, {}
    for task, (ids, predictions) in populations.items():
        truth = read(cfg['tests'][task]['truth'])
        meta = [truth[str(i)] for i in ids]
        labels = np.asarray([m['label'] for m in meta], dtype=np.int64)
        groups[task] = {'transmitter': np.asarray(cfg['classes'])[labels],
                        'receiver': np.asarray([str(m['receiver']) for m in meta]),
                        'date': np.asarray([str(m['day']) for m in meta])}
        correct[task] = {name: {cell: values == labels for cell, values in views.items()}
                         for name, views in predictions.items()}
        for name, (_, overall) in entries.items():
            for (route, view), hits in correct[task][name].items():
                score = overall[task, route, view]
                if score['count'] != len(ids) or int(hits.sum()) != round(score['accuracy'] * len(ids)):
                    raise ValueError(f'Prediction/truth counts disagree with scores: {name} {task} {route} {view}')

    combined = 'combined_target'
    correct[combined] = {name: {c: np.concatenate([correct[t][name][c] for t in TASKS]) for c in correct[TASKS[0]][name]}
                         for name in entries}
    groups[combined] = {dim: np.concatenate([groups[t][dim] for t in TASKS]) for dim in groups[TASKS[0]]}
    rows = []
    for task, by_model in correct.items():
        masks = [('overall', 'all', np.ones(len(by_model['candidate']['raw', 'clean']), dtype=bool))]
        for dim, values in groups[task].items():
            masks.extend((dim, str(value), values == value) for value in np.unique(values))
        for baseline in (name for name in entries if name != 'candidate'):
            for (route, view), a in by_model['candidate'].items():
                b = by_model[baseline][route, view]
                for dim, stratum, mask in masks:
                    rows.append(dict(seed=cfg['model_seed'], baseline=baseline, task=task, readout=route, view=view,
                                     dimension=dim, stratum=stratum, **paired_counts(a, b, mask)))
    return rows


def summarize(prepared):
    seeds = [cfg['model_seed'] for cfg, _ in prepared]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Expected one candidate per distinct model seed')
    rows = [r for cfg, entries in prepared for r in compare_row(cfg, entries)]
    summary = []
    for baseline in dict.fromkeys(r['baseline'] for r in rows):
        main = [r for r in rows if r['baseline'] == baseline and r['task'] == 'combined_target'
                and r['dimension'] == 'overall']
        for route in READOUTS:
            for view in (*VIEWS, 'six_practical_mean'):
                matching = [[r for r in main if r['seed'] == seed and r['readout'] == route and
                             (r['view'] in VIEWS[1:] if view == 'six_practical_mean' else r['view'] == view)]
                            for seed in seeds]
                differences = [mean(r['delta_pp'] for r in group) for group in matching]
                summary.append(dict(baseline=baseline, readout=route, view=view, paired_delta_pp=differences,
                                    candidate_mean_accuracy=mean(mean(r['candidate_accuracy'] for r in g) for g in matching),
                                    baseline_mean_accuracy=mean(mean(r['baseline_accuracy'] for r in g) for g in matching),
                                    mean_delta_pp=mean(differences), min_delta_pp=min(differences),
                                    sd_delta_pp=stdev(differences) if len(seeds) > 1 else None))
    return dict(status='PAIRED_COMPLETE', seeds=seeds, primary_readout='DURA-LU',
                truth_last=True, rows=rows, summary=summary)


def run(spec):
    prepared = [prepare_row(Path(root), row) for root, row in spec]
    protocols = list(dict.fromkeys(cfg['protocol_id'] for cfg, _ in prepared))
    if len(protocols) == 1:
        return summarize(prepared)
    return dict(status='PAIRED_BY_PROTOCOL_COMPLETE', protocols={pid: summarize(
        [(cfg, entries) for cfg, entries in prepared if cfg['protocol_id'] == pid]) for pid in protocols})


def compare_baselines(runs):
    prepared = []
    for seed in (2026092701, 2026092702, 2026092703):
        row, tri = tri_entry(runs, seed)
        folder = runs / PL_RUN / 'readouts' / f'F0-ordinary_PL-s{seed}'
        pl = dict(predictions=folder / 'predictions', scores=folder / 'scores')
        config_folder = runs / PAPER_RUN / 'configs'
        cfg = read(config_folder / f'score__{row}.json')
        inputs = read(config_folder / f'predict__{row}.json')
        prepared.append((cfg, {name: frozen_row(entry, seed, runs, inputs)
                               for name, entry in {'candidate': tri, 'PL': pl}.items()}))
    return dict(summarize(prepared), candidate='Tri-SIG-TIME')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--row', action='append', nargs=2, metavar=('RUN_ROOT', 'ROW_ID'))
    target.add_argument('--baseline-reference', type=Path, metavar='RUNS_ROOT')
    args = parser.parse_args()
    result = run(args.row) if args.row else compare_baselines(args.baseline_reference)
    print(json.dumps(result, allow_nan=False))
