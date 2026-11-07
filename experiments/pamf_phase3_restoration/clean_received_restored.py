"""Fixed-prediction truth-last comparison of clean, received and restored IQ."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


RUN = '20261108-analysis-Restore-CleanReceivedRestored-Main-s2026092701-r01'
MODELS = {
    'joint': ('20261107-phase1-Restore-JointPrior-E30-s2026092701-r01', 'TF_JointPrior',
              '4bba9df3bb2c909dc333dce0cf0102cb641b1682'),
    'projected': ('20261107-phase1-Restore-ProjectedPriorDetail-E30-s2026092701-r01',
                  'TF_ProjectedPriorDetail', '8cf3669c4a5d98f6c4930117952203afd56f2f6e'),
    'mismatch': ('20261107-phase1-Restore-MismatchedPrior-E30-s2026092701-r01',
                 'TF_MismatchedPrior', '14e7315d6da5d6fe242e7568f20c97ade064e8d9'),
}
TASKS = {'cross_rx': 126000, 'cross_both': 42000}
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
STREAMS = ('raw', 'DURA_v1', 'candidate')
F0_COMMIT = '3ff8f4b7e85b5608f5b45350aa18081499a7ef1d'


def joint_counts(truth, clean, received, restored):
    arrays = [np.asarray(x) for x in (truth, clean, received, restored)]
    if not arrays[0].ndim == 1 or any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError('Matched one-dimensional truth and predictions required')
    if any(x.dtype.kind not in 'iu' or not np.isin(x, np.arange(6)).all() for x in arrays):
        raise ValueError('Six-class integer truth and predictions required')
    correct = [x == arrays[0] for x in arrays[1:]]
    code = correct[0].astype(np.int8)*4 + correct[1].astype(np.int8)*2 + correct[2].astype(np.int8)
    counts = np.bincount(code, minlength=8)
    cells = {f'{i:03b}': int(counts[i]) for i in range(8)}
    return dict(count=len(code), cells=cells,
                clean_wrong_received_wrong_restored_correct=cells['001'],
                clean_wrong_received_correct_restored_correct=cells['011'],
                clean_correct_received_correct_restored_wrong=cells['110'])


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_fixed_predictions(root):
    all_predictions, identities, published = {}, [], {}
    baseline = {}
    for name, (run, arm, commit) in MODELS.items():
        row = root/'runs'/run/'rows'/arm
        train = read(row/'training/complete.json')
        if (train['status'], train['code_commit'], train['epoch'], train['updates']) != ('SOURCE_TRAINED', commit, 30, 7410):
            raise ValueError('Original final30 training identity required')
        complete = read(row/'prediction/complete.json')
        if (complete['status'], complete['truth_read'], complete['query_fit']) != ('ALL_PREDICTIONS_COMPLETE', False, False):
            raise ValueError('All candidate predictions must be fixed before truth')
        scored = read(row/'score/complete.json')
        if scored['status'] != 'SCORED_COMPLETE':
            raise ValueError('Previously completed independent truth-last scoring required')
        published[name] = row/'score/scores.json'
        all_predictions[name] = {}
        identifiers = set()
        for task, n in TASKS.items():
            folder = row/'prediction'/task
            marker = read(folder/'complete.json')
            if ((marker['status'], marker['code_commit'], marker['count']) != ('PREDICTIONS_COMPLETE', commit, n)
                    or marker['views'] != list(VIEWS) or marker['readouts'] != list(STREAMS)
                    or marker['truth_read'] or marker['query_fit'] or marker['model_state_updates']):
                raise ValueError('Complete matched physical prediction matrix required')
            if marker['F0_identity'] != train['F0_identity'] or marker['F0_identity']['source_code_commit'] != F0_COMMIT:
                raise ValueError('Same original frozen F0 required')
            identities.append(marker['F0_identity'])
            with np.load(folder/'predictions.npz', allow_pickle=False) as source:
                values = {key: source[key].copy() for key in source.files}
            if set(values) != {'ids'} | {s+'__'+v for s in STREAMS for v in VIEWS}:
                raise ValueError('All original prediction streams required')
            ids = values['ids']
            if ids.shape != (n,) or len(set(ids.tolist())) != n or identifiers.intersection(ids.tolist()):
                raise ValueError('Unique, disjoint original physical records required')
            identifiers.update(ids.tolist())
            for stream in STREAMS:
                for view in VIEWS:
                    p = values[stream+'__'+view]
                    if p.shape != (n,) or p.dtype != np.int64 or not np.isin(p, np.arange(6)).all():
                        raise ValueError('Complete six-class frozen predictions required')
            if task not in baseline:
                baseline[task] = values
            else:
                reference = baseline[task]
                if (not np.array_equal(ids, reference['ids']) or any(
                    not np.array_equal(values[stream+'__'+view], reference[stream+'__'+view])
                    for stream in ('raw', 'DURA_v1') for view in VIEWS)):
                    raise ValueError('Shared physical IDs and raw/DURA predictions required')
            all_predictions[name][task] = values
        if len(identifiers) != 168000:
            raise ValueError('All original Main records required')
    if any(x != identities[0] for x in identities):
        raise ValueError('F0 identity differs across candidates')
    return all_predictions, published, identities[0]


def score(root, commit):
    import fcntl

    root = Path(root)
    out = root/'runs'/RUN
    if out.exists():
        raise FileExistsError('Analysis output already exists; reconcile before retry')
    predictions, score_paths, identity = load_fixed_predictions(root)
    lock = (root/'runs/20261101-phase3-P3Restore-TF-IQ-E30-s2026092701-r01/score.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    published = {name: read(path) for name, path in score_paths.items()}
    if any((score['status'], score['truth_last'], score['target_feedback_forbidden']) !=
           ('SCORED', True, True) for score in published.values()):
        raise ValueError('Previously completed independent truth-last scoring required')
    config = read(root/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/configs/score__main_split-repl-shuffled_time-3view-s2026092701.json')
    rows = []
    for task, n in TASKS.items():
        reference = predictions['joint'][task]
        truth = read(Path(config['tests'][task]['truth']))
        y = np.asarray([truth[str(i)]['label'] for i in reference['ids']], dtype=np.int64)
        receivers = np.asarray([str(truth[str(i)]['receiver']) for i in reference['ids']])
        dates = np.asarray([str(truth[str(i)]['day']) for i in reference['ids']])
        masks = {('overall', 'ALL'): np.ones(n, dtype=bool)}
        for dimension, labels in [('transmitter', y.astype(str)), ('receiver', receivers), ('date', dates)]:
            for value in np.unique(labels):
                masks[dimension, str(value)] = labels == value
        for receiver in np.unique(receivers):
            for date in np.unique(dates):
                mask = (receivers == receiver) & (dates == date)
                if mask.any():
                    masks['receiver_date', receiver+'|'+date] = mask
        methods = [(name, name, 'candidate') for name in MODELS] + [('DURA_v1', 'joint', 'DURA_v1')]
        for name, source_name, stream in methods:
            values = predictions[source_name][task]
            scores = published[source_name]
            by_group = {(q['test'], q['readout'], q['view'], q['dimension'], q['stratum']): q
                        for q in scores['results']}
            by_pair = {(q['test'], q['readout'], q['view'], q['dimension'], q['stratum']): q
                       for q in scores['paired_raw']}
            for view in VIEWS:
                clean, raw, restored = values['raw__clean'], values['raw__'+view], values[stream+'__'+view]
                for (dimension, stratum), mask in masks.items():
                    result = joint_counts(y[mask], clean[mask], raw[mask], restored[mask])
                    cells = result['cells']
                    reference = by_group[task, stream, view, dimension, stratum]
                    raw_score = by_group[task, 'raw', view, dimension, stratum]
                    clean_score = by_group[task, 'raw', 'clean', dimension, stratum]
                    old_correct = sum(v for k, v in cells.items() if k[0] == '1')
                    raw_correct = sum(v for k, v in cells.items() if k[1] == '1')
                    new_correct = sum(v for k, v in cells.items() if k[2] == '1')
                    for q, correct in [(reference, new_correct), (raw_score, raw_correct), (clean_score, old_correct)]:
                        if q['count'] != result['count'] or int(np.trace(np.asarray(q['confusion']))) != correct:
                            raise ValueError('Existing truth-last score does not match triple counts')
                    paired = by_pair[task, stream, view, dimension, stratum]
                    repairs = cells['001']+cells['101']
                    damage = cells['010']+cells['110']
                    if (paired['repairs'], paired['damage']) != (repairs, damage):
                        raise ValueError('Existing raw/restored pair does not match triple counts')
                    rows.append(dict(method=name, task=task, view=view, dimension=dimension,
                                     stratum=stratum, **result))
    out.mkdir()
    result = dict(status='ANALYZED', run_id=RUN, code_commit=commit, F0_identity=identity,
                  model_runs=MODELS, tasks=TASKS, views=VIEWS, rows=rows,
                  predictions_fixed_before_truth=True, previously_scored=True,
                  new_neural_forwards=0, optimizer_updates=0, target_fitting=False,
                  caveat='Post-hoc descriptive analysis of previously exposed test; no routing or selection.')
    (out/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/'complete.json').write_text(json.dumps(dict(status='ARTIFACTS_COMPLETE', code_commit=commit,
                                                      rows=len(rows), target_fitting=False))+'\n', encoding='utf-8')
    print('CLEAN_RECEIVED_RESTORED_ANALYSIS_COMPLETE', len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU truth-side scoring required')
    if (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('Actual code commit marker required')
    score(args.root, args.commit)


if __name__ == '__main__':
    main()
