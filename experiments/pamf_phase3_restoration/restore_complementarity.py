"""Post-hoc fixed-prediction complementarity, never a deployable selector."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from . import clean_received_restored as previous

RUN = '20261108-analysis-Restore-FiveWayComplementarity-Main-s2026092701-r01'
ORDER = ('raw', 'DURA_v1', 'joint', 'projected', 'mismatch')
PREVIOUS_COMMIT = '77847f2398af93499d8c5fb258731003ded75b5c'


def correctness_cells(truth, predictions):
    arrays = [np.asarray(truth)] + [np.asarray(x) for x in predictions]
    if len(arrays) != 6 or arrays[0].ndim != 1 or any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError('Five predictions matched to one-dimensional truth required')
    if any(x.dtype.kind not in 'iu' or not np.isin(x, np.arange(6)).all() for x in arrays):
        raise ValueError('Six-class integer truth and predictions required')
    code = np.zeros(len(arrays[0]), dtype=np.uint8)
    for prediction in arrays[1:]:
        code = (code << 1) | (prediction == arrays[0]).astype(np.uint8)
    counts = np.bincount(code, minlength=32)
    return {format(i, '05b'): int(counts[i]) for i in range(32)}


def pair_marginal(cells, first, second):
    return {format(i, '02b'): sum(v for k, v in cells.items() if k[first]+k[second] == format(i, '02b'))
            for i in range(4)}


def score(root, commit):
    import fcntl
    root = Path(root)
    out = root/'runs'/RUN
    if out.exists():
        raise FileExistsError('Existing output: reconcile before retry')
    predictions, score_paths, identity = previous.load_fixed_predictions(root)
    lock = (root/'runs/20261101-phase3-P3Restore-TF-IQ-E30-s2026092701-r01/score.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    scores = {name: previous.read(path) for name, path in score_paths.items()}
    if any((q['status'], q['truth_last'], q['target_feedback_forbidden']) != ('SCORED', True, True)
           for q in scores.values()):
        raise ValueError('Completed truth-last scores required')
    prior = previous.read(root/'runs'/previous.RUN/'results.json')
    if (prior['status'], prior['code_commit'], prior['new_neural_forwards'], prior['target_fitting']) != (
            'ANALYZED', PREVIOUS_COMMIT, 0, False):
        raise ValueError('Matching independent three-way aggregate required')
    groups = {(q['method'], q['task'], q['view']): q for q in prior['rows']
              if q['dimension'] == 'overall' and q['stratum'] == 'ALL'}
    config = previous.read(root/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/configs/score__main_split-repl-shuffled_time-3view-s2026092701.json')
    rows = []
    for task, n in previous.TASKS.items():
        base = predictions['joint'][task]
        truth = previous.read(Path(config['tests'][task]['truth']))
        y = np.asarray([truth[str(i)]['label'] for i in base['ids']], dtype=np.int64)
        for view in previous.VIEWS:
            values = [base['raw__'+view], base['DURA_v1__'+view]]
            values += [predictions[method][task]['candidate__'+view] for method in ORDER[2:]]
            cells = correctness_cells(y, values)
            if sum(cells.values()) != n:
                raise ValueError('Physical count mismatch')
            for i, method in enumerate(ORDER[1:], 1):
                existing = groups[method, task, view]['cells']
                expected = {format(j, '02b'): sum(v for k, v in existing.items() if k[1:] == format(j, '02b'))
                            for j in range(4)}
                if pair_marginal(cells, 0, i) != expected:
                    raise ValueError('Existing truth-last paired counts differ')
            rows.append(dict(task=task, view=view, count=n, cells=cells))
    out.mkdir()
    result = dict(status='ANALYZED', run_id=RUN, code_commit=commit, F0_identity=identity, order=ORDER,
                  rows=rows, predictions_fixed_before_truth=True, previously_scored=True, target_fitting=False,
                  optimizer_updates=0, new_neural_forwards=0,
                  caveat='Retrospective correctness union upper bound, no trainable selector or method selection.')
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (out/'complete.json').write_text(json.dumps(dict(status='ARTIFACTS_COMPLETE', code_commit=commit,
                                                     rows=len(rows), target_fitting=False))+'\n', encoding='utf-8')
    print('FIVE_WAY_COMPLEMENTARITY_COMPLETE', len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only scoring required')
    if (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('Actual code commit marker required')
    score(args.root, args.commit)


if __name__ == '__main__':
    main()
