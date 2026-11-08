"""Independently score frozen Main predictions after missing-module failures."""
import argparse
import importlib
import json
import os
from pathlib import Path

from . import main_split_eval as main_eval
from .cost_probe import write
from .source_pair import read

F_RELEASE = '20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
F_COMMIT = '3ff8f4b7e85b5608f5b45350aa18081499a7ef1d'
STREAMS = ('raw', 'DURA_v1', 'candidate')
CASES = {
    'affine': dict(parent='20261108-phase1-Restore-MixtureAffine-E30-s2026092701-r02',
                   arm='TF_MixtureAffine', commit='f8a23fc9780c02a6884eb2a24ad4e6f3e3d16a57',
                   recovery='20261108-evaluation-Restore-ScoreRepairAffine-s2026092701-r01'),
    'bounded': dict(parent='20261108-phase1-Restore-BoundedProjected-E30-s2026092701-r03',
                    arm='TF_BoundedProjected', commit='28d1199fe66f05a7eaf4487716dab0c6fa5cd79a',
                    recovery='20261108-evaluation-Restore-ScoreRepairBounded-s2026092701-r01'),
}


def frozen_predictions(root, key):
    case = CASES[key]
    row = root / 'runs' / case['parent'] / 'rows' / case['arm']
    pred = row / 'prediction'
    whole = read(pred / 'complete.json')
    if any(whole.get(k) != v for k, v in dict(status='ALL_PREDICTIONS_COMPLETE',
            run_id=case['parent'], row_id=case['arm'], views=list(main_eval.VIEWS),
            readouts=list(STREAMS), truth_read=False, query_fit=False, code_commit=case['commit']).items()):
        raise ValueError('Original full prediction matrix must be frozen before truth')
    for task, count in main_eval.COUNTS.items():
        m = read(pred / task / 'complete.json')
        main_eval.valid_identity(m['F0_identity'])
        required = dict(status='PREDICTIONS_COMPLETE', run_id=case['parent'], arm=case['arm'],
                        task=task, count=count, code_commit=case['commit'], readouts=list(STREAMS),
                        views=list(main_eval.VIEWS), truth_read=False, query_fit=False, model_state_updates=0)
        if any(m.get(k) != v for k, v in required.items()) or not (pred / task / 'predictions.npz').is_file():
            raise ValueError('Original full task prediction must be frozen before truth')
    failed = read(root / 'runs' / case['parent'] / ('terminal-' + case['arm'] + '-score.json'))
    score_log = (root / 'runs' / case['parent'] / 'logs' / (case['arm'] + '-score.log')).read_text()
    if (failed['exit_code'] != 1 or
        "ModuleNotFoundError: No module named 'experiments.pamf_repl_view_shuffle'" not in score_log):
        raise ValueError('Original missing-module scoring failure required')
    if (row / 'score' / 'scores.json').exists() or (row / 'results.json').exists():
        raise ValueError('Existing score result requires reconciliation, not resubmission')
    return pred


def original_scorer(root):
    release = root / 'releases' / F_RELEASE
    if (release / 'release_commit.txt').read_text().strip() != F_COMMIT:
        raise ValueError('Original frozen F0 scorer release required')
    package = importlib.import_module('experiments')
    source = str(release / 'experiments')
    if source not in package.__path__: package.__path__.append(source)
    return importlib.import_module('experiments.pamf_repl_view_shuffle.evaluation').score


def run(root, key):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU truth scorer required')
    case = CASES[key]
    pred = frozen_predictions(root, key)
    scorer = original_scorer(root)
    c = main_eval.official(root, 'score')
    out = root / 'runs' / case['recovery']
    out.mkdir()
    c.update(run_id=case['parent'], row_id=case['arm'], readouts=list(STREAMS),
             prediction_output=str(pred), output_root=str(out / 'score'))
    import fcntl
    with (root / 'runs/20261101-phase3-P3Restore-TF-IQ-E30-s2026092701-r01/score.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        scorer(c)  # Original full-matrix preflight precedes any truth access.
    scores = read(out / 'score' / 'scores.json')
    write(out / 'results.json', dict(status='ANALYZED', run_id=case['parent'], arm=case['arm'],
          code_commit=case['commit'], score_recovery_run=case['recovery'],
          results=main_eval.summary(scores, streams=STREAMS), truth_last=True,
          default_DURA_changed=False, goal_complete=False))
    write(out / 'complete.json', dict(status='ARTIFACTS_COMPLETE', prediction_parent=case['parent'],
          score_recovery_run=case['recovery'], fixed_predictions_reused=True, new_model_forwards=0,
          new_training_updates=0, truth_last=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--candidate', choices=tuple(CASES), required=True)
    args = p.parse_args()
    run(args.root, args.candidate)


if __name__ == '__main__': main()
