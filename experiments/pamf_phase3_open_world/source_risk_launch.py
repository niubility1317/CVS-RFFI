"""CPU-only recalibration and scoring of a fixed analyzed source parent."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .launch import read, write
from .source_open_launch import _validate_marker as validate_source_stage, _integer
from .source_risk_run import PARENT_RUN, METHODS


FIELDS = {'run_id', 'run_root', 'release_root', 'python', 'code_commit', 'launch_owner',
          'parent_run_root', 'parent_oe_data_root', 'max_added_workers_per_gpu'}
STAGES = ('predict', 'score')


def validate_runtime(rt):
    if not isinstance(rt, dict) or set(rt) != FIELDS:
        raise ValueError('Only the nine declared source risk runtime fields are accepted')
    for key in FIELDS - {'max_added_workers_per_gpu'}:
        if not isinstance(rt[key], str) or not rt[key]:
            raise ValueError('Nonempty source risk runtime string required: ' + key)
    root = Path(rt['run_root'])
    if (type(rt['max_added_workers_per_gpu']) is not int or rt['max_added_workers_per_gpu'] != 2
            or len(rt['code_commit']) != 40 or any(ch not in '0123456789abcdef' for ch in rt['code_commit'])
            or not root.is_absolute() or root.name != rt['run_id']
            or any(not Path(rt[key]).is_absolute() for key in ('release_root', 'parent_run_root', 'parent_oe_data_root'))
            or Path(rt['parent_run_root']).name != PARENT_RUN or root == Path(rt['parent_run_root'])):
        raise ValueError('An immutable new run, fixed analyzed parent and two-worker allowance are required')
    return rt


def _validate_marker(kind, marker):
    if kind == 'score':
        return validate_source_stage('score', marker)
    if kind != 'predict' or not isinstance(marker, dict) or marker.get('source_only') is not True:
        raise ValueError('Completed source-only risk predictions required')
    for key, count in dict(target_reads=0, known_count=27000, novel_count=3150, count=30150,
            encoder_updates=0, identity_head_updates=0, readout_updates=0,
            unknown_calibration_scores_read=0, unknown_calibration_labels_read=0).items():
        _integer(marker, key, count)
    if (marker.get('status') != 'ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE' or marker.get('truth_read') is not False
            or marker.get('query_fit') is not False or marker.get('source_V_labels_read') is not True
            or marker.get('source_known_label_calibration') is not True
            or marker.get('independent_confirmation') is not False or marker.get('constraints_passed') is not True
            or marker.get('method_names') != list(METHODS)
            or marker.get('limits') != {'known_macro_frr': .04, 'known_macro_correct_rejection_loss': .015}):
        raise ValueError('Known-label risk calibration and unknown-truth boundaries differ')


def _commands(rt, root):
    prefix = [rt['python'], '-B', '-u', '-m']
    return dict(predict=prefix + ['experiments.pamf_phase3_open_world.source_risk_run', 'predict',
            '--parent-run-root', rt['parent_run_root'], '--output', str(root / 'predict')],
        score=prefix + ['experiments.pamf_phase3_open_world.source_open_scoring',
            '--predictions', str(root / 'predict/predictions.npz'),
            '--known-features', str(Path(rt['parent_run_root']) / 'export/source_V/features.npz'),
            '--novel-truth', str(Path(rt['parent_oe_data_root']) / 'novel_validation/truth/truth.npz'),
            '--output', str(root / 'score')])


def dispatch(rt):
    validate_runtime(rt)
    root = Path(rt['run_root']); root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    state = dict(status='PREPARING_SOURCE_ONLY', run_id=rt['run_id'], source_only=True, target_reads=0,
        owner=rt['launch_owner'], code_commit=rt['code_commit'], parent_run_root=rt['parent_run_root'],
        pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, started_at=time.time(), jobs={},
        added_GPU_workers=0, max_added_workers_per_gpu=2, foreign_processes_stopped=0,
        encoder_updates=0, identity_head_updates=0, readout_updates=0,
        independent_confirmation=False, source_novel_evaluation_reused=True, final_phase3_complete=False,
        risk_scope='empirical risk on the same known-source calibration samples',
        evidence_scope='repeated source development evaluation; not independent confirmation or formal target evidence')
    def record(): write(root / 'pipeline_state.json', state, exclusive=False)
    record(); active = None
    try:
        commands = _commands(rt, root)
        for kind in STAGES:
            active = kind; log = root / 'logs' / (kind + '.log')
            state['jobs'][kind] = dict(status='PREPARING', cwd=rt['release_root'], argv=commands[kind], log=str(log), started_at=time.time())
            record()
            env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES='',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            with log.open('x', encoding='utf-8') as stream:
                child = subprocess.Popen(commands[kind], cwd=rt['release_root'], env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                state['status'] = 'RUNNING_SOURCE_ONLY'; state['jobs'][kind].update(status='RUNNING', pid=child.pid)
                record(); status = child.wait()
            if status:
                raise RuntimeError(kind + ' failed with exit=' + str(status) + '; preserve ' + str(log))
            marker = read(root / kind / 'complete.json'); _validate_marker(kind, marker)
            state['jobs'][kind].update(status='COMPLETE', ended_at=time.time()); record()
        state.update(status='ANALYZED_SOURCE_ONLY', ended_at=time.time(), next_step=marker['next_step'],
            eligible_methods=marker['eligible_methods'], selected_method=marker['selected_method'])
        record(); print('ANALYZED_SOURCE_ONLY', rt['run_id'], state['next_step'], flush=True)
        return state
    except Exception as exc:
        if active is not None: state['jobs'][active].update(status='FAILED', ended_at=time.time())
        state.update(status='STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE', failure=str(exc), ended_at=time.time())
        record()
        with (root / 'logs/failure.log').open('x', encoding='utf-8') as stream: stream.write(traceback.format_exc())
        traceback.print_exc(); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--runtime', required=True)
    args = parser.parse_args(argv); dispatch(read(args.runtime)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
