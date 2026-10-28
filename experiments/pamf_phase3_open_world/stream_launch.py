"""One immutable source-only export/calibrate/predict/score launch owner."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .branch_export import KEYS as BRANCH_EXPORT_KEYS, validate_config as validate_branch_config
from .launch import choose_gpu, read, write


STAGES = ('export', 'calibrate', 'predict', 'score')
STATES = ('F', 'D0R0', 'D1R0', 'D0R1', 'D1R1')
ROW_ID = 'source-joint-fold0-s2026092701'
FIELDS = {'run_id', 'run_root', 'release_root', 'python', 'code_commit', 'launch_owner',
          'allowed_gpus', 'max_added_workers_per_gpu', 'minimum_free_mb', 'export'}
EXPORT_FIELDS = (BRANCH_EXPORT_KEYS - {'run_id', 'row_id', 'code_commit', 'output_root'}) | {'branch_export_root'}


def validate_runtime(rt):
    if not isinstance(rt, dict) or set(rt) != FIELDS:
        raise ValueError('Source stream runtime whitelist differs')
    for name in ('run_id', 'run_root', 'release_root', 'python', 'code_commit', 'launch_owner'):
        if not isinstance(rt[name], str) or not rt[name]:
            raise ValueError('Nonempty source stream runtime string required: ' + name)
    if (type(rt['max_added_workers_per_gpu']) is not int or rt['max_added_workers_per_gpu'] != 2
            or type(rt['minimum_free_mb']) is not int or rt['minimum_free_mb'] <= 0
            or not isinstance(rt['allowed_gpus'], list) or not rt['allowed_gpus']
            or any(type(value) is not int or value < 0 for value in rt['allowed_gpus'])
            or len(set(rt['allowed_gpus'])) != len(rt['allowed_gpus'])
            or len(rt['code_commit']) != 40):
        raise ValueError('Explicit code identity and at most two added GPU workers required')
    root = Path(rt['run_root'])
    if not root.is_absolute() or root.name != rt['run_id'] or not Path(rt['release_root']).is_absolute():
        raise ValueError('Absolute run-owned immutable output and code roots required')
    export = rt['export']
    if not isinstance(export, dict) or set(export) != EXPORT_FIELDS:
        raise ValueError('Source stream export input whitelist differs')
    if not isinstance(export['branch_export_root'], str) or not export['branch_export_root']:
        raise ValueError('Existing own branch export root required')
    branch = {key: value for key, value in export.items() if key != 'branch_export_root'}
    validate_branch_config(dict(branch, run_id=rt['run_id'], row_id=ROW_ID,
                               code_commit=rt['code_commit'], output_root=str(root / 'export')))
    return rt


def _integer(marker, key, expected):
    if type(marker.get(key)) is not int or marker[key] != expected:
        raise ValueError('Source stream completion integer differs: ' + key)


def _validate_marker(kind, marker):
    if not isinstance(marker, dict) or marker.get('source_only') is not True:
        raise ValueError(kind + ' source-only completion boundary differs')
    _integer(marker, 'target_reads', 0)
    expected = dict(export='SOURCE_STREAM_INPUTS_EXPORTED', calibrate='SOURCE_STREAM_CALIBRATED',
                    predict='ALL_STREAM_PREDICTIONS_COMPLETE', score='SCORED_SOURCE_STREAM_COMPLETE')
    if marker.get('status') != expected[kind]:
        raise ValueError(kind + ' completion artifact differs')
    if kind == 'export':
        for key, value in dict(train_count=3360, calibration_count=14400, discovery_count=1050,
                               future_count=5400, optimizer_updates=0, parent_epochs=20, parent_updates=540).items():
            _integer(marker, key, value)
        if marker.get('head_changed') is not False:
            raise ValueError('export changed the frozen complete head')
    elif kind == 'calibrate':
        for key, value in dict(train_count=3360, calibration_count=14400, class_count=4,
                               U_reads=0, future_reads=0).items():
            _integer(marker, key, value)
    elif kind == 'predict':
        _integer(marker, 'discovery_count', 1050)
        _integer(marker, 'future_count', 5400)
        if (marker.get('truth_read') is not False or marker.get('query_fit') is not False
                or marker.get('state_names') != list(STATES)):
            raise ValueError('predict must finish all five truth-blind frozen states')
    else:
        eligible = marker.get('eligible_states')
        if eligible not in ([], ['D1R1']):
            raise ValueError('Only the complete D1R1 joint candidate may be eligible')
        next_step = ('SOURCE_JOINT_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION' if eligible else 'NO_PROMOTION')
        if marker.get('next_step') != next_step:
            raise ValueError('score next_step differs from eligible source states')


def _commands(rt, root, config_path):
    prefix = [rt['python'], '-B', '-u', '-m']
    return dict(
        export=prefix + ['experiments.pamf_phase3_open_world.stream_export', '--config', str(config_path), '--device', 'cuda:0'],
        calibrate=prefix + ['experiments.pamf_phase3_open_world.stream_run', 'calibrate',
            '--source-inputs', str(root / 'export/source_inputs.npz'), '--output', str(root / 'calibration')],
        predict=prefix + ['experiments.pamf_phase3_open_world.stream_run', 'predict',
            '--bundle', str(root / 'calibration/bundle.npz'),
            '--discovery', str(root / 'export/discovery/public/features.npz'),
            '--future', str(root / 'export/evaluation/public/features.npz'),
            '--model-config', str(config_path), '--output', str(root / 'predict')],
        score=prefix + ['experiments.pamf_phase3_open_world.stream_scoring',
            '--future-predictions', str(root / 'predict/future_predictions.npz'),
            '--discovery-predictions', str(root / 'predict/discovery_predictions.npz'),
            '--registry', str(root / 'predict/registry.npz'),
            '--u-truth', str(root / 'export/discovery/truth/truth.npz'),
            '--future-truth', str(root / 'export/evaluation/truth/truth.npz'), '--output', str(root / 'score')])


def dispatch(rt):
    validate_runtime(rt)
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    (root / 'configs').mkdir()
    state = dict(status='PREPARING_SOURCE_ONLY', run_id=rt['run_id'], row_id=ROW_ID,
        source_only=True, target_reads=0, optimizer_updates=0, pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
        owner=rt['launch_owner'], code_commit=rt['code_commit'], started_at=time.time(), jobs={},
        foreign_processes_stopped=0, max_added_workers_per_gpu=2, target_performance_claim=False)

    def record():
        write(root / 'pipeline_state.json', state, exclusive=False)

    record()
    active_kind = None
    try:
        config = dict(rt['export'], run_id=rt['run_id'], row_id=ROW_ID,
                      code_commit=rt['code_commit'], output_root=str(root / 'export'))
        config_path = root / 'configs/export.json'
        write(config_path, config)
        commands = _commands(rt, root, config_path)
        for kind in STAGES:
            active_kind = kind
            log = root / 'logs' / (kind + '.log')
            state['jobs'][kind] = dict(status='PREPARING', cwd=rt['release_root'], argv=commands[kind],
                                       log=str(log), started_at=time.time())
            record()
            # Sequential ownership adds one CUDA worker at most. Other users' PIDs
            # are neither excluded nor terminated; choose_gpu only waits for memory.
            gpu = choose_gpu(rt) if kind == 'export' else None
            env = dict(os.environ, PYTHONPATH=rt['release_root'],
                       CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            with log.open('x', encoding='utf-8') as stream:
                child = subprocess.Popen(commands[kind], cwd=rt['release_root'], env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                state['status'] = 'RUNNING_SOURCE_ONLY'
                state['jobs'][kind].update(status='RUNNING', pid=child.pid, gpu=gpu)
                record()
                status = child.wait()
            if status:
                raise RuntimeError(kind + ' failed with exit=' + str(status) + '; preserve ' + str(log))
            folder = 'calibration' if kind == 'calibrate' else kind
            marker = read(root / folder / 'complete.json')
            _validate_marker(kind, marker)
            state['jobs'][kind].update(status='COMPLETE', ended_at=time.time())
            record()
        state.update(status='ANALYZED_SOURCE_ONLY', ended_at=time.time(),
                     next_step=marker['next_step'], eligible_states=marker['eligible_states'])
        record()
        print('ANALYZED_SOURCE_ONLY', rt['run_id'], state['next_step'], flush=True)
        return state
    except Exception as exc:
        if active_kind is not None:
            state['jobs'][active_kind].update(status='FAILED', ended_at=time.time())
        state.update(status='STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE', failure=str(exc),
                     ended_at=time.time(), performance_claim='NO_COMPLETE_PERFORMANCE_RESULT')
        record()
        failure = traceback.format_exc()
        with (root / 'logs/failure.log').open('x', encoding='utf-8') as stream:
            stream.write(failure)
        traceback.print_exc()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args(argv)
    dispatch(read(args.runtime))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
