"""One immutable owner for source-only export, OE fit, prediction and scoring."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import runner
from .source_open_export import validate_config as validate_export, COUNTS
from .source_open_run import METHODS
from .launch import choose_gpu, read, write


STAGES = ('export', 'fit', 'predict', 'score')
ROW_ID = 'source-open-readout-s2026092701'
FIELDS = {'run_id', 'run_root', 'release_root', 'python', 'code_commit', 'launch_owner',
          'allowed_gpus', 'max_added_workers_per_gpu', 'minimum_free_mb', 'export'}
EXPORT_FIELDS = ((runner.COMMON | runner.SOURCE) - {'run_id', 'row_id', 'code_commit', 'output_root'}) | {'oe_data_root'}
OE_METHODS = ('linear_oe', 'mlp16_oe')


def validate_runtime(rt):
    if not isinstance(rt, dict) or set(rt) != FIELDS:
        raise ValueError('Source open runtime field whitelist differs')
    for name in ('run_id', 'run_root', 'release_root', 'python', 'code_commit', 'launch_owner'):
        if not isinstance(rt[name], str) or not rt[name]:
            raise ValueError('Nonempty source open runtime string required: ' + name)
    if (type(rt['max_added_workers_per_gpu']) is not int or rt['max_added_workers_per_gpu'] != 2
            or type(rt['minimum_free_mb']) is not int or rt['minimum_free_mb'] <= 0
            or not isinstance(rt['allowed_gpus'], list) or not rt['allowed_gpus']
            or any(type(value) is not int or value < 0 for value in rt['allowed_gpus'])
            or len(set(rt['allowed_gpus'])) != len(rt['allowed_gpus'])
            or len(rt['code_commit']) != 40 or any(ch not in '0123456789abcdef' for ch in rt['code_commit'])):
        raise ValueError('A Git commit and the fixed two-added-worker GPU limit are required')
    root = Path(rt['run_root'])
    if not root.is_absolute() or root.name != rt['run_id'] or not Path(rt['release_root']).is_absolute():
        raise ValueError('Absolute immutable run-owned output and release roots required')
    export = rt['export']
    if not isinstance(export, dict) or set(export) != EXPORT_FIELDS or export.get('model_seed') != 2026092701:
        raise ValueError('Only the declared full six-class seed2026092701 source export is accepted')
    validate_export(dict(export, run_id=rt['run_id'], row_id=ROW_ID, code_commit=rt['code_commit'], output_root=str(root / 'export')))
    return rt


def _integer(marker, key, expected):
    if type(marker.get(key)) is not int or marker[key] != expected:
        raise ValueError('Source open completion integer differs: ' + key)


def _validate_marker(kind, marker):
    if kind not in STAGES or not isinstance(marker, dict) or marker.get('source_only') is not True:
        raise ValueError('The source-only stage completion boundary differs: ' + str(kind))
    _integer(marker, 'target_reads', 0)
    statuses = dict(export='SOURCE_OPEN_FEATURES_EXPORTED', fit='SOURCE_OPEN_READOUTS_FROZEN',
        predict='ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE', score='SCORED_SOURCE_OPEN_COMPLETE')
    if marker.get('status') != statuses[kind]:
        raise ValueError(kind + ' completion artifact differs')
    if kind == 'export':
        for key, expected in dict(model_classes=6, parameters=232993, optimizer_updates=0,
                                  original_U_s_reads=0, OE_truth_reads=0).items():
            _integer(marker, key, expected)
        if marker.get('head_changed') is not False or marker.get('counts') != COUNTS:
            raise ValueError('The frozen complete six-class source role identity differs')
        for key, count in COUNTS.items():
            if type(marker['counts'][key]) is not int or marker['counts'][key] != count:
                raise ValueError('Source export role count differs: ' + key)
    elif kind == 'fit':
        for key, expected in dict(encoder_updates=0, identity_head_updates=0, readout_steps_per_model=200,
                                  source_validation_reads=0, novel_validation_reads=0, OE_truth_reads=0).items():
            _integer(marker, key, expected)
        if (marker.get('method_names') != list(METHODS) or marker.get('learned_method_names') != list(OE_METHODS)
                or marker.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}):
            raise ValueError('Both declared source OE readouts must be frozen')
    elif kind == 'predict':
        for key, expected in dict(known_count=27000, novel_count=3150, count=30150, encoder_updates=0,
                identity_head_updates=0, readout_updates=0, source_L_reads=0, exposure_train_reads=0, OE_truth_reads=0).items():
            _integer(marker, key, expected)
        if (marker.get('truth_read') is not False or marker.get('query_fit') is not False
                or marker.get('source_V_labels_read') is not False or marker.get('method_names') != list(METHODS)):
            raise ValueError('All seven predictions must finish without truth or validation model fitting')
    else:
        eligible = marker.get('eligible_methods')
        if eligible not in ([], ['linear_oe'], ['mlp16_oe'], ['linear_oe', 'mlp16_oe']):
            raise ValueError('Only the two source OE readouts may be eligible; static controls cannot advance')
        selected = eligible[0] if eligible else None
        expected_next = 'SOURCE_OPEN_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION' if selected else 'NO_PROMOTION'
        if (marker.get('selected_method') != selected or marker.get('next_step') != expected_next
                or marker.get('truth_last') is not True or marker.get('final_phase3_complete') is not False):
            raise ValueError('Source candidate selection or Phase3 completion scope differs')


def _commands(rt, root, config_path):
    prefix = [rt['python'], '-B', '-u', '-m']
    return dict(export=prefix + ['experiments.pamf_phase3_open_world.source_open_export',
            '--config', str(config_path), '--device', 'cuda:0'],
        fit=prefix + ['experiments.pamf_phase3_open_world.source_open_run', 'fit',
            '--export-root', str(root / 'export'), '--model-seed', str(rt['export']['model_seed']), '--output', str(root / 'fit')],
        predict=prefix + ['experiments.pamf_phase3_open_world.source_open_run', 'predict',
            '--export-root', str(root / 'export'), '--fit-root', str(root / 'fit'), '--output', str(root / 'predict')],
        score=prefix + ['experiments.pamf_phase3_open_world.source_open_scoring',
            '--predictions', str(root / 'predict/predictions.npz'),
            '--known-features', str(root / 'export/source_V/features.npz'),
            '--novel-truth', str(Path(rt['export']['oe_data_root']) / 'novel_validation/truth/truth.npz'),
            '--output', str(root / 'score')])


def dispatch(rt):
    validate_runtime(rt)
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir(); (root / 'configs').mkdir()
    state = dict(status='PREPARING_SOURCE_ONLY', run_id=rt['run_id'], row_id=ROW_ID,
        source_only=True, target_reads=0, encoder_updates=0, identity_head_updates=0,
        pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, owner=rt['launch_owner'], code_commit=rt['code_commit'],
        started_at=time.time(), jobs={}, foreign_processes_stopped=0, max_added_workers_per_gpu=2,
        target_performance_claim=False, final_phase3_complete=False)
    def record():
        write(root / 'pipeline_state.json', state, exclusive=False)
    record()
    active_kind = None
    try:
        config = dict(rt['export'], run_id=rt['run_id'], row_id=ROW_ID,
                      code_commit=rt['code_commit'], output_root=str(root / 'export'))
        config_path = root / 'configs/export.json'; write(config_path, config)
        commands = _commands(rt, root, config_path)
        for kind in STAGES:
            active_kind = kind
            log = root / 'logs' / (kind + '.log')
            state['jobs'][kind] = dict(status='PREPARING', cwd=rt['release_root'], argv=commands[kind],
                                       log=str(log), started_at=time.time())
            record()
            # At most one added GPU worker, waited to completion before any next
            # stage. Existing users' PIDs never exclude a card and are not stopped.
            gpu = choose_gpu(rt) if kind == 'export' else None
            env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            with log.open('x', encoding='utf-8') as stream:
                child = subprocess.Popen(commands[kind], cwd=rt['release_root'], env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT)
                state['status'] = 'RUNNING_SOURCE_ONLY'
                state['jobs'][kind].update(status='RUNNING', pid=child.pid, gpu=gpu)
                record()
                status = child.wait()
            if status:
                raise RuntimeError(kind + ' failed with exit=' + str(status) + '; preserve ' + str(log))
            marker = read(root / kind / 'complete.json')
            _validate_marker(kind, marker)
            state['jobs'][kind].update(status='COMPLETE', ended_at=time.time()); record()
        state.update(status='ANALYZED_SOURCE_ONLY', ended_at=time.time(), next_step=marker['next_step'],
            eligible_methods=marker['eligible_methods'], selected_method=marker['selected_method'], final_phase3_complete=False)
        record()
        print('ANALYZED_SOURCE_ONLY', rt['run_id'], state['next_step'], flush=True)
        return state
    except Exception as exc:
        if active_kind is not None:
            state['jobs'][active_kind].update(status='FAILED', ended_at=time.time())
        state.update(status='STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE', failure=str(exc), ended_at=time.time(),
            performance_claim='NO_COMPLETE_PERFORMANCE_RESULT')
        record()
        with (root / 'logs/failure.log').open('x', encoding='utf-8') as stream:
            stream.write(traceback.format_exc())
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
