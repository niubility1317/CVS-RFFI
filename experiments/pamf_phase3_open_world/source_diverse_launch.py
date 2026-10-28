"""One export-GPU then CPU fit/predict/score owner for diverse source OE."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .launch import read, write, choose_gpu
from . import source_risk_launch as risk_launch
from .source_open_launch import _integer
from .source_consensus import DESCRIPTOR_SCHEMA
from .source_domain_oe_readout import REGIME
from .source_risk_calibration import METHODS


FIELDS = risk_launch.FIELDS | {'diverse_data_root', 'allowed_gpus', 'minimum_free_mb'}
STAGES = ('export', 'fit', 'predict', 'score')
ROW_ID = 'source-diverse-readout-s2026092701'


def validate_runtime(rt):
    if not isinstance(rt, dict) or set(rt) != FIELDS:
        raise ValueError('Only the twelve declared diverse source runtime fields are accepted')
    risk_launch.validate_runtime({key: rt[key] for key in risk_launch.FIELDS})
    if (not isinstance(rt['diverse_data_root'], str) or not Path(rt['diverse_data_root']).is_absolute()
            or not isinstance(rt['allowed_gpus'], list) or not rt['allowed_gpus']
            or any(type(value) is not int or value < 0 for value in rt['allowed_gpus'])
            or len(set(rt['allowed_gpus'])) != len(rt['allowed_gpus'])
            or type(rt['minimum_free_mb']) is not int or rt['minimum_free_mb'] <= 0):
        raise ValueError('An explicit relocated diverse data root and bounded GPU memory settings are required')
    return rt


def _validate_marker(kind, marker):
    if kind == 'score': return risk_launch._validate_marker('score', marker)
    if kind == 'predict':
        risk_launch._validate_marker('predict', marker)
        if (marker.get('descriptor_schema') != DESCRIPTOR_SCHEMA or marker.get('readout_training_regime') != REGIME
                or marker.get('domain_metadata_used_for_deployed_scores') is not False):
            raise ValueError('Fixed head-aligned/domain-balanced deployment scores required')
        return
    if kind not in ('export', 'fit') or not isinstance(marker, dict) or marker.get('source_only') is not True:
        raise ValueError('Complete source-only diverse stage required')
    _integer(marker, 'target_reads', 0)
    if kind == 'export':
        for key, value in dict(count=6134, model_classes=6, parameters=232993, optimizer_updates=0,
                              original_U_s_reads=0, OE_truth_reads=0).items(): _integer(marker, key, value)
        handle = marker.get('source_OE_handle')
        if (marker.get('status') != 'SOURCE_DIVERSE_OE_FEATURES_EXPORTED' or marker.get('head_changed') is not False
                or marker.get('feature_ref') != 'exposure_train/public/features.npz'
                or not isinstance(marker.get('parent_export_identity'), dict) or not isinstance(handle, dict)
                or set(handle) != {'capsule_id', 'split_id', 'count'} or handle.get('count') != 6134
                or any(not isinstance(handle[key], str) or not handle[key] for key in ('capsule_id', 'split_id'))):
            raise ValueError('Matching complete diverse source feature export required')
    else:
        for key, value in dict(model_seed=2026092701, encoder_updates=0, identity_head_updates=0,
                readout_steps_per_model=200, source_validation_reads=0, novel_validation_reads=0,
                OE_truth_reads=0, source_L_count=6300, source_OE_count=6134).items(): _integer(marker, key, value)
        if (marker.get('status') != 'SOURCE_DIVERSE_READOUTS_FROZEN' or marker.get('geometry_fitted') is not False
                or marker.get('descriptor_schema') != DESCRIPTOR_SCHEMA or marker.get('readout_training_regime') != REGIME
                or marker.get('method_names') != list(METHODS)
                or marker.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}):
            raise ValueError('The declared domain-balanced 200-step source readout fit differs')


def _commands(rt, root, config_path):
    prefix = [rt['python'], '-B', '-u', '-m']
    return dict(export=prefix + ['experiments.pamf_phase3_open_world.source_diverse_export',
            '--config', str(config_path), '--device', 'cuda:0'],
        fit=prefix + ['experiments.pamf_phase3_open_world.source_diverse_run', 'fit',
            '--parent-run-root', rt['parent_run_root'], '--export-root', str(root / 'export'),
            '--model-seed', '2026092701', '--output', str(root / 'fit')],
        predict=prefix + ['experiments.pamf_phase3_open_world.source_diverse_run', 'predict',
            '--parent-run-root', rt['parent_run_root'], '--export-root', str(root / 'export'),
            '--fit-root', str(root / 'fit'), '--output', str(root / 'predict')],
        score=risk_launch._commands(rt, root)['score'])


def dispatch(rt):
    validate_runtime(rt)
    root = Path(rt['run_root']); root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir(); (root / 'configs').mkdir()
    state = dict(status='PREPARING_SOURCE_ONLY', run_id=rt['run_id'], row_id=ROW_ID,
        descriptor_schema=DESCRIPTOR_SCHEMA, readout_training_regime=REGIME, source_only=True, target_reads=0,
        owner=rt['launch_owner'], code_commit=rt['code_commit'], parent_run_root=rt['parent_run_root'],
        diverse_data_root=rt['diverse_data_root'], pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
        started_at=time.time(), jobs={}, max_added_workers_per_gpu=2, foreign_processes_stopped=0,
        encoder_updates=0, identity_head_updates=0, readout_steps_per_model=200,
        independent_confirmation=False, source_novel_evaluation_reused=True, final_phase3_complete=False,
        evidence_scope='same-known-V empirical risk and repeated source novel evaluation; no formal target claim')
    def record(): write(root / 'pipeline_state.json', state, exclusive=False)
    record(); active = None
    try:
        config = dict(run_id=rt['run_id'], row_id=ROW_ID, code_commit=rt['code_commit'],
            parent_run_root=rt['parent_run_root'], diverse_data_root=rt['diverse_data_root'], output_root=str(root / 'export'))
        config_path = root / 'configs/export.json'; write(config_path, config)
        commands = _commands(rt, root, config_path)
        for kind in STAGES:
            active = kind; log = root / 'logs' / (kind + '.log')
            state['jobs'][kind] = dict(status='PREPARING', cwd=rt['release_root'], argv=commands[kind], log=str(log), started_at=time.time())
            record()
            # Sequential dispatch adds one export GPU worker at most. Existing
            # users' processes never exclude a card and are never terminated.
            gpu = choose_gpu(rt) if kind == 'export' else None
            env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            with log.open('x', encoding='utf-8') as stream:
                child = subprocess.Popen(commands[kind], cwd=rt['release_root'], env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                state['status'] = 'RUNNING_SOURCE_ONLY'; state['jobs'][kind].update(status='RUNNING', pid=child.pid, gpu=gpu)
                record(); status = child.wait()
            if status: raise RuntimeError(kind + ' failed with exit=' + str(status) + '; preserve ' + str(log))
            marker = read(root / kind / 'complete.json'); _validate_marker(kind, marker)
            state['jobs'][kind].update(status='COMPLETE', ended_at=time.time()); record()
        state.update(status='ANALYZED_SOURCE_ONLY', ended_at=time.time(), next_step=marker['next_step'],
            eligible_methods=marker['eligible_methods'], selected_method=marker['selected_method'])
        record(); print('ANALYZED_SOURCE_ONLY', rt['run_id'], state['next_step'], flush=True)
        return state
    except Exception as exc:
        if active is not None: state['jobs'][active].update(status='FAILED', ended_at=time.time())
        state.update(status='STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE', failure=str(exc), ended_at=time.time()); record()
        with (root / 'logs/failure.log').open('x', encoding='utf-8') as stream: stream.write(traceback.format_exc())
        traceback.print_exc(); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--runtime', required=True)
    args = parser.parse_args(argv); dispatch(read(args.runtime)); return 0


if __name__ == '__main__': raise SystemExit(main())
