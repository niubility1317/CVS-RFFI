"""Two independent fixed-2220 source rows through train/export/fit/predict/score."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from .launch import read, write
from .source_embedding_launch import select_gpus, _integer, FIELDS, MODEL_FIELDS
from . import source_open_embedding_train as trainer
from . import source_embedding_budget_export as exporter, source_embedding_budget_run as run
from .source_risk_calibration import METHODS


RUN_ID = '20261029-phase3-source-embedding-budget-s2026092701-r01'
ARMS = ('supervised_only', 'uniform_oe')
STAGES = ('train', 'export', 'fit', 'predict', 'score')
PASS_NEXT = 'SOURCE_EMBEDDING_BUDGET_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION'


def validate_runtime(rt):
    if not isinstance(rt, dict) or set(rt) != FIELDS:
        raise ValueError('Exactly the fourteen declared budget runtime fields are required; no budget or retention override')
    strings = FIELDS - {'model', 'allowed_gpus', 'max_added_workers_per_gpu', 'minimum_free_mb', 'max_concurrent_rows'}
    if (any(not isinstance(rt[k], str) or not rt[k].strip() for k in strings)
            or rt['run_id'] != RUN_ID or Path(rt['run_root']).name != RUN_ID
            or len(rt['code_commit']) != 40 or any(c not in '0123456789abcdef' for c in rt['code_commit'])
            or any(not Path(rt[k]).is_absolute() for k in ('run_root', 'release_root', 'python',
                'parent_run_root', 'parent_oe_data_root', 'diverse_data_root'))
            or Path(rt['parent_run_root']).name != run.PARENT_RUN):
        raise ValueError('Fixed budget run, original F6 parent, code commit and absolute immutable roots required')
    if (type(rt['max_added_workers_per_gpu']) is not int or rt['max_added_workers_per_gpu'] != 2
            or type(rt['max_concurrent_rows']) is not int or rt['max_concurrent_rows'] not in (1, 2)
            or type(rt['minimum_free_mb']) is not int or rt['minimum_free_mb'] < 4096
            or not isinstance(rt['allowed_gpus'], list) or not rt['allowed_gpus']
            or any(type(g) is not int or g < 0 for g in rt['allowed_gpus'])
            or len(set(rt['allowed_gpus'])) != len(rt['allowed_gpus'])):
        raise ValueError('At most two added row workers, two per card and >=4096MB are required')
    if not isinstance(rt['model'], dict) or set(rt['model']) != MODEL_FIELDS or rt['model'].get('model_seed') != 2026092701:
        raise ValueError('Only the original full-six-class source inputs and fixed model seed are allowed')
    for arm in ARMS:
        train, export = configurations(rt, Path(rt['run_root']), arm)
        trainer.validate_config(train); exporter.validate_config(export)
    return rt


def configurations(rt, root, arm):
    if arm not in ARMS: raise ValueError('Only the two matched budget arms are accepted')
    root = Path(root); row = root / 'rows' / arm
    train = dict(rt['model'], run_id=RUN_ID, row_id=arm + '-s2026092701', code_commit=rt['code_commit'],
        output_root=str(row / 'train'), arm=arm, steps=2220, lr=.0001, L_seed=2026092701, OE_seed=2026102907)
    if arm == 'uniform_oe': train['diverse_data_root'] = rt['diverse_data_root']
    export = dict(run_id=RUN_ID, row_id=arm + '-s2026092701', code_commit=rt['code_commit'],
        train_config=str(root / 'configs' / (arm + '.train.json')), checkpoint=str(row / 'train/final.pt'),
        parent_run_root=rt['parent_run_root'], oe_data_root=rt['parent_oe_data_root'], output_root=str(row / 'export'))
    return train, export


def commands(rt, root, arm):
    prefix = [rt['python'], '-B', '-u', '-m']; row = Path(root) / 'rows' / arm
    return dict(train=prefix + ['experiments.pamf_phase3_open_world.source_open_embedding_train',
            '--config', str(Path(root) / 'configs' / (arm + '.train.json')), '--device', 'cuda:0'],
        export=prefix + ['experiments.pamf_phase3_open_world.source_embedding_budget_export',
            '--config', str(Path(root) / 'configs' / (arm + '.export.json')), '--device', 'cuda:0'],
        fit=prefix + ['experiments.pamf_phase3_open_world.source_embedding_budget_run', 'fit',
            '--export-root', str(row / 'export'), '--model-seed', '2026092701', '--output', str(row / 'fit')],
        predict=prefix + ['experiments.pamf_phase3_open_world.source_embedding_budget_run', 'predict',
            '--export-root', str(row / 'export'), '--fit-root', str(row / 'fit'), '--output', str(row / 'predict')],
        score=prefix + ['experiments.pamf_phase3_open_world.source_embedding_budget_scoring',
            '--predictions', str(row / 'predict/predictions.npz'),
            '--known-features', str(Path(rt['parent_run_root']) / 'export/source_V/features.npz'),
            '--novel-truth', str(Path(rt['parent_oe_data_root']) / 'novel_validation/truth/truth.npz'), '--output', str(row / 'score')])


def _validate_marker(kind, marker, arm):
    if not isinstance(marker, dict) or marker.get('source_only') is not True or marker.get('arm') != arm:
        raise ValueError('Complete matching source-only row stage required')
    _integer(marker, 'target_reads', 0)
    if kind == 'train':
        if (marker.get('status') != 'SOURCE_OPEN_EMBEDDING_FROZEN' or marker.get('default_head_joint') is not True
                or marker.get('identity_head_replaced') is not False or marker.get('optimizer_reset') is not True
                or marker.get('lr') != .0001):
            raise ValueError('Native default-head fixed-2220 joint source training must complete')
        for key, expected in dict(updates=2220, parent_updates=11100, cumulative_updates=13320,
                model_classes=6, parameters=232993, U_reads=0, validation_reads=0, novel_validation_reads=0,
                OE_truth_reads=0, L_sampled_exposures=284160, As_generated_exposures=142080,
                L_forward_packet_views=1704960, OE_sampled_exposures=142080 if arm == 'uniform_oe' else 0).items():
            _integer(marker, key, expected)
    elif kind == 'export':
        if (marker.get('status') != 'SOURCE_BUDGET_BRANCH_FEATURES_EXPORTED' or marker.get('head_changed') is not False
                or marker.get('counts') != exporter.COUNTS): raise ValueError('Complete frozen E60 branch export required')
        exporter.validate_encoder_identity(marker.get('encoder_state_identity'), arm)
        for key, expected in dict(model_classes=6, parameters=232993, new_encoder_updates=0, new_identity_head_updates=0,
                optimizer_updates=0, original_U_s_reads=0, OE_truth_reads=0).items(): _integer(marker, key, expected)
    elif kind == 'fit':
        if (marker.get('status') != 'SOURCE_BUDGET_READOUTS_FROZEN' or marker.get('geometry_fitted') is not True
                or marker.get('descriptor_schema') != run.consensus.DESCRIPTOR_SCHEMA or marker.get('readout_training_regime') != run.REGIME
                or marker.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}): raise ValueError('New L-only geometry and matched two-readout fit required')
        exporter.validate_encoder_identity(marker.get('encoder_state_identity'), arm)
        for key, expected in dict(new_encoder_updates=0, new_identity_head_updates=0, readout_steps_per_model=200,
                source_validation_reads=0, novel_validation_reads=0, OE_truth_reads=0).items(): _integer(marker, key, expected)
    elif kind == 'predict':
        if (marker.get('status') != 'SOURCE_BUDGET_READOUT_PREDICTIONS_COMPLETE' or marker.get('unknown_truth_read') is not False
                or marker.get('query_fit') is not False or marker.get('default_head_replaced') is not False
                or marker.get('method_names') != list(METHODS) or marker.get('calibration_schema') != run.calibration.SCHEMA
                or marker.get('source_V_labels_read') is not True
                or marker.get('source_known_label_calibration') is not True): raise ValueError('Seven complete truth-blind source predictions required')
        exporter.validate_encoder_identity(marker.get('encoder_state_identity'), arm)
        for key, expected in dict(encoder_updates=2220, identity_head_updates=2220, new_encoder_updates=0,
                new_identity_head_updates=0, new_readout_updates=0, readout_steps_per_model=200, known_count=27000, novel_count=3150, count=30150).items(): _integer(marker, key, expected)
    elif kind == 'score':
        eligible = marker.get('eligible_methods')
        if (marker.get('status') != 'SCORED_SOURCE_BUDGET_READOUT_COMPLETE' or marker.get('truth_last') is not True
                or marker.get('final_phase3_complete') is not False or eligible not in ([], ['mlp16_oe'])
                or marker.get('selected_method') != ('mlp16_oe' if eligible else None)
                or marker.get('next_step') != (PASS_NEXT if eligible else 'NO_PROMOTION')): raise ValueError('Only matched MLP16 source scoring may advance')
    else: raise ValueError('Unknown stage')


def selection(outcomes):
    if any(not outcomes[arm]['complete'] for arm in ARMS): return None
    return next((arm + '::mlp16_oe' for arm in ARMS if outcomes[arm]['marker']['eligible_methods'] == ['mlp16_oe']), None)


def dispatch(rt):
    validate_runtime(rt); root = Path(rt['run_root']); root.mkdir(parents=True, exist_ok=False)
    (root / 'configs').mkdir(); (root / 'rows').mkdir(); lock = threading.RLock()
    state = dict(status='PREPARING_SOURCE_ONLY', run_id=rt['run_id'], code_commit=rt['code_commit'], owner=rt['launch_owner'],
        pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, started_at=time.time(), source_only=True, target_reads=0,
        foreign_processes_stopped=0, final_phase3_complete=False, independent_confirmation=False,
        source_novel_evaluation_reused=True, max_added_workers_per_gpu=2, added_updates_per_arm=2220,
        parent_updates=11100, cumulative_updates=13320, readout_steps_per_model=200, rows={arm: dict(status='PENDING', jobs={}) for arm in ARMS})
    def record():
        with lock:
            path = root / 'pipeline_state.json.tmp'; write(path, state, exclusive=False); os.replace(path, root / 'pipeline_state.json')
    record()
    for arm in ARMS:
        (root / 'rows' / arm).mkdir(); (root / 'rows' / arm / 'logs').mkdir()
        train, export = configurations(rt, root, arm)
        write(root / 'configs' / (arm + '.train.json'), train)
        write(root / 'configs' / (arm + '.export.json'), export)
    try:
        gpus = select_gpus(rt)
    except Exception as exc:
        state.update(status='PARTIAL_SOURCE_ONLY', failure=str(exc), ended_at=time.time(), selected_candidate=None,
            eligible_candidates=[], next_step='INCOMPLETE_SOURCE_MATRIX_NO_PROMOTION')
        record(); raise
    def row(arm, gpu):
        row_root = root / 'rows' / arm; stage_commands = commands(rt, root, arm); active = None
        try:
            for kind in STAGES:
                active = kind; log = row_root / 'logs' / (kind + '.log')
                with lock:
                    current = state['rows'][arm]; state['status'] = 'RUNNING_SOURCE_ONLY'; current['status'] = 'RUNNING'
                    current['jobs'][kind] = dict(status='PREPARING', argv=stage_commands[kind], log=str(log), cwd=rt['release_root'], started_at=time.time()); record()
                env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES=str(gpu) if kind in ('train', 'export') else '',
                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
                with log.open('x', encoding='utf-8') as stream:
                    child = subprocess.Popen(stage_commands[kind], cwd=rt['release_root'], env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    with lock:
                        current['jobs'][kind].update(status='RUNNING', pid=child.pid, gpu=gpu if kind in ('train', 'export') else None); record()
                    status = child.wait()
                if status: raise RuntimeError(kind + ' failed exit=' + str(status))
                marker = read(row_root / kind / 'complete.json'); _validate_marker(kind, marker, arm)
                with lock: current['jobs'][kind].update(status='COMPLETE', ended_at=time.time()); record()
            with lock:
                current.update(status='COMPLETE', next_step=marker['next_step'], eligible_methods=marker['eligible_methods']); record()
            return dict(complete=True, marker=marker)
        except Exception as exc:
            with lock:
                current = state['rows'][arm]
                if active is not None: current['jobs'][active].update(status='FAILED')
                current.update(status='FAILED', failure=str(exc)); record()
            (row_root / 'logs/failure.log').write_text(traceback.format_exc(), encoding='utf-8')
            print('SOURCE_EMBEDDING_BUDGET_ROW_FAILED', arm, str(exc), flush=True)
            return dict(complete=False)
    with ThreadPoolExecutor(max_workers=rt['max_concurrent_rows']) as pool:
        futures = {arm: pool.submit(row, arm, gpu) for arm, gpu in zip(ARMS, gpus)}
        outcomes = {arm: future.result() for arm, future in futures.items()}
    selected = selection(outcomes); complete = all(value['complete'] for value in outcomes.values())
    state.update(status='ANALYZED_SOURCE_ONLY' if complete else 'PARTIAL_SOURCE_ONLY', selected_candidate=selected,
        eligible_candidates=[arm + '::mlp16_oe' for arm in ARMS if outcomes[arm]['marker']['eligible_methods'] == ['mlp16_oe']] if complete else [], ended_at=time.time(),
        next_step=PASS_NEXT if selected else ('NO_PROMOTION' if complete else 'INCOMPLETE_SOURCE_MATRIX_NO_PROMOTION'))
    record(); print(state['status'], rt['run_id'], state['next_step'], flush=True); return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--runtime', required=True)
    args = parser.parse_args(argv); result = dispatch(read(args.runtime)); return 0 if result['status'] == 'ANALYZED_SOURCE_ONLY' else 1


if __name__ == '__main__': raise SystemExit(main())
