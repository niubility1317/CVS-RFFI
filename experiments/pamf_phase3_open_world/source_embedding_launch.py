"""Two independent source-only E50+222 rows, preserving all foreign work."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from .launch import read, write
from . import source_open_embedding_train as trainer
from . import source_embedding_eval as eval_module
from .scores import METHODS


FIELDS = {'run_id', 'run_root', 'release_root', 'python', 'code_commit', 'launch_owner', 'model',
          'parent_run_root', 'parent_oe_data_root', 'diverse_data_root', 'allowed_gpus',
          'max_added_workers_per_gpu', 'minimum_free_mb', 'max_concurrent_rows'}
MODEL_FIELDS = trainer.BASE_KEYS - {'run_id', 'row_id', 'code_commit', 'output_root', 'arm',
                                  'steps', 'lr', 'L_seed', 'OE_seed'}
ARMS = ('supervised_only', 'uniform_oe')
ROW_IDS = {arm: arm + '-s2026092701' for arm in ARMS}
STAGES = ('train', 'eval', 'score')
PAIR_METRICS = ('unknown_macro_rejection', 'known_macro_closed_accuracy', 'known_macro_open_accuracy',
                'known_macro_frr', 'encoder_closed_loss', 'total_open_loss')


def validate_runtime(rt):
    if not isinstance(rt, dict) or set(rt) != FIELDS:
        raise ValueError('Only the fourteen declared source embedding runtime fields are accepted')
    strings = FIELDS - {'model', 'allowed_gpus', 'max_added_workers_per_gpu', 'minimum_free_mb', 'max_concurrent_rows'}
    if any(not isinstance(rt[key], str) or not rt[key].strip() for key in strings):
        raise ValueError('Nonempty source embedding runtime strings required')
    if (not Path(rt['run_root']).is_absolute() or Path(rt['run_root']).name != rt['run_id']
            or any(not Path(rt[key]).is_absolute() for key in ('release_root', 'python', 'parent_run_root',
                'parent_oe_data_root', 'diverse_data_root'))
            or Path(rt['parent_run_root']).name != eval_module.PARENT_RUN):
        raise ValueError('Absolute immutable run roots and the fixed original source-open parent are required')
    if (type(rt['max_added_workers_per_gpu']) is not int or rt['max_added_workers_per_gpu'] != 2
            or type(rt['max_concurrent_rows']) is not int or rt['max_concurrent_rows'] not in (1, 2)
            or type(rt['minimum_free_mb']) is not int or rt['minimum_free_mb'] < 4096
            or not isinstance(rt['allowed_gpus'], list) or not rt['allowed_gpus']
            or any(type(gpu) is not int or gpu < 0 for gpu in rt['allowed_gpus'])
            or len(set(rt['allowed_gpus'])) != len(rt['allowed_gpus'])):
        raise ValueError('At most two added row workers, two slots/card and >=4096MB are required')
    if not isinstance(rt['model'], dict) or set(rt['model']) != MODEL_FIELDS:
        raise ValueError('Only the original full-F6 model paths/source cache are accepted; no retention')
    train, _ = configurations(rt, Path(rt['run_root']), 'supervised_only')
    trainer.validate_config(train)
    return rt


def configurations(rt, root, arm):
    if arm not in ARMS:
        raise ValueError('Only the two predeclared source embedding arms are allowed')
    row = root / 'rows' / arm
    train = dict(rt['model'], run_id=rt['run_id'], row_id=ROW_IDS[arm], code_commit=rt['code_commit'],
        output_root=str(row / 'train'), arm=arm, steps=222, lr=.0001, L_seed=2026092701, OE_seed=2026102907)
    if arm == 'uniform_oe':
        train['diverse_data_root'] = rt['diverse_data_root']
    evaluation = dict(run_id=rt['run_id'], row_id=ROW_IDS[arm], code_commit=rt['code_commit'],
        train_config=str(root / 'configs' / (arm + '.train.json')), checkpoint=str(row / 'train/final.pt'),
        parent_run_root=rt['parent_run_root'],
        novel_public_root=str(Path(rt['parent_oe_data_root']) / 'novel_validation/public'), output_root=str(row / 'eval'))
    return train, evaluation


def commands(rt, root, arm):
    prefix = [rt['python'], '-B', '-u', '-m']
    row = root / 'rows' / arm
    return dict(train=prefix + ['experiments.pamf_phase3_open_world.source_open_embedding_train',
            '--config', str(root / 'configs' / (arm + '.train.json')), '--device', 'cuda:0'],
        eval=prefix + ['experiments.pamf_phase3_open_world.source_embedding_eval',
            '--config', str(root / 'configs' / (arm + '.eval.json')), '--device', 'cuda:0'],
        score=prefix + ['experiments.pamf_phase3_open_world.source_embedding_scoring',
            '--predictions', str(row / 'eval/predictions.npz'),
            '--known-features', str(Path(rt['parent_run_root']) / 'export/source_V/features.npz'),
            '--novel-truth', str(Path(rt['parent_oe_data_root']) / 'novel_validation/truth/truth.npz'),
            '--output', str(row / 'score')])


def select_gpus(rt):
    while True:
        snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
            '--format=csv,noheader,nounits'], text=True)
        available = []
        for line in snapshot.splitlines():
            gpu, free = [int(part.strip()) for part in line.split(',')]
            if gpu in rt['allowed_gpus'] and free >= rt['minimum_free_mb']:
                available.append((free, gpu))
        available.sort(key=lambda value: (-value[0], value[1]))
        if available:
            # The only owned workers are these two rows. Foreign PIDs neither
            # exclude a card nor cause any stop operation.
            return [available[0][1], available[1][1] if len(available) > 1 else available[0][1]]
        print('SOURCE_EMBEDDING_WAITING_FOR_MEMORY', 'no processes stopped', flush=True)
        time.sleep(20)


def _integer(marker, name, expected):
    if type(marker.get(name)) is not int or marker[name] != expected:
        raise ValueError('Source embedding stage completion integer differs: ' + name)


def _validate_marker(kind, marker, arm):
    if kind not in STAGES or arm not in ARMS or not isinstance(marker, dict):
        raise ValueError('Declared source embedding row/stage completion required')
    if marker.get('source_only') is not True or marker.get('arm') != arm:
        raise ValueError('The completed stage belongs to another arm or data scope')
    _integer(marker, 'target_reads', 0)
    if kind == 'train':
        if (marker.get('status') != 'SOURCE_OPEN_EMBEDDING_FROZEN' or marker.get('default_head_joint') is not True
                or marker.get('identity_head_replaced') is not False or marker.get('optimizer_reset') is not True):
            raise ValueError('Complete joint default-head source increment required')
        for key, expected in dict(updates=222, parent_updates=11100, cumulative_updates=11322, model_classes=6,
                parameters=232993, U_reads=0, validation_reads=0, novel_validation_reads=0, OE_truth_reads=0).items():
            _integer(marker, key, expected)
    elif kind == 'eval':
        if (marker.get('status') != 'SOURCE_EMBEDDING_PREDICTIONS_COMPLETE'
                or marker.get('default_head_replaced') is not False or marker.get('method_names') != list(METHODS)
                or marker.get('unknown_truth_read') is not False or marker.get('query_fit') is not False
                or marker.get('source_V_labels_read') is not True or marker.get('source_known_label_calibration') is not True):
            raise ValueError('All five predictions must finish with only known-source risk calibration')
        for key, expected in dict(updates=222, encoder_updates=222, identity_head_updates=222,
                known_count=27000, novel_count=3150, count=30150).items():
            _integer(marker, key, expected)
    else:
        eligible = marker.get('eligible_methods')
        if (marker.get('status') != 'SCORED_SOURCE_EMBEDDING_COMPLETE' or marker.get('truth_last') is not True
                or marker.get('final_phase3_complete') is not False or eligible not in ([], ['msp'])
                or (arm == 'supervised_only' and eligible)
                or marker.get('selected_method') != ('msp' if eligible else None)
                or marker.get('next_step') != ('SOURCE_EMBEDDING_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION' if eligible else 'NO_PROMOTION')):
            raise ValueError('Only uniform_oe::msp may advance after independent source scoring')


def _paired(results):
    rows = {arm: results[arm]['methods']['msp'] for arm in ARMS}
    groups = ('overall', 'by_receiver')
    selected = {arm: {'overall': {key: rows[arm]['overall'][key] for key in PAIR_METRICS},
        'by_receiver': {str(rx): {key: rows[arm]['by_receiver'][str(rx)][key] for key in PAIR_METRICS}
                        for rx in (1, 3, 4, 6, 8)}} for arm in ARMS}
    def difference(left, right):
        return {key: None if left[key] is None or right[key] is None else float(right[key] - left[key])
                for key in PAIR_METRICS}
    delta = dict(overall=difference(selected['supervised_only']['overall'], selected['uniform_oe']['overall']),
        by_receiver={str(rx): difference(selected['supervised_only']['by_receiver'][str(rx)],
            selected['uniform_oe']['by_receiver'][str(rx)]) for rx in (1, 3, 4, 6, 8)})
    return dict(status='PAIRED_SOURCE_EMBEDDING_METRICS', method='msp', source_only=True, target_reads=0,
        independent_confirmation=False, source_novel_evaluation_reused=True, units='proportions',
        rows=selected, difference_uniform_oe_minus_supervised_only=delta,
        evidence_scope='matched source development; no formal target or independent confirmation')


def dispatch(rt):
    validate_runtime(rt)
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'configs').mkdir(); (root / 'rows').mkdir()
    lock = threading.RLock()
    state = dict(status='PREPARING_SOURCE_ONLY', run_id=rt['run_id'], owner=rt['launch_owner'],
        code_commit=rt['code_commit'], pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, started_at=time.time(),
        source_only=True, target_reads=0, max_added_workers_per_gpu=2, max_concurrent_rows=rt['max_concurrent_rows'],
        foreign_processes_stopped=0, final_phase3_complete=False, independent_confirmation=False,
        source_novel_evaluation_reused=True, eligible_candidates=[], rows={arm: dict(row_id=ROW_IDS[arm], status='PENDING', jobs={}) for arm in ARMS})
    def record():
        with lock:
            temporary = root / 'pipeline_state.json.tmp'
            write(temporary, state, exclusive=False)
            os.replace(temporary, root / 'pipeline_state.json')
    record()
    for arm in ARMS:
        (root / 'rows' / arm).mkdir(); (root / 'rows' / arm / 'logs').mkdir()
        train, evaluation = configurations(rt, root, arm)
        trainer.validate_config(train); eval_module.validate_config(evaluation)
        write(root / 'configs' / (arm + '.train.json'), train)
        write(root / 'configs' / (arm + '.eval.json'), evaluation)
    try:
        gpus = select_gpus(rt)
    except Exception as exc:
        state.update(status='PARTIAL_SOURCE_ONLY', failure=str(exc), ended_at=time.time(), next_step='INCOMPLETE_SOURCE_MATRIX_NO_PROMOTION')
        record(); raise
    with lock:
        state['status'] = 'RUNNING_SOURCE_ONLY'
        for arm, gpu in zip(ARMS, gpus): state['rows'][arm]['gpu'] = gpu
        record()
    def run_row(arm, gpu):
        row_root = root / 'rows' / arm
        row_commands = commands(rt, root, arm)
        active = None
        try:
            for kind in STAGES:
                active = kind
                log = row_root / 'logs' / (kind + '.log')
                with lock:
                    row_state = state['rows'][arm]
                    row_state['status'] = 'RUNNING'
                    row_state['jobs'][kind] = dict(status='PREPARING', cwd=rt['release_root'], argv=row_commands[kind], log=str(log), started_at=time.time())
                    record()
                visible = '' if kind == 'score' else str(gpu)
                env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES=visible,
                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                    PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
                with log.open('x', encoding='utf-8') as stream:
                    child = subprocess.Popen(row_commands[kind], cwd=rt['release_root'], env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    with lock:
                        row_state['jobs'][kind].update(status='RUNNING', pid=child.pid, gpu=None if kind == 'score' else gpu)
                        record()
                    result = child.wait()
                if result:
                    raise RuntimeError(kind + ' exit=' + str(result) + '; preserve ' + str(log))
                marker = read(row_root / kind / 'complete.json')
                _validate_marker(kind, marker, arm)
                with lock:
                    row_state['jobs'][kind].update(status='COMPLETE', ended_at=time.time())
                    record()
            scores = read(row_root / 'score/results.json')
            with lock:
                row_state.update(status='COMPLETE', ended_at=time.time(), eligible_methods=marker['eligible_methods'],
                    selected_method=marker['selected_method'], next_step=marker['next_step'])
                record()
            return dict(complete=True, marker=marker, results=scores)
        except Exception as exc:
            with lock:
                row_state = state['rows'][arm]
                if active is not None: row_state['jobs'][active].update(status='FAILED', ended_at=time.time())
                row_state.update(status='FAILED', failure=str(exc), ended_at=time.time())
                record()
            with (row_root / 'logs/failure.log').open('x', encoding='utf-8') as stream:
                stream.write(traceback.format_exc())
            print('SOURCE_EMBEDDING_ROW_TECHNICAL_FAILURE', arm, str(exc), flush=True)
            return dict(complete=False)
    with ThreadPoolExecutor(max_workers=rt['max_concurrent_rows']) as executor:
        futures = {arm: executor.submit(run_row, arm, gpu) for arm, gpu in zip(ARMS, gpus)}
        outcomes = {arm: future.result() for arm, future in futures.items()}
    if all(value['complete'] for value in outcomes.values()):
        write(root / 'paired_source_metrics.json', _paired({arm: value['results'] for arm, value in outcomes.items()}))
        promoted = outcomes['uniform_oe']['marker']['eligible_methods'] == ['msp']
        state.update(status='ANALYZED_SOURCE_ONLY', next_step=outcomes['uniform_oe']['marker']['next_step'],
            eligible_candidates=['uniform_oe::msp'] if promoted else [],
            selected_candidate='uniform_oe::msp' if promoted else None, paired_metrics_ref='paired_source_metrics.json')
    else:
        state.update(status='PARTIAL_SOURCE_ONLY', next_step='INCOMPLETE_SOURCE_MATRIX_NO_PROMOTION',
            eligible_candidates=[], selected_candidate=None)
    state['ended_at'] = time.time(); record()
    print(state['status'], rt['run_id'], state['next_step'], flush=True)
    return deepcopy(state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args(argv)
    result = dispatch(read(args.runtime))
    return 0 if result['status'] == 'ANALYZED_SOURCE_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
