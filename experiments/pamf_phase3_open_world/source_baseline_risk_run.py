"""CPU-only frozen readout threshold recovery against the original F6 baseline."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

from . import source_baseline_risk as calibration, source_embedding_readout_run as readout
from . import source_embedding_readout_launch as prior_launch
from . import source_open_run as parent, source_embedding_eval as embedding
from .source_risk_run import _known_labels


RUN_ID = '20261029-phase3-source-baseline-risk-s2026092701-r01'
PARENT_RUN = '20261029-phase3-source-embedding-readout-s2026092701-r01'
ARMS = ('supervised_only', 'uniform_oe')
PASS_NEXT = prior_launch.PASS_NEXT
FIELDS = {'run_id', 'run_root', 'release_root', 'python', 'code_commit', 'launch_owner', 'parent_run_root', 'max_concurrent_rows'}


def _parent_row(root):
    root = Path(root); arm = root.name
    if arm not in ARMS or root.parent.name != 'rows' or root.parent.parent.name != PARENT_RUN:
        raise ValueError('Only a completed row of the fixed frozen encoder/readout combo is allowed')
    export, fitted = root / 'export', root / 'fit'
    marker = readout._export_marker(export); readout._fit_marker(fitted, marker)
    if marker['arm'] != arm: raise ValueError('Parent row and encoder arm differ')
    completed = parent._read(root / 'predict/complete.json')
    prior_launch._validate_marker('predict', completed, arm)
    return dict(marker=marker, export_root=export, fit_root=fitted, arm=arm)


def predict(parent_row_root, output):
    output = parent._new(output); ctx = _parent_row(parent_row_root)
    marker, export, arm = ctx['marker'], ctx['export_root'], ctx['arm']
    geometry, state = readout.serialization._load_new(ctx['fit_root'])
    output.mkdir(parents=True, exist_ok=False)
    labeled = parent._load_features(export, 'source_L', False)
    smoke = readout._score({key: value[:2] for key, value in labeled.items()}, geometry, state)
    if smoke.shape != (2, 7) or not np.isfinite(smoke).all(): raise ValueError('Actual frozen readout source-L2 smoke failed')
    parent._write(output / 'smoke.json', dict(status='PASS', actual_frozen_readouts=True,
        source_L_smoke_samples=2, encoder_forwards=0, new_encoder_updates=0, new_identity_head_updates=0,
        new_readout_updates=0, source_V_reads=0, novel_validation_reads=0, target_reads=0))
    print('SOURCE_BASELINE_RISK_SMOKE_PASS', arm, 2, flush=True); del labeled
    known = parent._load_features(export, 'source_V', False)
    knownness = readout._score(known, geometry, state); closed_known = known['logits'].argmax(axis=1).astype(np.int64)
    baseline_root = Path(marker['config']['parent_run_root']) / 'export'
    baseline_known = embedding._baseline(baseline_root / parent.FEATURE_REFS['source_V'], known['ids'], 'source_V')
    risk = calibration.calibrate_known(known['ids'], _known_labels(export, known), closed_known, baseline_known, knownness, known['rx_i'])
    frozen = output / 'calibration'; frozen.mkdir()
    with (frozen / 'thresholds.npy').open('xb') as stream: np.save(stream, risk['thresholds'], allow_pickle=False)
    report = {key: value for key, value in risk.items() if key != 'thresholds'}
    report.update(status='SOURCE_ORIGINAL_BASELINE_RISK_THRESHOLDS_FROZEN', calibration_schema=calibration.SCHEMA,
        source_only=True, target_reads=0, source_V_labels_read=True, source_known_label_calibration=True,
        unknown_calibration_scores_read=0, unknown_calibration_labels_read=0, novel_validation_reads=0,
        new_encoder_updates=0, new_identity_head_updates=0, new_readout_updates=0, thresholds_ref='thresholds.npy')
    parent._write(frozen / 'complete.json', report)
    print('SOURCE_ORIGINAL_BASELINE_RISK_THRESHOLDS_FROZEN', arm, len(known['ids']), flush=True)
    novel = parent._load_features(export, 'novel_validation', False)
    if set(known['ids'].tolist()) & set(novel['ids'].tolist()): raise ValueError('Known/novel source roles overlap')
    novelness = readout._score(novel, geometry, state)
    baseline_novel = embedding._baseline(baseline_root / parent.FEATURE_REFS['novel_validation'], novel['ids'], 'novel_validation')
    ids = np.r_[known['ids'], novel['ids']]; values = np.concatenate((knownness, novelness))
    closed = np.r_[closed_known, novel['logits'].argmax(axis=1)].astype(np.int64)
    baseline = np.r_[baseline_known, baseline_novel].astype(np.int64)
    accepted = values >= risk['thresholds']; order = np.argsort(ids, kind='stable')
    parent._npz(output / 'predictions.npz', dict(ids=ids[order], method_names=np.asarray(calibration.METHODS),
        closed_prediction=closed[order], baseline_closed_prediction=baseline[order], knownness=values[order],
        accepted=accepted[order], prediction=np.where(accepted, closed[:, None], -1).astype(np.int64)[order],
        rx_i=np.r_[known['rx_i'], novel['rx_i']][order], day_i=np.r_[known['day_i'], novel['day_i']][order]))
    run_id = os.environ.get('PAMF_BASELINE_RISK_RUN_ID', RUN_ID)
    new_identity = dict(run_id=run_id, row_id=arm + '-s2026092701', arm=arm,
        code_commit=os.environ.get('PAMF_BASELINE_RISK_CODE_COMMIT'), output_root=str(output))
    result = dict(status='SOURCE_EMBEDDING_READOUT_PREDICTIONS_COMPLETE', source_only=True, target_reads=0,
        arm=arm, run_id=run_id, new_run_identity=new_identity, parent_row_root=str(parent_row_root),
        encoder_state_identity=marker['encoder_state_identity'], parent_export_identity=marker['parent_export_identity'],
        encoder_updates=222, identity_head_updates=222, readout_steps_per_model=200,
        update_scope='historical frozen parent only; this run has no model/readout updates',
        new_encoder_updates=0, new_identity_head_updates=0, new_readout_updates=0, default_head_replaced=False,
        descriptor_schema=readout.consensus.DESCRIPTOR_SCHEMA, readout_training_regime=readout.REGIME,
        calibration_schema=calibration.SCHEMA, source_V_labels_read=True, source_known_label_calibration=True,
        unknown_truth_read=False, truth_read=False, query_fit=False, unknown_calibration_scores_read=0,
        unknown_calibration_labels_read=0, known_count=len(known['ids']), novel_count=len(novel['ids']), count=len(ids),
        method_names=list(calibration.METHODS), thresholds_ref='calibration/thresholds.npy', predictions_ref='predictions.npz',
        independent_confirmation=False, source_novel_evaluation_reused=True,
        evidence_scope='same-known-V original-baseline empirical risk and repeated source novel evaluation; target0')
    parent._write(output / 'complete.json', result); return result


def validate_runtime(rt):
    if (not isinstance(rt, dict) or set(rt) != FIELDS or any(not isinstance(rt[key], str) or not rt[key].strip() for key in FIELDS - {'max_concurrent_rows'})
            or type(rt['max_concurrent_rows']) is not int or rt['max_concurrent_rows'] not in (1, 2)
            or len(rt['code_commit']) != 40 or any(c not in '0123456789abcdef' for c in rt['code_commit'])
            or Path(rt['run_root']).name != rt['run_id'] or Path(rt['parent_run_root']).name != PARENT_RUN
            or any(not Path(rt[key]).is_absolute() for key in ('run_root', 'release_root', 'python', 'parent_run_root'))):
        raise ValueError('Eight-field immutable CPU runtime with the fixed parent combo required')
    return rt


def _score_command(rt, root, arm):
    ctx = _parent_row(Path(rt['parent_run_root']) / 'rows' / arm); config = ctx['marker']['config']
    return [rt['python'], '-B', '-u', '-m', 'experiments.pamf_phase3_open_world.source_embedding_readout_scoring',
        '--predictions', str(root / 'rows' / arm / 'predict/predictions.npz'),
        '--known-features', str(Path(config['parent_run_root']) / 'export/source_V/features.npz'),
        '--novel-truth', str(Path(config['oe_data_root']) / 'novel_validation/truth/truth.npz'),
        '--output', str(root / 'rows' / arm / 'score')]


def _validate_marker(kind, marker, arm):
    prior_launch._validate_marker(kind, marker, arm)
    if kind == 'predict' and (marker.get('calibration_schema') != calibration.SCHEMA or marker.get('new_readout_updates') != 0):
        raise ValueError('Only new original-baseline-risk frozen predictions may be scored')


def dispatch(rt):
    validate_runtime(rt); root = Path(rt['run_root']); root.mkdir(parents=True, exist_ok=False)
    (root / 'rows').mkdir(); lock = threading.RLock()
    state = dict(status='PREPARING_SOURCE_ONLY', run_id=rt['run_id'], owner=rt['launch_owner'], code_commit=rt['code_commit'],
        pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, source_only=True, target_reads=0, calibration_schema=calibration.SCHEMA,
        added_GPU_workers=0, foreign_processes_stopped=0, new_encoder_updates=0, new_identity_head_updates=0, new_readout_updates=0,
        final_phase3_complete=False, independent_confirmation=False, source_novel_evaluation_reused=True,
        rows={arm: dict(status='PENDING', jobs={}) for arm in ARMS})
    def record():
        with lock:
            temp = root / 'pipeline_state.json.tmp'
            temp.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            os.replace(temp, root / 'pipeline_state.json')
    record()
    def row(arm):
        rowroot = root / 'rows' / arm; rowroot.mkdir(); (rowroot / 'logs').mkdir(); active = None
        try:
            commands = dict(predict=[rt['python'], '-B', '-u', '-m', 'experiments.pamf_phase3_open_world.source_baseline_risk_run',
                'predict', '--parent-row-root', str(Path(rt['parent_run_root']) / 'rows' / arm), '--output', str(rowroot / 'predict')],
                score=_score_command(rt, root, arm))
            for kind in ('predict', 'score'):
                active = kind; log = rowroot / 'logs' / (kind + '.log')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=rt['release_root'], OMP_NUM_THREADS='2',
                    MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                    PAMF_BASELINE_RISK_RUN_ID=rt['run_id'], PAMF_BASELINE_RISK_CODE_COMMIT=rt['code_commit'])
                with lock:
                    current = state['rows'][arm]; state['status'] = 'RUNNING_SOURCE_ONLY'; current['status'] = 'RUNNING'
                    current['jobs'][kind] = dict(status='PREPARING', argv=commands[kind], cwd=rt['release_root'], log=str(log)); record()
                with log.open('x', encoding='utf-8') as stream:
                    child = subprocess.Popen(commands[kind], cwd=rt['release_root'], env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    with lock: current['jobs'][kind].update(status='RUNNING', pid=child.pid); record()
                    code = child.wait()
                if code: raise RuntimeError(kind + ' exit=' + str(code))
                marker = parent._read(rowroot / kind / 'complete.json'); _validate_marker(kind, marker, arm)
                with lock: current['jobs'][kind].update(status='COMPLETE'); record()
            with lock: current.update(status='COMPLETE', eligible_methods=marker['eligible_methods']); record()
            return dict(complete=True, marker=marker)
        except Exception as exc:
            with lock:
                current = state['rows'][arm]
                if active is not None: current['jobs'][active].update(status='FAILED')
                current.update(status='FAILED', failure=str(exc)); record()
            (rowroot / 'logs/failure.log').write_text(traceback.format_exc(), encoding='utf-8')
            return dict(complete=False)
    with ThreadPoolExecutor(max_workers=rt['max_concurrent_rows']) as pool:
        futures = {arm: pool.submit(row, arm) for arm in ARMS}; outcomes = {arm: future.result() for arm, future in futures.items()}
    chosen = prior_launch.selection(outcomes); complete = all(v['complete'] for v in outcomes.values())
    state.update(status='ANALYZED_SOURCE_ONLY' if complete else 'PARTIAL_SOURCE_ONLY', ended_at=time.time(), selected_candidate=chosen,
        eligible_candidates=[chosen] if chosen else [], next_step=PASS_NEXT if chosen else ('NO_PROMOTION' if complete else 'INCOMPLETE_SOURCE_MATRIX_NO_PROMOTION'))
    record(); return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('predict'); p.add_argument('--parent-row-root', required=True); p.add_argument('--output', required=True)
    d = commands.add_parser('dispatch'); d.add_argument('--runtime', required=True)
    args = parser.parse_args(argv)
    result = predict(args.parent_row_root, args.output) if args.action == 'predict' else dispatch(parent._read(args.runtime))
    print(json.dumps(dict(status=result['status']))); return 1 if result['status'] == 'PARTIAL_SOURCE_ONLY' else 0


if __name__ == '__main__': raise SystemExit(main())
