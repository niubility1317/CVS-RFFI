"""One CPU-only head-aligned source fit/predict/score owner."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .launch import read, write
from . import source_risk_launch as risk_launch
from .source_open_launch import _integer
from .source_consensus import DESCRIPTOR_SCHEMA
from .source_risk_calibration import METHODS


FIELDS = risk_launch.FIELDS
STAGES = ('fit', 'predict', 'score')
validate_runtime = risk_launch.validate_runtime


def _validate_marker(kind, marker):
    if kind in ('predict', 'score'):
        risk_launch._validate_marker(kind, marker)
        if kind == 'predict' and marker.get('descriptor_schema') != DESCRIPTOR_SCHEMA:
            raise ValueError('Predictions must use the new head-aligned descriptor')
        return
    if kind != 'fit' or not isinstance(marker, dict) or marker.get('source_only') is not True:
        raise ValueError('A complete source-only head-aligned fit is required')
    for key, expected in dict(target_reads=0, model_seed=2026092701, encoder_updates=0,
            identity_head_updates=0, readout_steps_per_model=200, source_validation_reads=0,
            novel_validation_reads=0, OE_truth_reads=0).items():
        _integer(marker, key, expected)
    if (marker.get('status') != 'SOURCE_CONSENSUS_READOUTS_FROZEN'
            or marker.get('descriptor_schema') != DESCRIPTOR_SCHEMA or marker.get('geometry_fitted') is not False
            or marker.get('method_names') != list(METHODS)
            or marker.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}):
        raise ValueError('The fixed new descriptor/readout budget or inherited geometry identity differs')


def _commands(rt, root):
    prefix = [rt['python'], '-B', '-u', '-m']
    return dict(fit=prefix + ['experiments.pamf_phase3_open_world.source_consensus_run', 'fit',
            '--parent-run-root', rt['parent_run_root'], '--model-seed', '2026092701', '--output', str(root / 'fit')],
        predict=prefix + ['experiments.pamf_phase3_open_world.source_consensus_run', 'predict',
            '--parent-run-root', rt['parent_run_root'], '--fit-root', str(root / 'fit'), '--output', str(root / 'predict')],
        score=risk_launch._commands(rt, root)['score'])


def dispatch(rt):
    validate_runtime(rt)
    root = Path(rt['run_root']); root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    state = dict(status='PREPARING_SOURCE_ONLY', run_id=rt['run_id'], descriptor_schema=DESCRIPTOR_SCHEMA,
        source_only=True, target_reads=0, owner=rt['launch_owner'], code_commit=rt['code_commit'],
        parent_run_root=rt['parent_run_root'], pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
        started_at=time.time(), jobs={}, added_GPU_workers=0, max_added_workers_per_gpu=2,
        foreign_processes_stopped=0, encoder_updates=0, identity_head_updates=0, readout_steps_per_model=200,
        independent_confirmation=False, source_novel_evaluation_reused=True, final_phase3_complete=False,
        risk_scope='empirical risk on the same known-source calibration samples',
        evidence_scope='repeated source development evaluation; no established domain invariance or formal target claim')
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
