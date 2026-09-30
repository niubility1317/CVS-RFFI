"""Two CPU lanes for the fixed support-only sequential residual pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time
import json

import run_d92_registration_diagnostic as baseline
from run_d92_branch_support_probe import read, write, launch

ROOT=Path(__file__).resolve().parents[1]
CHANNEL=baseline.CHANNEL
KS=baseline.KS
NEW_COUNTS=baseline.NEW_COUNTS
SCENARIOS=baseline.SCENARIOS
require=baseline.require
validate_selection=baseline.validate_selection
PROBE_CONFIG=read(ROOT/'configs/d92_sequential_residual_head_frozen_20260930.json')['algorithm']
CONFIG_NAMES={c:f'configs/d92_sequential_residual_support_{c}_20260930.json' for c in ('rx3','rx1')}
STATUS='SEQUENTIAL_RESIDUAL_PROBE_COMPLETE'
COUNTERS=('episodes','k1_episodes','oof_episodes','proxy_anchor_count','sequence_paths',
          'head_fit_count','factorization_count','residual_training_stages','optimizer_steps')


def validate_spec(spec):
    p=spec['probe']
    require(p['algorithm']==PROBE_CONFIG and p['channel']==CHANNEL,'Frozen sequential residual algorithm/channel mismatch')
    require(p['expected_residual_training_stages']==4576 and p['expected_optimizer_steps']==292864,'Residual training budget mismatch')
    require(p['candidate']=='R_seq' and p['controls']==['R0','R_reset'],'Fixed candidate/control mismatch')
    for name,co in p['cohorts'].items():
        require(co['evaluation_config']==str(PurePosixPath(spec['code']['cwd'])/CONFIG_NAMES[name]),'Residual config escaped release')
    # Reuse the established support identity/permission checks without changing that module.
    view=deepcopy(spec)
    view['probe']['algorithm']=baseline.DIAGNOSTIC_CONFIG
    for name,co in view['probe']['cohorts'].items():
        co['evaluation_config']=str(PurePosixPath(spec['code']['cwd'])/baseline.CONFIG_NAMES[name])
    baseline.validate_spec(view)


def command(spec,row):
    co=spec['probe']['cohorts'][row['cohort']]
    return [sys.executable,'-u',str(Path(spec['code']['cwd'])/'tools/evaluate_d92_sequential_residual_probe.py'),
        '--support-features',row['support_features'],'--capsule',co['capsule'],
        '--output',str(Path(row['output_root'])/'probe'),'--config',co['evaluation_config'],
        '--expected-capsule-id',co['capsule_id'],'--expected-checkpoint-sha256',row['expected_checkpoint_sha256'],
        '--expected-model-seed',str(row['seeds']['model'])]


def verify_marker(path,spec,row):
    marker=read(path);co=spec['probe']['cohorts'][row['cohort']]
    expected=dict(status=STATUS,capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],
        model_seed=row['seeds']['model'],algorithm=PROBE_CONFIG,selection=co['selection'],
        episodes=40,k1_episodes=10,oof_episodes=30,proxy_anchor_count=350,sequence_paths=440,
        head_fit_count=792,residual_training_stages=1144,optimizer_steps=73216,query_rows_used=0,source_rows_used=0)
    require(all(marker.get(k)==v for k,v in expected.items()),'Incomplete or incorrectly bound residual pilot')
    require(type(marker.get('factorization_count')) is int and 0<=marker['factorization_count']<=792,'Invalid actual base factorization count')
    for key in COUNTERS+('query_rows_used','source_rows_used'):
        require(type(marker.get(key)) is int,'Invalid counter type: '+key)
    require(marker.get('producer_matrix')==co['matrix'],'Producer matrix marker mismatch')
    return marker


def run(spec,commit,launch_fn=launch):
    validate_spec(spec)
    for co in spec['probe']['cohorts'].values():
        require(read(co['evaluation_config'])==dict(algorithm=PROBE_CONFIG,producer_matrix=co['matrix'],selection=co['selection']),
                'Evaluator config/spec mismatch')
    root=Path(spec['execution']['remote_run_root']);root.mkdir(parents=True,exist_ok=False)
    rows=spec['rows'];state={r['row_id']:dict(status='PENDING') for r in rows};lock=threading.Lock()
    write(root/'startup.json',dict(spec=spec,commit=commit,pid=os.getpid(),argv=sys.argv,
        cpu_lanes=2,blas_threads_per_lane=2,query_access=False,source_sample_access=False,
        checkpoint_loaded=False,gpu_use=False,adapter_training=True,adapted_state_inherited=True,
        actual_A=None,optimizer_steps_planned=292864,started=time.time()))
    def update(row,status,**fields):
        with lock:
            state[row['row_id']]=dict(status=status,updated=time.time(),**fields)
            write(root/'state.json',state)
        print(json.dumps(dict(row=row['row_id'],status=status,**fields)),flush=True)
    def work(row):
        try:
            out=Path(row['output_root']);out.mkdir(exist_ok=False);update(row,'PROBING_SUPPORT')
            launch_fn(command(spec,row),out/'probe.log',Path(spec['code']['cwd']),'probe')
            marker=verify_marker(out/'probe/probe_complete.json',spec,row)
            update(row,STATUS,**{k:marker[k] for k in COUNTERS})
        except Exception as exc:
            update(row,'FAILED',error_type=type(exc).__name__,error=str(exc))
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(work,rows))
    complete=all(v['status']==STATUS for v in state.values())
    write(root/'complete.json',dict(status=STATUS if complete else 'FAILED',commit=commit,model_rows=4,
        completed_rows=sum(v['status']==STATUS for v in state.values()),
        **{k:sum(v.get(k,0) for v in state.values()) for k in COUNTERS},
        query_access=False,source_sample_access=False,checkpoint_loaded=False,gpu_use=False,finished=time.time()))
    if not complete:raise RuntimeError('Residual pilot lane failed; healthy lanes retained, no retry')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();run(read(a.spec),a.commit)
