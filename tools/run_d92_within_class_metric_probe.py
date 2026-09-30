"""Two CPU lanes for the fixed support-only within-class metric diagnostic."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time

import run_d92_registration_diagnostic as baseline
from run_d92_branch_support_probe import read, write, launch

ROOT=Path(__file__).resolve().parents[1]
CHANNEL=baseline.CHANNEL
KS=baseline.KS
NEW_COUNTS=baseline.NEW_COUNTS
SCENARIOS=baseline.SCENARIOS
require=baseline.require
validate_selection=baseline.validate_selection
PROBE_CONFIG=read(ROOT/'configs/d92_within_class_metric_frozen_20260930.json')['algorithm']
CONFIG_NAMES={c:f'configs/d92_within_metric_diagnostic_{c}_20260930.json' for c in ('rx3','rx1')}
STATUS='WITHIN_CLASS_METRIC_PROBE_COMPLETE'
COUNTERS=('episodes','k1_episodes','oof_episodes','proxy_anchor_count','sequence_paths',
          'baseline_head_fit_count','metric_fit_count','metric_nonidentity_count',
          'metric_head_fit_count','head_fit_count','factorization_count','metric_factorization_count',
          'diagnostic_fit_count','diagnostic_factorization_count','optimizer_steps')


def validate_spec(spec):
    p=spec['probe']
    require(p['algorithm']==PROBE_CONFIG and p['channel']==CHANNEL,'Frozen within-class metric/channel mismatch')
    require(p['expected_baseline_head_fits']==3168 and p['expected_metric_fit_count']==1760
        and p['maximum_metric_nonidentity_count']==360 and p['maximum_metric_head_fits']==648
        and p['expected_head_fits']==3816 and p['expected_diagnostic_fit_count']==2160
        and p['expected_optimizer_steps']==0,'Metric diagnostic budget mismatch')
    require(p['candidate']=='R_metric' and p['controls']==['R0'],'Fixed metric/control mismatch')
    require(p['interpretation']=='mechanism_pilot_no_direct_promotion','Mechanism interpretation mismatch')
    for name,co in p['cohorts'].items():
        require(co['evaluation_config']==str(PurePosixPath(spec['code']['cwd'])/CONFIG_NAMES[name]),'Metric config escaped release')
    view=deepcopy(spec)
    view['probe'].update(algorithm=baseline.DIAGNOSTIC_CONFIG,expected_head_fits=3168)
    for name,co in view['probe']['cohorts'].items():
        co['evaluation_config']=str(PurePosixPath(spec['code']['cwd'])/baseline.CONFIG_NAMES[name])
    baseline.validate_spec(view)


def command(spec,row):
    co=spec['probe']['cohorts'][row['cohort']]
    return [sys.executable,'-u',str(Path(spec['code']['cwd'])/'tools/evaluate_d92_within_class_metric_probe.py'),
        '--support-features',row['support_features'],'--capsule',co['capsule'],
        '--output',str(Path(row['output_root'])/'probe'),'--config',co['evaluation_config'],
        '--expected-capsule-id',co['capsule_id'],'--expected-checkpoint-sha256',row['expected_checkpoint_sha256'],
        '--expected-model-seed',str(row['seeds']['model'])]


def verify_marker(path,spec,row):
    marker=read(path);co=spec['probe']['cohorts'][row['cohort']]
    expected=dict(status=STATUS,capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],
        model_seed=row['seeds']['model'],algorithm=PROBE_CONFIG,selection=co['selection'],producer_matrix=co['matrix'],
        episodes=40,k1_episodes=10,oof_episodes=30,proxy_anchor_count=350,sequence_paths=440,
        baseline_head_fit_count=792,metric_fit_count=440,diagnostic_fit_count=540,
        optimizer_steps=0,query_rows_used=0,source_rows_used=0)
    require(all(marker.get(k)==v for k,v in expected.items()),'Incomplete or incorrectly bound metric pilot')
    for key in COUNTERS+('query_rows_used','source_rows_used'):
        require(type(marker.get(key)) is int and marker[key]>=0,'Invalid counter: '+key)
    require(marker['metric_nonidentity_count']<=90 and marker['metric_head_fit_count']<=162,'Metric fit budget exceeded')
    require(marker['metric_nonidentity_count']<=marker['metric_head_fit_count']<=2*marker['metric_nonidentity_count'],
        'Nonidentity/head accounting mismatch')
    require(marker['head_fit_count']==marker['baseline_head_fit_count']+marker['metric_head_fit_count'], 'Head count identity mismatch')
    require(marker['factorization_count']<=marker['head_fit_count'],'Invalid actual factorization count')
    require(marker['metric_factorization_count']<=marker['metric_nonidentity_count']
        and marker['diagnostic_factorization_count']<=marker['diagnostic_fit_count'],
        'Invalid metric/diagnostic Cholesky counts')
    require(marker.get('truth_read') is False,'Query truth access forbidden')
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
        checkpoint_loaded=False,gpu_use=False,adapter_training=False,adapted_state_inherited=True,
        actual_A=None,optimizer_steps_planned=0,started=time.time()))
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
    if not complete:raise RuntimeError('Metric pilot lane failed; healthy lanes retained, no retry')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();run(read(a.spec),a.commit)
