"""One launch owner: serial frozen support export, bounded CPU-only probe lanes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def validate_spec(spec):
    rows=spec['rows'];out=Path(spec['execution']['remote_run_root']).resolve()
    if spec['permissions']['query_use']!='none; query IQ/labels/truth/scores never read' or spec['probe']['query_access'] is not False:
        raise ValueError('Support-only permission mismatch')
    if len(rows)!=8 or len({r['row_id'] for r in rows})!=8:
        raise ValueError('All eight model/cohort rows required')
    if spec['execution']['cpu_lanes']!=4 or spec['execution']['blas_threads_per_lane']!=2:
        raise ValueError('Frozen resource contract mismatch')
    if spec['execution']['export_device']!='cuda:0' or spec['execution']['export_batch_size']!=32:
        raise ValueError('Frozen export contract mismatch')
    if spec['probe'].get('view_count')!=4 or spec['probe'].get('reuse_support_cache') is not False:
        raise ValueError('Four received views require a new support-only cache')
    pairs=set()
    for r in rows:
        p=Path(r['output_root']).resolve()
        if p.parent!=out or p.name!=r['row_id']:
            raise ValueError('Output outside unique row')
        co=spec['probe']['cohorts'][r['cohort']]
        if r['data_overrides']!=dict(capsule=co['capsule'],capsule_id=co['capsule_id'],expected_split_count=co['expected_split_count']):
            raise ValueError('Cohort binding mismatch')
        pairs.add((r['cohort'],r['seeds']['model']))
        digest=r['expected_checkpoint_sha256']
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Missing checkpoint binding')
    if pairs!={(c,s) for c in ('rx3','rx1') for s in range(2026092701,2026092705)}:
        raise ValueError('Fixed model/cohort matrix mismatch')


def commands(spec,row):
    probe=spec['probe'];co=probe['cohorts'][row['cohort']];out=Path(row['output_root']);release=Path(spec['code']['cwd'])
    common=['--capsule',co['capsule'],'--expected-capsule-id',co['capsule_id'],
            '--expected-checkpoint-sha256',row['expected_checkpoint_sha256']]
    export=[sys.executable,'-u',str(release/'tools/export_d92_branch_orbit_support_features.py'),
        '--checkpoint-root',row['source_root'],'--native-code',probe['native_code'],
        '--source-contract',probe['source_contract'],'--source-receivers',json.dumps(probe['source_receivers']),
        '--seed',str(row['seeds']['model']),'--output',str(out/'feature_cache'),
        '--device',spec['execution']['export_device'],'--batch-size',str(spec['execution']['export_batch_size']),*common]
    evaluate=[sys.executable,'-u',str(release/'tools/evaluate_d92_branch_orbit_ce_probe.py'),
        '--support-features',str(out/'feature_cache'),'--output',str(out/'probe'),
        '--config',co['evaluation_config'],'--expected-model-seed',str(row['seeds']['model']),*common]
    return export,evaluate


def verify_marker(path,spec,row,stage):
    data=read(path);co=spec['probe']['cohorts'][row['cohort']]
    expected='BRANCH_ORBIT_SUPPORT_FEATURES_COMPLETE' if stage=='export' else 'SUPPORT_PROBE_COMPLETE'
    if data['status']!=expected or data['capsule_id']!=co['capsule_id'] or data['checkpoint_sha256']!=row['expected_checkpoint_sha256'] or data['model_seed']!=row['seeds']['model']:
        raise ValueError('Artifact completion/binding mismatch')
    if stage=='probe' and (data['episodes']!=co['expected_split_count'] or data['query_rows_used']!=0 or data['source_rows_used']!=0
            or data['k1_episodes']!=co['expected_split_count']//4
            or data['oof_episodes']!=3*co['expected_split_count']//4
            or data['proxy_anchor_count']!=35*co['expected_split_count']//4):
        raise ValueError('Incomplete or non-support probe')
    if stage=='export' and (data['count']<1 or data['query_rows_read']!=0):
        raise ValueError('Invalid support export count or query access')
    return data


def launch(argv,log,cwd,device):
    env=dict(os.environ,PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0' if device=='export' else '',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with Path(log).open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        write(str(log)+'.process.json',dict(pid=child.pid,argv=argv,cwd=str(cwd),started=time.time(),kind=device))
        rc=child.wait()
    if rc!=0:
        raise RuntimeError(f'{device} failed with exit code {rc}; preserve log {log}; no automatic retry')


def run(spec,commit,launch_fn=launch):
    validate_spec(spec)
    root=Path(spec['execution']['remote_run_root']);root.mkdir(parents=True,exist_ok=False)
    rows=spec['rows'];state={r['row_id']:dict(status='PENDING') for r in rows};lock=threading.Lock()
    write(root/'startup.json',dict(pid=os.getpid(),argv=sys.argv,cwd=os.getcwd(),python=sys.executable,
        commit=commit,spec=spec,started=time.time(),query_access=False,source_sample_access=False))
    def update(row,status,**fields):
        with lock:
            state[row['row_id']]=dict(status=status,updated=time.time(),**fields)
            write(root/'state.json',state)
        print(json.dumps(dict(row=row['row_id'],**state[row['row_id']])),flush=True)
    def evaluate(row,argv):
        out=Path(row['output_root'])
        try:
            update(row,'PROBING_SUPPORT')
            launch_fn(argv,out/'probe.log',Path(spec['code']['cwd']),'probe')
            marker=verify_marker(out/'probe/probe_complete.json',spec,row,'probe')
            update(row,'SUPPORT_PROBE_COMPLETE',**{key:marker[key] for key in
                ('episodes','k1_episodes','oof_episodes','proxy_anchor_count','factorization_count','optimizer_steps')})
            return True
        except Exception as exc:
            update(row,'FAILED',stage='probe',error=str(exc));return False
    futures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in rows:
            out=Path(row['output_root']);out.mkdir(exist_ok=False)
            export,probe=commands(spec,row)
            try:
                update(row,'EXTRACTING_SUPPORT')
                launch_fn(export,out/'export.log',Path(spec['code']['cwd']),'export')
                marker=verify_marker(out/'feature_cache/features_complete.json',spec,row,'export')
                update(row,'EXPORTED_SUPPORT',count=marker['count'])
                futures.append(pool.submit(evaluate,row,probe))
            except Exception as exc:
                update(row,'FAILED',stage='export',error=str(exc))
        for future in as_completed(futures):
            future.result()
    success=all(r['status']=='SUPPORT_PROBE_COMPLETE' for r in state.values())
    write(root/'complete.json',dict(status='SUPPORT_PROBE_COMPLETE' if success else 'FAILED',commit=commit,
        model_rows=len(rows),completed_rows=sum(r['status']=='SUPPORT_PROBE_COMPLETE' for r in state.values()),
        episodes=sum(r.get('episodes',0) for r in state.values()),query_access=False,source_sample_access=False,
        k1_episodes=sum(r.get('k1_episodes',0) for r in state.values()),
        oof_episodes=sum(r.get('oof_episodes',0) for r in state.values()),
        proxy_anchor_count=sum(r.get('proxy_anchor_count',0) for r in state.values()),
        factorization_count=sum(r.get('factorization_count',0) for r in state.values()),
        optimizer_steps=sum(r.get('optimizer_steps',0) for r in state.values()),
        query_performance_claim=False,finished=time.time()))
    if not success:
        raise RuntimeError('One or more support probe lanes failed; healthy lanes retained; no automatic retry')


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();run(read(a.spec),a.commit)


if __name__=='__main__':main()
