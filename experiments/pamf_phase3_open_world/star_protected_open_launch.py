"""Three source-only protected160 rows, then independent truth-side scoring."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .star_representation_launch import _read, _write, _env, _gpu_free, _job

RUN='20261030-phase3-star-protected-open-s2026092701-r01'
ARMS=('N0','R0','R1')
PRIMARY='R1::relation_max'
CACHE_RUN='20261029-phase3-star-representation-s2026092701-r01'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus',
    'minimum_free_mb','max_added_workers_per_gpu','max_concurrent_rows','cache_config','rows','novel_truth'}


def validate(r):
    from .star_protected_open_cache import validate_config as cache_validate
    from .star_protected_open_train import validate_config as train_validate
    from .star_protected_open_eval import validate_config as eval_validate
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in
                ('code_commit','run_root','release_root','python','launch_owner','novel_truth'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None
            or r.get('allowed_gpus')!=list(range(8)) or r.get('minimum_free_mb')!=4096
            or r.get('max_added_workers_per_gpu')!=2 or type(r.get('max_concurrent_rows')) is not int
            or r['max_concurrent_rows']!=3 or not isinstance(r.get('rows'),dict) or set(r['rows'])!=set(ARMS)):
        raise ValueError('Only the three fixed source-only arms and scorer-only source truth allowed')
    root=Path(r['run_root']);source=root.parent/CACHE_RUN/'cache';cache_validate(r['cache_config'])
    cc=r['cache_config']
    if (cc['run_id']!=RUN or cc['code_commit']!=r['code_commit']
            or Path(cc['output_root'])!=root/'features'
            or Path(cc['L_root'])!=source/'L' or Path(cc['OE_root'])!=source/'OE'):
        raise ValueError('One owned source-only feature cache from exact existing L/OE roles required')
    for arm,row in r['rows'].items():
        if not isinstance(row,dict) or set(row)!={'train_config','eval_config'}:
            raise ValueError('One independent train/eval pair per arm required')
        t,e=row['train_config'],row['eval_config'];train_validate(t);eval_validate(e)
        if (t['arm']!=arm or e['arm']!=arm or any(c['run_id']!=RUN or c['code_commit']!=r['code_commit'] for c in (t,e))
                or Path(t['feature_root'])!=root/'features' or Path(t['output_root'])!=root/'rows'/arm/'train'
                or Path(e['feature_root'])!=root/'features' or Path(e['cache_root'])!=source
                or Path(e['output_root'])!=root/'rows'/arm/'predict' or e['train_config']!=t):
            raise ValueError('Owned row/code/budget and source feature/P1 paths differ')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe existing output; never relaunch or overwrite')
    root.mkdir();(root/'configs').mkdir();(root/'logs').mkdir();(root/'rows').mkdir()
    _write(root/'configs/cache.json',r['cache_config'])
    for arm,row in r['rows'].items():
        (root/'rows'/arm).mkdir()
        for role,c in row.items():_write(root/'configs'/(arm+'_'+role+'.json'),c)
    command=[r['python'],'-B','-u','-m',__package__+'.star_protected_open_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,cwd=r['release_root'],
        time=time.time(),launch_owner=r['launch_owner'])
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def _cache_terminal(r):
    m=_read(Path(r['run_root'])/'features/complete.json')
    required=dict(schema='pamf_phase3_protected_source_features_v1',status='STAR_PROTECTED_SOURCE_FEATURES_COMPLETE',
        run_id=RUN,code_commit=r['code_commit'],counts=dict(L=6300,OE=6134),source_only=True,target_reads=0,
        private_truth_reads=0,original_U_s_reads=0,G_forward_rows=0)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Completed matching source-only feature cache required')
    return m


def _terminal(r,arm,stage):
    folder=dict(train='train',eval='predict',score='score')[stage]
    m=_read(Path(r['run_root'])/'rows'/arm/folder/'complete.json')
    status=dict(train='STAR_PROTECTED_SOURCE_FROZEN',eval='STAR_PROTECTED_PREDICTIONS_COMPLETE',score='STAR_PROTECTED_SCORES_COMPLETE')[stage]
    if any(m.get(k)!=v for k,v in dict(status=status,run_id=RUN,code_commit=r['code_commit'],arm=arm).items()):
        raise ValueError(stage+' complete identity differs')
    if stage=='train':
        if (m.get('updates')!=2220 or m.get('model_native_state_unchanged') is not True
                or any(m.get(k)!=0 for k in ('target_reads','validation_reads','novel_validation_reads','original_U_s_reads','private_truth_reads'))):
            raise ValueError('Frozen-F0/source-only training budget or permissions differ')
    elif stage=='eval':
        if (m.get('known_count')!=27000 or m.get('novel_count')!=3150
                or any(m.get(k)!=0 for k in ('source_truth_reads','target_reads'))
                or m.get('thresholds_frozen_before_novel') is not True
                or m.get('frozen_models_verified') is not True or m.get('source_aggregates_unchanged') is not True):
            raise ValueError('All frozen source predictions must precede truth-side scoring')
    elif (type(m.get('eligible_for_target_prereg')) is not bool or m.get('truth_last') is not True
            or m.get('final_phase3_complete') is not False or not isinstance(m.get('next_step'),str)):
        raise ValueError('Independent source score terminal fields differ')
    return m


def row_driver(runtime,arm,gpu):
    r=validate(_read(runtime))
    if arm not in ARMS or gpu not in r['allowed_gpus']:raise ValueError('Only declared owned row/GPU allowed')
    root=Path(r['run_root']);path=root/'rows'/arm/'state.json'
    if path.exists():raise FileExistsError('Observe existing owned row state; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],arm=arm,pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={})
    _write(path,state)
    commands=[
        ('train',[r['python'],'-B','-u','-m',__package__+'.star_protected_open_train','--config',str(root/'configs'/(arm+'_train_config.json')),'--device','cuda:0'],gpu),
        ('eval',[r['python'],'-B','-u','-m',__package__+'.star_protected_open_eval','--config',str(root/'configs'/(arm+'_eval_config.json')),'--device','cuda:0'],gpu),
        ('score',[r['python'],'-B','-u','-m',__package__+'.star_protected_open_score','--prediction-root',str(root/'rows'/arm/'predict'),
            '--novel-truth',r['novel_truth'],'--output',str(root/'rows'/arm/'score')],None)]
    try:
        for stage,command,device in commands:
            if device is not None:
                while device not in _gpu_free(r):
                    state['status']='WAITING_MEMORY';_write(path,state);time.sleep(20)
                state['status']='RUNNING';_write(path,state)
            if not _job(r,state,path,arm+'_'+stage,command,device):raise RuntimeError(stage+' failed; preserve output, no automatic retry')
            result=_terminal(r,arm,stage)
        state.update(status='ANALYZED_SOURCE_ONLY',eligible_for_target_prereg=result['eligible_for_target_prereg'],
            next_step=result['next_step'],target_reads=0,final_phase3_complete=False)
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],arm,flush=True)


def _available_gpu(r,active):
    counts={g:sum(v['gpu']==g for v in active.values()) for g in r['allowed_gpus']}
    eligible=[g for g in _gpu_free(r) if counts[g]<r['max_added_workers_per_gpu']]
    return min(eligible,key=lambda g:counts[g]) if eligible else None


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe existing pipeline state; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={})
    _write(path,state);pending=list(ARMS);active={}
    try:
        available=_gpu_free(r)
        while not available:
            state['status']='WAITING_MEMORY';_write(path,state);time.sleep(20);available=_gpu_free(r)
        state['status']='RUNNING';_write(path,state)
        command=[r['python'],'-B','-u','-m',__package__+'.star_protected_open_cache','--config',str(root/'configs/cache.json'),'--device','cuda:0']
        if not _job(r,state,path,'cache',command,available[0]):raise RuntimeError('Cache failed; no row may start')
        _cache_terminal(r)
        while pending or active:
            while pending and len(active)<r['max_concurrent_rows']:
                gpu=_available_gpu(r,active)
                if gpu is None:break
                arm=pending.pop(0)
                command=[r['python'],'-B','-u','-m',__package__+'.star_protected_open_launch','row','--runtime',str(runtime),'--arm',arm,'--gpu',str(gpu)]
                log=(root/'logs'/(arm+'_driver.log')).open('xb')
                try:child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                except Exception:log.close();raise
                active[arm]=dict(process=child,gpu=gpu,log=log)
                state['jobs'][arm]=dict(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,cwd=r['release_root'],started_at=time.time())
                _write(path,state)
            for arm,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();sp=root/'rows'/arm/'state.json';row=_read(sp) if sp.is_file() else {}
                passed=code==0 and row.get('status')=='ANALYZED_SOURCE_ONLY'
                state['jobs'][arm].update(status='COMPLETE' if passed else 'FAILED',returncode=code,finished_at=time.time())
                del active[arm]
                if not passed:
                    for skipped in pending:state['jobs'][skipped]=dict(status='NOT_STARTED_AFTER_TECHNICAL_FAILURE')
                    pending.clear()
                _write(path,state)
            if pending or active:time.sleep(20 if pending and not active else 2)
        complete=all(state['jobs'].get(arm,{}).get('status')=='COMPLETE' for arm in ARMS)
        primary=_terminal(r,'R1','score') if complete else None
        state.update(status='ANALYZED_SOURCE_ONLY' if complete else 'PARTIAL_FAILED_PRESERVED',primary=PRIMARY,
            all_rows_complete=complete,eligible_for_target_prereg=bool(complete and primary['eligible_for_target_prereg']),
            next_step=primary['next_step'] if complete else 'NO_PROMOTION',target_reads=0,final_phase3_complete=False)
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),eligible_for_target_prereg=False,final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver','row'));p.add_argument('--runtime',required=True)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='row':
        if a.arm is None or a.gpu is None:p.error('row requires --arm and --gpu')
        return row_driver(a.runtime,a.arm,a.gpu)
    return {'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
