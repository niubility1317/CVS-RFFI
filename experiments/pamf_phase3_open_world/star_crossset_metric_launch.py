"""Two cached-feature learners with per-row automatic eval and one CPU scorer."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261030-phase3-star-crossset-metric-s2026092701-r01'
ARMS=('S_SEP','S_SEP_TOPO')
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb',
 'max_added_workers_per_gpu','max_concurrent_rows','prepare_config','rows','novel_truth'}


def validate(r):
    from .star_crossset_metric_train import validate_prepare_config,validate_config as train_validate
    from .star_crossset_metric_eval import validate_config as eval_validate
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('code_commit','run_root','release_root','python','launch_owner','novel_truth'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None or r.get('allowed_gpus')!=list(range(8))
            or r.get('minimum_free_mb')!=4096 or r.get('max_added_workers_per_gpu')!=2 or r.get('max_concurrent_rows')!=2
            or not isinstance(r.get('rows'),dict) or set(r['rows'])!=set(ARMS)):
        raise ValueError('Exact two-arm source metric runtime required')
    root=Path(r['run_root']);c=validate_prepare_config(r['prepare_config'])
    if c['run_id']!=RUN or c['code_commit']!=r['code_commit'] or Path(c['output_root'])!=root/'features':
        raise ValueError('Owned common derived feature cache required')
    for arm,row in r['rows'].items():
        if not isinstance(row,dict) or set(row)!={'train_config','eval_config'}:raise ValueError('One train/eval pair per fixed arm')
        t,e=row['train_config'],row['eval_config'];train_validate(t);eval_validate(e)
        if (any(x['run_id']!=RUN or x['code_commit']!=r['code_commit'] or x['arm']!=arm for x in (t,e))
                or Path(t['cache_root'])!=root/'features' or Path(e['cache_root'])!=root/'features' or e['train_config']!=t
                or Path(t['output_root'])!=root/'rows'/arm/'train' or Path(e['output_root'])!=root/'rows'/arm/'predict'):
            raise ValueError('Owned row/code and source cache identities differ')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe existing run; never overwrite or restart')
    root.mkdir();(root/'logs').mkdir();(root/'configs').mkdir();(root/'rows').mkdir()
    _write(root/'configs/prepare.json',r['prepare_config'])
    for arm,row in r['rows'].items():
        (root/'rows'/arm).mkdir()
        for stage,c in row.items():_write(root/'configs'/(arm+'_'+stage+'.json'),c)
    cmd=[r['python'],'-B','-u','-m',__package__+'.star_crossset_metric_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(cmd,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=cmd,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time())
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def _cache_terminal(r):
    m=_read(Path(r['run_root'])/'features/complete.json')
    required=dict(status='STAR_CROSSSET_METRIC_FEATURES_COMPLETE',run_id=RUN,code_commit=r['code_commit'],counts=dict(L=6300,OE=6134),
        source_only=True,target_reads=0,private_truth_reads=0,original_U_s_reads=0,G_forward_rows=0,real_F_forward_rows=0,synthetic_F_forward_rows=2)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Complete matching derived source feature cache required')
    return m


def _terminal(r,arm,stage):
    folder=dict(train='train',eval='predict',score='score')[stage]
    m=_read(Path(r['run_root'])/'rows'/arm/folder/'complete.json')
    required=dict(run_id=RUN,code_commit=r['code_commit'],arm=arm,source_only=True,target_reads=0,final_phase3_complete=False)
    if stage=='train':required.update(status='STAR_CROSSSET_METRIC_SOURCE_FROZEN',updates=2220,native_updates=0,
        validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,private_truth_reads=0)
    elif stage=='eval':required.update(status='STAR_CROSSSET_METRIC_PREDICTIONS_COMPLETE',known_count=27000,novel_count=3150,
        source_L_count=6300,source_OE_count=6134,private_truth_reads=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,
        G_forward_rows=0,G_checkpoint_loaded=False,identity_predictions_unchanged=True,models_frozen_before_V=True,
        all_thresholds_frozen_before_novel=True,source_bank_deployed=False,per_record_source_rows_deployed=0)
    else:required.update(status='STAR_CROSSSET_METRIC_SCORES_COMPLETE',truth_last=True,full_goal_status='ACTIVE')
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Metric '+stage+' terminal differs')
    if stage=='score' and (type(m.get('source_gate_passed')) is not bool or type(m.get('eligible_for_target_prereg')) is not bool or not isinstance(m.get('next_step'),str)):
        raise ValueError('Independent source score eligibility fields required')
    return m


def _command(r,arm,stage):
    root=Path(r['run_root']);prefix=[r['python'],'-B','-u','-m']
    if stage=='train':return prefix+[__package__+'.star_crossset_metric_train','train','--config',str(root/'configs'/(arm+'_train_config.json')),'--device','cuda:0']
    if stage=='eval':return prefix+[__package__+'.star_crossset_metric_eval','--config',str(root/'configs'/(arm+'_eval_config.json')),'--device','cuda:0']
    return prefix+[__package__+'.star_crossset_metric_score','--prediction-root',str(root/'rows'/arm/'predict'),'--novel-truth',r['novel_truth'],'--output',str(root/'rows'/arm/'score')]


def _spawn(r,state,path,active,arm,stage,gpu):
    key=arm+'_'+stage;command=_command(r,arm,stage);log=(Path(r['run_root'])/'logs'/(key+'.log')).open('xb')
    try:child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r,gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    except Exception:log.close();raise
    active[key]=dict(process=child,log=log,arm=arm,stage=stage,gpu=gpu)
    state['jobs'][key]=dict(status='RUNNING',pid=child.pid,argv=command,cwd=r['release_root'],gpu=gpu,started_at=time.time())
    state['rows'][arm]['status']='RUNNING_'+stage.upper();_write(path,state)


def _available_gpu(r,active):
    counts={g:sum(v['gpu']==g for v in active.values()) for g in r['allowed_gpus']}
    values=[g for g in _gpu_free(r) if counts[g]<r['max_added_workers_per_gpu']]
    return min(values,key=lambda g:counts[g]) if values else None


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe original pipeline; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={},
        rows={arm:dict(status='PENDING_TRAIN') for arm in ARMS});_write(path,state);active={}
    try:
        available=_gpu_free(r)
        while not available:state.update(status='WAITING_MEMORY');_write(path,state);time.sleep(20);available=_gpu_free(r)
        command=[r['python'],'-B','-u','-m',__package__+'.star_crossset_metric_train','prepare','--config',str(root/'configs/prepare.json'),'--device','cuda:0']
        if not _job(r,state,path,'prepare',command,available[0]):raise RuntimeError('Prepare failed; no row may start')
        _cache_terminal(r);state['status']='RUNNING';_write(path,state)
        while any(v['status'] not in ('ANALYZED_SOURCE_ONLY','FAILED_PRESERVED') for v in state['rows'].values()):
            for key,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();del active[key];arm=item['arm'];stage=item['stage'];error=None;result=None
                try:
                    if code!=0:raise RuntimeError(stage+' process failed')
                    result=_terminal(r,arm,stage)
                except Exception as exc:error=str(exc)
                state['jobs'][key].update(status='FAILED' if error else 'COMPLETE',returncode=code,finished_at=time.time())
                if error:state['rows'][arm].update(status='FAILED_PRESERVED',error=error)
                elif stage=='train':state['rows'][arm]['status']='WAITING_EVAL'
                elif stage=='eval':state['rows'][arm]['status']='WAITING_SCORE'
                else:state['rows'][arm].update(status='ANALYZED_SOURCE_ONLY',source_gate_passed=result['source_gate_passed'],
                    eligible_for_target_prereg=result['eligible_for_target_prereg'],next_step=result['next_step'])
                _write(path,state)
            # Evaluating a frozen row takes priority over another pending train.
            for wanted,stage in (('WAITING_EVAL','eval'),('PENDING_TRAIN','train')):
                for arm in ARMS:
                    if state['rows'][arm]['status']!=wanted or sum(v['gpu'] is not None for v in active.values())>=r['max_concurrent_rows']:continue
                    gpu=_available_gpu(r,active)
                    if gpu is not None:_spawn(r,state,path,active,arm,stage,gpu)
            if not any(v['stage']=='score' for v in active.values()):
                for arm in ARMS:
                    if state['rows'][arm]['status']=='WAITING_SCORE':_spawn(r,state,path,active,arm,'score',None);break
            if active:time.sleep(2)
            elif any(v['status'] in ('PENDING_TRAIN','WAITING_EVAL') for v in state['rows'].values()):
                state['status']='WAITING_MEMORY';_write(path,state);time.sleep(20);state['status']='RUNNING'
        complete=all(v['status']=='ANALYZED_SOURCE_ONLY' for v in state['rows'].values());primary=state['rows']['S_SEP_TOPO']
        state.update(status='ANALYZED_SOURCE_ONLY' if complete else 'PARTIAL_FAILED_PRESERVED',all_rows_complete=complete,
            primary='S_SEP_TOPO::learned_known_support',eligible_for_target_prereg=bool(complete and primary.get('eligible_for_target_prereg')),
            next_step=primary.get('next_step','NO_PROMOTION') if complete else 'NO_PROMOTION',target_reads=0,final_phase3_complete=False,full_goal_status='ACTIVE')
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),eligible_for_target_prereg=False,final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args()
    {'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
