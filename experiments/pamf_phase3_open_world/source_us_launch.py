"""Fresh receiver data, fixed PCD student, frozen controls, independent scoring."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261031-phase3-source-us-replay-s2026092701-r01'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb','max_added_workers_per_gpu',
    'cache_config','train_config','eval_config','source_novel_truth','clean_truth','star_truth'}


def validate(r):
    from . import source_us_cache as cache
    from . import source_us_train as train
    from . import source_us_run as predict
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('code_commit','run_root','release_root','python','launch_owner','source_novel_truth','clean_truth','star_truth'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None or r['allowed_gpus']!=list(range(8)) or r['minimum_free_mb']!=4096 or r['max_added_workers_per_gpu']!=2):raise ValueError('Exact fresh-development-rx3-2 runtime required')
    k=cache.validate_config(r['cache_config']);t=train.validate_config(r['train_config']);e=predict.validate_config(r['eval_config'])
    if k['run_id']!=RUN or k['code_commit']!=r['code_commit'] or Path(k['output_root'])!=Path(r['run_root'])/'cache' or Path(t['Us_public_root'])!=Path(k['output_root'])/'U_s':raise ValueError('Independent source Us cache/run path differs')
    if (t['run_id']!=RUN or e['run_id']!=RUN or t['code_commit']!=r['code_commit'] or e['code_commit']!=r['code_commit'] or e['train_config']!=t or e['source_control_config']['source_control_config']!=t['reference_config']
            or Path(t['output_root'])!=Path(r['run_root'])/'train' or Path(e['output_root'])!=Path(r['run_root'])/'predict'):
        raise ValueError('Actual neural/evaluation identity and immutable roots must agree')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe original run; never duplicate or replace')
    root.mkdir();(root/'logs').mkdir();(root/'configs').mkdir();_write(root/'configs/cache.json',r['cache_config']);_write(root/'configs/train.json',r['train_config']);_write(root/'configs/eval.json',r['eval_config'])
    argv=[r['python'],'-B','-u','-m',__package__+'.source_us_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:child=subprocess.Popen(argv,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    v=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=argv,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time());_write(root/'launch.json',v);print(json.dumps(v),flush=True)


def _command(r,stage):
    root=Path(r['run_root']);p=[r['python'],'-B','-u','-m']
    if stage=='cache':return p+[__package__+'.source_us_cache','--config',str(root/'configs/cache.json'),'--device','cuda:0']
    if stage=='train':return p+[__package__+'.source_us_train','--config',str(root/'configs/train.json'),'--device','cuda:0']
    if stage=='predict':return p+[__package__+'.source_us_run','--config',str(root/'configs/eval.json'),'--device','cuda:0']
    return p+[__package__+'.source_us_score','--prediction-root',str(root/'predict'),'--source-novel-truth',r['source_novel_truth'],'--clean-truth',r['clean_truth'],'--star-truth',r['star_truth'],'--output',str(root/'score')]


def _terminal(r,stage):
    m=_read(Path(r['run_root'])/stage/'complete.json' if stage!='predict' else Path(r['run_root'])/'predict/complete.json')
    expected=dict(run_id=RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='cache':expected.update(status='SOURCE_US_G10_CACHE_COMPLETE',original_U_s_reads=56700,Us_labels_read=0,private_truth_reads=0,target_reads=0,G_forward_rows=56700,real_F_forward_rows=0,original_models_unchanged=True)
    elif stage=='train':expected.update(status='SOURCE_US_REPLAY_TRAINING_COMPLETE',updates=444,total_optimizer_updates=444,source_sampling_matched=True,source_only=True,
        original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,Us_ground_truth_reads=0,target_U_reads=0,target_Q_reads=0,private_truth_reads=0)
    elif stage=='predict':expected.update(status='SOURCE_US_REPLAY_PREDICTIONS_COMPLETE',models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,private_truth_reads=0,target_U_reads=0,classification_unchanged=True,source_density_models_fitted=1,new_threshold_sets_fitted=2,source_control_readouts_reused_exactly=True)
    else:expected.update(status='SOURCE_US_REPLAY_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE')
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Fresh development '+stage+' terminal differs')
    if stage=='score' and (m.get('adaptation_success') is not None or type(m.get('development_gate_passed')) is not bool or not isinstance(m.get('next_step'),str)):raise ValueError('Independent UDA and joint-protection verdicts required')
    return m


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe original pipeline; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        for stage in ('cache','train','predict','score'):
            gpu=None
            if stage!='score':
                available=_gpu_free(r)
                while not available:state.update(status='WAITING_MEMORY');_write(path,state);time.sleep(20);available=_gpu_free(r)
                gpu=available[0]
            state.update(status='RUNNING_'+stage.upper());_write(path,state)
            if not _job(r,state,path,stage,_command(r,stage),gpu):raise RuntimeError(stage+' failed; preserve outputs')
            m=_terminal(r,stage)
        state.update(status='ANALYZED_DEVELOPMENT',all_stages_complete=True,adaptation_success=None,development_gate_passed=m['development_gate_passed'],
            primary='SOURCE_US_predicted_ratio',next_step=m['next_step'],final_phase3_complete=False,full_goal_status='ACTIVE')
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),adaptation_success=False,development_gate_passed=False,final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args();{'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
