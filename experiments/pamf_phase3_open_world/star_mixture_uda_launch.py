"""One U-only adaptation producer followed by independent future-Q scoring."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261031-phase3-mixture-uda-s2026092701-r01'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb',
        'max_added_workers_per_gpu','run_config','source_novel_truth','clean_truth','star_truth'}


def validate(r):
    from .star_mixture_uda_run import validate_config
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in
                   ('code_commit','run_root','release_root','python','launch_owner','source_novel_truth','clean_truth','star_truth'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None or r['allowed_gpus']!=list(range(8))
            or r['minimum_free_mb']!=4096 or r['max_added_workers_per_gpu']!=2):
        raise ValueError('Exact single-U-only-adaptation runtime required')
    c=validate_config(r['run_config'])
    if c['run_id']!=RUN or c['code_commit']!=r['code_commit'] or Path(c['output_root'])!=Path(r['run_root'])/'predict':
        raise ValueError('Actual adaptation identity and immutable output differ')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe original run; never duplicate or replace')
    root.mkdir();(root/'logs').mkdir();(root/'configs').mkdir();_write(root/'configs/run.json',r['run_config'])
    argv=[r['python'],'-B','-u','-m',__package__+'.star_mixture_uda_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(argv,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=argv,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time())
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def _command(r,stage):
    root=Path(r['run_root']);prefix=[r['python'],'-B','-u','-m']
    if stage=='predict':return prefix+[__package__+'.star_mixture_uda_run','--config',str(root/'configs/run.json'),'--device','cuda:0']
    return prefix+[__package__+'.star_mixture_uda_score','--prediction-root',str(root/'predict'),
        '--source-novel-truth',r['source_novel_truth'],'--clean-truth',r['clean_truth'],'--star-truth',r['star_truth'],'--output',str(root/'score')]


def _terminal(r,stage):
    m=_read(Path(r['run_root'])/('predict' if stage=='predict' else 'score')/'complete.json')
    expected=dict(run_id=RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='predict':
        expected.update(status='MIXTURE_UDA_PREDICTIONS_COMPLETE',source_models_frozen_before_U=True,
            adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,private_truth_reads=0,
            target_U_reads=384,new_neural_updates=0,threshold_fit_updates=0,classification_unchanged=True)
    else:expected.update(status='MIXTURE_UDA_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE')
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Mixture-UDA '+stage+' terminal differs')
    if stage=='score' and (type(m.get('adaptation_success')) is not bool or type(m.get('development_gate_passed')) is not bool
            or not isinstance(m.get('next_step'),str)):raise ValueError('Independent adaptation and protection verdicts required')
    return m


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe existing pipeline; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        for stage in ('predict','score'):
            gpu=None
            if stage=='predict':
                available=_gpu_free(r)
                while not available:
                    state.update(status='WAITING_MEMORY');_write(path,state);time.sleep(20);available=_gpu_free(r)
                gpu=available[0]
            state.update(status='RUNNING_'+stage.upper());_write(path,state)
            if not _job(r,state,path,stage,_command(r,stage),gpu):raise RuntimeError(stage+' failed; preserve outputs')
            m=_terminal(r,stage)
        state.update(status='ANALYZED_DEVELOPMENT',all_stages_complete=True,adaptation_success=m['adaptation_success'],
            development_gate_passed=m['development_gate_passed'],primary='DA_M_ratio',next_step=m['next_step'],final_phase3_complete=False,full_goal_status='ACTIVE')
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),adaptation_success=False,development_gate_passed=False,final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args()
    {'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
