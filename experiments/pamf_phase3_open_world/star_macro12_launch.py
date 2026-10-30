"""One fixed-score source calibration followed by independent truth-side scoring."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261030-phase3-star-macro12-risk-s2026092701-r01'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb','max_added_workers_per_gpu','run_config','novel_truth'}


def validate(r):
    from .star_macro12_run import validate_config
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('code_commit','run_root','release_root','python','launch_owner','novel_truth'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None or r['allowed_gpus']!=list(range(8))
            or r['minimum_free_mb']!=4096 or r['max_added_workers_per_gpu']!=2):raise ValueError('Exact macro12 source runtime required')
    c=validate_config(r['run_config'])
    if c['run_id']!=RUN or c['code_commit']!=r['code_commit'] or Path(c['output_root'])!=Path(r['run_root'])/'predict':
        raise ValueError('Owned source prediction config/root must match')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe original run; never replace or restart')
    root.mkdir();(root/'configs').mkdir();(root/'logs').mkdir();_write(root/'configs/run.json',r['run_config'])
    command=[r['python'],'-B','-u','-m',__package__+'.star_macro12_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time())
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def _terminal(r,stage):
    m=_read(Path(r['run_root'])/stage/'complete.json')
    expected=dict(run_id=RUN,code_commit=r['code_commit'],source_only=True,target_reads=0,final_phase3_complete=False)
    if stage=='predict':expected.update(status='STAR_MACRO12_PREDICTIONS_COMPLETE',known_count=27000,novel_count=3150,private_truth_reads=0,
        native_updates=0,neural_updates=0,real_signal_forward_rows=0,synthetic_signal_forward_rows=2,real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,controls_reused_exactly=True,model_parameters_buffers_unchanged=True,
        source_bank_deployed=False,per_record_source_rows_deployed=0)
    else:expected.update(status='STAR_MACRO12_SCORES_COMPLETE',truth_last=True,full_goal_status='ACTIVE')
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Macro12 '+stage+' terminal differs')
    if stage=='score' and (type(m.get('source_gate_passed')) is not bool or type(m.get('eligible_for_target_prereg')) is not bool or not isinstance(m.get('next_step'),str)):
        raise ValueError('Independent source score eligibility fields required')
    return m


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe original pipeline; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        available=_gpu_free(r)
        while not available:state.update(status='WAITING_MEMORY');_write(path,state);time.sleep(20);available=_gpu_free(r)
        state.update(status='RUNNING',gpu=available[0]);_write(path,state)
        command=[r['python'],'-B','-u','-m',__package__+'.star_macro12_run','--config',str(root/'configs/run.json'),'--device','cuda:0']
        if not _job(r,state,path,'predict',command,available[0]):raise RuntimeError('Source calibration failed; preserve without scoring')
        _terminal(r,'predict')
        command=[r['python'],'-B','-u','-m',__package__+'.star_macro12_score','--prediction-root',str(root/'predict'),'--novel-truth',r['novel_truth'],'--output',str(root/'score')]
        if not _job(r,state,path,'score',command,None):raise RuntimeError('Independent score failed; preserve predictions')
        m=_terminal(r,'score');state.update(status='ANALYZED_SOURCE_ONLY',all_stages_complete=True,source_gate_passed=m['source_gate_passed'],eligible_for_target_prereg=m['eligible_for_target_prereg'],
            primary='signal_known12',next_step=m['next_step'],target_reads=0,final_phase3_complete=False,full_goal_status='ACTIVE')
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),eligible_for_target_prereg=False,final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args()
    {'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
