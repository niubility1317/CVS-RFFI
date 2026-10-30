"""One source-only frozen-F0 density run, followed by independent CPU scoring."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261030-phase3-star-contribution-density-s2026092701-r01'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb',
    'max_added_workers_per_gpu','run_config','novel_truth'}


def validate(r):
    from .star_contribution_density_run import validate_config
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in
                ('code_commit','run_root','release_root','python','launch_owner','novel_truth'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None or r.get('allowed_gpus')!=list(range(8))
            or r.get('minimum_free_mb')!=4096 or r.get('max_added_workers_per_gpu')!=2):
        raise ValueError('Exact one-worker density source runtime required')
    validate_config(r['run_config']);c=r['run_config']
    if c['run_id']!=RUN or c['code_commit']!=r['code_commit'] or Path(c['output_root'])!=Path(r['run_root'])/'predict':
        raise ValueError('Owned source prediction config/root must match the fixed run')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe original run; never replace or relaunch outputs')
    root.mkdir();(root/'configs').mkdir();(root/'logs').mkdir();_write(root/'configs/run.json',r['run_config'])
    command=[r['python'],'-B','-u','-m',__package__+'.star_contribution_density_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time())
    _write(root/'launch.json',value);print(value,flush=True)


def _terminal(r,stage):
    folder='predict' if stage=='predict' else 'score';m=_read(Path(r['run_root'])/folder/'complete.json')
    required=dict(run_id=RUN,code_commit=r['code_commit'],source_only=True,target_reads=0,final_phase3_complete=False)
    if stage=='predict':required.update(status='STAR_CONTRIBUTION_DENSITY_PREDICTIONS_COMPLETE',known_count=27000,novel_count=3150,source_L_count=6300,source_OE_count=6134,
        private_truth_reads=0,real_F_forward_rows=30150,synthetic_F_forward_rows=2,G_forward_rows=0,G_checkpoint_loaded=False,
        identity_predictions_unchanged=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        source_bank_deployed=False,per_record_source_rows_deployed=0)
    else:required.update(status='STAR_CONTRIBUTION_DENSITY_SCORES_COMPLETE',truth_last=True,full_goal_status='ACTIVE')
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Density '+stage+' terminal permission/identity differs')
    return m


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe original pipeline; never restart it')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        available=_gpu_free(r)
        while not available:
            state['status']='WAITING_MEMORY';_write(path,state);time.sleep(20);available=_gpu_free(r)
        state.update(status='RUNNING',gpu=available[0]);_write(path,state)
        command=[r['python'],'-B','-u','-m',__package__+'.star_contribution_density_run','--config',str(root/'configs/run.json'),'--device','cuda:0']
        if not _job(r,state,path,'predict',command,available[0]):raise RuntimeError('Source density run failed; preserve output and do not score')
        _terminal(r,'predict')
        command=[r['python'],'-B','-u','-m',__package__+'.star_contribution_density_score','--prediction-root',str(root/'predict'),
            '--novel-truth',r['novel_truth'],'--output',str(root/'score')]
        if not _job(r,state,path,'score',command,None):raise RuntimeError('Independent score failed; preserve predictions')
        m=_terminal(r,'score');state.update(status='ANALYZED_SOURCE_ONLY',all_stages_complete=True,
            source_gate_passed=m['source_gate_passed'],next_step=m['next_step'],target_reads=0,final_phase3_complete=False,full_goal_status='ACTIVE')
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args()
    {'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
