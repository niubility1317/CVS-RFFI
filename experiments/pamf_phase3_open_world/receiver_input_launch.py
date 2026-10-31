"""Evaluation-only recovery of frozen source/U input states and DURA tests."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261031-phase3-receiver-input-s2026092701-r02'
TRAIN_RUN='20261031-phase3-receiver-input-s2026092701-r01'
TRAIN_CODE='ac8bbd27277fcb1f16806c7b8b94e8c3e27e390b'
SOURCE_COUNT=6300
U_COUNT=576
STEPS=200
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb','max_added_workers_per_gpu',
    'train_config','eval_config','source_novel_truth','clean_truth','star_truth'}


def validate(r):
    from . import receiver_input_train as train
    from . import receiver_input_run as predict
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('code_commit','run_root','release_root','python','launch_owner','source_novel_truth','clean_truth','star_truth'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None or r['allowed_gpus']!=list(range(8)) or r['minimum_free_mb']!=4096 or r['max_added_workers_per_gpu']!=2):raise ValueError('Exact receiver-input runtime required')
    t=train.validate_config(r['train_config']);e=predict.validate_config(r['eval_config'])
    if (t['run_id']!=TRAIN_RUN or t['code_commit']!=TRAIN_CODE or e['run_id']!=RUN or e['code_commit']!=r['code_commit'] or e['train_config']!=t or e['reference_config']['train_config']!=t['reference_config'] or e['reference_config']['output_root']!=t['reference_prediction_root']
            or Path(t['output_root']).resolve().is_relative_to(Path(r['run_root']).resolve()) or Path(e['output_root'])!=Path(r['run_root'])/'predict'):
        raise ValueError('Actual neural/evaluation identity and immutable roots must agree')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe original run; never duplicate or replace')
    root.mkdir();(root/'logs').mkdir();(root/'configs').mkdir();_write(root/'configs/train.json',r['train_config']);_write(root/'configs/eval.json',r['eval_config'])
    argv=[r['python'],'-B','-u','-m',__package__+'.receiver_input_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:child=subprocess.Popen(argv,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    v=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=argv,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time());_write(root/'launch.json',v);print(json.dumps(v),flush=True)


def _command(r,stage):
    root=Path(r['run_root']);p=[r['python'],'-B','-u','-m']
    if stage=='predict':return p+[__package__+'.receiver_input_run','--config',str(root/'configs/eval.json'),'--device','cuda:0']
    return p+[__package__+'.receiver_input_score','--prediction-root',str(root/'predict'),'--source-novel-truth',r['source_novel_truth'],'--clean-truth',r['clean_truth'],'--star-truth',r['star_truth'],'--output',str(root/'score')]


def _terminal(r,stage):
    m=_read(Path(r['run_root'])/stage/'complete.json' if stage!='predict' else Path(r['run_root'])/'predict/complete.json')
    expected=dict(run_id=RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='predict':expected.update(status='RECEIVER_INPUT_PREDICTIONS_COMPLETE',all_adapters_frozen_before_Q=True,
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,all_thresholds_inherited_unchanged=True,
        controls_reused_exactly=True,original_models_unchanged=True,classification_unchanged=False,new_density_fits=0,new_threshold_sets_fitted=0,
        private_truth_reads=0,U_truth_reads=0,target_U_reads=0)
    else:expected.update(status='RECEIVER_INPUT_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE')
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Receiver input '+stage+' terminal differs')
    if stage=='score' and (type(m.get('adaptation_success')) is not bool or type(m.get('development_gate_passed')) is not bool or not isinstance(m.get('next_step'),str)):raise ValueError('Independent UDA and joint-protection verdicts required')
    return m


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe original pipeline; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        for stage in ('predict','score'):
            gpu=None
            if stage!='score':
                available=_gpu_free(r)
                while not available:state.update(status='WAITING_MEMORY');_write(path,state);time.sleep(20);available=_gpu_free(r)
                gpu=available[0]
            state.update(status='RUNNING_'+stage.upper());_write(path,state)
            if not _job(r,state,path,stage,_command(r,stage),gpu):raise RuntimeError(stage+' failed; preserve outputs')
            m=_terminal(r,stage)
        state.update(status='ANALYZED_DEVELOPMENT',all_stages_complete=True,adaptation_success=m['adaptation_success'],development_gate_passed=m['development_gate_passed'],
            primary='TARGET_INPUT_predicted_ratio',next_step=m['next_step'],final_phase3_complete=False,full_goal_status='ACTIVE')
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),adaptation_success=False,development_gate_passed=False,final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args();{'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
