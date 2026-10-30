"""One independent signal branch: fixed train, source eval, truth-side score."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261030-phase3-star-signal-open-s2026092701-r03'
ARM='E_SIGNAL'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb',
 'max_added_workers_per_gpu','train_config','eval_config','novel_truth'}


def validate(r):
    from .star_signal_open_train import validate_config as train_validate
    from .star_signal_open_eval import validate_config as eval_validate
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('code_commit','run_root','release_root','python','launch_owner','novel_truth'))
            or re.fullmatch('[0-9a-f]{40}',r['code_commit']) is None or r['allowed_gpus']!=list(range(8))
            or r['minimum_free_mb']!=4096 or r['max_added_workers_per_gpu']!=2):raise ValueError('Exact single-arm signal runtime required')
    train_validate(r['train_config']);eval_validate(r['eval_config']);t,e=r['train_config'],r['eval_config']
    root=Path(r['run_root'])/'rows'/ARM
    if (any(c['run_id']!=RUN or c['code_commit']!=r['code_commit'] or c['arm']!=ARM for c in (t,e))
            or e['train_config']!=t or t['IQ_root']!=e['IQ_root'] or t['feature_root']!=e['feature_root']
            or t['control_root']!=e['reference_prediction_root'] or Path(t['output_root'])!=root/'train'
            or Path(e['output_root'])!=root/'predict'):raise ValueError('Owned train/eval/parent paths differ')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe existing run; never overwrite or relaunch')
    root.mkdir();(root/'logs').mkdir();(root/'configs').mkdir();(root/'rows'/ARM).mkdir(parents=True)
    for k in ('train_config','eval_config'):_write(root/'configs'/(ARM+'_'+k+'.json'),r[k])
    command=[r['python'],'-B','-u','-m',__package__+'.star_signal_open_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=command,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time())
    _write(root/'launch.json',value);print(json.dumps(value),flush=True)


def _terminal(r,stage):
    path=Path(r['run_root'])/'rows'/ARM/dict(train='train',eval='predict',score='score')[stage]/'complete.json';m=_read(path)
    expected=dict(run_id=RUN,code_commit=r['code_commit'],arm=ARM,source_only=True,target_reads=0,final_phase3_complete=False)
    if stage=='train':expected.update(status='STAR_SIGNAL_OPEN_SOURCE_FROZEN',updates=2220,native_updates=0,open_signal_updates=2220,
        validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,private_truth_reads=0)
    elif stage=='eval':expected.update(status='STAR_SIGNAL_OPEN_PREDICTIONS_COMPLETE',known_count=27000,novel_count=3150,source_L_count=6300,source_OE_count=6134,
        native_updates=0,open_signal_updates=2220,private_truth_reads=0,real_signal_forward_rows=42584,synthetic_signal_forward_rows=2,
        real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0,identity_predictions_unchanged=True,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_bank_deployed=False,per_record_source_rows_deployed=0)
    else:expected.update(status='STAR_SIGNAL_OPEN_SCORES_COMPLETE',truth_last=True,full_goal_status='ACTIVE')
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Signal '+stage+' terminal differs')
    if stage=='score' and (type(m.get('source_gate_passed')) is not bool or type(m.get('eligible_for_target_prereg')) is not bool or not isinstance(m.get('next_step'),str)):
        raise ValueError('Independent source eligibility required')
    return m


def _command(r,stage):
    root=Path(r['run_root']);prefix=[r['python'],'-B','-u','-m']
    if stage in ('train','eval'):return prefix+[__package__+'.star_signal_open_'+stage,'--config',str(root/'configs'/(ARM+'_'+stage+'_config.json')),'--device','cuda:0']
    return prefix+[__package__+'.star_signal_open_score','--prediction-root',str(root/'rows'/ARM/'predict'),'--novel-truth',r['novel_truth'],'--output',str(root/'rows'/ARM/'score')]


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe original pipeline; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        for stage in ('train','eval','score'):
            gpu=None
            if stage!='score':
                available=_gpu_free(r)
                while not available:state.update(status='WAITING_MEMORY');_write(path,state);time.sleep(20);available=_gpu_free(r)
                gpu=available[0]
            state.update(status='RUNNING_'+stage.upper());_write(path,state)
            if not _job(r,state,path,stage,_command(r,stage),gpu):raise RuntimeError(stage+' failed; preserve original output')
            m=_terminal(r,stage)
        state.update(status='ANALYZED_SOURCE_ONLY',all_stages_complete=True,source_gate_passed=m['source_gate_passed'],eligible_for_target_prereg=m['eligible_for_target_prereg'],
            primary='E_SIGNAL::signal_known_support',next_step=m['next_step'],target_reads=0,final_phase3_complete=False,full_goal_status='ACTIVE')
    except Exception as exc:state.update(status='FAILED_PRESERVED',error=str(exc),eligible_for_target_prereg=False,final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args()
    {'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
