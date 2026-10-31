"""Owned source-meta training, complete predictions and CPU truth score."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261031-phase3-source-meta-s2026092701-r01'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb','max_added_workers_per_gpu',
        'train_config','eval_config','source_novel_truth','clean_truth','star_truth'}


def _train():
    from . import source_meta_train
    return source_meta_train


def _predict():
    from . import source_meta_run
    return source_meta_run


def validate(r):
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN or re.fullmatch('[0-9a-f]{40}',str(r.get('code_commit'))) is None
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('run_root','release_root','python','launch_owner','source_novel_truth','clean_truth','star_truth'))
            or r['allowed_gpus']!=list(range(8)) or r['minimum_free_mb']!=4096 or r['max_added_workers_per_gpu']!=2):
        raise ValueError('Exact source meta runtime required')
    t=_train().validate_config(r['train_config']);e=_predict().validate_config(r['eval_config'])
    if (t['run_id']!=RUN or e['run_id']!=RUN or t['code_commit']!=r['code_commit'] or e['code_commit']!=r['code_commit'] or e['train_config']!=t
            or e['reference_config']!=t['reference_config'] or Path(t['output_root'])!=Path(r['run_root'])/'train' or Path(e['output_root'])!=Path(r['run_root'])/'predict'):
        raise ValueError('Native training/evaluation ancestry and immutable roots differ')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe original run; never replace or relaunch it')
    root.mkdir();(root/'logs').mkdir();(root/'configs').mkdir();_write(root/'configs/train.json',r['train_config']);_write(root/'configs/eval.json',r['eval_config'])
    argv=[r['python'],'-B','-u','-m',__package__+'.source_meta_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(argv,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    _write(root/'launch.json',dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=argv,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time()))


def _command(r,stage):
    root=Path(r['run_root']);p=[r['python'],'-B','-u','-m']
    if stage in ('train','predict'):
        return p+[__package__+('.source_meta_train' if stage=='train' else '.source_meta_run'),'--config',str(root/'configs'/('train.json' if stage=='train' else 'eval.json')),'--device','cuda:0']
    if stage!='score':raise ValueError('Fixed train/predict/score stage required')
    return p+[__package__+'.source_meta_score','--prediction-root',str(root/'predict'),'--source-novel-truth',r['source_novel_truth'],'--clean-truth',r['clean_truth'],'--star-truth',r['star_truth'],'--output',str(root/'score')]


def _available(r):
    available=_gpu_free(r)
    gpu_map={u.strip():int(i) for i,u in (line.split(',') for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
    counts={g:0 for g in available}
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
        pid,uuid=line.split(',')
        try:argv=(Path('/proc')/pid.strip()/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
        except FileNotFoundError:continue
        g=gpu_map[uuid.strip()]
        if g in counts and 'experiments.pamf_phase3_open_world' in argv:counts[g]+=1
    return [g for g in available if counts[g]<r['max_added_workers_per_gpu']]


def _terminal(r,stage):
    m=_read(Path(r['run_root'])/stage/'complete.json');expected=dict(run_id=RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='train':
        expected.update(status='SOURCE_META_TRAIN_COMPLETE',updates_per_arm=444,total_optimizer_updates=888,
            source_sampling_matched=True,full_second_order=True,all_receiver_splits_disjoint=True,virtual_updates_committed=False,meta_alpha=.001,meta_beta=.5,original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
            target_U_reads=0,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0)
    elif stage=='predict':
        expected.update(status='SOURCE_META_PREDICTIONS_COMPLETE',source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
            all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,source_control_readouts_reused_exactly=True,classification_unchanged=False,
            source_density_models_fitted=2,new_threshold_sets_fitted=4,training_target_U_count=0,U_ID_metadata_reads=0,private_truth_reads=0,target_U_reads=0,U_truth_reads=0)
    elif stage=='score':expected.update(status='SOURCE_META_SCORES_COMPLETE',truth_last=True,all_three_predictions_before_truth=True,full_goal_status='ACTIVE',new_target_DA=False,adaptation_success=None)
    else:raise ValueError('Fixed train/predict/score stage required')
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError(stage+' terminal artifacts differ')
    if stage=='score' and any(type(m.get(k)) is not bool for k in ('source_meta_success','scientific_success','development_gate_passed')):raise ValueError('Independent source-meta scientific verdict required')
    return m


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe original driver; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        for stage in ('train','predict','score'):
            gpu=None
            if stage!='score':
                available=_available(r)
                while not available:
                    state.update(status='WAITING_MEMORY');_write(path,state);time.sleep(20);available=_available(r)
                gpu=available[0]
            state.update(status='RUNNING_'+stage.upper());_write(path,state)
            if not _job(r,state,path,stage,_command(r,stage),gpu):raise RuntimeError(stage+' failed; preserve artifacts')
            m=_terminal(r,stage)
        state.update(status='ANALYZED_DEVELOPMENT',all_stages_complete=True,source_meta_success=m['source_meta_success'],scientific_success=m['scientific_success'],development_gate_passed=m['development_gate_passed'],
            adaptation_success=None,new_target_DA=False,primary='META_NATIVE_predicted_ratio',next_step=m['next_step'],full_goal_status='ACTIVE',final_phase3_complete=False)
    except Exception as exc:
        state.update(status='FAILED_PRESERVED',error=str(exc),source_meta_success=False,scientific_success=False,adaptation_success=None,new_target_DA=False,development_gate_passed=False,final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args();{'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
