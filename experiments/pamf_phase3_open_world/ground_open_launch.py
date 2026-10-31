"""Owned ground open-set training, complete predictions and CPU truth score."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261101-phase3-ground-open-s2026092701-r02'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb','max_added_workers_per_gpu',
        'data_config','train_config','eval_config','source_novel_truth','clean_truth','star_truth'}


def _train():
    from . import ground_open_train
    return ground_open_train


def _predict():
    from . import ground_open_run
    return ground_open_run


def validate(r):
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN or re.fullmatch('[0-9a-f]{40}',str(r.get('code_commit'))) is None
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('run_root','release_root','python','launch_owner','source_novel_truth','clean_truth','star_truth'))
            or r['allowed_gpus']!=list(range(8)) or r['minimum_free_mb']!=4096 or r['max_added_workers_per_gpu']!=2):
        raise ValueError('Exact ground open-set runtime required')
    from . import ground_open_data
    d=ground_open_data.validate_config(r['data_config']);t=_train().validate_config(r['train_config']);e=_predict().validate_config(r['eval_config'])
    if (t['run_id']!=RUN or e['run_id']!=RUN or t['code_commit']!=r['code_commit'] or e['code_commit']!=r['code_commit'] or e['train_config']!=t
            or d['run_id']!=RUN or d['code_commit']!=r['code_commit'] or d['reference_config']!=t['reference_config'] or Path(d['output_root'])!=Path(r['run_root'])/'data' or t['data_root']!=d['output_root'] or e['data_root']!=d['output_root'] or e['reference_config']!=t['reference_config'] or Path(t['output_root'])!=Path(r['run_root'])/'train' or Path(e['output_root'])!=Path(r['run_root'])/'predict'):
        raise ValueError('Native training/evaluation ancestry and immutable roots differ')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe original run; never replace or relaunch it')
    root.mkdir();(root/'logs').mkdir();(root/'configs').mkdir();_write(root/'configs/data.json',r['data_config']);_write(root/'configs/train.json',r['train_config']);_write(root/'configs/eval.json',r['eval_config'])
    argv=[r['python'],'-B','-u','-m',__package__+'.ground_open_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(argv,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    _write(root/'launch.json',dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=argv,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time()))


def _command(r,stage):
    root=Path(r['run_root']);p=[r['python'],'-B','-u','-m']
    if stage in ('data','train','predict'):
        return p+[__package__+('.ground_open_data' if stage=='data' else '.ground_open_train' if stage=='train' else '.ground_open_run'),'--config',str(root/'configs'/('data.json' if stage=='data' else 'train.json' if stage=='train' else 'eval.json')),'--device','cuda:0']
    if stage!='score':raise ValueError('Fixed train/predict/score stage required')
    return p+[__package__+'.ground_open_score','--prediction-root',str(root/'predict'),'--source-known-truth',str(root/'data/private/known_test_truth.npz'),'--source-novel-truth',r['source_novel_truth'],'--clean-truth',r['clean_truth'],'--star-truth',r['star_truth'],'--output',str(root/'score')]


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
    if stage=='data':
        expected.update(status='GROUND_OPEN_DATA_COMPLETE',phase3_data_status='VALIDATED_ONCE',source_only=True,role_input_path='G(clean)',
            source_roles_disjoint=True,V_calibration_test_disjoint=True,paired_query_ID_order_equal=True,known_TEST_labels_public=False,
            real_F_forward_rows=36450,G_forward_rows=99284,private_truth_reads=0,target_U_reads=0,target_Q_reads=0,channel_observations_generated=0)
    elif stage=='train':
        expected.update(status='GROUND_OPEN_TRAIN_COMPLETE',updates_per_arm=2220,total_optimizer_updates=4440,real_student_forward_rows=1420800,
            source_L_OE_sampling_matched=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
            target_U_reads=0,target_Q_reads=0,private_truth_reads=0,source_U_GT_reads=0)
    elif stage=='predict':
        expected.update(status='GROUND_OPEN_PREDICTIONS_COMPLETE',neural_models_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
            all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,
            private_truth_reads=0,target_U_reads=0,source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,
            source_query_count=16650,cross_query_count=10238,calibration_known_count=13500,real_student_forward_rows=134552,G_forward_rows=20476)
    elif stage=='score':expected.update(status='GROUND_OPEN_SCORES_COMPLETE',truth_last=True,all_four_predictions_before_truth=True,all_12_points_before_truth=True,full_goal_status='ACTIVE',new_target_DA=False,adaptation_success=None)
    else:raise ValueError('Fixed data/train/predict/score stage required')
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError(stage+' terminal artifacts differ')
    if stage=='score' and any(type(m.get(k)) is not bool for k in ('ground_training_success','scientific_success','development_gate_passed')):raise ValueError('Independent ground scientific verdict required')
    return m


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe original driver; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        for stage in ('data','train','predict','score'):
            gpu=None
            if stage!='score':
                available=_available(r)
                while not available:
                    state.update(status='WAITING_MEMORY');_write(path,state);time.sleep(20);available=_available(r)
                gpu=available[0]
            state.update(status='RUNNING_'+stage.upper());_write(path,state)
            if not _job(r,state,path,stage,_command(r,stage),gpu):raise RuntimeError(stage+' failed; preserve artifacts')
            m=_terminal(r,stage)
        state.update(status='ANALYZED_DEVELOPMENT',all_stages_complete=True,ground_training_success=m['ground_training_success'],scientific_success=m['scientific_success'],development_gate_passed=m['development_gate_passed'],
            adaptation_success=None,new_target_DA=False,primary='SSL_energy',next_step=m['next_step'],full_goal_status='ACTIVE',final_phase3_complete=False)
    except Exception as exc:
        state.update(status='FAILED_PRESERVED',error=str(exc),ground_training_success=False,scientific_success=False,adaptation_success=None,new_target_DA=False,development_gate_passed=False,final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args();{'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
