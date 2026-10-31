"""Owned source geometry training, complete predictions and CPU truth score."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read,_write,_env,_gpu_free,_job

RUN='20261101-phase3-ground-geometry-s2026092701-r01'
FIELDS={'run_id','code_commit','run_root','release_root','python','launch_owner','allowed_gpus','minimum_free_mb','max_added_workers_per_gpu',
        'eval_config','source_novel_truth','clean_truth','star_truth'}


def _predict():
    from . import ground_geometry_run
    return ground_geometry_run


def validate(r):
    if (not isinstance(r,dict) or set(r)!=FIELDS or r.get('run_id')!=RUN or re.fullmatch('[0-9a-f]{40}',str(r.get('code_commit'))) is None
            or any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('run_root','release_root','python','launch_owner','source_novel_truth','clean_truth','star_truth'))
            or r['allowed_gpus']!=list(range(8)) or r['minimum_free_mb']!=4096 or r['max_added_workers_per_gpu']!=2):
        raise ValueError('Exact source geometry runtime required')
    e=_predict().validate_config(r['eval_config'])
    if e['run_id']!=RUN or e['code_commit']!=r['code_commit'] or Path(e['output_root'])!=Path(r['run_root'])/'predict':
        raise ValueError('Geometry source fitting and immutable prediction roots differ')
    return r


def start(runtime):
    r=validate(_read(runtime));root=Path(r['run_root'])
    if root.exists():raise FileExistsError('Observe original run; never replace or relaunch it')
    root.mkdir();(root/'logs').mkdir();(root/'configs').mkdir();_write(root/'configs/eval.json',r['eval_config'])
    argv=[r['python'],'-B','-u','-m',__package__+'.ground_geometry_launch','driver','--runtime',str(runtime)]
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(argv,cwd=r['release_root'],env=_env(r),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    _write(root/'launch.json',dict(run_id=RUN,code_commit=r['code_commit'],pid=child.pid,argv=argv,cwd=r['release_root'],launch_owner=r['launch_owner'],time=time.time()))


def _command(r,stage):
    root=Path(r['run_root']);p=[r['python'],'-B','-u','-m']
    if stage=='predict':return p+[__package__+'.ground_geometry_run','--config',str(root/'configs/eval.json'),'--device','cuda:0']
    if stage!='score':raise ValueError('Fixed source-fit/predict and CPUscore stages required')
    return p+[__package__+'.ground_geometry_score','--prediction-root',str(root/'predict'),'--source-known-truth',str(Path(r['eval_config']['reference_config']['data_root'])/'private/known_test_truth.npz'),'--source-novel-truth',r['source_novel_truth'],'--clean-truth',r['clean_truth'],'--star-truth',r['star_truth'],'--output',str(root/'score')]


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
    if stage=='predict':
        expected.update(status='GROUND_GEOMETRY_PREDICTIONS_COMPLETE',all_16_points_complete=True,all_four_DURA_conditions_complete=True,
            all_conditions_through_DURA=True,neural_models_frozen=True,source_subspaces_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
            source_controls_reused_exactly=True,source_subspace_fit_count=2,nuisance_SVD_fit_count=1,source_subspace_fit_role='Us',source_subspace_fit_rows=56700,
            source_Us_reads=56700,source_U_GT_reads=0,source_L_reads=0,source_OE_reads=0,target_U_reads=0,U_ID_metadata_reads=0,
            private_truth_reads=0,source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,optimizer_updates=0,
            calibration_known_count=13500,source_query_count=16650,cross_query_count=10238,real_F_forward_rows=123976,G_forward_rows=20476,
            real_SSL_forward_rows=0,real_student_forward_rows=0,new_threshold_values=12,reused_threshold_values=12,threshold_values=24,
            original_models_unchanged=True,original_default_head_unchanged=True,exact_native_classifier_coordinates=True)
    elif stage=='score':expected.update(status='GROUND_GEOMETRY_SCORES_COMPLETE',truth_last=True,all_four_predictions_before_truth=True,
        all_16_points_before_truth=True,legacy_energy_overall_exact=True,full_goal_status='ACTIVE',new_target_DA=False,adaptation_success=None)
    else:raise ValueError('Fixed predict/score stages required')
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError(stage+' terminal artifacts differ')
    if stage=='score' and any(type(m.get(k)) is not bool for k in ('ground_stage_success','frozen_rejection_success','scientific_success','development_gate_passed')):
        raise ValueError('Independent formal CK95/RU95 geometry verdict required')
    return m


def driver(runtime):
    r=validate(_read(runtime));root=Path(r['run_root']);path=root/'pipeline_state.json'
    if path.exists():raise FileExistsError('Observe original driver; never restart')
    state=dict(run_id=RUN,code_commit=r['code_commit'],pid=os.getpid(),status='RUNNING',started_at=time.time(),jobs={});_write(path,state)
    try:
        for stage in ('predict','score'):
            gpu=None
            if stage!='score':
                available=_available(r)
                while not available:
                    state.update(status='WAITING_MEMORY');_write(path,state);time.sleep(20);available=_available(r)
                gpu=available[0]
            state.update(status='RUNNING_'+stage.upper());_write(path,state)
            if not _job(r,state,path,stage,_command(r,stage),gpu):raise RuntimeError(stage+' failed; preserve artifacts')
            m=_terminal(r,stage)
        state.update(status='ANALYZED_DEVELOPMENT',all_stages_complete=True,ground_stage_success=m['ground_stage_success'],frozen_rejection_success=m['frozen_rejection_success'],scientific_success=m['scientific_success'],development_gate_passed=m['development_gate_passed'],
            adaptation_success=None,new_target_DA=False,primary='F0_domain_vim',next_step=m['next_step'],full_goal_status='ACTIVE',final_phase3_complete=False)
    except Exception as exc:
        state.update(status='FAILED_PRESERVED',error=str(exc),ground_stage_success=False,frozen_rejection_success=False,scientific_success=False,adaptation_success=None,new_target_DA=False,development_gate_passed=False,final_phase3_complete=False)
    state['finished_at']=time.time();_write(path,state);print(state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','driver'));p.add_argument('--runtime',required=True);a=p.parse_args();{'start':start,'driver':driver}[a.mode](a.runtime)


if __name__=='__main__':main()
