"""Detached single-owner source practical readout and frozen development driver."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .launch import choose_gpu, read, write
from .source_practical_development import RUN, validate_config

FIELDS={'run_id','run_root','release_root','python','code_commit','launch_owner','allowed_gpus',
    'max_added_workers_per_gpu','minimum_free_mb','targets_path','export_config','source_fit_root','novel_truth','rows'}


def validate_runtime(rt):
    from .source_practical_export import validate_config as validate_export
    if not isinstance(rt,dict) or set(rt)!=FIELDS:raise ValueError('Fixed source-practical runtime required')
    for k in FIELDS-{'rows','export_config','allowed_gpus','max_added_workers_per_gpu','minimum_free_mb'}:
        if not isinstance(rt[k],str) or not rt[k].strip():raise ValueError('Nonempty runtime field '+k)
    root=Path(rt['run_root'])
    if (rt['run_id']!=RUN or root.name!=RUN or not root.is_absolute() or not Path(rt['release_root']).is_absolute()
            or len(rt['code_commit'])!=40 or any(x not in '0123456789abcdef' for x in rt['code_commit'])
            or type(rt['max_added_workers_per_gpu']) is not int or rt['max_added_workers_per_gpu']!=2
            or type(rt['minimum_free_mb']) is not int or rt['minimum_free_mb']<4096
            or not isinstance(rt['allowed_gpus'],list) or not rt['allowed_gpus']
            or any(type(x) is not int or x<0 for x in rt['allowed_gpus'])
            or len(set(rt['allowed_gpus']))!=len(rt['allowed_gpus'])):raise ValueError('Immutable identities and two-added-worker policy required')
    e=validate_export(rt['export_config'])
    if (e['run_id']!=RUN or e['code_commit']!=rt['code_commit'] or Path(e['output_root'])!=root/'source/export'
            or Path(rt['source_fit_root'])!=Path(e['source_row_root'])/'fit'
            or not Path(rt['novel_truth']).is_absolute()
            or not isinstance(rt['rows'],dict) or set(rt['rows'])!={'C0','P1'}):raise ValueError('Source paths or exact paired development rows differ')
    for condition in ('C0','P1'):
        row=rt['rows'][condition]
        if not isinstance(row,dict) or set(row)!={'predict_config','truth_path'}:raise ValueError('Keep scorer truth separate')
        c=validate_config(row['predict_config'])
        if (c['code_commit']!=rt['code_commit'] or c['condition']!=condition
                or Path(c['output_root'])!=root/'rows'/condition/'predict'
                or Path(c['fit_root'])!=root/'source/fit' or Path(c['calibration_root'])!=root/'source/calibration'
                or c['source_row_root']!=e['source_row_root'] or c['risk_prediction_root']!=e['risk_prediction_root']
                or not isinstance(row['truth_path'],str) or not Path(row['truth_path']).is_absolute()):raise ValueError('Shared frozen source fit/calibration and row identities required')
        if condition=='P1' and any(c[k]!=e[k] for k in ('dura_checkpoint','dura_release')):raise ValueError('Training and test G must match')
    return rt


def commands(rt):
    root=Path(rt['run_root']);source=root/'source';prefix=[rt['python'],'-B','-u','-m']
    readout=prefix+['experiments.pamf_phase3_open_world.source_practical_readout']
    jobs=[]
    def add(name,argv,marker,gpu=False):jobs.append(dict(name=name,argv=argv,marker=str(marker),gpu=gpu))
    add('export',prefix+['experiments.pamf_phase3_open_world.source_practical_export','--config',str(root/'configs/export.json'),'--device','cuda:0'],source/'export/complete.json',True)
    add('fit',readout+['fit','--export-root',str(source/'export'),'--source-fit-root',rt['source_fit_root'],'--output',str(source/'fit')],source/'fit/complete.json')
    add('calibrate',readout+['calibrate','--export-root',str(source/'export'),'--fit-root',str(source/'fit'),'--output',str(source/'calibration')],source/'calibration/complete.json')
    add('predict_source',readout+['predict-source','--export-root',str(source/'export'),'--fit-root',str(source/'fit'),'--calibration-root',str(source/'calibration'),'--output',str(source/'predict')],source/'predict/complete.json')
    add('score_source',readout+['score-source','--predictions-root',str(source/'predict'),'--novel-truth',rt['novel_truth'],'--output',str(source/'score')],source/'score/complete.json')
    dev=prefix+['experiments.pamf_phase3_open_world.source_practical_development']
    for c in ('C0','P1'):
        row=root/'rows'/c
        add(c+'_predict',dev+['predict','--config',str(root/'configs'/(c+'.predict.json')),'--device','cuda:0'],row/'predict/complete.json',True)
        add(c+'_score',dev+['score','--predictions-root',str(row/'predict'),'--truth',rt['rows'][c]['truth_path'],'--targets',rt['targets_path'],'--output',str(row/'score')],row/'score/complete.json')
    return jobs


def dispatch(rt):
    validate_runtime(rt);root=Path(rt['run_root'])
    state=dict(status='RUNNING',run_id=rt['run_id'],code_commit=rt['code_commit'],owner=rt['launch_owner'],
        pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,started_at=time.time(),jobs={},foreign_processes_stopped=0,
        max_added_workers_per_gpu=2,max_concurrent_gpu_workers=1,target_U_reads=0,formal_target_reads=0,final_phase3_complete=False)
    def record():write(root/'pipeline_state.json',state,exclusive=False)
    record()
    try:
        write(root/'configs/export.json',rt['export_config'])
        for c in ('C0','P1'):write(root/'configs'/(c+'.predict.json'),rt['rows'][c]['predict_config'])
        for job in commands(rt):
            gpu=choose_gpu(rt) if job['gpu'] else None
            env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            current=dict(status='STARTING',argv=job['argv'],cwd=rt['release_root'],gpu=gpu,started_at=time.time(),log=str(root/'logs'/(job['name']+'.log')))
            state['jobs'][job['name']]=current;record()
            with Path(current['log']).open('xb') as stream:
                process=subprocess.Popen(job['argv'],cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                current.update(status='RUNNING',pid=process.pid);record();exitcode=process.wait()
            current.update(returncode=exitcode,finished_at=time.time())
            if exitcode:
                current['status']='FAILED';raise RuntimeError(job['name']+' failed; preserve partial and stop without retry')
            marker=read(job['marker'])
            current.update(status='COMPLETE',completion_status=marker['status']);record()
        state.update(status='ANALYZED_DEVELOPMENT_ONLY',finished_at=time.time())
    except Exception as exc:
        state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),finished_at=time.time())
    record();return state


def launch(rt,runtime_path):
    validate_runtime(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    (root/'configs').mkdir();(root/'logs').mkdir()
    argv=[rt['python'],'-B','-u','-m','experiments.pamf_phase3_open_world.source_practical_launch','--runtime',str(Path(runtime_path).resolve()),'--dispatch']
    env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with (root/'logs/driver.log').open('xb') as stream:
        child=subprocess.Popen(argv,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(status='LAUNCH_RETURNED_REQUIRES_POST_STATE',run_id=rt['run_id'],pid=child.pid,cwd=rt['release_root'],argv=argv,
        owner=rt['launch_owner'],code_commit=rt['code_commit'],foreign_processes_stopped=0)
    write(root/'launch.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True);p.add_argument('--dispatch',action='store_true')
    a=p.parse_args(argv);rt=read(a.runtime);result=dispatch(rt) if a.dispatch else launch(rt,a.runtime)
    print(result['status'],flush=True);return int(result['status']=='FAILED')


if __name__=='__main__':raise SystemExit(main())
