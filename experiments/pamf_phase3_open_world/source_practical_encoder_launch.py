"""Single-owner cached-view build and two independent automatic source/dev rows."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .launch import read,write
from .source_practical_encoder import RUN,ARMS,validate_config as validate_train

FIELDS={'run_id','run_root','release_root','python','code_commit','launch_owner','allowed_gpus',
    'max_added_workers_per_gpu','minimum_free_mb','max_concurrent_rows','targets_path','cache_config','rows'}


def validate_runtime(rt):
    from .source_practical_encoder_cache import validate_config as cache_validate
    from .source_practical_encoder_readout import validate_config as source_validate
    from .source_practical_encoder_development import validate_config as dev_validate
    if not isinstance(rt,dict) or set(rt)!=FIELDS:raise ValueError('Fixed source-encoder runtime fields required')
    root=Path(rt['run_root'])
    if (rt['run_id']!=RUN or root.name!=RUN or not root.is_absolute() or not Path(rt['release_root']).is_absolute()
            or len(rt['code_commit'])!=40 or any(x not in '0123456789abcdef' for x in rt['code_commit'])
            or type(rt['max_concurrent_rows']) is not int or rt['max_concurrent_rows']!=2
            or type(rt['max_added_workers_per_gpu']) is not int or rt['max_added_workers_per_gpu']!=2
            or type(rt['minimum_free_mb']) is not int or rt['minimum_free_mb']<4096
            or not isinstance(rt['allowed_gpus'],list) or not rt['allowed_gpus']
            or any(type(g) is not int or g<0 for g in rt['allowed_gpus'])
            or len(set(rt['allowed_gpus']))!=len(rt['allowed_gpus']) or set(rt['rows'])!=set(ARMS)):
        raise ValueError('Immutable identity and two-added-worker/free-memory policy required')
    cache=cache_validate(rt['cache_config'])
    if cache['output_root']!=str(root/'cache') or cache['run_id']!=RUN or cache['code_commit']!=rt['code_commit']:
        raise ValueError('One shared source-only training-view cache required')
    for arm in ARMS:
        row=rt['rows'][arm];base=root/'rows'/arm
        if set(row)!={'train_config','source_config','source_novel_truth','development'} or set(row['development'])!={'C0','P1'}:
            raise ValueError('Independent training/source/development row required')
        train=validate_train(row['train_config']);source=source_validate(row['source_config'])
        for cfg,output in ((train,base/'train'),(source,base/'source')):
            if (cfg['run_id']!=RUN or cfg['code_commit']!=rt['code_commit'] or cfg['arm']!=arm or Path(cfg['output_root'])!=output
                    or any(cfg[k]!=cache[k] for k in ('source_row_root','risk_prediction_root'))):raise ValueError('Row source identity/output differs')
        if (train['cache_root']!=cache['output_root'] or source['train_root']!=train['output_root']
                or any(source[k]!=cache[k] for k in ('dura_checkpoint','dura_release'))
                or not Path(row['source_novel_truth']).is_absolute()):raise ValueError('Shared cache/train/G or scorer path differs')
        for c in ('C0','P1'):
            d=row['development'][c]
            if set(d)!={'predict_config','truth_path'}:raise ValueError('Keep development truth separate')
            cfg=dev_validate(d['predict_config'])
            if (cfg['run_id']!=RUN or cfg['code_commit']!=rt['code_commit'] or cfg['arm']!=arm or cfg['condition']!=c
                    or cfg['train_root']!=train['output_root'] or cfg['source_root']!=source['output_root']
                    or Path(cfg['output_root'])!=base/'development'/c/'predict' or not Path(d['truth_path']).is_absolute()):
                raise ValueError('Development source/row identity differs')
            if c=='P1' and any(cfg[k]!=cache[k] for k in ('dura_checkpoint','dura_release')):raise ValueError('G must match across training/source/development')
    return rt


def choose_gpu(rt,reservations):
    while True:
        snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        choices=[]
        for line in snapshot.splitlines():
            g,free=map(int,line.split(','));own=reservations.count(g)
            if g in rt['allowed_gpus'] and own<2 and free>=rt['minimum_free_mb']*(own+1):
                choices.append((free-rt['minimum_free_mb']*own,g))
        if choices:return max(choices)[1]
        print('WAITING_FOR_MEMORY no foreign process changes',flush=True);time.sleep(20)


def row_commands(rt,arm):
    root=Path(rt['run_root']);base=root/'rows'/arm;p=[rt['python'],'-B','-u','-m'];jobs=[]
    def add(name,argv,marker,gpu):jobs.append(dict(name=name,argv=argv,marker=str(marker),gpu=gpu))
    add('train',p+['experiments.pamf_phase3_open_world.source_practical_encoder','--config',str(root/'configs'/(arm+'.train.json')),'--device','cuda:0'],base/'train/complete.json',True)
    src=p+['experiments.pamf_phase3_open_world.source_practical_encoder_readout']
    add('source',src+['run','--config',str(root/'configs'/(arm+'.source.json')),'--device','cuda:0'],base/'source/complete.json',True)
    add('source_score',src+['score','--source-root',str(base/'source'),'--novel-truth',rt['rows'][arm]['source_novel_truth'],'--output',str(base/'source/score')],base/'source/score/complete.json',False)
    dev=p+['experiments.pamf_phase3_open_world.source_practical_encoder_development']
    for c in ('C0','P1'):
        row=base/'development'/c
        add(c+'_predict',dev+['predict','--config',str(root/'configs'/(arm+'.'+c+'.json')),'--device','cuda:0'],row/'predict/complete.json',True)
        add(c+'_score',dev+['score','--predictions-root',str(row/'predict'),'--truth',rt['rows'][arm]['development'][c]['truth_path'],
            '--targets',rt['targets_path'],'--output',str(row/'score')],row/'score/complete.json',False)
    return jobs


def _env(rt,gpu):
    return dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')


def _run_job(rt,root,job,gpu,state,record):
    current=dict(status='STARTING',argv=job['argv'],cwd=rt['release_root'],gpu=gpu,
        log=str(root/'logs'/(job['name']+'.log')),started_at=time.time())
    state['jobs'][job['name']]=current;record()
    with Path(current['log']).open('xb') as log:
        child=subprocess.Popen(job['argv'],cwd=rt['release_root'],env=_env(rt,gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        current.update(status='RUNNING',pid=child.pid);record();exitcode=child.wait()
    current.update(returncode=exitcode,finished_at=time.time())
    if exitcode:
        current['status']='FAILED';record();raise RuntimeError(job['name']+' failed; preserve partial without retry')
    done=read(job['marker']);current.update(status='COMPLETE',completion_status=done['status']);record()


def row_dispatch(rt,arm,gpu):
    validate_runtime(rt);root=Path(rt['run_root'])/'rows'/arm;root.mkdir(exist_ok=False);(root/'logs').mkdir()
    state=dict(status='RUNNING',arm=arm,run_id=RUN,code_commit=rt['code_commit'],pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        gpu=gpu,started_at=time.time(),jobs={},foreign_processes_stopped=0,target_U_reads=0,formal_target_reads=0,final_Phase3_complete=False)
    def record():write(root/'pipeline_state.json',state,exclusive=False)
    record()
    try:
        for job in row_commands(rt,arm):_run_job(rt,root,job,gpu if job['gpu'] else None,state,record)
        state.update(status='ANALYZED_DEVELOPMENT_ONLY',finished_at=time.time())
    except Exception as exc:state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),finished_at=time.time())
    record();return state


def dispatch(rt,runtime_path):
    validate_runtime(rt);root=Path(rt['run_root'])
    state=dict(status='RUNNING',run_id=RUN,code_commit=rt['code_commit'],owner=rt['launch_owner'],pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        started_at=time.time(),jobs={},rows={},foreign_processes_stopped=0,max_concurrent_GPU_workers=2,max_added_workers_per_gpu=2,
        target_U_reads=0,formal_target_reads=0,final_Phase3_complete=False)
    def record():write(root/'pipeline_state.json',state,exclusive=False)
    record();children=[];streams=[]
    try:
        write(root/'configs/cache.json',rt['cache_config'])
        for arm in ARMS:
            row=rt['rows'][arm]
            for kind in ('train','source'):write(root/'configs'/(arm+'.'+kind+'.json'),row[kind+'_config'])
            for c in ('C0','P1'):write(root/'configs'/(arm+'.'+c+'.json'),row['development'][c]['predict_config'])
        job=dict(name='cache',argv=[rt['python'],'-B','-u','-m','experiments.pamf_phase3_open_world.source_practical_encoder_cache',
            '--config',str(root/'configs/cache.json'),'--device','cuda:0'],marker=str(root/'cache/complete.json'))
        _run_job(rt,root,job,choose_gpu(rt,[]),state,record)
        for arm in ARMS:
            gpu=choose_gpu(rt,[g for p,g in children if p.poll() is None])
            command=[rt['python'],'-B','-u','-m','experiments.pamf_phase3_open_world.source_practical_encoder_launch',
                '--runtime',str(Path(runtime_path).resolve()),'--row',arm,'--gpu',str(gpu)]
            stream=(root/'logs'/(arm+'.driver.log')).open('xb');streams.append(stream)
            child=subprocess.Popen(command,cwd=rt['release_root'],env=_env(rt,gpu),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            children.append((child,gpu));state['rows'][arm]=dict(status='RUNNING',pid=child.pid,gpu=gpu,cwd=rt['release_root'],argv=command);record()
        for arm,(child,gpu) in zip(ARMS,children):
            exitcode=child.wait();done=read(root/'rows'/arm/'pipeline_state.json')
            state['rows'][arm].update(status=done['status'],returncode=exitcode,finished_at=time.time());record()
        state.update(status='ANALYZED_DEVELOPMENT_ONLY' if all(r['status']=='ANALYZED_DEVELOPMENT_ONLY' for r in state['rows'].values()) else 'FAILED_PARTIAL',finished_at=time.time())
    except Exception as exc:
        # Any already healthy row is allowed to finish. Never kill foreign/other rows.
        for child,_ in children:child.wait()
        state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),finished_at=time.time())
    finally:
        for stream in streams:stream.close()
    record();return state


def launch(rt,runtime_path):
    validate_runtime(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    for name in ('logs','configs','rows'):(root/name).mkdir()
    command=[rt['python'],'-B','-u','-m','experiments.pamf_phase3_open_world.source_practical_encoder_launch','--runtime',str(Path(runtime_path).resolve()),'--dispatch']
    with (root/'logs/driver.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=rt['release_root'],env=_env(rt,None),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(status='LAUNCH_RETURNED_REQUIRES_POST_STATE',run_id=RUN,pid=child.pid,cwd=rt['release_root'],argv=command,code_commit=rt['code_commit'],owner=rt['launch_owner'])
    write(root/'launch.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True);p.add_argument('--dispatch',action='store_true');p.add_argument('--row',choices=ARMS);p.add_argument('--gpu',type=int)
    a=p.parse_args(argv);rt=read(a.runtime)
    result=row_dispatch(rt,a.row,a.gpu) if a.row else dispatch(rt,a.runtime) if a.dispatch else launch(rt,a.runtime)
    print(result['status'],flush=True);return int(result['status'] in ('FAILED','FAILED_PARTIAL'))


if __name__=='__main__':raise SystemExit(main())
