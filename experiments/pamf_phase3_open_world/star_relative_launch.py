"""One sequential owner: source freeze, complete four-state prediction, independent scoring."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .launch import choose_gpu,read,write
from .star_relative_core import RUN
from .source_practical_encoder_launch import _env,_run_job

FIELDS={'run_id','run_root','release_root','python','code_commit','launch_owner','allowed_gpus',
    'max_added_workers_per_gpu','minimum_free_mb','targets_path','source_config','rows'}


def validate_runtime(rt):
    from .star_relative_source import validate_config as source_validate
    from .star_relative_target import validate_config as target_validate
    if not isinstance(rt,dict) or set(rt)!=FIELDS:raise ValueError('Exact sequential registration runtime fields required')
    root=Path(rt['run_root'])
    if (rt['run_id']!=RUN or root.name!=RUN or not root.is_absolute() or not Path(rt['release_root']).is_absolute()
            or not Path(rt['targets_path']).is_absolute() or not rt['launch_owner']
            or len(rt['code_commit'])!=40 or any(x not in '0123456789abcdef' for x in rt['code_commit'])
            or type(rt['max_added_workers_per_gpu']) is not int or rt['max_added_workers_per_gpu']!=2
            or type(rt['minimum_free_mb']) is not int or rt['minimum_free_mb']<4096
            or not isinstance(rt['allowed_gpus'],list) or not rt['allowed_gpus']
            or any(type(g) is not int or g<0 for g in rt['allowed_gpus'])
            or len(set(rt['allowed_gpus']))!=len(rt['allowed_gpus']) or set(rt['rows'])!={'P0','P1'}):
        raise ValueError('Immutable identity and actual-memory/two-added-worker policy required')
    source=source_validate(rt['source_config'])
    if source['code_commit']!=rt['code_commit'] or Path(source['output_root'])!=root/'source':raise ValueError('Source identity/output differs')
    for c in ('P0','P1'):
        row=rt['rows'][c]
        if set(row)!={'predict_config','U_truth','Q_truth'}:raise ValueError('Public predictor and separate scorer paths required')
        cfg=target_validate(row['predict_config'])
        if (cfg['code_commit']!=rt['code_commit'] or cfg['condition']!=c or cfg['source_root']!=source['output_root']
                or Path(cfg['output_root'])!=root/'rows'/c/'predict'
                or any(not isinstance(row[k],str) or not Path(row[k]).is_absolute() for k in ('U_truth','Q_truth'))):
            raise ValueError('Condition/public-source identity or scorer path differs')
        if c=='P1' and any(cfg[k]!=source[k] for k in ('dura_checkpoint','dura_release')):raise ValueError('Frozen G differs')
    return rt


def commands(rt):
    root=Path(rt['run_root']);p=[rt['python'],'-B','-u','-m'];jobs=[]
    def add(name,argv,marker,gpu):jobs.append(dict(name=name,argv=argv,marker=str(marker),gpu=gpu))
    add('source',p+['experiments.pamf_phase3_open_world.star_relative_source','run','--config',str(root/'configs/source.json'),'--device','cuda:0'],root/'source/complete.json',True)
    for c in ('P0','P1'):
        base=root/'rows'/c
        add(c+'_predict',p+['experiments.pamf_phase3_open_world.star_relative_target','--config',str(root/'configs'/(c+'.json')),'--device','cuda:0'],base/'predict/complete.json',True)
        add(c+'_score',p+['experiments.pamf_phase3_open_world.star_relative_score','--predictions-root',str(base/'predict'),
            '--U-truth',rt['rows'][c]['U_truth'],'--Q-truth',rt['rows'][c]['Q_truth'],'--targets',rt['targets_path'],'--output',str(base/'score')],base/'score/complete.json',False)
    return jobs


def dispatch(rt):
    validate_runtime(rt);root=Path(rt['run_root'])
    state=dict(status='RUNNING',run_id=RUN,code_commit=rt['code_commit'],owner=rt['launch_owner'],pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        started_at=time.time(),jobs={},foreign_processes_stopped=0,max_added_workers_per_gpu=2,max_concurrent_gpu_workers=1,
        formal_target_reads=0,final_phase3_complete=False)
    def record():write(root/'pipeline_state.json',state,exclusive=False)
    record()
    try:
        write(root/'configs/source.json',rt['source_config'])
        for c in ('P0','P1'):write(root/'configs'/(c+'.json'),rt['rows'][c]['predict_config'])
        for job in commands(rt):_run_job(rt,root,job,choose_gpu(rt) if job['gpu'] else None,state,record)
        state.update(status='ANALYZED_DEVELOPMENT_ONLY',finished_at=time.time())
    except Exception as exc:state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),finished_at=time.time())
    record();return state


def launch(rt,runtime_path):
    validate_runtime(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    for name in ('configs','logs','rows'):(root/name).mkdir()
    argv=[rt['python'],'-B','-u','-m','experiments.pamf_phase3_open_world.star_relative_launch','--runtime',str(Path(runtime_path).resolve()),'--dispatch']
    with (root/'logs/driver.log').open('xb') as stream:
        child=subprocess.Popen(argv,cwd=rt['release_root'],env=_env(rt,None),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(status='LAUNCH_RETURNED_REQUIRES_POST_STATE',run_id=RUN,pid=child.pid,cwd=rt['release_root'],argv=argv,code_commit=rt['code_commit'],owner=rt['launch_owner'])
    write(root/'launch.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True);p.add_argument('--dispatch',action='store_true')
    a=p.parse_args(argv);rt=read(a.runtime);result=dispatch(rt) if a.dispatch else launch(rt,a.runtime)
    print(result['status'],flush=True);return int(result['status']=='FAILED')


if __name__=='__main__':raise SystemExit(main())
