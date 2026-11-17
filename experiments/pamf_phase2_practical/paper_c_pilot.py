"""One-seed paper-version A/C pilot, with a separate truth-last scorer."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .io import read, require, write

PROFILES=('lora_paper_v2','orthogonal_paper_v2')


def run_stage(rt,stage):
    root=Path(rt['run_root'])
    profile=rt['pilot_profile']
    seed=int(rt['pilot_seed'])
    if stage=='source':
        from .group1 import source_scope
        from .group1_source import train
        return train(source_scope(rt),profile,seed,root/'source'/f'{profile}-s{seed}')
    if stage=='predict':
        from .group1_target import predict
        return predict(rt,dict(profile=profile,seed=seed,capsule=rt['pilot_capsule'],
                               paper_c_points=[[5,5]],output=str(root/'predictions')))
    if stage=='score':
        from .episodic_score import run
        return run(rt,dict(prediction=str(root/'predictions'),output=str(root/'scores')))
    raise ValueError('Unknown pilot stage')


def dispatch(rt,runtime_path):
    root=Path(rt['run_root'])
    profile=rt['pilot_profile']
    require(profile in PROFILES,'Unexpected pilot profile')
    require(rt.get('pilot_capsule')=='capsules/P01/MIX01/1-1','Unexpected pilot capsule')
    require(int(rt.get('pilot_seed',0))==2026092701,'Unexpected pilot seed')
    require(not rt.get('clean_root'),'Pilot uses satellite-received data')
    root.mkdir(parents=True,exist_ok=False)
    write(root/'owner.json',dict(pid=os.getpid(),profile=profile,runtime=str(runtime_path)))
    artifacts={'source':(root/'source'/f"{profile}-s{rt['pilot_seed']}"/'group1_complete.json','GROUP1_SOURCE_READY'),
               'predict':(root/'predictions'/'complete.json','PREDICTIONS_COMPLETE'),
               'score':(root/'scores'/'complete.json','SCORED_COMPLETE')}
    for stage in ('source','predict','score'):
        env=dict(os.environ)
        if stage=='score':env['CUDA_VISIBLE_DEVICES']=''
        command=[sys.executable,'-u','-m','experiments.pamf_phase2_practical.paper_c_pilot',
                 '--runtime',str(runtime_path),'--stage',stage]
        print('PILOT_STAGE_START',profile,stage,flush=True)
        subprocess.run(command,cwd=rt['release_root'],env=env,check=True)
        artifact,expected=artifacts[stage]
        require(artifact.is_file() and read(artifact).get('status')==expected,
                f'{stage} did not produce its expected completion')
        print('PILOT_STAGE_COMPLETE',profile,stage,flush=True)
    write(root/'complete.json',dict(status='SCORED_COMPLETE',profile=profile,seed=rt['pilot_seed'],
                                   capsule=rt['pilot_capsule'],truth_last=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',required=True)
    p.add_argument('--stage',choices=('source','predict','score'))
    a=p.parse_args()
    rt=read(a.runtime)
    if a.stage:run_stage(rt,a.stage)
    else:dispatch(rt,a.runtime)


if __name__=='__main__':
    main()
