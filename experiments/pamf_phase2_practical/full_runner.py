"""Data-only controller: no identity training, prediction or scorer dispatch."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import read,atomic


def run(runtime):
    rt=read(runtime);release=Path.cwd();status=release/'preparation_status.json'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    started=time.time()
    for mode in ('inventory','select','audit','build'):
        command=[sys.executable,'-u','-m','experiments.pamf_phase2_practical.full_data',mode,'--runtime',str(Path(runtime).resolve())]
        with (release/(mode+'.log')).open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic(status,dict(status='RUNNING',stage=mode,pid=child.pid,controller_pid=os.getpid(),
                               run_root=rt['run_root'],code_commit=rt['code_commit'],elapsed_seconds=time.time()-started))
            result=child.wait()
        if result:
            atomic(status,dict(status='FAILED',stage=mode,exit_code=result,controller_pid=os.getpid(),
                               run_root=rt['run_root'],code_commit=rt['code_commit'],elapsed_seconds=time.time()-started))
            raise SystemExit(result)
    atomic(status,dict(status='DATA_COMPLETE',run_root=rt['run_root'],code_commit=rt['code_commit'],
                       elapsed_seconds=time.time()-started,identity_training=0,identity_predictions=0,scoring=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True)
    run(p.parse_args().runtime)
