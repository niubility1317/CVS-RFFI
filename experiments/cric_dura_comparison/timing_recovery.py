"""One failed timing recovery; preserve the running scientific controller."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .io import read,write


def run(rt):
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    from .source_queue import capacity
    # A real allocation holds an otherwise idle card below the already-agreed
    # 12GB admission floor. Existing workers are never stopped or joined.
    # The reservation is recorded separately from model/incremental memory.
    write(root/'controller.json',dict(pid=os.getpid(),cwd=os.getcwd(),status='WAITING_EXCLUSIVE_GPU'))
    while True:
        gpu=next((g for g in (0,1,4,5,6,7) if capacity(g)[0]==0 and capacity(g)[1]>=22000),None)
        if gpu is None:time.sleep(60);continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        cmd=[rt['python'],'-u','-m','experiments.cric_dura_comparison.timing_recovery','worker','--config',rt['runtime_file']]
        with (root/'timing.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'submit.json',dict(pid=child.pid,gpu=gpu,cwd=rt['release_root'],argv=cmd,owner_pid=os.getpid()))
        return


def worker(rt):
    from .bootstrap import activate,numerical_policy
    activate(rt);numerical_policy()
    import torch
    from .timing import exclusive,run as measure
    torch.cuda.init();torch.cuda.set_per_process_memory_fraction(.95)
    exclusive()
    reservation=torch.empty(13*1024**3,dtype=torch.uint8,device='cuda:0')
    exclusive()
    c=read(rt['original_config']);c.update(run_id=rt['run_id'],output_root=str(Path(rt['run_root'])/'measurement'),code_commit=rt['code_commit'],
        scheduler_reservation_bytes=reservation.numel(),frozen_sources=rt['frozen_sources'])
    write(Path(rt['run_root'])/'gpu_ready.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],reservation_bytes=reservation.numel(),scheduler_policy='Other dispatchers require free>=12000MiB; reservation prevents sharing without terminating workloads'))
    measure(c)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['dispatch','worker']);p.add_argument('--config',required=True)
    a=p.parse_args();(run if a.mode=='dispatch' else worker)(read(a.config))
