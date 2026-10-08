"""Reserve a GPU before CUDA initialization, then exec with the same PID."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from experiments.cvs_receiver_residual import design as d
from experiments.cvs_receiver_residual.dispatch import capacity


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['source','predict'],required=True)
    p.add_argument('--config',required=True);p.add_argument('--row',required=True);a=p.parse_args()
    c=d.read(a.config);d.validate_config(c)
    if a.row!=d.row_id(c['arm'],c['model_seed']): raise ValueError('Worker row/config differs')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);pid=os.getpid();stable=0
    lease=Path(d.PROJECT)/'runs'/d.RUN/'capacity_leases'/(a.kind+'-'+a.row+'.json')
    while stable<3:
        caps=capacity();entry=caps[gpu];pids=entry['pids']|{pid}
        ready=len(pids)<=2 and entry['free_mb']>=12000
        stable=stable+1 if ready else 0
        d.write(lease,dict(pid=pid,gpu=gpu,observed_pids=sorted(pids),stable_checks=stable,
            status='CAPACITY_READY' if stable==3 else 'WAITING_CAPACITY',cuda_initialized=False,updated_at=time.time()))
        if stable<3: time.sleep(1 if ready else 5)
    command=[sys.executable,'-u','-m','experiments.cvs_receiver_residual.'+('source' if a.kind=='source' else 'evaluate')]
    command+=['--config',a.config] if a.kind=='source' else ['--mode','predict','--row',a.row]
    print('CAPACITY_READY '+json.dumps(dict(pid=pid,gpu=gpu)),flush=True)
    os.execv(sys.executable,command)


if __name__=='__main__': main()
