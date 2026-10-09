import argparse
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.cvs_multi_action_audit import design as d
from experiments.cvs_multi_disentangle.dispatch import capacity
from experiments.cvs_receiver_residual_capacity4.control import lock

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--row',required=True);a=p.parse_args()
    row=next(r for r in d.rows() if r['row_id']==a.row);c=d.config(row)
    if d.read(d.BASE/'configs'/(a.row+'.json'))!=c:raise ValueError('Worker config changed')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);stable=0
    while stable<3:
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            cap=capacity()[gpu];ready=len(cap['pids']|{os.getpid()})<=4 and cap['free_mb']>=12000
            stable=stable+1 if ready else 0
        if stable<3:time.sleep(1 if ready else 5)
    print('CAPACITY_READY',os.getpid(),gpu,flush=True)
    from experiments.cvs_multi_action_audit.runner import run
    run(c)
