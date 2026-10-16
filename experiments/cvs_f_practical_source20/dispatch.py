import os
import subprocess
import sys
import time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def dispatch():
    with lock(d.ROOT/'F_practical_source.owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Preserve immutable diagnostic output')
        spec=d.read(d.ROOT/'experiments/cvs_f_practical_source20/registration.json')
        if [v['config'] for v in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Fixed source diagnostic matrix differs')
        from .provenance import check
        for row in d.rows():check(d.config(row))
        d.BASE.mkdir();(d.BASE/'logs').mkdir();d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        child=None
        while child is None:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                if choices:
                    gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g))
                    cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_f_practical_source20/runner.py'),'--worker']
                    log=d.BASE/'logs/worker.log'
                    with log.open('x') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    d.write(d.BASE/'launch.json',dict(rows=[r['row_id'] for r in d.rows()],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)))
            if child is None:time.sleep(3)
        code=child.wait()
        if code or not (d.BASE/'completion.json').is_file():
            d.write(d.BASE/'dispatcher_failure.json',dict(status='FAILED',exit_code=code,no_automatic_retry=True));raise RuntimeError('Own frozen diagnostic failure preserved')
